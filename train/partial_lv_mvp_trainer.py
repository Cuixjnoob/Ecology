from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.optim import AdamW

from models.partial_lv_recovery_model import PartialLVRecoveryModel
from train.trainer import resolve_device


@dataclass
class FitResult:
    history: List[Dict[str, float]]
    best_epoch: int
    best_val_score: float


def _mean_pearson(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.reshape(-1, prediction.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    prediction_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(
        prediction_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)
    ).clamp_min(1e-8)
    values = (prediction_centered * target_centered).sum(dim=0) / denominator
    return float(values.mean().item())


def _rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt((prediction - target).square().mean()).item())


def _corr_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)
    numerator = (a_centered * b_centered).mean(dim=1)
    denominator = torch.sqrt(a_centered.square().mean(dim=1) * b_centered.square().mean(dim=1)).clamp_min(1e-6)
    return numerator / denominator


class PartialLVMVPTrainer:
    def __init__(
        self,
        model: PartialLVRecoveryModel,
        true_interaction_matrix: torch.Tensor,
        visible_log_mean: torch.Tensor,
        visible_log_std: torch.Tensor,
        visible_series_raw: torch.Tensor,
        hidden_series_raw: torch.Tensor,
        split_ranges: Dict[str, tuple[int, int]],
        history_length: int,
        config: Dict[str, Any],
        noise_profile: Dict[str, float],
        particle_rollout_k: int,
    ) -> None:
        self.model = model
        self.config = config
        self.train_cfg = config["train"]
        self.loss_cfg = config["loss"]
        self.data_cfg = config["data"]
        self.noise_cfg = config["noise"]
        self.noise_profile = noise_profile
        self.particle_rollout_k = int(particle_rollout_k)
        self.train_particles = int(self.noise_cfg.get("train_particles", 1))
        self.eval_particles = int(self.noise_cfg.get("eval_particles", min(self.particle_rollout_k, 2)))

        self.device = resolve_device(str(self.train_cfg["device"]))
        self.model.to(self.device)
        self.model.set_visible_normalization(
            mean=visible_log_mean.to(self.device),
            std=visible_log_std.to(self.device),
        )
        self.true_interaction_matrix = true_interaction_matrix.to(self.device)
        self.visible_series_raw = visible_series_raw.to(torch.float32)
        self.hidden_series_raw = hidden_series_raw.to(torch.float32)
        self.split_ranges = split_ranges
        self.history_length = int(history_length)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.train_cfg["learning_rate"]),
            weight_decay=float(self.train_cfg["weight_decay"]),
        )
        self.current_epoch = 0
        self.total_epochs = max(int(self.train_cfg["epochs"]), 1)

    def _log_values(self, values: torch.Tensor) -> torch.Tensor:
        return torch.log1p(values.clamp_min(0.0))

    def _apply_history_noise(self, history_visible_raw: torch.Tensor, training: bool) -> torch.Tensor:
        if not training:
            return history_visible_raw
        std = float(self._current_noise_profile().get("history_jitter", 0.0))
        if std <= 0.0:
            return history_visible_raw
        scale = history_visible_raw.std(dim=1, keepdim=True, unbiased=False) + 0.05 * history_visible_raw.mean(
            dim=1,
            keepdim=True,
        )
        noise = std * scale * torch.randn_like(history_visible_raw)
        return torch.clamp(history_visible_raw + noise, min=1e-4)

    def _forward_model(
        self,
        history_visible_raw: torch.Tensor,
        rollout_horizon: int,
        training: bool,
        num_particles: int | None = None,
        stochastic_override: bool | None = None,
        full_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        history_input = self._apply_history_noise(history_visible_raw, training=training)
        current_noise = self._current_noise_profile(full_context=full_context) if training else self.noise_profile
        process_noise_scale = float(current_noise.get("rollout_process_noise", 0.0))
        latent_perturb_scale = float(current_noise.get("latent_perturb", 0.0)) if training else 0.0
        stochastic = process_noise_scale > 0.0 if stochastic_override is None else stochastic_override
        return self.model(
            history_visible_raw=history_input.to(self.device),
            rollout_horizon=rollout_horizon,
            num_particles=int(num_particles or 1),
            stochastic=stochastic,
            process_noise_scale=process_noise_scale if stochastic else 0.0,
            latent_perturb_scale=latent_perturb_scale,
        )

    def _current_noise_profile(self, full_context: bool = False) -> Dict[str, float]:
        progress = float(self.current_epoch) / max(self.total_epochs - 1, 1)
        anneal = max(0.25, 1.0 - 0.70 * progress)
        profile = {
            "history_jitter": float(self.noise_profile.get("history_jitter", 0.0)) * anneal,
            "rollout_process_noise": float(self.noise_profile.get("rollout_process_noise", 0.0)) * anneal,
            "latent_perturb": float(self.noise_profile.get("latent_perturb", 0.0)) * anneal,
        }
        if full_context:
            profile["history_jitter"] *= 0.8
            profile["rollout_process_noise"] *= 0.55
            profile["latent_perturb"] *= 0.65
        return profile

    def _peak_weights(self, future_visible: torch.Tensor) -> torch.Tensor:
        quantile = float(self.loss_cfg.get("peak_quantile", 0.75))
        thresholds = torch.quantile(future_visible, q=quantile, dim=1, keepdim=True)
        return 1.0 + 2.0 * (future_visible >= thresholds).float()

    def _peak_visible_error(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        weights = self._peak_weights(target)
        squared = (self._log_values(prediction) - self._log_values(target)).square()
        weighted = (squared * weights).sum() / weights.sum().clamp_min(1.0)
        return float(torch.sqrt(weighted).item())

    def _amplitude_collapse_score(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        pred_range = prediction.max(dim=1).values - prediction.min(dim=1).values
        true_range = target.max(dim=1).values - target.min(dim=1).values
        ratio = pred_range / (true_range + 1e-6)
        collapse = torch.relu(1.0 - ratio)
        return float(collapse.mean().item())

    def _sequence_diagnostics(
        self,
        hidden_series: torch.Tensor,
        environment_series: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if hidden_series.dim() == 2:
            hidden_series = hidden_series.unsqueeze(0)
        if environment_series.dim() == 2:
            environment_series = environment_series.unsqueeze(0)

        hidden_log = self._log_values(hidden_series)
        environment = environment_series

        hidden_correlation = _corr_tensor(hidden_log, environment).mean()

        if hidden_log.shape[1] > 1:
            hidden_lag = _corr_tensor(hidden_log[:, 1:, :], hidden_log[:, :-1, :]).mean()
            environment_lag = _corr_tensor(environment[:, 1:, :], environment[:, :-1, :]).mean()
            hidden_diff = hidden_log[:, 1:, :] - hidden_log[:, :-1, :]
            environment_diff = environment[:, 1:, :] - environment[:, :-1, :]
            hidden_roughness = hidden_diff.abs().mean()
            environment_roughness = environment_diff.abs().mean()
            diff_correlation = _corr_tensor(hidden_diff, environment_diff).mean()
        else:
            zero = hidden_log.new_tensor(0.0)
            hidden_lag = zero
            environment_lag = zero
            hidden_roughness = zero
            environment_roughness = zero
            diff_correlation = zero

        hidden_std = hidden_log.std(dim=1, unbiased=False).mean()
        environment_std = environment.std(dim=1, unbiased=False).mean()
        return {
            "hidden_environment_correlation": hidden_correlation,
            "hidden_autocorrelation": hidden_lag,
            "environment_autocorrelation": environment_lag,
            "hidden_roughness": hidden_roughness,
            "environment_roughness": environment_roughness,
            "hidden_std": hidden_std,
            "environment_std": environment_std,
            "diff_correlation": diff_correlation,
        }

    def _visible_loss_terms(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_log = self._log_values(prediction)
        target_log = self._log_values(target)
        weights = self._peak_weights(target).to(prediction.device)

        visible_one_step = torch.nn.functional.mse_loss(pred_log[:, 0, :], target_log[:, 0, :])
        visible_rollout = torch.nn.functional.mse_loss(pred_log, target_log)
        peak_visible = (weights * (pred_log - target_log).square()).mean()

        if prediction.shape[1] > 1:
            pred_diff = pred_log[:, 1:, :] - pred_log[:, :-1, :]
            true_diff = target_log[:, 1:, :] - target_log[:, :-1, :]
            slope_magnitude = torch.nn.functional.l1_loss(pred_diff, true_diff)
            slope_direction = torch.relu(-(pred_diff * true_diff)).mean()
            slope_loss = slope_magnitude + 0.7 * slope_direction
        else:
            slope_loss = prediction.new_tensor(0.0)

        pred_range = prediction.max(dim=1).values - prediction.min(dim=1).values
        true_range = target.max(dim=1).values - target.min(dim=1).values
        pred_std = prediction.std(dim=1, unbiased=False)
        true_std = target.std(dim=1, unbiased=False)
        amplitude_loss = (
            torch.relu(true_range - pred_range).mean() / (true_range.mean() + 1e-6)
            + 0.5 * torch.relu(true_std - pred_std).mean() / (true_std.mean() + 1e-6)
        )

        # Multi-scale difference loss
        multiscale_loss = prediction.new_tensor(0.0)
        for scale in [2, 4]:
            if prediction.shape[1] > scale:
                pred_diff_s = pred_log[:, scale:, :] - pred_log[:, :-scale, :]
                true_diff_s = target_log[:, scale:, :] - target_log[:, :-scale, :]
                multiscale_loss = multiscale_loss + torch.nn.functional.l1_loss(pred_diff_s, true_diff_s)
        multiscale_loss = multiscale_loss / 2.0

        # Local variance preservation loss
        window_size = min(6, prediction.shape[1])
        if prediction.shape[1] >= window_size:
            num_windows = prediction.shape[1] - window_size + 1
            sample_indices = list(range(0, num_windows, max(1, num_windows // 4)))
            pred_local_var = torch.stack([
                pred_log[:, i:i + window_size, :].var(dim=1, unbiased=False)
                for i in sample_indices
            ], dim=1)
            true_local_var = torch.stack([
                target_log[:, i:i + window_size, :].var(dim=1, unbiased=False)
                for i in sample_indices
            ], dim=1)
            local_var_loss = torch.relu(true_local_var - pred_local_var).mean() / (true_local_var.mean() + 1e-6)
        else:
            local_var_loss = prediction.new_tensor(0.0)

        return {
            "visible_one_step": visible_one_step,
            "visible_rollout": visible_rollout,
            "peak_visible": peak_visible,
            "slope": slope_loss,
            "amplitude": amplitude_loss,
            "multiscale": multiscale_loss,
            "local_variance": local_var_loss,
        }

    def _compose_visible_loss(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            float(self.loss_cfg["lambda_visible_one_step"]) * terms["visible_one_step"]
            + float(self.loss_cfg["lambda_visible_rollout"]) * terms["visible_rollout"]
            + float(self.loss_cfg["lambda_peak_visible"]) * terms["peak_visible"]
            + float(self.loss_cfg["lambda_slope"]) * terms["slope"]
            + float(self.loss_cfg["lambda_amplitude"]) * terms["amplitude"]
            + float(self.loss_cfg.get("lambda_multiscale", 0.0)) * terms.get("multiscale", terms["slope"].new_tensor(0.0))
            + float(self.loss_cfg.get("lambda_local_variance", 0.0)) * terms.get("local_variance", terms["slope"].new_tensor(0.0))
        )

    def _hidden_terms(
        self,
        hidden_initial: torch.Tensor,
        hidden_predictions: torch.Tensor,
        current_hidden: torch.Tensor,
        future_hidden: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden_init_log = self._log_values(hidden_initial)
        hidden_current_log = self._log_values(current_hidden)
        hidden_pred_log = self._log_values(hidden_predictions)
        hidden_true_log = self._log_values(future_hidden)

        hidden_initial_loss = torch.nn.functional.mse_loss(hidden_init_log, hidden_current_log)
        hidden_rollout_loss = torch.nn.functional.mse_loss(hidden_pred_log, hidden_true_log)
        if hidden_predictions.shape[1] > 2:
            hidden_smooth = (
                hidden_pred_log[:, 2:, :]
                - 2 * hidden_pred_log[:, 1:-1, :]
                + hidden_pred_log[:, :-2, :]
            ).square().mean()
        else:
            hidden_smooth = hidden_predictions.new_tensor(0.0)
        return {
            "hidden_initial": hidden_initial_loss,
            "hidden_rollout": hidden_rollout_loss,
            "hidden_smooth": hidden_smooth,
        }

    def _interaction_terms(self, interaction_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        interaction_true = self.true_interaction_matrix
        hidden_edge_pred = torch.cat([interaction_pred[:5, 5], interaction_pred[5, :5]], dim=0)
        hidden_edge_true = torch.cat([interaction_true[:5, 5], interaction_true[5, :5]], dim=0)
        interaction_hidden = torch.nn.functional.mse_loss(hidden_edge_pred, hidden_edge_true)
        interaction_sparse = (interaction_pred - torch.diag(torch.diagonal(interaction_pred))).abs().mean()
        return {
            "interaction_hidden": interaction_hidden,
            "interaction_sparse": interaction_sparse,
        }

    def _environment_terms(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        diagnostics = self._sequence_diagnostics(
            hidden_series=outputs["hidden_predictions"],
            environment_series=outputs["environment_predictions"],
        )
        initial_diagnostics = self._sequence_diagnostics(
            hidden_series=outputs["hidden_initial"],
            environment_series=outputs["environment_initial"],
        )
        environment = outputs["environment_predictions"]
        if environment.shape[1] > 1:
            env_smooth = (environment[:, 1:, :] - environment[:, :-1, :]).square().mean()
            env_curvature = (
                environment[:, 2:, :]
                - 2 * environment[:, 1:-1, :]
                + environment[:, :-2, :]
            ).square().mean() if environment.shape[1] > 2 else environment.new_tensor(0.0)
        else:
            env_smooth = environment.new_tensor(0.0)
            env_curvature = environment.new_tensor(0.0)
        env_stability = torch.relu(environment.abs() - 1.8).square().mean()

        orthogonality = diagnostics["diff_correlation"].square()
        hidden_floor = float(self.loss_cfg.get("hidden_variance_floor", 0.09))
        environment_floor = float(self.loss_cfg.get("environment_variance_floor", 0.08))
        variance_floor = (
            torch.relu(hidden_floor - diagnostics["hidden_std"]).square()
            + torch.relu(environment_floor - diagnostics["environment_std"]).square()
            + 0.5 * torch.relu(hidden_floor - initial_diagnostics["hidden_std"]).square()
            + 0.5 * torch.relu(environment_floor - initial_diagnostics["environment_std"]).square()
        )
        smoother_ratio = float(self.loss_cfg.get("environment_smoother_ratio", 0.70))
        timescale_prior = torch.relu(
            diagnostics["environment_roughness"] - smoother_ratio * diagnostics["hidden_roughness"]
        ).square()
        autocorr_margin = float(self.loss_cfg.get("environment_autocorr_margin", 0.05))
        timescale_prior = timescale_prior + torch.relu(
            diagnostics["hidden_autocorrelation"] - diagnostics["environment_autocorrelation"] + autocorr_margin
        ).square()
        timescale_prior = timescale_prior + 0.5 * torch.relu(
            initial_diagnostics["hidden_autocorrelation"]
            - initial_diagnostics["environment_autocorrelation"]
            + autocorr_margin
        ).square()
        timescale_prior = timescale_prior + 0.5 * env_curvature

        disentangle = (
            diagnostics["hidden_environment_correlation"].square()
            + 0.85 * initial_diagnostics["hidden_environment_correlation"].square()
            + 0.7 * orthogonality
        )
        return {
            "env_smooth": env_smooth,
            "env_stability": env_stability,
            "disentangle": disentangle,
            "orthogonality": orthogonality,
            "variance_floor": variance_floor,
            "timescale_prior": timescale_prior,
        }

    def _lv_terms(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lv_contrib = outputs["lv_contribution_history"]
        residual_contrib = outputs["residual_contribution_history"]
        deterministic_next = outputs["deterministic_prediction_history"]
        lv_only_next = outputs["lv_only_prediction_history"]

        lv_norm = lv_contrib.norm(dim=-1)
        residual_norm = residual_contrib.norm(dim=-1)
        residual_ratio = residual_norm / (lv_norm + 1e-6)
        # Energy-based LV/residual balance
        lv_energy = lv_norm.square().mean()
        res_energy = residual_norm.square().mean()
        total_energy = lv_energy + res_energy + 1e-8
        lv_energy_fraction = lv_energy / total_energy
        residual_energy_penalty = torch.relu(lv_norm.new_tensor(0.55) - lv_energy_fraction).square()

        residual_magnitude = (
            torch.relu(residual_ratio - 0.85).square().mean()
            + 0.8 * (residual_ratio > 1.0).float().mean()
            + residual_energy_penalty
        )

        lv_guidance = torch.nn.functional.mse_loss(
            self._log_values(deterministic_next),
            self._log_values(lv_only_next),
        )
        return {
            "lv_guidance": lv_guidance,
            "residual_magnitude": residual_magnitude,
            "residual_energy": residual_energy_penalty,
            "lv_energy_fraction": lv_energy_fraction.detach(),
        }

    def _particle_terms(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        visible_particles = outputs["visible_particles"]
        if visible_particles.shape[1] <= 1:
            zero = visible_particles.new_tensor(0.0)
            return {"particle_consistency": zero}
        particle_std = visible_particles.std(dim=1, unbiased=False)
        particle_consistency = (
            torch.relu(particle_std.mean() - 0.30).square()
            + torch.relu(0.01 - particle_std.mean()).square()
        )
        return {"particle_consistency": particle_consistency}

    def _loss_bundle(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        future_visible = batch["future_raw"].to(self.device)
        future_hidden = batch["future_hidden"].to(self.device)
        current_hidden = batch["history_hidden"][:, -1, :].to(self.device)

        visible_terms = self._visible_loss_terms(outputs["visible_predictions"], future_visible)
        hidden_terms = self._hidden_terms(
            hidden_initial=outputs["hidden_initial"],
            hidden_predictions=outputs["hidden_predictions"],
            current_hidden=current_hidden,
            future_hidden=future_hidden,
        )
        interaction_terms = self._interaction_terms(outputs["interaction_matrix"])
        environment_terms = self._environment_terms(outputs)
        lv_terms = self._lv_terms(outputs)
        particle_terms = self._particle_terms(outputs)

        total = (
            self._compose_visible_loss(visible_terms)
            + float(self.loss_cfg["lambda_hidden_initial"]) * hidden_terms["hidden_initial"]
            + float(self.loss_cfg["lambda_hidden_rollout"]) * hidden_terms["hidden_rollout"]
            + float(self.loss_cfg["lambda_interaction_hidden"]) * interaction_terms["interaction_hidden"]
            + float(self.loss_cfg["lambda_interaction_sparse"]) * interaction_terms["interaction_sparse"]
            + float(self.loss_cfg["lambda_hidden_smooth"]) * hidden_terms["hidden_smooth"]
            + float(self.loss_cfg["lambda_env_smooth"]) * environment_terms["env_smooth"]
            + float(self.loss_cfg["lambda_env_stability"]) * environment_terms["env_stability"]
            + float(self.loss_cfg["lambda_disentangle"]) * environment_terms["disentangle"]
            + float(self.loss_cfg.get("lambda_orthogonality", 0.0)) * environment_terms["orthogonality"]
            + float(self.loss_cfg.get("lambda_variance_floor", 0.0)) * environment_terms["variance_floor"]
            + float(self.loss_cfg.get("lambda_timescale_prior", 0.0)) * environment_terms["timescale_prior"]
            + float(self.loss_cfg["lambda_lv_guidance"]) * lv_terms["lv_guidance"]
            + float(self.loss_cfg["lambda_residual_magnitude"]) * lv_terms["residual_magnitude"]
            + float(self.loss_cfg.get("lambda_residual_energy", 0.0)) * lv_terms["residual_energy"]
            + float(self.loss_cfg.get("lambda_particle_consistency", 0.0)) * particle_terms["particle_consistency"]
        )

        return {
            "total": total,
            **visible_terms,
            **hidden_terms,
            **interaction_terms,
            **environment_terms,
            **lv_terms,
            **particle_terms,
        }

    def _full_context_train_segment(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, train_end = self.split_ranges["train"]
        ratio = float(self.data_cfg.get("full_context_train_ratio", 0.72))
        context_end = int(train_end * ratio)
        context_end = max(self.history_length, min(context_end, train_end - 6))
        history_visible = self.visible_series_raw[:context_end]
        future_visible = self.visible_series_raw[context_end:train_end]
        future_hidden = self.hidden_series_raw[context_end:train_end]
        return history_visible, future_visible, future_hidden

    def _full_context_eval_segment(self, split_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_start, train_end = self.split_ranges["train"]
        _, val_end = self.split_ranges["val"]
        split_start, split_end = self.split_ranges[split_name]

        if split_name == "val":
            history_visible = self.visible_series_raw[train_start:train_end]
        elif split_name == "test":
            history_visible = self.visible_series_raw[train_start:val_end]
        else:
            raise ValueError(f"Unsupported split for full-context evaluation: {split_name}")

        future_visible = self.visible_series_raw[split_start:split_end]
        future_hidden = self.hidden_series_raw[split_start:split_end]
        return history_visible, future_visible, future_hidden

    def _full_context_train_step(self) -> float:
        history_visible, future_visible, future_hidden = self._full_context_train_segment()
        outputs = self._forward_model(
            history_visible_raw=history_visible.unsqueeze(0),
            rollout_horizon=int(future_visible.shape[0]),
            training=True,
            num_particles=self.train_particles,
            full_context=True,
        )
        visible_terms = self._visible_loss_terms(outputs["visible_predictions"], future_visible.unsqueeze(0).to(self.device))
        hidden_terms = self._hidden_terms(
            hidden_initial=outputs["hidden_initial"],
            hidden_predictions=outputs["hidden_predictions"],
            current_hidden=self.hidden_series_raw[history_visible.shape[0] - 1 : history_visible.shape[0]].to(self.device),
            future_hidden=future_hidden.unsqueeze(0).to(self.device),
        )
        environment_terms = self._environment_terms(outputs)
        lv_terms = self._lv_terms(outputs)

        full_context_loss = (
            float(self.loss_cfg["lambda_full_context_visible"]) * self._compose_visible_loss(visible_terms)
            + float(self.loss_cfg["lambda_full_context_hidden"]) * hidden_terms["hidden_rollout"]
            + float(self.loss_cfg["lambda_hidden_smooth"]) * 0.5 * hidden_terms["hidden_smooth"]
            + float(self.loss_cfg["lambda_env_smooth"]) * 0.5 * environment_terms["env_smooth"]
            + float(self.loss_cfg["lambda_disentangle"]) * 0.5 * environment_terms["disentangle"]
            + float(self.loss_cfg.get("lambda_orthogonality", 0.0)) * 0.5 * environment_terms["orthogonality"]
            + float(self.loss_cfg.get("lambda_timescale_prior", 0.0)) * 0.5 * environment_terms["timescale_prior"]
            + float(self.loss_cfg["lambda_lv_guidance"]) * 0.5 * lv_terms["lv_guidance"]
            + float(self.loss_cfg.get("lambda_residual_energy", 0.0)) * 0.5 * lv_terms["residual_energy"]
        )

        self.optimizer.zero_grad()
        full_context_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_cfg["grad_clip"]))
        self.optimizer.step()
        return float(full_context_loss.detach().cpu().item())

    def _iterate_loader(self, data_loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        aggregate = {
            "total": 0.0,
            "visible_one_step": 0.0,
            "visible_rollout": 0.0,
            "peak_visible": 0.0,
            "slope": 0.0,
            "amplitude": 0.0,
            "hidden_initial": 0.0,
            "hidden_rollout": 0.0,
            "interaction_hidden": 0.0,
            "interaction_sparse": 0.0,
            "hidden_smooth": 0.0,
            "env_smooth": 0.0,
            "env_stability": 0.0,
            "disentangle": 0.0,
            "orthogonality": 0.0,
            "variance_floor": 0.0,
            "timescale_prior": 0.0,
            "lv_guidance": 0.0,
            "residual_magnitude": 0.0,
            "residual_energy": 0.0,
            "lv_energy_fraction": 0.0,
            "multiscale": 0.0,
            "local_variance": 0.0,
            "particle_consistency": 0.0,
        }
        batches = 0

        for batch in data_loader:
            outputs = self._forward_model(
                history_visible_raw=batch["history_raw"],
                rollout_horizon=int(batch["future_raw"].shape[1]),
                training=training,
                num_particles=self.train_particles if training else self.eval_particles,
            )
            losses = self._loss_bundle(batch, outputs)

            if training:
                self.optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_cfg["grad_clip"]))
                self.optimizer.step()

            for key in aggregate:
                aggregate[key] += float(losses[key].detach().cpu().item())
            batches += 1

        return {key: value / max(batches, 1) for key, value in aggregate.items()}

    @torch.no_grad()
    def _collect_ratio_metrics(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        lv_norm = outputs["lv_contribution_history"].norm(dim=-1)
        residual_norm = outputs["residual_contribution_history"].norm(dim=-1)
        ratio = residual_norm / (lv_norm + 1e-6)
        lv_energy = lv_norm.square().mean()
        res_energy = residual_norm.square().mean()
        lv_energy_fraction = float((lv_energy / (lv_energy + res_energy + 1e-8)).item())
        return {
            "lv_residual_ratio_mean": float(ratio.mean().item()),
            "lv_residual_ratio_std": float(ratio.std(unbiased=False).item()),
            "residual_dominates_fraction": float((ratio > 1.0).float().mean().item()),
            "lv_energy_fraction": lv_energy_fraction,
        }

    @torch.no_grad()
    def _validation_metrics(self, data_loader, num_particles: int | None = None) -> Dict[str, float]:
        self.model.eval()
        visible_pred = []
        visible_true = []
        hidden_pred = []
        hidden_true = []
        env_pred = []
        ratio_mean = []
        ratio_std = []
        dominate = []

        for batch in data_loader:
            outputs = self._forward_model(
                history_visible_raw=batch["history_raw"],
                rollout_horizon=int(batch["future_raw"].shape[1]),
                training=False,
                num_particles=num_particles or self.eval_particles,
            )
            visible_pred.append(outputs["visible_predictions"].cpu())
            visible_true.append(batch["future_raw"].cpu())
            hidden_pred.append(outputs["hidden_predictions"].cpu())
            hidden_true.append(batch["future_hidden"].cpu())
            env_pred.append(outputs["environment_predictions"].cpu())
            ratio_metrics = self._collect_ratio_metrics(outputs)
            ratio_mean.append(ratio_metrics["lv_residual_ratio_mean"])
            ratio_std.append(ratio_metrics["lv_residual_ratio_std"])
            dominate.append(ratio_metrics["residual_dominates_fraction"])

        visible_pred_tensor = torch.cat(visible_pred, dim=0)
        visible_true_tensor = torch.cat(visible_true, dim=0)
        hidden_pred_tensor = torch.cat(hidden_pred, dim=0)
        hidden_true_tensor = torch.cat(hidden_true, dim=0)
        env_pred_tensor = torch.cat(env_pred, dim=0)
        disent_metrics = self._sequence_diagnostics(hidden_pred_tensor, env_pred_tensor)

        return {
            "visible_rollout_rmse": _rmse(visible_pred_tensor, visible_true_tensor),
            "visible_rollout_pearson": _mean_pearson(visible_pred_tensor, visible_true_tensor),
            "hidden_rollout_rmse": _rmse(hidden_pred_tensor, hidden_true_tensor),
            "hidden_rollout_pearson": _mean_pearson(hidden_pred_tensor, hidden_true_tensor),
            "peak_visible_error": self._peak_visible_error(visible_pred_tensor, visible_true_tensor),
            "amplitude_collapse_score": self._amplitude_collapse_score(visible_pred_tensor, visible_true_tensor),
            "hidden_environment_correlation": float(disent_metrics["hidden_environment_correlation"].item()),
            "hidden_autocorrelation": float(disent_metrics["hidden_autocorrelation"].item()),
            "environment_autocorrelation": float(disent_metrics["environment_autocorrelation"].item()),
            "hidden_roughness": float(disent_metrics["hidden_roughness"].item()),
            "environment_roughness": float(disent_metrics["environment_roughness"].item()),
            "lv_residual_ratio_mean": float(sum(ratio_mean) / max(len(ratio_mean), 1)),
            "lv_residual_ratio_std": float(sum(ratio_std) / max(len(ratio_std), 1)),
            "residual_dominates_fraction": float(sum(dominate) / max(len(dominate), 1)),
        }

    @torch.no_grad()
    def evaluate_loader(self, data_loader, num_particles: int | None = None) -> Dict[str, float]:
        return self._validation_metrics(data_loader, num_particles=num_particles)

    @torch.no_grad()
    def evaluate_full_context(self, split_name: str, num_particles: int | None = None) -> Dict[str, Any]:
        self.model.eval()
        history_visible, future_visible, future_hidden = self._full_context_eval_segment(split_name)
        outputs = self._forward_model(
            history_visible_raw=history_visible.unsqueeze(0),
            rollout_horizon=int(future_visible.shape[0]),
            training=False,
            num_particles=num_particles or self.particle_rollout_k,
        )
        visible_pred = outputs["visible_predictions"][0].cpu()
        hidden_pred = outputs["hidden_predictions"][0].cpu()
        environment_pred = outputs["environment_predictions"][0].cpu()
        disent_metrics = self._sequence_diagnostics(hidden_pred, environment_pred)

        return {
            "metrics": {
                "visible_rmse": _rmse(visible_pred, future_visible),
                "visible_pearson": _mean_pearson(visible_pred, future_visible),
                "peak_visible_error": self._peak_visible_error(visible_pred.unsqueeze(0), future_visible.unsqueeze(0)),
                "amplitude_collapse_score": self._amplitude_collapse_score(visible_pred.unsqueeze(0), future_visible.unsqueeze(0)),
                "hidden_rmse": _rmse(hidden_pred, future_hidden),
                "hidden_pearson": _mean_pearson(hidden_pred, future_hidden),
                "hidden_environment_correlation": float(disent_metrics["hidden_environment_correlation"].item()),
                **self._collect_ratio_metrics(outputs),
            },
            "visible_pred": visible_pred,
            "visible_true": future_visible.clone(),
            "hidden_pred": hidden_pred,
            "hidden_true": future_hidden.clone(),
            "environment_pred": environment_pred,
            "interaction_pred": outputs["interaction_matrix"].detach().cpu(),
        }

    def fit(self, train_loader, val_loader) -> FitResult:
        best_state = None
        best_epoch = -1
        best_val_score = float("inf")
        patience = int(self.train_cfg["early_stopping_patience"])
        no_improvement = 0
        history: List[Dict[str, float]] = []

        for epoch in range(int(self.train_cfg["epochs"])):
            self.current_epoch = epoch
            self.model.residual_curriculum_progress = min(1.0, float(epoch) / max(self.total_epochs * 0.6, 1))
            train_losses = self._iterate_loader(train_loader, training=True)
            train_full_context_loss = self._full_context_train_step()
            val_losses = self._iterate_loader(val_loader, training=False)
            val_metrics = self._validation_metrics(val_loader, num_particles=self.eval_particles)
            val_full_context = self.evaluate_full_context("val", num_particles=self.eval_particles)
            val_hidden_recovery = self.recover_hidden_on_split("val")

            val_score = (
                0.30 * val_metrics["visible_rollout_rmse"]
                + 0.22 * val_full_context["metrics"]["visible_rmse"]
                + 0.15 * val_metrics["peak_visible_error"]
                + 0.12 * val_metrics["amplitude_collapse_score"]
                + 0.08 * val_hidden_recovery["metrics"]["hidden_recovery_rmse"]
                + 0.08 * abs(val_hidden_recovery["metrics"]["hidden_environment_correlation"])
                + 0.05 * val_metrics["residual_dominates_fraction"]
            )
            epoch_record = {
                "epoch": float(epoch),
                "train_total": train_losses["total"] + train_full_context_loss,
                "val_total": val_losses["total"] + float(self.loss_cfg["lambda_full_context_visible"]) * val_full_context["metrics"]["visible_rmse"],
                "val_visible_rollout_rmse": val_metrics["visible_rollout_rmse"],
                "val_full_context_visible_rmse": val_full_context["metrics"]["visible_rmse"],
                "val_hidden_recovery_rmse": val_hidden_recovery["metrics"]["hidden_recovery_rmse"],
                "val_hidden_environment_correlation": val_hidden_recovery["metrics"]["hidden_environment_correlation"],
                "val_amplitude_collapse_score": val_metrics["amplitude_collapse_score"],
                "val_score": val_score,
            }
            history.append(epoch_record)

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= patience:
                break

        if best_state is None:
            raise RuntimeError("Training finished without a valid best state.")

        self.model.load_state_dict(best_state)
        return FitResult(history=history, best_epoch=best_epoch, best_val_score=best_val_score)

    @torch.no_grad()
    def forecast_case(
        self,
        history_visible_raw: torch.Tensor,
        future_visible_true: torch.Tensor,
        future_hidden_true: torch.Tensor,
        num_particles: int | None = None,
    ) -> Dict[str, Any]:
        self.model.eval()
        outputs = self._forward_model(
            history_visible_raw=history_visible_raw.unsqueeze(0),
            rollout_horizon=int(future_visible_true.shape[0]),
            training=False,
            num_particles=num_particles or self.particle_rollout_k,
        )
        visible_pred = outputs["visible_predictions"][0].cpu()
        hidden_pred = outputs["hidden_predictions"][0].cpu()
        environment_pred = outputs["environment_predictions"][0].cpu()
        interaction_pred = outputs["interaction_matrix"].detach().cpu()
        disent_metrics = self._sequence_diagnostics(hidden_pred, environment_pred)

        hidden_edge_pred = torch.cat([interaction_pred[:5, 5], interaction_pred[5, :5]], dim=0)
        hidden_edge_true = torch.cat(
            [self.true_interaction_matrix[:5, 5].cpu(), self.true_interaction_matrix[5, :5].cpu()],
            dim=0,
        )
        mask = hidden_edge_true.abs() > 0.05
        if int(mask.sum().item()) > 0:
            sign_accuracy = float((torch.sign(hidden_edge_pred[mask]) == torch.sign(hidden_edge_true[mask])).float().mean().item())
        else:
            sign_accuracy = 0.0

        metrics = {
            "visible_rollout_rmse": _rmse(visible_pred, future_visible_true),
            "visible_rollout_pearson": _mean_pearson(visible_pred, future_visible_true),
            "peak_visible_error": self._peak_visible_error(visible_pred.unsqueeze(0), future_visible_true.unsqueeze(0)),
            "amplitude_collapse_score": self._amplitude_collapse_score(visible_pred.unsqueeze(0), future_visible_true.unsqueeze(0)),
            "hidden_recovery_rmse": _rmse(hidden_pred, future_hidden_true),
            "hidden_recovery_pearson": _mean_pearson(hidden_pred, future_hidden_true),
            "hidden_environment_correlation": float(disent_metrics["hidden_environment_correlation"].item()),
            "interaction_hidden_sign_accuracy": sign_accuracy,
            **self._collect_ratio_metrics(outputs),
        }
        return {
            "metrics": metrics,
            "visible_pred": visible_pred,
            "hidden_pred": hidden_pred,
            "environment_pred": environment_pred,
            "interaction_pred": interaction_pred,
        }

    @torch.no_grad()
    def recover_hidden_sequence(
        self,
        visible_series_raw: torch.Tensor,
        hidden_series_true: torch.Tensor,
        history_length: int,
        global_offset: int = 0,
    ) -> Dict[str, Any]:
        self.model.eval()
        predictions = []
        targets = []
        env_predictions = []
        indices = []

        for time_index in range(history_length - 1, int(visible_series_raw.shape[0])):
            history_visible = visible_series_raw[time_index - history_length + 1 : time_index + 1]
            outputs = self._forward_model(
                history_visible_raw=history_visible.unsqueeze(0),
                rollout_horizon=1,
                training=False,
                num_particles=self.eval_particles,
            )
            predictions.append(outputs["hidden_initial"][0].cpu())
            targets.append(hidden_series_true[time_index].cpu())
            env_predictions.append(outputs["environment_initial"][0].cpu())
            indices.append(global_offset + time_index)

        prediction_tensor = torch.stack(predictions, dim=0)
        target_tensor = torch.stack(targets, dim=0)
        env_prediction_tensor = torch.stack(env_predictions, dim=0)
        disent_metrics = self._sequence_diagnostics(prediction_tensor, env_prediction_tensor)

        metrics = {
            "hidden_recovery_rmse": _rmse(prediction_tensor, target_tensor),
            "hidden_recovery_pearson": _mean_pearson(prediction_tensor, target_tensor),
            "hidden_environment_correlation": float(disent_metrics["hidden_environment_correlation"].item()),
            "hidden_autocorrelation": float(disent_metrics["hidden_autocorrelation"].item()),
            "environment_autocorrelation": float(disent_metrics["environment_autocorrelation"].item()),
            "hidden_roughness": float(disent_metrics["hidden_roughness"].item()),
            "environment_roughness": float(disent_metrics["environment_roughness"].item()),
        }
        return {
            "metrics": metrics,
            "hidden_pred": prediction_tensor,
            "hidden_true": target_tensor,
            "environment_pred": env_prediction_tensor,
            "indices": torch.tensor(indices, dtype=torch.long),
        }

    @torch.no_grad()
    def recover_hidden_on_split(self, split_name: str) -> Dict[str, Any]:
        split_start, split_end = self.split_ranges[split_name]
        slice_start = max(0, split_start - self.history_length + 1)
        return self.recover_hidden_sequence(
            visible_series_raw=self.visible_series_raw[slice_start:split_end],
            hidden_series_true=self.hidden_series_raw[slice_start:split_end],
            history_length=self.history_length,
            global_offset=slice_start,
        )
