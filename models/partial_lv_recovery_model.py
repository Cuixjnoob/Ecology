from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.encoders import MLP


class PartialLVRecoveryModel(nn.Module):
    def __init__(
        self,
        num_visible: int,
        delay_length: int,
        delay_stride: int,
        delay_embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        encoder_layers: int,
        residual_layers: int,
        use_environment_latent: bool = True,
        use_lv_guidance: bool = False,
        max_state_value: float = 5.5,
        base_visible_noise: float = 0.015,
        base_hidden_noise: float = 0.012,
        base_environment_noise: float = 0.020,
    ) -> None:
        super().__init__()
        self.num_visible = num_visible
        self.total_species = num_visible + 1
        self.delay_length = delay_length
        self.delay_stride = delay_stride
        self.max_state_value = max_state_value
        self.use_environment_latent = use_environment_latent
        self.use_lv_guidance = use_lv_guidance
        self.context_dim = context_dim
        self.base_visible_noise = float(base_visible_noise)
        self.base_hidden_noise = float(base_hidden_noise)
        self.base_environment_noise = float(base_environment_noise)

        self.delay_encoder = MLP(
            input_dim=delay_length,
            hidden_dim=hidden_dim,
            output_dim=delay_embedding_dim,
            num_layers=max(encoder_layers, 2),
            dropout=0.0,
        )
        self.history_encoder = nn.GRU(
            input_size=num_visible,
            hidden_size=context_dim,
            batch_first=True,
        )
        self.rollout_memory = nn.GRUCell(
            input_size=num_visible,
            hidden_size=context_dim,
        )
        self.context_refiner = MLP(
            input_dim=num_visible * delay_embedding_dim + context_dim + num_visible,
            hidden_dim=hidden_dim,
            output_dim=context_dim,
            num_layers=max(encoder_layers, 2),
            dropout=0.0,
        )
        self.hidden_head = MLP(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=max(encoder_layers, 2),
            dropout=0.0,
        )
        self.environment_head = MLP(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=max(encoder_layers, 2),
            dropout=0.0,
        )

        residual_input_dim = 2 * self.total_species + 1 + 2 * context_dim
        self.residual_network = MLP(
            input_dim=residual_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.total_species,
            num_layers=max(residual_layers, 2),
            dropout=0.0,
        )
        self.environment_target_network = MLP(
            input_dim=self.total_species + 1 + 2 * context_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=max(residual_layers, 2),
            dropout=0.0,
        )
        self.hidden_fast_network = MLP(
            input_dim=num_visible + 1 + context_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            dropout=0.0,
        )
        self.growth_rates = nn.Parameter(0.08 * torch.ones(self.total_species, dtype=torch.float32))
        self.off_diagonal = nn.Parameter(0.035 * torch.randn(self.total_species, self.total_species))
        self.diagonal_unconstrained = nn.Parameter(torch.full((self.total_species,), 0.32))
        self.environment_to_species = nn.Parameter(0.04 * torch.randn(self.total_species, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.11, dtype=torch.float32))
        self.alpha_res_unconstrained = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.alpha_lv_unconstrained = nn.Parameter(torch.tensor(0.60, dtype=torch.float32))
        self.lv_drift_scale_unconstrained = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.tau_env_unconstrained = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.hidden_fast_scale_unconstrained = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.residual_curriculum_progress = 0.0

        self.visible_noise_unconstrained = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.hidden_noise_unconstrained = nn.Parameter(torch.tensor(-2.1, dtype=torch.float32))
        self.environment_noise_unconstrained = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        self.visible_log_mean: torch.Tensor | None = None
        self.visible_log_std: torch.Tensor | None = None

    def set_visible_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.visible_log_mean = mean.detach().clone()
        self.visible_log_std = std.detach().clone()

    def required_history_steps(self) -> int:
        return 1 + (self.delay_length - 1) * self.delay_stride

    def get_interaction_matrix(self) -> torch.Tensor:
        interaction = self.off_diagonal.clone()
        diagonal = -torch.nn.functional.softplus(self.diagonal_unconstrained) - 0.05
        interaction.diagonal().copy_(diagonal)
        return interaction

    def alpha_res(self) -> torch.Tensor:
        return 0.08 + 0.82 * torch.sigmoid(self.alpha_res_unconstrained)

    def alpha_lv(self) -> torch.Tensor:
        if not self.use_lv_guidance:
            return self.alpha_res_unconstrained.new_tensor(0.0)
        return 0.10 + 0.85 * torch.sigmoid(self.alpha_lv_unconstrained)

    def _standardize_visible(self, visible_raw: torch.Tensor) -> torch.Tensor:
        if self.visible_log_mean is None or self.visible_log_std is None:
            raise RuntimeError("Visible normalization statistics must be set before calling the model.")
        mean = self.visible_log_mean.to(visible_raw.device)
        std = self.visible_log_std.to(visible_raw.device)
        log_values = torch.log1p(visible_raw.clamp_min(0.0))
        return (log_values - mean) / (std + 1e-6)

    def _build_delay_features(self, visible_standardized: torch.Tensor) -> torch.Tensor:
        required_steps = self.required_history_steps()
        if visible_standardized.shape[1] < required_steps:
            raise ValueError(
                f"history length {visible_standardized.shape[1]} is shorter than required {required_steps}"
            )
        indices = [
            visible_standardized.shape[1] - 1 - step * self.delay_stride
            for step in range(self.delay_length)
        ]
        selected = visible_standardized[:, indices, :]
        return selected.permute(0, 2, 1).contiguous()

    def _encode_context(self, history_visible_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visible_standardized = self._standardize_visible(history_visible_raw)
        delay_features = self._build_delay_features(visible_standardized)
        batch_size = delay_features.shape[0]

        encoded_delays = self.delay_encoder(delay_features.reshape(batch_size * self.num_visible, self.delay_length))
        encoded_delays = encoded_delays.reshape(batch_size, self.num_visible, -1)

        _, hidden_state = self.history_encoder(visible_standardized)
        memory_state = hidden_state[-1]
        history_slope = visible_standardized[:, 1:, :] - visible_standardized[:, :-1, :]
        slope_summary = history_slope.mean(dim=1)
        refined_context = self.context_refiner(
            torch.cat([encoded_delays.reshape(batch_size, -1), memory_state, slope_summary], dim=-1)
        )
        return refined_context, memory_state

    def _expand_particles(self, tensor: torch.Tensor, num_particles: int) -> torch.Tensor:
        if num_particles == 1:
            return tensor
        repeated = tensor.unsqueeze(1).expand(-1, num_particles, -1)
        return repeated.reshape(tensor.shape[0] * num_particles, tensor.shape[1])

    def forward(
        self,
        history_visible_raw: torch.Tensor,
        rollout_horizon: int,
        num_particles: int = 1,
        stochastic: bool = True,
        process_noise_scale: float = 1.0,
        latent_perturb_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        context, memory_state = self._encode_context(history_visible_raw)
        hidden_initial = torch.nn.functional.softplus(self.hidden_head(context)) + 1e-4
        environment_initial = torch.tanh(self.environment_head(context))

        if stochastic and latent_perturb_scale > 0.0:
            hidden_initial = torch.clamp(
                hidden_initial + latent_perturb_scale * 0.08 * torch.randn_like(hidden_initial),
                min=1e-4,
            )
            environment_initial = environment_initial + latent_perturb_scale * 0.12 * torch.randn_like(
                environment_initial
            )

        batch_size = history_visible_raw.shape[0]
        static_context = self._expand_particles(context, num_particles)
        current_memory = self._expand_particles(memory_state, num_particles)
        hidden_state = self._expand_particles(hidden_initial, num_particles)
        env_state = self._expand_particles(environment_initial, num_particles)
        visible_state = self._expand_particles(history_visible_raw[:, -1, :], num_particles)
        state = torch.cat([visible_state, hidden_state], dim=-1)
        interaction_matrix = self.get_interaction_matrix()

        alpha_res = self.alpha_res()
        alpha_lv = self.alpha_lv()
        lv_drift_scale = 0.08 + 0.20 * torch.sigmoid(self.lv_drift_scale_unconstrained)
        tau_env = 0.03 + 0.09 * torch.sigmoid(self.tau_env_unconstrained)
        hidden_fast_scale = 0.03 + 0.12 * torch.sigmoid(self.hidden_fast_scale_unconstrained)
        curriculum_factor = 0.3 + 0.7 * min(1.0, self.residual_curriculum_progress)

        visible_predictions = []
        hidden_predictions = []
        environment_predictions = []
        lv_contributions = []
        residual_contributions = []
        noise_contributions = []
        deterministic_predictions = []
        lv_only_predictions = []
        hidden_fast_contributions = []

        visible_noise_std = self.base_visible_noise * (
            0.4 + torch.nn.functional.softplus(self.visible_noise_unconstrained)
        )
        hidden_noise_std = self.base_hidden_noise * (
            0.4 + torch.nn.functional.softplus(self.hidden_noise_unconstrained)
        )
        environment_noise_std = self.base_environment_noise * (
            0.4 + torch.nn.functional.softplus(self.environment_noise_unconstrained)
        )

        for _ in range(rollout_horizon):
            current_state = state
            interaction_feature = current_state @ interaction_matrix.T
            log_state = torch.log1p(current_state)
            residual_inputs = torch.cat(
                [log_state, interaction_feature, env_state, current_memory, static_context],
                dim=-1,
            )
            residual_raw = self.residual_network(residual_inputs)
            residual_delta = torch.tanh(residual_raw) * (0.10 + current_state) * torch.tanh(
                self.residual_scale
            )
            residual_contribution = curriculum_factor * alpha_res * residual_delta

            if self.use_lv_guidance:
                lv_raw = self.growth_rates + interaction_feature + env_state * self.environment_to_species.view(1, -1)
                lv_drift = lv_drift_scale * current_state * torch.tanh(lv_raw)
                lv_contribution = alpha_lv * lv_drift
            else:
                lv_contribution = torch.zeros_like(current_state)

            if stochastic:
                visible_noise = (
                    process_noise_scale
                    * visible_noise_std
                    * torch.randn_like(current_state[:, : self.num_visible])
                    * (0.08 + current_state[:, : self.num_visible])
                )
                hidden_noise = (
                    process_noise_scale
                    * hidden_noise_std
                    * torch.randn_like(current_state[:, self.num_visible :])
                    * (0.08 + current_state[:, self.num_visible :])
                )
                noise_contribution = torch.cat([visible_noise, hidden_noise], dim=-1)
                env_noise = process_noise_scale * environment_noise_std * torch.randn_like(env_state)
            else:
                noise_contribution = torch.zeros_like(current_state)
                env_noise = torch.zeros_like(env_state)

            # Hidden fast innovation (only affects hidden species)
            hidden_fast_input = torch.cat([
                current_state[:, :self.num_visible],
                env_state,
                current_memory,
            ], dim=-1)
            hidden_fast_delta = torch.zeros_like(current_state)
            hidden_fast_delta[:, self.num_visible:] = (
                hidden_fast_scale
                * (0.08 + current_state[:, self.num_visible:])
                * torch.tanh(self.hidden_fast_network(hidden_fast_input))
            )

            deterministic_next = current_state + lv_contribution + residual_contribution + hidden_fast_delta
            lv_only_next = current_state + lv_contribution
            next_state = torch.clamp(
                deterministic_next + noise_contribution,
                min=1e-4,
                max=self.max_state_value,
            )

            env_inputs = torch.cat([log_state, env_state, current_memory, static_context], dim=-1)
            env_target = torch.tanh(self.environment_target_network(env_inputs))
            next_env_state = torch.clamp(
                env_state + tau_env * (env_target - env_state) + env_noise,
                min=-2.5, max=2.5,
            )

            visible_predictions.append(next_state[:, : self.num_visible])
            hidden_predictions.append(next_state[:, self.num_visible :])
            environment_predictions.append(next_env_state)
            lv_contributions.append(lv_contribution)
            residual_contributions.append(residual_contribution)
            noise_contributions.append(noise_contribution)
            deterministic_predictions.append(torch.clamp(deterministic_next, min=1e-4, max=self.max_state_value))
            lv_only_predictions.append(torch.clamp(lv_only_next, min=1e-4, max=self.max_state_value))
            hidden_fast_contributions.append(hidden_fast_delta)

            standardized_visible = self._standardize_visible(next_state[:, : self.num_visible])
            current_memory = self.rollout_memory(standardized_visible, current_memory)
            env_state = next_env_state
            state = next_state

        def _reshape_particles(series: torch.Tensor) -> torch.Tensor:
            reshaped = series.reshape(batch_size, num_particles, rollout_horizon, -1)
            return reshaped

        visible_particles = _reshape_particles(torch.stack(visible_predictions, dim=1))
        hidden_particles = _reshape_particles(torch.stack(hidden_predictions, dim=1))
        environment_particles = _reshape_particles(torch.stack(environment_predictions, dim=1))
        lv_particles = _reshape_particles(torch.stack(lv_contributions, dim=1))
        residual_particles = _reshape_particles(torch.stack(residual_contributions, dim=1))
        noise_particles = _reshape_particles(torch.stack(noise_contributions, dim=1))
        deterministic_particles = _reshape_particles(torch.stack(deterministic_predictions, dim=1))
        lv_only_particles = _reshape_particles(torch.stack(lv_only_predictions, dim=1))
        hidden_fast_particles = _reshape_particles(torch.stack(hidden_fast_contributions, dim=1))

        return {
            "visible_predictions": visible_particles.mean(dim=1),
            "hidden_predictions": hidden_particles.mean(dim=1),
            "environment_predictions": environment_particles.mean(dim=1),
            "visible_particles": visible_particles,
            "hidden_particles": hidden_particles,
            "environment_particles": environment_particles,
            "hidden_initial": hidden_initial,
            "environment_initial": environment_initial,
            "interaction_matrix": interaction_matrix,
            "lv_contribution_history": lv_particles.mean(dim=1),
            "residual_contribution_history": residual_particles.mean(dim=1),
            "noise_contribution_history": noise_particles.mean(dim=1),
            "deterministic_prediction_history": deterministic_particles.mean(dim=1),
            "lv_only_prediction_history": lv_only_particles.mean(dim=1),
            "hidden_fast_contribution_history": hidden_fast_particles.mean(dim=1),
            "tau_env": tau_env.detach(),
            "alpha_lv": alpha_lv.detach(),
            "alpha_res": alpha_res.detach(),
            "visible_noise_std": visible_noise_std.detach(),
            "hidden_noise_std": hidden_noise_std.detach(),
            "environment_noise_std": environment_noise_std.detach(),
        }
