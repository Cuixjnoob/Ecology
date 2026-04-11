# Codex Iteration Log

## Current Best Run
- Run: [20260411_115901_partial_lv_lv_guided_stochastic_refined](/Users/cuixingji/Desktop/生态模拟/runs/20260411_115901_partial_lv_lv_guided_stochastic_refined)
- Summary: [summary.json](/Users/cuixingji/Desktop/生态模拟/runs/20260411_115901_partial_lv_lv_guided_stochastic_refined/results/summary.json)
- Sliding-window visible RMSE: `0.7937`
- Full-context visible RMSE: `0.8075`
- Hidden RMSE: `0.1660`
- Hidden Pearson: `0.9020`
- Amplitude collapse score: `0.0580`
- Hidden/environment correlation: `0.0993`
- LV/residual ratio mean: `1.6452`
- Residual dominates fraction: `0.8711`

## Current Best Bottleneck Explanation
- Hidden recovery is no longer the dominant failure mode.
- Full-context visible prediction is still the main bottleneck.
- Residual remains too dominant over the structured LV-guided branch.
- Hidden/environment correlation is much improved, but role separation in the visible generator is still not strong enough.
- Particle rollout is not currently helping.

## Pre-Iteration Hypotheses
1. **Visible residual budget is too loose.**
   Hypothesis: the residual branch can still directly drive visible species too strongly, so the model defaults to residual-heavy safe trajectories. Adding a visible-specific residual budget and visible-specific LV/residual penalty should improve long-horizon visible dynamics and reduce structured-branch underuse.
2. **Full-context training still under-pressures multi-regime long rollout behavior.**
   Hypothesis: the current full-context training segment is too limited, so visible long-horizon prediction remains template-like. Training on multiple long-context cut points inside the train segment should improve full-context visible quality.
3. **Hidden/environment disentanglement is numerically improved but functionally under-specified.**
   Hypothesis: visible generation still lacks explicit channel responsibility separation between hidden-driven and environment-driven effects. A partially separated decoder path could improve role separation and long-horizon visible realism.

## Iteration 1
- Status: completed
- Selected hypothesis: **#1 Visible residual budget is too loose**
- Reason: it directly targets the real bottleneck that remains after amplitude collapse and basic disentanglement were improved, while being smaller and more reversible than redesigning the decoder.
- Code files changed:
  - [models/partial_lv_recovery_model.py](/Users/cuixingji/Desktop/生态模拟/models/partial_lv_recovery_model.py)
  - [train/partial_lv_mvp_trainer.py](/Users/cuixingji/Desktop/生态模拟/train/partial_lv_mvp_trainer.py)
  - [scripts/run_partial_lv_mvp.py](/Users/cuixingji/Desktop/生态模拟/scripts/run_partial_lv_mvp.py)
- Exact change summary:
  - Added an explicit visible residual budget and a separate hidden residual budget.
  - Added visible-specific LV/residual diagnostics and penalties.
  - Updated noise selection and LV-activity checks to consider visible-specific structured usage.
- Commands run:
  - `python3 -m py_compile models/partial_lv_recovery_model.py train/partial_lv_mvp_trainer.py scripts/run_partial_lv_mvp.py`
  - `./run_all.sh`
- Train/eval outputs:
  - Run: [20260411_181512_partial_lv_lv_guided_stochastic_refined](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181512_partial_lv_lv_guided_stochastic_refined)
  - Summary: [summary.json](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181512_partial_lv_lv_guided_stochastic_refined/results/summary.json)
- Key metrics:
  - Sliding-window visible RMSE: `0.8505` vs previous best `0.7937`
  - Full-context visible RMSE: `0.9066` vs previous best `0.8075`
  - Hidden RMSE: `0.4013` vs previous best `0.1660`
  - Hidden Pearson: `-0.5667` vs previous best `0.9020`
  - Amplitude collapse: `0.6747` vs previous best `0.0580`
  - LV/residual ratio mean: `0.7380`
  - Residual dominates fraction: `0.2310`
  - Visible residual dominates fraction: `0.2731`
- Plots produced:
  - [fig2_hidden_test_overlay.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181512_partial_lv_lv_guided_stochastic_refined/results/fig2_hidden_test_overlay.png)
  - [fig3_visible_rollout_compare.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181512_partial_lv_lv_guided_stochastic_refined/results/fig3_visible_rollout_compare.png)
  - [fig4_visible_fullcontext_compare.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181512_partial_lv_lv_guided_stochastic_refined/results/fig4_visible_fullcontext_compare.png)
  - [fig6_diagnostics.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181512_partial_lv_lv_guided_stochastic_refined/results/fig6_diagnostics.png)
- Keep/revert decision: **revert structural residual budget change, keep the new diagnostics**
- Short interpretation:
  - The visible residual budget was too aggressive. It successfully forced the structured branch to dominate, but at the cost of hidden collapse, visible oversmoothing, and renewed amplitude collapse.
  - This indicates that “reduce residual” is directionally correct, but hard-clamping visible residual capacity is too blunt.
- Next-step hypothesis:
  - Move pressure from architecture hard-budgeting to training strategy: improve full-context learning with multiple long-context cut points while preserving the more flexible decoder.

## Iteration 2
- Status: completed
- Selected hypothesis: **#2 Full-context training still under-pressures multi-regime long rollout behavior**
- Reason:
  - Iteration 1 showed that hard residual suppression can improve LV usage but destroys hidden and visible quality.
  - The next most promising move is to improve long-horizon training pressure without hard-clamping the decoder.
- Planned code files:
  - [models/partial_lv_recovery_model.py](/Users/cuixingji/Desktop/生态模拟/models/partial_lv_recovery_model.py)
  - [train/partial_lv_mvp_trainer.py](/Users/cuixingji/Desktop/生态模拟/train/partial_lv_mvp_trainer.py)
  - [configs/partial_lv_mvp.yaml](/Users/cuixingji/Desktop/生态模拟/configs/partial_lv_mvp.yaml)
- Planned change summary:
  - Revert the overly aggressive structural residual budget.
  - Keep the new visible-specific LV/residual diagnostics.
  - Train on multiple full-context cut points inside the train split to expose the model to more long-horizon regimes.
  - Shift validation selection slightly toward full-context visible quality.
- Commands run:
  - `python3 -m py_compile models/partial_lv_recovery_model.py train/partial_lv_mvp_trainer.py scripts/run_partial_lv_mvp.py`
  - `./run_all.sh`
- Train/eval outputs:
  - Run: [20260411_181816_partial_lv_lv_guided_stochastic_refined](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181816_partial_lv_lv_guided_stochastic_refined)
  - Summary: [summary.json](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181816_partial_lv_lv_guided_stochastic_refined/results/summary.json)
- Key metrics:
  - Sliding-window visible RMSE: `1.1159`
  - Full-context visible RMSE: `1.0489`
  - Hidden RMSE: `0.3760`
  - Hidden Pearson: `0.1145`
  - Amplitude collapse: `0.1613`
  - LV/residual ratio mean: `1.3768`
  - Residual dominates fraction: `0.7576`
  - Visible residual dominates fraction: `0.7151`
- Plots produced:
  - [fig2_hidden_test_overlay.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181816_partial_lv_lv_guided_stochastic_refined/results/fig2_hidden_test_overlay.png)
  - [fig3_visible_rollout_compare.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181816_partial_lv_lv_guided_stochastic_refined/results/fig3_visible_rollout_compare.png)
  - [fig4_visible_fullcontext_compare.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_181816_partial_lv_lv_guided_stochastic_refined/results/fig4_visible_fullcontext_compare.png)
- Keep/revert decision: **revert**
- Short interpretation:
  - Increasing full-context training pressure via multiple cut points did not fix the visible generator.
  - The model became worse on both visible and hidden, suggesting that the bottleneck is not simply “more long-horizon training”, but “how hidden/environment actually enter visible generation”.
- Next-step hypothesis:
  - Add partially separated structured visible drivers for hidden and environment, so visible future does not need to rely on the residual branch for these roles.

## Iteration 3
- Status: completed
- Selected hypothesis: **#3 Hidden/environment need more explicit visible-generation roles**
- Reason:
  - Hidden recovery is already reasonably good, so the problem is more about how hidden and environment are translated into visible future.
  - The current residual branch still absorbs too much of this translation burden.
  - A small structured hidden-to-visible and environment-to-visible pathway should improve role separation without reintroducing the hard residual clamp from Iteration 1.
- Planned code files:
  - [models/partial_lv_recovery_model.py](/Users/cuixingji/Desktop/生态模拟/models/partial_lv_recovery_model.py)
  - [train/partial_lv_mvp_trainer.py](/Users/cuixingji/Desktop/生态模拟/train/partial_lv_mvp_trainer.py)
  - [configs/partial_lv_mvp.yaml](/Users/cuixingji/Desktop/生态模拟/configs/partial_lv_mvp.yaml)
- Planned change summary:
  - Revert the Iteration 2 multi-cut full-context training path.
  - Keep visible-specific diagnostics.
  - Add a structured `hidden -> visible` driver and a structured `environment -> visible` driver.
  - Evaluate residual dominance against the combined structured contribution rather than pure LV only.
- Commands run:
  - `python3 -m py_compile models/partial_lv_recovery_model.py train/partial_lv_mvp_trainer.py scripts/run_partial_lv_mvp.py`
  - `./run_all.sh`
- Train/eval outputs:
  - Run: [20260411_182236_partial_lv_lv_guided_stochastic_refined](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182236_partial_lv_lv_guided_stochastic_refined)
  - Summary: [summary.json](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182236_partial_lv_lv_guided_stochastic_refined/results/summary.json)
- Key metrics:
  - Sliding-window visible RMSE: `0.3537` vs current best `0.7937`
  - Full-context visible RMSE: `0.7534` vs current best `0.8075`
  - Full-context visible Pearson: `0.3081` vs current best `0.2421`
  - Hidden RMSE: `0.3816` vs current best `0.1660`
  - Hidden Pearson: `-0.0857` vs current best `0.9020`
  - Amplitude collapse: `0.0367` vs current best `0.0580`
  - LV/residual ratio mean: `0.8469`
  - Residual dominates fraction: `0.3516`
  - Visible residual dominates fraction: `0.2843`
- Plots produced:
  - [fig2_hidden_test_overlay.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182236_partial_lv_lv_guided_stochastic_refined/results/fig2_hidden_test_overlay.png)
  - [fig3_visible_rollout_compare.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182236_partial_lv_lv_guided_stochastic_refined/results/fig3_visible_rollout_compare.png)
  - [fig4_visible_fullcontext_compare.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182236_partial_lv_lv_guided_stochastic_refined/results/fig4_visible_fullcontext_compare.png)
  - [fig6_diagnostics.png](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182236_partial_lv_lv_guided_stochastic_refined/results/fig6_diagnostics.png)
- Keep/revert decision: **partial keep**
- Short interpretation:
  - This is the first change that meaningfully improved the real bottleneck: full-context visible prediction.
  - It also improved amplitude stability and made the structured branch much healthier.
  - However, hidden recovery collapsed, which suggests the new hidden-to-visible path can explain visible future without preserving a faithful hidden latent.
- Next-step hypothesis:
  - Keep the structured visible channels, but prevent the hidden-to-visible path from bypassing the actual hidden latent by restricting its inputs and increasing hidden supervision.

## Iteration 4
- Status: completed
- Selected hypothesis: **Structured hidden-to-visible path is bypassing hidden identity**
- Reason:
  - Iteration 3 strongly improved visible dynamics, so that direction is promising.
  - The likely new failure mode is that the hidden-visible structured driver can rely too much on visible/context features instead of a meaningful hidden state.
  - Constraining the hidden-visible path to depend primarily on hidden state, plus slightly stronger hidden supervision, should try to keep the visible gains while restoring hidden recovery.
- Commands run:
  - `python3 -m py_compile models/partial_lv_recovery_model.py train/partial_lv_mvp_trainer.py scripts/run_partial_lv_mvp.py`
  - `./run_all.sh`
- Train/eval outputs:
  - Run: [20260411_182835_partial_lv_lv_guided_stochastic_refined](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182835_partial_lv_lv_guided_stochastic_refined)
  - Summary: [summary.json](/Users/cuixingji/Desktop/生态模拟/runs/20260411_182835_partial_lv_lv_guided_stochastic_refined/results/summary.json)
- Key metrics:
  - Sliding-window visible RMSE: `0.8458`
  - Full-context visible RMSE: `0.8001`
  - Hidden RMSE: `0.3110`
  - Hidden Pearson: `0.8416`
  - Amplitude collapse: `0.7139`
  - LV/residual ratio mean: `0.6280`
  - Residual dominates fraction: `0.1054`
  - Visible residual dominates fraction: `0.1457`
- Keep/revert decision: **revert**
- Short interpretation:
  - Restricting the hidden-visible path to hidden-only inputs did improve hidden relative to Iteration 3, but it destroyed amplitude stability and did not beat the current best run overall.
  - This suggests the structured-channel idea is promising for visible dynamics, but it needs a more careful hidden-consistency design than the quick constrained patch used here.
- Current decision:
  - Keep [20260411_115901_partial_lv_lv_guided_stochastic_refined](/Users/cuixingji/Desktop/生态模拟/runs/20260411_115901_partial_lv_lv_guided_stochastic_refined) as the accepted best run.
  - Revert the unsuccessful structural channel changes in the working code, while keeping the improved diagnostic instrumentation and research log.
