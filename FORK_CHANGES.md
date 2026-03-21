# torad-labs/verl — Fork Changes

Base: volcengine/verl @ 4aaac63e

## Changes

### 1. skip_special_tokens=False in naive.py
- **File:** `verl/workers/reward_manager/naive.py`
- **Why:** Enables loop detection via `<|im_end|>` count in reward functions that inspect raw
  model output. Qwen3 chat format emits `<|im_end|>` after each turn; stripping special tokens
  removes this signal before the reward function can see it.
- **Config:** No config key — always on in this fork (one-line behavioral change).
- **Upstream:** Not submitted (Qwen3 chat-format specific).

### 2. filter_groups — dynamic sampling to eliminate zero-advantage batches
- **File:** `verl/trainer/ppo/ray_trainer.py` — `_apply_filter_groups()` + call in `fit()`
- **File:** `verl/trainer/config/ppo_trainer.yaml` — default block under `algorithm.filter_groups`
- **Why:** DAPO paper (arXiv:2409.07236 §3.2). Zero-advantage groups (all completions have
  identical reward) produce zero gradient. Without filtering, 20–40% of steps can be wasted on
  batches that contribute nothing to learning. This implementation ports the core filtering logic
  into the main trainer (without the full retry mechanism, which requires dataloader-level
  integration not yet available in the main trainer).
- **Config:** `algorithm.filter_groups.enable: True` (default `False` — backward compatible)
- **Upstream:** Candidate for upstreaming as PR.

### 3. LLDS — Lazy Likelihood Displacement Stabilization
- **File:** `verl/trainer/ppo/core_algos.py` — `compute_llds_loss()` function (appended at end)
- **File:** `verl/workers/actor/dp_actor.py` — call when `llds_coef > 0`
- **File:** `verl/workers/config/actor.py` — `llds_coef: float = 0.0` field in `ActorConfig`
- **Why:** arXiv:2512.04220 (Search-R1). Correct-response log-prob silently decays under RL
  training even when reward curves look stable. LLDS regularizes displaced tokens in
  positive-advantage completions only (triple-gate: trajectory + token + action). Official code
  not released as of 2026-03-15 — implementation derived from paper algorithm description.
- **Config:** `actor_rollout_ref.actor.llds_coef: 0.05` (default `0.0` = disabled — backward compatible)
- **Upstream:** Candidate pending official code release for comparison.
  Monitor `actor/llds_loss` and `actor/llds_mask_ratio` in MLflow — if `llds_loss` exceeds
  `policy_loss`, reduce `llds_coef` to `0.01`.
