# Training & serving notes (GPU host)

This device has no GPU; this file records how the GPU/network-bound pieces
are intended to be wired so they can be finished on the training host without
re-deriving the design.

## Standing rules (enforced in code)

Three rules every run obeys. They are checked by code, not just documented,
so a misconfigured run fails fast (often on a laptop, before any GPU time).

1. **Every run writes to the result store.** One layout for all runs:
   `runs/<experiment>/<run_id>/` with `run.json` (id, git sha, host, status,
   W&B url), `config.json`, `env.json`, `metrics.json` (final headline
   numbers), `metrics.jsonl` (append-only stream), and `artifacts/`
   (checkpoints/plots). Single source of truth: `gps.results.ResultStore`.
   `runs/` is git-ignored. See `gps/results.py` for the full format.

2. **Every *training* run logs to Weights & Biases — mandatory, no opt-out.**
   `gps.tracking.require_wandb` reads `WANDB_API_KEY` from the environment and
   **raises `TrackingError` if it is missing/blank**, aborting the run before
   any work. Set it on the host:

   ```bash
   export WANDB_API_KEY=<your-key>     # from https://wandb.ai/authorize
   # optional overrides:
   export WANDB_PROJECT=grounded-player-sim
   export WANDB_ENTITY=<team-or-user>
   ```

   `wandb` is in the `train` extra. The W&B run id/url is written back into
   the local `run.json`, so a result dir always points at its W&B run.

3. **LLM training uses slime; LLM inference uses sglang** (`gps.backends`).
   - Serving an open-weight model for logprob scoring / rollouts → `sglang`
     (`SGLangBackbone`). A *closed* API model (`APIBackbone`) is the RQ4
     baseline, not a served model, and is exempt.
   - RL / post-training of a served LLM → `slime` (`SlimeRLTrainer`), which
     pairs its training backend with the sglang rollout engine. SFT of the
     injector on top of a *frozen* LLM still serves its rollouts via sglang.
   - The **board-native CNN control is exempt** — it is plain torch, no LLM,
     so neither slime nor sglang is required (this is the cheap Milestone-A
     control; see `milestone_a.md`).

   `Trainer.begin_run` applies all three at the top of `fit`: it validates the
   backend pairing, requires the W&B key, creates the run dir, and starts the
   tracked W&B run. Concrete trainers stream metrics via `wandb_run.log(...)`,
   write headline numbers via `wandb_run.summary(...)`, and end with
   `wandb_run.finish(...)` + `handle.finalize(...)`.

## Environment

```bash
pip install -e '.[serve,train,api,chess,dev]'
# slime is installed from source on the GPU host (not on PyPI):
#   git clone <slime repo> && pip install -e ./slime
```

`gps info` prints which backends are importable — use it to confirm the host
is set up.

## Serving the LLM policy (sglang)

`gps.policy.sglang_backbone.SGLangBackbone` launches an sglang engine for
Qwen3-8B (or any open-weight model). To finish:

1. In `_engine()`, launch `sgl.Engine(model_path=...)` with logprob return
   enabled.
2. In `predict()`, use constrained / regex decoding to restrict generation
   to `dp.legal_actions`, then read per-move token logprobs into a
   `MoveDistribution` normalized over the legal set.
3. For the **hidden** injection kind, attach `injection.vector` as a prefix
   embedding / soft prompt (requires the soft-prompt-enabled serving path;
   set `enable_hidden=True`). The **verbal** kind already works via
   `build_prompt`, which is implemented and unit-tested.

## Closed-source baseline (API)

`gps.policy.api_backbone.APIBackbone` (RQ4 foil). `build_messages` is
implemented and tested; `predict` needs the live request + a parse of the
returned move (and optional think-time). Prefer providers that expose token
logprobs; otherwise fall back to multi-sample frequency estimation and flag
it in the prediction `meta`. Accepts **verbal only** by design.

## Board-native backbone (the controlled comparison)

`gps.policy.board_native.BoardNativeBackbone` — a Maia/KataGo-style CNN with
move + timing heads, latent conditioning via FiLM/concat (hidden injection
only). Build/load the trunk + heads in `_network()`. This backbone is how we
prove the latent helps independent of an LLM (see `design.md` §2).

## Training the injector (SFT — default)

`gps.train.sft.SFTTrainer` maximizes the likelihood of observed moves/timing
under the simulator w.r.t. the injector params `phi` (and optionally the
backbone). The loop is sketched in the source. Note the **no-op guard**: the
parameter-free structured reference injector reports `status="no-op"` rather
than pretending to train — swap in a neural (differentiable) injector
variant to actually exercise SFT.

## Training with rewards (slime RL)

`gps.train.slime_rl.SlimeRLTrainer` for behavior-matching reward, the
opponent-prep loop, or fine-tuning the agent to fit the latent. The rollout
is `Simulator(injector, backbone).run_trajectory`; the reward
(`behavior_match_reward`) compares rollout vs. target-player statistics
(move-quality, time-allocation, error-by-phase) via KL/JS/Wasserstein
(proposal §Phase 4.4). slime pairs a training backend with the sglang
rollout engine.

## Data ingestion (next)

* **Chess:** Lichess open DB (CC0) PGN with `[%clk]`; parse with
  `python-chess` (`GameNode.clock()`); Stockfish per-position eval →
  centipawn loss. Record Stockfish depth (centipawn-loss is
  settings-dependent — report it).
* **Go:** KGS/OGS SGF with per-move time + byo-yomi; KataGo points-lost.
  Confirm per-move timing availability before committing scope (proposal
  Risk 1).
* **Sessions:** `gps.data.sessions.segment_sessions` — sweep the gap
  threshold as an ablation.
* **Splits:** `gps.eval.splits.temporal_split` — strict chronological only.
