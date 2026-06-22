# Training & serving notes (GPU host)

This device has no GPU; this file records how the GPU/network-bound pieces
are intended to be wired so they can be finished on the training host without
re-deriving the design.

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
