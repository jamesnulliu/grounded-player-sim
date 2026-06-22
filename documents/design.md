# Design notes

How the code maps onto `proposal_v2.md`, and the decisions made while
implementing it. Read alongside the proposal.

## 1. The contribution lives in two interfaces

The proposal's contribution is a *dynamic, per-individual latent behavioral
state injected into a policy*. In code that is exactly two abstractions:

* `gps.latent.base.LatentStateInjector` — maintains `z_t`, renders it as an
  `Injection`, and advances `z_{t-1} → z_t` from the player's own
  trajectory. This is `f_phi` from proposal §4.2.
* `gps.policy.base.PolicyBackbone` — the agent that consumes the injection
  and emits a move + timing distribution.

`gps.simulator.Simulator` composes them and threads `z_t` through a
trajectory. Everything else (data, eval, training) is scaffolding around
these two.

## 2. Decision: the LLM is the policy (revised from proposal v2)

The proposal as written says "the LLM is not the decision policy." We
**revised** this with the author: the LLM (Qwen3-8B local via sglang, or a
closed model via API) **is** the backbone agent, and the dynamic latent is a
trainable *injector* on top of it. The latent may be verbalized ("memory in
words") or a hidden vector (soft prompt / prefix) — both are first-class and
share one interface (`InjectionKind`).

**Why this is defensible despite the move.** The headline metric is next-move
likelihood, where an LLM is a *weaker* board-move predictor than a
board-native model like Maia. Reviewers from the Maia line will benchmark us
there. The mitigation is baked into the architecture: `PolicyBackbone` is
swappable, so the *same* latent-injection experiment runs on (a) an LLM
backbone and (b) a strong board-native backbone (`BoardNativeBackbone`).
That turns "why an LLM?" from a liability into a controlled variable — we can
show the latent helps *on top of* a strong board policy, so the claim is
about the dynamic-state mechanism, not the backbone.

## 3. Decision: verbal vs. hidden latent, one interface

`Injection` carries either `text` (verbal) or `vector` (hidden); a backbone
advertises which `InjectionKind`s it `accepts`, and the simulator rejects an
incompatible pairing up front. Consequence: the closed-source API backbone
accepts **verbal only** (you cannot attach soft-prompt vectors to a hosted
API), which is precisely the RQ4 contrast — the strongest a closed model
gets is a verbal profile, while the open-weight core can also receive a
learned hidden latent.

## 4. Decision: SFT is the default trainer; slime-RL is for rewards

"Imitate this player's moves/timing" is naturally maximum-likelihood, so
`SFTTrainer` is the default path for fitting the injector (and optionally the
backbone). `SlimeRLTrainer` is reserved for objectives that are *rewards*
rather than per-move likelihood: matching distributional rollout statistics,
the stretch opponent-preparation loop (§5 Phase 4.7), or fine-tuning the
agent to *fit* the latent. Both implement one `Trainer` interface.

## 5. Phase-0 is the falsifiable core, and we kept it honest

The proposal leans on Phase-0 synthetic players with known mechanisms. Two
implementation choices keep the experiment from being circular:

* **An untrained heuristic injector is NOT asserted to beat static.** A
  hand-specified injector can easily *mismatch* a player (e.g. an always-on
  fatigue term applied to a pure-tilt player), and net improvement is what a
  *trained* injector must earn. We report it, we don't assert it.
* **An `OracleInjector` that reads the true injected state IS asserted to
  beat static (P0.2).** This is non-circular: the gain is true by
  construction wherever the mechanism fires, so it confirms (a) the dynamics
  carry predictive signal and (b) the eval can detect it, and it
  upper-bounds the achievable gain. The recovery probe (P0.1) then asks
  whether a *linear* probe can recover the mechanism from the latent.

### Two bugs found and fixed during Phase-0 bring-up (recorded so they
### don't recur)

1. **Aliased mutable history.** `play_session` originally bound the *same*
   mutable `OutcomeStream` to every `DecisionPoint`, then kept mutating it —
   so every decision saw the *final* session history and all temporal signal
   vanished. Fixed by snapshotting history per game.
2. **Nobody ever lost.** The toy `game_won` used a 0.5 quality cut, which a
   competent player clears every game, so the post-loss tilt mechanism could
   never fire (probe `R^2` was exactly 0). Fixed with a logistic win model
   centered at a strong player's expected quality, giving a realistic loss
   rate. A test (`test_phase0_mechanism_actually_fires`) now guards against
   the degenerate "mechanism never fires" case.

## 6. Open methodological decisions surfaced in code (from the proposal
## critique)

* **Sessions are unlabeled.** `gps.data.sessions.segment_sessions` makes the
  wall-clock gap threshold an explicit, ablatable parameter rather than a
  silent default — because Lichess gives a timestamp stream, not session
  labels.
* **Temporal split only.** `gps.eval.splits.temporal_split` refuses to do a
  random split and raises if any partition is empty, because an empty test
  set silently invalidates the RQ3 future-behavior claim.
* **Probes show presence, not use.** `state_recovery_probe` is documented as
  a *presence* test; a causal/intervention check (clamp a latent dimension,
  measure prediction change) is the stronger RQ2 evidence and is noted as
  the next step.
* **The "is it just history-conditioning?" baseline is the dangerous one.**
  Per the proposal review, the most threatening missing baseline is an
  expressive history-conditioned policy with no structured latent (same
  inputs, latent inductive bias removed). The `PolicyBackbone` /
  `LatentStateInjector` split is arranged so this baseline is a no-latent
  backbone fed engineered history features — wire it before claiming the
  latent earns its keep.

## 7. What is real vs. stubbed

| Component | Status |
|-----------|--------|
| Interface, prediction, simulator | real, tested |
| Latent injectors (structured verbal/hidden, oracle) | real, tested |
| Mock backbone | real, tested (a test double, not a baseline) |
| Eval (NLL/Brier/ECE/top-k, timing, probes, splits) | real, tested |
| Phase-0 experiment | real, tested, runnable on CPU |
| Session segmentation | real, tested |
| sglang / API / board-native backbones | interface real + prompt
  construction tested; `predict` is a documented GPU/network stub |
| SFT / slime-RL trainers | interface + orchestration real; tensor/RL loop
  is a documented GPU stub |
| Lichess PGN + Stockfish, SGF + KataGo ingestion | not yet (next on GPU/data
  host) |
