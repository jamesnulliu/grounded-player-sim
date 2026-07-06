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

## 8. Positioning vs. the closest competitors (see `documents/related_work.md`)

> **Canonical version: `documents/related_work.md`** (2026-07 sweep). This
> section keeps only the design-relevant summary of the three novelties; the
> detailed competitor notes, axes table, and shared-territory list live there.

**We no longer claim novelty by conjunction.** Each axis is now occupied by 2026
prior art (per-individual chess style, cohort move+clock, dynamic emotional
chess, timing>choice in psychometrics, evolving latent + future split), so the
contribution is the **controlled empirical synthesis on real humans** those lines
lack — three results that lead:

1. **The when-not-what finding** — the evolving state is legible in *timing*,
   near-null in *move choice*; robust across a 6-year span and reproduced in a
   non-game domain (the raw timing>choice fact is old in response-time
   psychometrics; the evolving-state + oracle + future-split *form* is ours).
2. **The equal-capacity evolving-vs-memoryless control on a strict future
   split** — isolates *dynamics/individualization* from habit; the rigor the
   human-chess and user-simulation lines skip. Settles the #1 objection.
3. **The hidden-vs-verbal channel ordering is backbone-dependent** (measured) —
   the trained *hidden* latent wins with *no* language prior (board-native RQ6,
   −0.069/−0.117), the verbal note wins *inside an LLM* (G3), while injected
   state helps think-time either way.

The full, cited competitor comparison (ChessMimic, Elo-Disentangled, Ailed,
Allie, Maia-3, HumanLM, LATTE + the response-time literature), the axes table,
and the shared-territory list now live in **`documents/related_work.md`** (the
canonical version). Do not re-derive them here. Caution: the chess/LLM
competitors are all 2026 preprints — "first to" on any single *axis* is fragile
and several are already lost, which is why we anchor on the three results above,
not axis-ownership.

## 9. Decision: verbal-vs-hidden is a HEADLINE, not a side experiment

HumanLM is text-latent only; LATTE is vector-latent only. **Nobody compares
the two channels head-to-head as interchangeable ways to inject the same
per-individual state.** Our `InjectionKind` split already supports both, so
this comparison is nearly free and is genuinely unclaimed — promote it to a
first-class research question (the new RQ6), not an ablation footnote.

## 10. Decision: population generation is a DEMONSTRATION, not a 2nd pillar

The author proposed generating human-like behavior by (a) modifying
profile/history and (b) adding noise to the trained injector, for *user
simulation*. Resolved as follows:

* These are **two different axes**, not two ways to do one thing: a
  **control** axis (who is this player? — set by conditioning) and a
  **stochastic** axis (natural variation — set by sampling). Keep them
  distinct.
* **Drop "edit history arbitrarily."** `z_t` is causally downstream of
  history, so editing history steers the latent indirectly through a learned
  map and can produce globally **incoherent** players. Instead, **intervene
  directly on the structured latent dimensions** (we already have anchored
  time-pressure / post-loss / fatigue dims + the `OracleInjector` clamp).
* **Isotropic noise will not work** — a discriminatively-trained latent has
  no valid neighborhood, so Gaussian noise pushes **off-manifold** (degraded
  play, not a different human). Valid sampling instead: a **variational /
  KL-regularized** latent (sample the prior), **fit + sample the empirical
  per-individual latent distribution** over the real population, or
  **interpolate** between real players' latents. Also disambiguate noise on
  the latent *output* (per-step jitter) vs. on injector *weights*
  (systematic, "different coherent player").
* **Validation flips from pointwise to distributional.** Invented players
  have no ground truth, so this can only be validated at the
  population/distributional level (KL/JS/Wasserstein on behavioral stats +
  **precision/recall for generative models**: coverage of real human
  diversity without implausible outliers / an off-manifold realism rate).
  This is strictly weaker ground than the future-behavior test — state it
  explicitly.
* **Fidelity and diversity trade off.** Pillar-1 wants pinpoint, low-variance
  replication; this wants diverse plausible non-existent humans. Keep the
  experiments separate so the diversity objective never contaminates the
  fidelity eval.
* **Why it is worth doing:** done right it attacks the field's named-but-
  unsolved problem — LLM simulators collapsing to a *"positive average
  person"* with lost long-tail/individual variance. A calibrated, diverse,
  heterogeneity-recovering population is a contribution to *that*.
* **Build/scope decision:** build it as a **downstream demonstration**
  ("what faithful per-individual modeling unlocks"), but **instrument it with
  the full distributional + precision/recall eval**. The demo framing and the
  pillar framing share the same code and experiments; only the abstract-level
  emphasis differs, and that is a one-paragraph call made **last**, once the
  numbers show whether the result is strong (promote to pillar) or merely
  suggestive (keep as demo).

## 11. Decision: chess is primary; generality goes to an oracle-preserving
## non-game domain, never an oracle-less one (June 2026)

We pressure-tested "is chess/Go the right dataset, or should we pivot?" The
answer: **keep chess as the primary domain, do not pivot.** The reasoning,
recorded so it is not relitigated:

* **The engine oracle is the moat, not a convenience.** The entire
  contribution is "reproduce a player's *deviation from optimal*, conditioned
  on their evolving state." That target only has meaning where "optimal" is
  measurable *per decision*. Stockfish/KataGo give that for free at
  human-identity granularity. Remove the oracle and (a) RQ2 degrades from "the
  latent predicts a measured centipawn swing" to "the latent correlates with a
  hand-built proxy," (b) the suboptimality-matching metrics
  (`P(blunder | low time)`) vanish, and (c) we lose the one axis that
  separates us from every LLM-simulation paper that can only do distribution
  matching. **The oracle is the differentiation from LATTE / HumanLM /
  Agent4Rec.**

* **Six properties any replacement/added domain must satisfy** (chess has all
  six; this is the rubric, not a vibe):
  1. stable per-individual identity at volume,
  2. multi-session / longitudinal history (so the *future-behavior* split
     RQ3 — our decisive test — is even definable),
  3. per-decision discrete action (we model `P(a_t | state, z_t)`),
  4. **a per-decision quality oracle** (the rare one),
  5. a timing signal (our cleanest behavioral fingerprint, the Allie axis),
  6. genuine within-session non-stationarity (tilt/fatigue/pressure must
     actually exist in the data).

* **Generality: add a non-game oracle domain, don't replace Go with an
  oracle-less one.** "Two board games" is weak generality evidence (proposal
  §2). The fix that *keeps the moat* is a non-game domain that still has an
  oracle:
  - **Preferred — knowledge tracing / intelligent tutoring** (EdNet,
    ASSISTments): massive per-student longitudinal data; IRT gives a per-item
    difficulty+correctness oracle; engagement/fatigue/"gaming-the-system"
    dynamics are real and studied; per-item timing exists. Bonus: the
    entrenched BKT/DKT literature is a ready-made set of *static* baselines to
    beat.
  - **Alternative — competitive programming** (Codeforces): contest sessions,
    per-submission verdict + problem-rating oracle, submission timestamps,
    plausible tilt/fatigue. Caveat: the action (code) is unstructured, so model
    verdict / difficulty-choice / timing rather than a move distribution.
  - A **game + ITS** pairing is materially stronger generality than chess+Go
    (a game *and* a non-game, both oracle-graded) and is the upgrade most
    likely to help a main-track case.

* **Explicitly rejected: recommendation / dialogue / social-media
  simulation.** They have identity and longitudinality but **no per-decision
  oracle** — pivoting there deletes our differentiation and lands us in the
  exact crowded pool (LATTE/HumanLM/Agent4Rec) we are differentiating from.

* **On "use a strong model to learn the user profile."** Two distinct ideas,
  both demoted from the thesis: (a) an LLM that *summarizes a trajectory into
  a profile* is our **persona-prompting baseline** (B5/B6) and the **verbal
  injection channel** — and an LLM RL-trained to emit a NL latent state from
  history is *literally HumanLM's headline*, so it must never be our core; (b)
  an LLM *as the backbone policy* is fine (the design.md §2 branch) but both
  its flavors (in-context trace, per-player fine-tune) are **static**. The
  thing that is ours is the delta on top of any backbone: a state that
  **evolves**, is validated on **future** games, and is graded against an
  **oracle**.

* **Ordering (load-bearing):** chess headline first (Milestone A → E-C1/C2/C3).
  Only *after* the core mechanism is shown to work on chess do we spend effort
  on generality (Go, or better, a non-game oracle domain) and the population
  demo. Generality/breadth is polish on a proven mechanism, never a substitute
  for proving it.
