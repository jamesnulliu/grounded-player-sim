# Modeling the Player, Not Just the Move: A Dynamic Latent-State Framework for Personalized Human Behavior Simulation

*The original research proposal (the vision). For **what actually landed**, read
`documents/paper_draft.md` (synthesis), `documents/results_ec.md` (detailed
results), and `documents/related_work.md` (positioning). This doc is kept for the
motivation, the research questions, and the framework design — trimmed of the
phase-by-phase plan, baseline list, and reference dump that the landed docs now
own.*

> **Status — how the landed work diverged from this proposal (read first).**
> - **Go became an honest negative, not a co-domain.** Under a board-size
>   homogeneity control *and* a power check, the evolving latent does **not** beat
>   the memoryless twin on real-Go think-time (E-D2). Generality now rests on
>   **chess + knowledge tracing** (both real). The "chess and Go" framing below is
>   historical.
> - **The core results are *oracle-free*.** The landed chess backbone is a
>   board-native head (FEN→from/to, no Stockfish); KT uses an IRT
>   difficulty+correctness oracle. The "Stockfish/KataGo oracle is the moat"
>   emphasis below is only partly realized — the moat that held is the
>   *future-per-individual split* + the *equal-capacity control*, not engine grading.
> - **Two headlines this proposal did not anticipate:** (1) a **heterogeneity
>   scaling law** — the latent's edge grows with population/among-player/context
>   variability (Pearson 0.89 across 8 real KT datasets); (2) **G4** — the latent
>   adds think-time value over *released SOTA* models' own predictions (Maia-2
>   difficulty ≈ ChessMimic Spearman; Allie's actual think-time, Spearman 0.62–0.65).
> - **The when-not-what asymmetry** (state legible in *timing*, near-null in *move
>   choice*) is the empirical spine — not fully anticipated here.

---

## 1. Motivation

Learned policies and LLMs are increasingly used as *simulated humans* — to
evaluate agents, train opponents, populate social simulations, stand in for
users. The dominant approach hands the model a **static** description (persona,
trait, skill bin, style embedding) and asks it to behave accordingly. Two
failures are already **settled background, not contributions**: (a) static
persona prompting is behaviorally miscalibrated (personas explain little variance;
long-horizon simulations collapse to a "positive average person"); (b) grounding
in a specific individual's real behavioral data beats persona prompting.

What remains open is the **third step**: real human behavior is not only
individual, it is *non-stationary*. The same person plays differently ahead vs.
behind, fresh vs. fatigued, under time pressure vs. not, after a bad loss vs. a
clean win. Existing individual models represent a person as a **static** object.
Almost no work models an **evolving latent behavioral state** that is (a) fit to a
specific real individual, (b) updated move-by-move from that individual's own
trajectory, and (c) validated against that individual's **future** behavior.

> **Central hypothesis.** A personalized simulator that maintains a dynamic,
> per-individual latent behavioral state — inferred from the player's own
> action+timing trajectory — reproduces individual human behavior (move choice,
> timing, within-session deviation) better than (i) static persona prompting,
> (ii) static individual models, and (iii) population models.

## 2. Why board games / oracle-graded domains

The contribution is domain-independent but must first be shown where it can be
*measured*: domains with **stable per-individual identity at volume**,
**multi-session longitudinal history** (so the future-behavior split is
definable), **per-decision discrete actions**, **per-move timing**, genuine
**within-session non-stationarity**, and — the rare one — a **per-decision quality
reference**. Chess (Lichess: permanent usernames, per-move `[%clk]`, huge volume)
satisfies all six. The generality domain must *preserve* an oracle-graded,
longitudinal structure — which is why the second domain became **knowledge
tracing** (ASSISTments/EdNet: IRT difficulty+correctness oracle, per-student
longitudinal traces, timing), not a second board game. (The full six-property
rubric and the "no oracle-less pivot to recsys/dialogue" decision are in
`design.md §11`.)

## 3. Research questions (as they landed)

- **RQ1 — dynamics vs. static & memoryless.** Does an evolving per-individual
  latent beat a *static* individual model **and** an *equal-capacity memoryless*
  control fed the identical history features? *(Landed: yes — the equal-capacity
  future-split control is the #1-objection defense; see milestone_a.md, E-C.)*
- **RQ2 — recovering interpretable state.** Does the latent recover known
  phenomena (time pressure, post-loss tilt, fatigue), per-individual? *(Landed:
  on a synthetic player with a known hidden state, probe R²=0.93 vs 0.65, plus a
  causal clamp; on real chess the timing edge concentrates under time pressure.)*
- **RQ3 — future-behavior validation.** Trained on earlier sessions, does it
  predict *later* sessions under a strict temporal split? *(Landed: yes,
  future-*sessions* split.)*
- **RQ4 — persona-prompting contrast.** *(Landed as the frozen-LLM negative
  control: a verbal state note ≈ irrelevant filler; naive persona prompting does
  not use the state.)*
- **RQ5 — cross-domain framework generality.** Same framework, encoder swapped —
  does it reproduce in a non-game domain? *(Landed: yes, knowledge tracing, on
  real students across 8 datasets / 3 subjects.)*
- **RQ6 (added) — hidden vs. verbal channel.** Which injection channel is richer?
  *(Landed: backbone-dependent — hidden wins with no language prior, verbal wins
  inside an LLM.)*

## 4. Framework

A **game-agnostic personalized simulator with a dynamic latent behavioral
state**. Only the state encoder (and any oracle) is domain-specific; the
personalization, latent, and prediction modules are shared.

**Shared decision-point interface.** Each decision is a `DecisionPoint`:
`(state, legal_actions, engine/IRT reference, time_signal, recent_outcomes,
context)`. The reference makes the target well-posed: the model is not asked to
play *well* but to **reproduce this player's deviation from optimal, conditioned
on their evolving state**.

**The core contribution — the dynamic latent.**
```
z_t = f_phi(z_{t-1}, state_{t-1}, action_{t-1}, outcome_{t-1},
            time_signal_{t-1}, result_stream_{t-1})
```
`z_t` evolves **within a game** (deliberation, time pressure) and **across games
in a session** (tilt after a loss, momentum, fatigue). It is a sequential latent
(recurrent / structured with anchored dimensions), **not required to be
verbalized** — interpretability is tested post hoc (probes + clamp, RQ2).

**Prediction heads.** A calibrated move distribution `P(a_t | state, player, z_t,
ref)` and a **first-class timing head** (think-time is one of the clearest
behavioral fingerprints and where aggregate models — Maia ignores time, Allie is
Elo-aggregate — leave room). Evaluation emphasizes **likelihood/calibration**, not
top-1, because humans mix among reasonable moves.

**Where an LLM is used.** The LLM is a *swappable backbone* (open-weight via
sglang, or closed via API) with the latent as a trainable injector on top; the
persona-prompt is the RQ4 baseline. The core policy is a probabilistic model
grounded in observed moves+timing; if the LLM components don't help, that is
itself a reportable finding. *(Landed: board-native is the headline; the LLM is an
honest secondary — see milestone_g.md.)*

## 5. Positioning + contributions

> Prior individual game models answer *"what move does this skill level / this
> player typically make?"* (static). We answer *"what move does this player make
> **right now**, given how their session has gone so far?"* (dynamic), validated
> against their **future** play, across engine/oracle-graded domains.

Contributions (as realized): (1) the task formulation — *dynamic personalized
behavior simulation*, validated against future behavior; (2) a game-agnostic
simulator with a dynamic latent + joint move+timing prediction; (3) an
engine/IRT-graded, temporally-split, per-individual benchmark with a synthetic
ground-truth check; (4) empirical findings — the **when-not-what asymmetry**, the
equal-capacity evolving-vs-memoryless control, the heterogeneity scaling law, the
backbone-dependent channel ordering, and the G4 released-SOTA add-on.

*Full cited related work + the axes/shared-territory analysis:
`documents/related_work.md`. Detailed results + reproduction:
`documents/results_ec.md`. Synthesis: `documents/paper_draft.md`.*
