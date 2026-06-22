# Modeling the Player, Not Just the Move: A Dynamic Latent-State Framework for Personalized Human Behavior Simulation in Chess and Go

## Proposed Title

**Modeling the Player, Not Just the Move: A Dynamic Latent-State Framework for Personalized Human Behavior Simulation in Chess and Go**

Alternative titles:

- **Beyond Static Skill: Dynamic Within-Session Behavioral State for Individual Human Game Simulation**
- **Playing Like You, Right Now: Inferring Evolving Behavioral State from Individual Game Trajectories**

> **Revision note (v2).** This version is a substantial reframing of the original "Grounded User Simulation in Imperfect-Information Games (poker)" proposal, following a literature investigation. Three things changed and the reasons are stated up front so they are not rediscovered later by a reviewer:
>
> 1. **The contribution moved from "grounding vs. persona prompting" to "dynamic latent state vs. static individual model."** A 2024–2026 literature already establishes that (a) static persona prompting is behaviorally miscalibrated and (b) grounding simulation in real individual trajectories beats persona prompting. Those are settled motivation, not novelty. What is *not* claimed by any verified paper is a **dynamic, temporally evolving, per-individual latent behavioral state, validated against the same individual's future behavior.** That is the defensible whitespace.
> 2. **The domain moved from poker to Chess + Go.** Public poker hand histories lack stable per-player identity, are bot-contaminated, and are session-discontinuous — fatal for per-individual longitudinal modeling. Chess (Lichess) and Go (KGS/OGS/GoGoD) provide stable identity, high per-player volume, low contamination, per-move clock data, and — critically — **engine oracles (Stockfish, KataGo)** that give a measurable per-move suboptimality baseline. We knowingly trade away the imperfect-information setting to gain clean longitudinal data and a normative baseline.
> 3. **A single framework is evaluated on two games.** Chess gives strong named baselines (Maia family, Allie) to beat; Go provides novelty cover because no per-individual dynamic-state model exists there. Two domains sharing one engine-graded interface substantiate the "framework," not "one model," claim.

---

## 1. Motivation

Large language models and learned policies are increasingly used as *simulated humans* — to evaluate agents, train opponents, populate social simulations, and stand in for users. The dominant approach is to hand the model a static description (a persona, profile, trait, or emotional label) and ask it to behave accordingly. This is convenient and, by now, known to be wrong in a specific way.

Two claims about this failure are already established in the literature and should be treated as **settled background, not contributions**:

- **Static persona prompting is behaviorally miscalibrated.** Personas explain a small fraction of behavioral variance, models over-interpret directives into caricatured extremes ("Directive Amplification"), and long-horizon simulations collapse toward a "positive average person," losing individual differences and long-tail behavior.
- **Grounding simulation in a specific individual's real behavioral data beats persona prompting.** Interview-grounded and corpus-grounded individual simulators substantially outperform demographic- or persona-only baselines on reproducing real human behavior.

What remains genuinely open is the **third step**: real human behavior is not only individual, it is *non-stationary*. The same person plays differently when ahead vs. behind, fresh vs. fatigued, under time pressure vs. not, and after a bad loss vs. a clean win. Existing individual models — even the strong ones — represent a person as a **static** object: a fixed skill level, a fixed style embedding, or a one-time interview snapshot. Almost no work models an **evolving latent behavioral state** that is (a) fit to a specific real individual, (b) updated move-by-move from that individual's own trajectory, and (c) validated against that individual's *future* behavior rather than held-out random samples of past behavior.

This proposal targets exactly that gap, in two domains where it can be measured cleanly.

The central hypothesis is:

> **A personalized simulator that maintains a dynamic, per-individual latent behavioral state — inferred from the player's own action and timing trajectory — reproduces individual human behavior (move choice, timing, and within-session deviation) better than (i) static persona prompting, (ii) static individual models that fix a single skill level or style embedding, and (iii) population models.**

---

## 2. Why Chess and Go

The contribution is domain-independent, but it must first be demonstrated where it can be *measured*. Chess and Go are chosen for five concrete reasons, all of which fail in poker:

1. **Stable individual identity at volume.** Lichess usernames are permanent; prolific players have tens of thousands of games. Go records on KGS/OGS and the curated GoGoD professional collection give multi-year per-player histories.
2. **Low contamination.** Bots are flagged/segregated on Lichess; human game corpora are cleanly separable.
3. **Per-move timing signals.** Lichess open data ships per-move clock annotations (`[%clk ...]`); online Go servers record per-move time and byo-yomi consumption. Timing is the most direct observable proxy for the dynamic states we care about (time pressure, deliberation, tilt-driven speed-up).
4. **An engine oracle for suboptimality.** Stockfish (chess) and KataGo (Go) provide a per-move value/policy reference, so each human move can be scored by *points lost vs. optimal*. This converts "style" and "mistake" from vague labels into measurable, per-decision quantities — something no poker dataset offers at human-identity granularity.
5. **A shared abstraction.** Both are sequential, perfect-information games whose decision points expose the same interface: `(board state, legal action set, engine value/policy reference, time-remaining signal, running outcome stream)`. This common interface is what makes a *single framework* a legitimate claim rather than two separate models.

**Explicit scope decision.** We drop the imperfect-information setting from the original poker proposal. Perfect-information board games are where clean longitudinal data and an engine oracle co-exist; that combination is worth more to this specific contribution than partial observability. Imperfect-information generalization is named as future work, not promised.

**Honest limitation, stated up front.** Chess and Go are both board games. Demonstrating a framework on two similar domains is weaker generality evidence than a game + a non-game domain. We therefore (a) design the framework with a game-agnostic core so a third domain (e.g., longitudinal learning traces, forecasting belief-updates) can be added, and (b) list that cross-modality test as future work.

---

## 3. Research Questions

**RQ1 (Dynamics vs. static individual).** Does a dynamic, per-individual latent behavioral state improve next-move prediction, timing prediction, and calibration over a *static* individual model (fixed skill level or fixed style embedding) of the same player?

**RQ2 (Recovering interpretable states).** Does the learned latent state recover known behavioral phenomena — time-pressure degradation, post-loss change in risk/aggression and move quality (tilt), within-session fatigue/drift — and does it do so per-individual rather than only on aggregate?

**RQ3 (Future-behavior validation).** When trained on a player's earlier sessions, does the dynamic model predict the player's *later* sessions better than static baselines, under a strict temporal split? (This is the test that distinguishes a real dynamic model from one that merely memorizes a stable habit.)

**RQ4 (Persona-prompting contrast).** In a head-to-head where an LLM is prompted to "play as player X" (persona/profile) vs. a trajectory-grounded dynamic model of X, which better reproduces X's move and timing distribution? (Rides the settled persona-shallowness literature into a domain where the contrast has never been run.)

**RQ5 (Cross-game framework generality).** Does the *same* framework, with only the game-specific state encoder swapped, reproduce these results in both chess and Go? Do the learned latent-state dimensions transfer or correlate across games for players who play both?

---

## 4. Proposed Framework

The system is a **game-agnostic personalized simulator with a dynamic latent behavioral state**. Only the state encoder and the engine oracle are game-specific; the personalization, latent-state, and prediction modules are shared between chess and Go.

### 4.1 Shared decision-point interface

Each decision point, in either game, is represented by:

```text
game:               chess | go
player_id:          stable account identifier
state:              board encoding (game-specific encoder)
legal_actions:      legal move set
engine_reference:   oracle value + policy (Stockfish / KataGo) for this state
                    -> per-candidate-move value, and post-hoc points-lost of chosen move
time_signal:        time remaining, increment, time spent on this move,
                    byo-yomi state (Go), move number, phase of game
recent_outcomes:    running stream of recent results within and across the session
                    (win/loss, margin, engine-scored swings, blunders, time scrambles)
context:            time control, rating gap to opponent, color, session position
                    (1st game of session vs. 20th), wall-clock gaps between games
```

The engine reference is what makes the target well-posed: the model is not asked to play well, it is asked to **reproduce this player's deviation from optimal**, conditioned on their evolving state.

### 4.2 Modules

**1. Game-state encoder (game-specific).** Encodes the board. Reuses established architectures (e.g., the convolutional/transformer encoders used by Maia/Maia-2 for chess; KataGo-style board encoders for Go). This is deliberately *not* the contribution.

**2. Player representation (shared).** A per-player embedding capturing stable individual tendencies, learned with parameter-efficient adaptation so that low-data players are reachable (cf. the 20-games-vs-5000 efficiency results in recent individual-chess work).

**3. Dynamic latent behavioral state — the core contribution (shared).** A latent state evolving over the player's trajectory:

```text
z_t = f_phi(z_{t-1}, state_{t-1}, action_{t-1}, engine_outcome_{t-1},
            time_signal_{t-1}, result_stream_{t-1})
```

`z_t` is updated both **within a game** (move to move: deliberation, time pressure, accumulating pressure) and **across games within a session** (tilt after a loss, momentum after wins, fatigue late in a session). It is modeled as a sequential latent variable (candidate parameterizations: a state-space/HMM-style latent, a recurrent latent, or a structured latent with semantically anchored dimensions for time-pressure / post-loss / fatigue). It is **not required to be verbalized**; interpretability is tested post hoc by probing whether dimensions of `z_t` correlate with engineered indicators (time-trouble, recent loss margin, session depth).

**4. Probabilistic prediction heads (shared).**
- **Move head:** a calibrated distribution over legal moves, `P(a_t | state_t, player, z_t, engine_reference_t)`.
- **Timing head:** a distribution over time spent on the move (and, in Go, byo-yomi usage). Timing is a first-class prediction target, not a feature — it is one of the clearest behavioral fingerprints and a place where current models (Maia ignores time; Allie predicts only aggregate, Elo-conditioned think-time) leave room.

Evaluation emphasizes **likelihood and calibration**, not top-1 accuracy, because humans legitimately mix among reasonable moves.

### 4.3 Where (and whether) an LLM is used

The LLM is **not** the decision policy. Its justified roles are: (a) the persona-prompting baseline for RQ4; (b) optional generation of natural-language player summaries / scouting notes from trajectories, evaluated for whether they improve few-shot personalization or only add cost; (c) interpretability narration of latent-state dimensions. The core policy is a probabilistic model grounded in observed moves and timing. If the LLM components do not help, that is itself a reportable finding consistent with the existing "memory/RAG give only modest gains" results.

---

## 5. Experimental Plan

### Phase 0: Controlled synthetic players (ground truth known)

Before touching real data, construct synthetic players with **known dynamic mechanisms** — e.g., "plays optimally until a loss, then increases blunder rate for N games," "degrades move quality below T seconds remaining," "fatigues after game 15." Implement as perturbations of an engine policy. Because the generating mechanism is known, we can verify:

```text
P0.1  Does the dynamic latent state recover the injected mechanism?
P0.2  Does a static individual model provably fail where dynamics matter?
P0.3  Does persona prompting over-/under-shoot the injected trait?
P0.4  Calibration and identifiability of z_t under known ground truth.
```

This phase makes the central claims falsifiable independent of messy real data, and is fully feasible with OpenSpiel / python-chess / KataGo self-play.

### Phase 1: Chess (Lichess + Stockfish)

```text
Data:   Lichess open database (CC0); per-move SAN + [%clk] timing; ratings;
        bot accounts excluded. Focus on faster time controls (blitz/rapid)
        for high within-session dynamic density.
Oracle: Stockfish per-position eval (or the published Lichess Stockfish
        evaluation set) -> per-move centipawn loss.
Players: select humans with sufficient volume AND enough multi-game sessions
        to expose within-session dynamics.
Split:  strict temporal split per player (train = earlier sessions,
        val = middle, test = later sessions). No random split (it leaks
        stable habits and inflates results).
```

### Phase 2: Go (KGS/OGS + KataGo)

```text
Data:   KGS / OGS game records (SGF) with per-move time data; GoGoD for
        deep professional career histories. Prefer servers/time settings
        that preserve per-move timing and byo-yomi.
Oracle: KataGo per-move points-lost (winrate/score drop) on human moves.
Players: humans with multi-year or multi-session histories.
Split:  same per-player temporal protocol as chess.
Note:   Go games are longer and fewer-per-session than blitz chess, so
        cross-game tilt has less data per session; compensate by favoring
        faster Go and by leaning on rich within-game byo-yomi time dynamics.
```

### Phase 3: Baselines

```text
B1. Population model (no personalization)               -- e.g., skill-conditioned Maia / Maia-2 (chess);
                                                            KataGo human-SL or rank-conditioned model (Go)
B2. Static individual model                             -- per-player fine-tuned / embedded model with
                                                            NO dynamic state (Maia-Individual / Maia4All style)
B3. Static-skill state-space                            -- time-varying SKILL across matches, fixed within
                                                            (e.g., Glicko/TrueSkill-style or state-space skill);
                                                            tests "is the only dynamic just rating drift?"
B4. Aggregate time model                                -- Allie-style think-time, Elo-conditioned, not
                                                            per-individual (chess)
B5. LLM persona prompt                                  -- "play as player X" from a profile (RQ4)
B6. LLM persona + recent history                        -- profile + last K games
B7. Static-covariate tilt model                         -- recent win-ratio as a fixed covariate
                                                            (the Gee et al. 2025 formulation; mostly-null
                                                            aggregate result is the foil)
B8. Proposed dynamic latent-state model
```

### Phase 4: Evaluation

```text
1. Move prediction:   NLL, macro-F1, Brier, expected calibration error,
                      top-k likelihood, agreement-with-human rate.
2. Timing prediction: per-move time NLL / calibration; rank correlation of
                      predicted vs. actual think-time, PER INDIVIDUAL
                      (vs. Allie's aggregate correlation).
3. Suboptimality matching: predicted vs. actual centipawn/points-lost
                      distribution, including the conditional
                      P(blunder | low time), P(quality | post-loss).
4. Style / aggregate stats: per-player rollout statistics vs. real
                      (aggression, time-allocation profile, error rate by
                      phase/time-pressure), compared by KL / JS / Wasserstein.
5. Future-behavior fidelity (RQ3): all of the above on the temporally
                      held-out LATER sessions -> the decisive test that
                      the dynamic state generalizes forward, not memorizes.
6. State recovery (RQ2): post-hoc probes of z_t against engineered
                      indicators (time-trouble, recent loss, session depth);
                      does z_t predict the player's next-move quality swing?
7. (Optional, stretch) Opponent-preparation utility: tune an agent against
                      the simulator; test vs. held-out real games of the
                      target player. Scope as stretch — requires more
                      per-player data and an RL loop.
```

---

## 6. Expected Results

1. **Static individual models leave dynamic signal on the table.** B2 (per-player but static) will predict aggregate style well but mispredict exactly in the high-variance moments — time scrambles, post-loss games — where the dynamic state matters. The proposed model's gains will concentrate there.
2. **The latent state recovers interpretable phenomena per-individual.** Where aggregate static tilt studies found mostly-null effects (B7 / Gee et al.), a per-individual *dynamic* formulation should find heterogeneous, individually-real effects — turning a published null into a contribution.
3. **Timing is individually predictable.** Per-player think-time and time-allocation will be reproducible beyond what an aggregate, Elo-conditioned think-time model (B4/Allie) achieves.
4. **Persona prompting is coherent but miscalibrated** (RQ4), consistent with the established persona-shallowness literature; the contribution is demonstrating it specifically against move/timing distributions with an engine-graded yardstick.
5. **The framework transfers across games** (RQ5): the same architecture, re-encoded, reproduces the pattern in Go, where no prior individual-dynamic model exists — establishing that the contribution is the dynamic-state mechanism, not a chess-specific artifact.

A reportable *negative* outcome is also acceptable and valuable: if dynamics add little beyond static individual modeling once volume is high, that bounds the phenomenon and still answers RQ1–RQ3 cleanly (especially given Phase 0's ground-truth check).

---

## 7. Related Work and Positioning

### Individual human modeling in chess (the prior art to beat, and to differentiate from)

The Toronto CSSLab / Microsoft Research program owns *static* individual chess modeling:
- **Maia** (move prediction by skill level), **Maia-2** (unified skill-aware model), **Maia-Individual** and **Maia4All / "Learning to Imitate with Less"** (per-player modeling, data-efficient), **Behavioral Stylometry** (player identification from moves), **Designing Skill-Compatible AI**.
- **Allie** (human-aligned chess with search) is the closest timing prior art: it predicts pondering time and resignations and runs time-adaptive search — but **conditioned on Elo/skill, aggregate, not per-individual**, and with no evolving psychological state.

**Differentiation:** every one of these represents a player as a *static* object (a skill bin, a fixed embedding, or aggregate think-time). None maintain a latent state that evolves within a session and is validated against the player's future behavior. That is our axis.

### Dynamic behavioral state in games (the open gap)

- The one peer-reviewed chess "experiential effects / tilt" study enters history as a **static covariate** and reports **mostly-null aggregate effects**, explicitly flagging the absence of temporal-dynamics modeling — a published invitation.
- State-space skill models capture **skill drift across matches, fixed within a match** — not a within-session psychological state.
- No per-individual dynamic-state model exists for Go.

### LLM individual simulation (settled motivation; scoop risks to cite)

- **Grounding beats persona** is established (interview-grounded individual simulation; corpus-grounded user simulators). Treat as motivation, not novelty.
- **Scoop risks to cite and differentiate explicitly:**
  - **RealUserSim** — owns "grounding beats prompting" and coined "Directive Amplification"; but grounds in a *corpus* into reusable *static* profiles, with no per-individual binding and no dynamic state.
  - **Agent4Rec** — has an emotion+fatigue loop on real-data-initialized agents; but the dynamics are a *generic heuristic loop*, not a per-user learned/calibrated latent variable, and target session disengagement, not behavioral drift.
  - **HumanLM** — RL-learns latent states, but *population*-grounded and *per-context*, not per-individual and temporally evolving.
  - **OmniBehavior** — the friendly citation: it *measures* real per-individual interest drift, shows current methods miss it, and explicitly *calls for* structure-aware mechanisms. Quote as the invitation.
  - **BehaviorChain** — sequential behavior simulation, but over *fictional* personas, not real trajectories, and with no latent-state fix for its observed "snowball" errors.

### The one-sentence positioning

> Prior individual game models answer *"what move does this skill level / this player typically make?"* (static). We answer *"what move does this player make **right now**, given how their session has gone so far?"* (dynamic), and we validate it against their **future** play — across two engine-graded games.

---

## 8. Contributions

1. **Task formulation:** *dynamic personalized behavior simulation* — predicting an individual's evolving move-and-timing distribution conditioned on an inferred latent state, validated against future behavior — as distinct from optimal play, static individual modeling, and persona-prompted simulation.
2. **Framework:** a game-agnostic simulator with a dynamic latent behavioral state, engine-grounded suboptimality targets, and joint move+timing prediction, instantiated on chess and Go through a shared interface.
3. **Benchmark + protocol:** an engine-graded, temporally-split, per-individual benchmark on Lichess and Go records, with calibration, suboptimality-matching, future-behavior fidelity, and latent-state-recovery metrics — and controlled synthetic players with known dynamics for falsifiable validation.
4. **Empirical findings:** dynamic state vs. static individual vs. persona prompting, including per-individual recovery of phenomena (time pressure, post-loss tilt, fatigue) that aggregate static studies report as null, demonstrated to transfer across two games.

---

## 9. Feasibility and Risks

**Confirmed feasible:** Lichess open data (CC0) ships per-move `[%clk]` timing, ratings, and stable usernames, with a separate large Stockfish evaluation set; KataGo (MIT) provides an engine oracle for arbitrary Go positions; KGS/OGS/GoGoD provide per-player Go histories. The Maia line provides reusable encoders and strong, named baselines.

**Risk 1 — Go per-move timing coverage.** Per-move clock data is standard on some Go servers/time settings and weaker in others; SGF time annotations vary. *Mitigation:* confirm timing availability before committing Phase 2 scope; prefer servers/time controls that log per-move time and byo-yomi; if timing is thin, Go can still support move-quality dynamics (post-loss, fatigue) even with coarser timing, and chess carries the timing-specific claims.

**Risk 2 — Within-session density.** Dynamic states need multiple decisions under varying conditions per session. *Mitigation:* favor faster time controls (blitz/bullet chess; fast Go) where many games and vivid time scrambles occur per session; require minimum session-depth when selecting players.

**Risk 3 — "Isn't this just Maia / just rating drift?"** *Mitigation:* baselines B2 (static individual) and B3 (state-space skill drift) are designed precisely to absorb those explanations; the contribution must beat both, and the future-behavior split (RQ3) plus synthetic ground truth (Phase 0) make the dynamic claim falsifiable.

**Risk 4 — Two-board-game generality.** As noted, this is weaker than game + non-game. *Mitigation:* game-agnostic core; explicit future-work path to a non-game longitudinal domain; the cross-game transfer analysis (RQ5) extracts maximal generality evidence from the two domains chosen.

**Risk 5 — Identifiability of the latent state.** A flexible recurrent latent can fit anything and explain nothing. *Mitigation:* prefer structured/probeable latents; require Phase 0 recovery of injected mechanisms; report state-recovery probes (RQ2) as a first-class result, not an afterthought.

---

## 10. Suggested Venue and Framing

Frame as an **empirical + benchmark + method** paper, led by the **temporal-dynamics-of-individual-behavior** thesis — *not* "anti-persona-prompting" (true but claimed) and *not* "individual style" (owned by Maia).

- **Strong fit:** IEEE Conference on Games (CoG), Computers and Games, AIIDE; agents / human-modeling / behavior-simulation tracks and workshops at NeurIPS / ICML / ICLR / AAMAS.
- **Stretch:** a main-track submission is plausible if the future-behavior fidelity result and the per-individual recovery of a previously-null tilt effect are strong, because those are genuinely new claims rather than a domain port.

Minimal first paper:

```text
1. Phase 0 synthetic players with known dynamic mechanisms (falsifiable core).
2. Chess (Lichess + Stockfish): full baseline suite incl. Maia-individual, Allie, persona-LLM.
3. Go (KataGo): same framework, re-encoded -> generality + Maia-objection defense.
4. Headline results: dynamic > static individual on future-behavior split;
   per-individual recovery of time-pressure / post-loss effects;
   timing predictable per-individual; persona prompting miscalibrated.
```

---

## References to Verify and Cite Precisely

The following were checked during the investigation; confirm IDs/venues at draft time (some are recent preprints).

**Individual chess modeling (differentiate against):**
- Maia — "Aligning Superhuman AI with Human Behavior," McIlroy-Young et al., KDD 2020 (arXiv:2006.01855).
- Behavioral Stylometry, McIlroy-Young et al., NeurIPS 2021 (arXiv:2208.01366).
- Maia-Individual — "Learning Models of Individual Behavior in Chess," KDD 2022 (arXiv:2008.10086).
- Designing Skill-Compatible AI, Hamade et al., ICLR 2024 (arXiv:2405.05066).
- Maia-2 — "A Unified Model for Human-AI Alignment in Chess," NeurIPS 2024 (arXiv:2409.20553).
- Maia4All — "Learning to Imitate with Less," Tang et al., 2025 / TMLR (arXiv:2507.21488).
- Allie — "Human-Aligned Chess With a Bit of Search," Zhang et al., ICLR 2025 (arXiv:2410.03893).

**Dynamics / state-space (the gap):**
- Gee, Seese, Curley, Ward — "Experiential Effects in Online Chess," JQAS 2025 (arXiv:2503.21713). [Static-covariate, mostly-null foil.]
- Duffield, Power, Rimella — "A state-space perspective on online skill rating," JRSS-C 2024. [Across-match skill drift, fixed within match.]
- Chess rating from moves + clock (CNN-LSTM), arXiv:2409.11506. [Clock used for skill, not psychological state.]

**Engine oracles / encoders:**
- Stockfish; Lichess Stockfish evaluation dataset (database.lichess.org, CC0).
- KataGo — Wu, "Accelerating Self-Play Learning in Go," 2019 (verify arXiv ID, ~1902.10565); MIT-licensed.
- DeepMind "Grandmaster-Level Chess Without Search," Ruoss et al., NeurIPS 2024 (arXiv:2402.04494). [Mimics Stockfish, not humans.]
- Karvonen, "Emergent World Models … in Chess-Playing LMs," COLM 2024 (arXiv:2403.15498).

**LLM individual simulation (settled motivation + scoop risks):**
- RealUserSim (arXiv:2605.20204) — cite + differentiate.
- OmniBehavior (arXiv:2604.08362) — the invitation.
- BehaviorChain, ACL 2025 Findings (arXiv:2502.14642).
- HumanLM (arXiv:2603.03303) — cite + differentiate.
- Agent4Rec, SIGIR 2024 (arXiv:2310.10108) — cite + differentiate.
- Generative Agents, Park et al. (arXiv:2304.03442); "Generative Agent Simulations of 1,000+ Individuals" (arXiv:2411.10109) — static individual grounding.
- τ-bench (arXiv:2406.12045) — persona-prompted user-sim baseline lineage.

**Tooling / data:**
- OpenSpiel (synthetic-player phase); python-chess (`%clk` parsing, `GameNode.clock()`).
- Lichess open database (CC0); KGS/OGS records; GoGoD professional collection.

**Future-work generality domains (non-game):**
- Duolingo Half-Life Regression traces; Good Judgment Project forecasting data (CC0).
