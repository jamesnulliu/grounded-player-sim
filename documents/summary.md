# Project Summary: Beyond Static Personas

## The story, directly

**The problem.** Systems that simulate specific humans freeze the person
into a static profile — a persona paragraph ("this user is impatient"), a
rating, a fixed embedding — and reason over it. Under that design the
simulated policy is *static attribute + reasoning*, and everything a real
person carries **between** decisions (tilt after a loss, fatigue late in a
session, panic under a ticking clock — the emotion-like state the
environment writes into them) is unrepresentable by construction.

**The one-line pitch.** *Everyone now says users have dynamic emotion and
static personas are not enough — nobody has measured it. This paper is the
measurement.*

**Why the measurement is not a foregone conclusion.** Five bets an informed
practitioner loses against this data: (1) "tilt shows in which moves you
pick" — no, moves are near-null; it shows in think-time; (2) "updating
always helps" — no, it fails in Go and on both real education timers;
(3) "a fitted per-student ability score helps" — no, frozen it is *worse
than no profile* on 6/8 datasets; (4) "a learned neural state beats
hand-designed running stats" — no, at equal information they tie;
(5) "just prompt the LLM dynamically" — no, inside the LLM an updating
persona *ties* a frozen fitted one on a low-drift horizon; the gap that
matters is fitted-vs-guessed, not frozen-vs-updating. An intrinsically-true
claim cannot lose bets or have failure boundaries; this one has both,
mapped.

**The one law that ties it together.** *Freezing a person's description
fails in proportion to how fast the person drifts.* Education students
learn over weeks → the frozen profile fails on every dataset. A blitz
player's typical speed barely moves in a month → the frozen persona ties
the updating one in the LLM. The fast, emotion-like state (clock panic,
tilt) is real but small and lives in chess think-time, where the sharp
board-native instrument resolves it and the blunt LLM probe cannot.

**The hypothesis.** A person's policy decomposes as **static attribute +
evolving behavioral state**. We never claim to measure emotion (no emotion
labels exist in our data); we claim the decomposition's *dynamic term*
exists, is separately measurable, and improves prediction of real
individuals' future behavior.

**Why games and tutoring.** Free-form interaction has no fixed action space
to score likelihoods over. Chess and knowledge tracing have discrete
actions, logged per-decision timing, and years of per-individual public
data — the one place the decomposition is cleanly testable, and a lower
bound any user simulator must answer to.

**The controls that make it believable.** The dynamic term is isolated
five ways, each killing one objection:

- an equal-capacity, same-input **memoryless twin** (same parameters, same
  inputs, recurrence reset every step) — kills "it just sees history";
- a learned **static per-player embedding** over a released SOTA think-time
  model (Allie) — kills "it's just identity/individualization";
- a training-free **structured memory** (running statistics of the raw
  history, linearly read out) — kills "an evolving latent is just a memory"
  by *building the memory and racing it*;
- a **fitted frozen profile** on KT (one number per student from their own
  training answers) — kills "then a static profile also works": it fails
  on every dataset;
- the **persona ladder inside the LLM** (G5: none / frozen fitted persona /
  updating text scorecard / soft-vector) — carries the whole question into
  the channel today's LLM user simulators actually use.

## The model: input, output, and the twin

One task: given everything a player did so far, predict their next
action — both the move and how long they will think.

**Input, per decision point** (both arms see exactly the same thing):

- the position (12×64 board planes for the from-scratch backbone; text
  for the LLM arm);
- the clock (time remaining);
- four engineered history features summarizing the session so far:
  `time_pressure` (0→1 as the clock drains), `post_loss` (spikes after
  a loss, decays over ~3 games), `fatigue` (ramps with games played
  this session), `momentum` (signed win rate over the last 5 games).

**The model (D, evolving).** A small GRU folds those features, step by
step, into a latent state `z_t`, updated after every action. `z_t` is
injected into a swappable backbone — the from-scratch board CNN
(headline results) or Qwen3 (as a hidden prefix or a verbal note).

**The twin (B, memoryless).** The same network, same parameter count,
same inputs — but the recurrence is reset before every step, so it sees
the same instantaneous features and cannot accumulate them.

**The memory arm (M, structured memory).** No learned parameters: a
hand-designed vector of causal running statistics (running/EWMA/lagged
log think-time, residuals against Allie's own predictions, premove rate,
recent outcomes, session position), linearly read out. Deliberately
*richer-input* than the GRU (it reads raw past think-times; the GRU never
does) — the strongest thing a "memory module" could store for this task.

**Output, two heads** — scored by held-out NLL (lower = better) on a
strict per-player future split:

- **what:** a distribution over the player's legal next moves;
- **when:** a distribution over think-time — a zero-inflated log-normal.
  The timing head reads *only* the latent, which is why the timing result
  is backbone-independent by construction.

In the education arm the same machinery reads a student's exercise
stream and predicts response correctness instead of moves.

## What we found

1. **The dynamic term is real — under two different instruments.** The
   evolving model beats its memoryless twin on future think-time everywhere
   tested, still adds value on top of Allie (a released SOTA think-time
   model), and beats Allie + a static per-player embedding on 2/3 cohorts
   (pooled −0.0126, every-seed audit). The training-free structured memory
   beats the static profile on **3/3** cohorts at 2–3× that margin
   (pooled −0.0276) — including 2019-07, where the learned arm was null.
   The "2019 null" was a learning failure, not absent dynamics.
2. **What matters is carrying dynamic state, not the mechanism.** The
   richer-input structured memory matches or beats the four-feature learned
   latent on chess timing (pooled evolving−memory = +0.0150), and the
   input-matched control settles it: a GRU over the *same* 15 memory
   statistics ties the linear memory readout (pooled +0.0096, CI crossing
   zero) while still beating static (−0.0180, significant). Given equal
   information, learned and hand-designed dynamics coincide. The honest
   claim is story-first: *any* faithful carrier of within-person dynamics
   beats the static profile; the information the state carries matters more
   than whether its update rule is learned. (`results/memory_gru_arm.txt`)
   **This transfers to education:** the KT memory arm (running stats of the
   student's answers, logistic readout, zero trained parameters) beats the
   memoryless twin on 7/8 datasets — the same profile as the learned
   latent — ties it on 5, and *repairs* ASSISTments 15, where the learned
   arm's training reversed (D−B +0.0063) but memory still wins (M−B
   −0.0058). "Memory suffices" is cross-domain. (`results/kt_memory_arm.txt`)
3. **The state is legible in timing, nearly invisible in choices.** Move
   prediction shows almost no gain, and a probe from the state to
   "deviation from a strong move model" recovers essentially nothing
   (R² = 0.009 vs 0.93 on a known synthetic state).
4. **The edge lives where behavior is least average — where an emotion-like
   state should matter.** ~3× larger under time pressure (after variance
   control) and ~3× larger for the weakest players; flat for post-loss and
   fatigue buckets (specific to the clock).
5. **It generalizes to education — as individualization that must keep
   updating.** Response-channel wins on 8 real datasets (22/24 seed
   cells); a temporal-shuffle control identifies that edge as
   order-invariant individualization — but order-invariant ≠ freezable: a
   fitted frozen per-student profile is beaten by the updating memory on
   **8/8** datasets (−0.011…−0.098, largest where students drift most)
   and loses even to the memoryless twin on 6/8. Real education response
   *times* do not transfer (two honest negatives).
   (`results/kt_static_arm.txt`)
6. **Only the learned state generates populations.** Sampling novel latents
   from the fitted prior recovers a population's accuracy distribution
   (W1 2× closer than average-person; generated recall 1.00 vs 0.00). No
   static persona set and no memory store defines this operation — there is
   no distribution over individuals to sample.
7. **How you inject the state depends on the backbone.** A hidden vector
   beats a text note in a from-scratch model; inside an LLM, that advantage
   disappears (the LLM reads the note semantically) — replicated on fresh
   data by G5, where the same scorecard numbers helped significantly more
   as language than as a trained soft prefix.
8. **Inside the LLM, fitted person-information is the gap that matters.**
   The G5 persona ladder (Qwen3-1.7B, 99 players, paired bootstrap): a
   fitted frozen persona and an updating text scorecard both beat
   no-information clearly (≈ −0.010, the project's largest LLM
   person-effect) and *tie each other* (+0.0009, CI spans zero) — the
   low-drift regime of the freezing law, and an upper bound on
   hand-written-persona practice, which fits personas from no data at
   all. Practical rule: fit the persona from the person's data, keep it
   updating when the person can drift, deliver it as text.
   (`results/g5_persona_ladder.txt`)

## Contributions

1. **The decomposition test:** policy = static attribute + evolving state,
   with the three controls (memoryless twin, static embedding over released
   SOTA, structured memory) that make the dynamic term attributable — on
   real human behavior with a timing target.
2. **The finding:** the dynamic term is robust across 6 years of real
   chess, survives released baselines, and reappears larger under a
   training-free memory instrument.
3. **The characterization:** when-not-what — the state shows in timing,
   not choices — plus the concentration signature (time pressure, weak
   players) that anchors the state reading.
4. **Generality + the generative capability:** response wins on 8 real KT
   datasets; population recovery and generation that persona libraries and
   memory modules structurally lack.
5. **The channel result:** hidden vs verbal injection ordering flips with
   the backbone's language prior.

## Difference from previous work

| Previous work | What it does | What it lacks (that we add) |
|---|---|---|
| Maia / Maia-2 / Allie | Human-like chess: rating-conditioned moves; Allie adds a think-time head | Static — no per-individual evolving state; no per-player future split |
| ChessMimic | Per-Elo-band move + clock transformers | Cohort-level, not per-individual; no memoryless-twin control |
| Matilda (v1 "Elo-Disentangled"), Maia4All | Per-player style conditioning of a strong policy | The style vector is frozen per player; no timing target |
| Ailed | Chess engine: static personality + dynamic "psyche" scalar | Our decomposition as a *generative design* — hand-tuned, engine-vs-engine, no human validation; we learn and validate it on real players |
| DASKT | "Dynamic affect" for knowledge tracing | Hand-engineered affect pipeline; no static-vs-dynamic ablation, no future-split validation |
| AnnaAgent, Customer-R1, TWICE | LLM user simulators arguing static personas are insufficient | Prompt/scheduler-delivered state, qualitative evaluation — the *premise* without the controlled test |
| LATTE | Evolving user state injected into a frozen LLM (recsys) | No timing target, no capacity-matched twin; baselines are static profiles |
| HumanLM, Duan et al. | LLM emits verbal psychological state / student profiles | Verbal-only channel; no timing; no equal-capacity control |
| MemGPT / generative-agents memory; Imperfect Learner, ContextSim | Memory modules for agents / user simulators | Not trained for behavior prediction; no population sampling; we build the strongest summary-statistic memory and race it |
| van der Linden, LaRT (psychometrics) | Stable per-person speed + item difficulty predicts RT | Speed is *stable*; we beat a stable-speed control on 3/5 cohorts (all 5 point estimates favor evolving) |

One line: **the premise (users drift; static personas can't) is now common;
the controlled, quantitative demonstration — released-SOTA / static /
memory ladder, future splits, timing target, population generation — exists
nowhere else.**

## Results (headline numbers)

D = evolving latent, B = memoryless twin, M = structured memory. All gaps
are held-out NLL in nats; **negative favors the first-named arm**;
significant = 95% player-bootstrap CI excludes zero.

### Chess: the dynamic term survives every control

| Test | Result | Significant? |
|---|---|---|
| Think-time, 4 eras (2017–2023) × 2 backbones, 5 seeds | D−B = −0.021 to −0.033 | Yes, all 8/8 conditions |
| Add-on over Allie locked (Spearman 0.62–0.65) | −0.018 to −0.033 | Yes, 3/3 cohorts |
| Strictest: Allie + static embedding vs + evolving | −0.017 / −0.001 / −0.019 | 2/3 cohorts; pooled −0.0126, significant |
| **Memory vs static (training-free dynamics)** | **−0.0148 / −0.0350 / −0.0327** | **Yes, 3/3 cohorts; pooled −0.0276** |
| Evolving vs memory | −0.002 / +0.034 / +0.013 | Memory wins pooled (+0.0150) — see finding 2 |

### Chess: when, not what

| Test | Result | Significant? |
|---|---|---|
| Move choice, same sweep | −0.0005 to −0.018 | Small (conv backbone) or null (MLP) |
| Probe: state → deviation-from-Maia-2 | R² = 0.009 (vs 0.93 on known synthetic state) | Near-null |
| Time-pressure terciles (variance-controlled) | High-pressure edge 2.7–3.6× larger | Every CI excludes zero |
| Rating terciles (480 players) | Weakest −0.0404 vs strongest −0.0139 (~3×) | Difference significant |

### Education (knowledge tracing) and population recovery

| Test | Result |
|---|---|
| Response prediction, 8 real datasets × 3 seeds | D wins 22/24 cells; 7/8 dataset means favor D |
| KT memory arm (training-free) vs twin | M−B significant on 7/8 datasets (same profile as D); D−M ties 5/8, M wins 2 (incl. repairing the ASSISTments-15 reversal), D wins 1 |
| KT fitted FROZEN profile (strongest static persona) | Beaten by the updating memory **8/8** (−0.011…−0.098); loses even to the twin 6/8 → "order-invariant" ≠ "freezable"; the estimate must keep updating |
| Population recovery (500 real students) | Wasserstein 0.074 vs average-person 0.147 (2× closer); rank corr 0.96 |
| Diversity coverage (recall) | Latent 0.75–1.00 vs average-person 0.00 |
| Generation (sampled new students, synthetic cohort) | Precision 0.93, recall 1.00 |

### LLM arm (Qwen3, SFT probe)

| Test | Result |
|---|---|
| State helps think-time, 0.6B → 8B | Δ = −0.011 to −0.014 at every scale |
| State helps moves | Null under LoRA at ≥4B; small (−0.007/−0.008) under full fine-tuning → timing ≈ 1.5× move |
| Hidden vs verbal channel | Board-native: hidden wins by 0.07–0.12 nats; inside the LLM: advantage disappears |
| G5 persona ladder: fitted info vs none | static−none −0.0104, memory−none −0.0096, both significant (99 players, paired bootstrap) |
| G5: updating vs frozen persona | memory−static +0.0009, CI [−0.0036, +0.0052] — tie (low-drift regime) |
| G5: text vs vector channel | hidden−memory +0.0045, hidden−static +0.0054 — language wins, replicating G3 |

### Honest negatives

| Negative | What it means |
|---|---|
| Real education response *times*: ASSISTments inconsistent, EdNet null | The timing result does not transfer to incidental UI timers; we hypothesize it needs a strategically managed clock |
| Go: null at every board size once board size is controlled | The naive mixed-cohort "positive" was a regime confound |
| KT temporal shuffle: edge undiminished | The education *response* edge is individualization, not order-tracking |
| Evolving (4-feature GRU) loses to richer-input memory | The learned instrument under-extracts; carrying dynamic state is the claim, not the GRU |
| KT frozen fitted profile: worse than no profile 6/8 | Freezing a drifting person mis-calibrates; the estimate must keep updating |
| G5: updating persona ties frozen persona in the LLM | The dynamic increment sits below the LLM probe's resolution on a low-drift horizon; "just prompt dynamically" is not supported at this scale |
