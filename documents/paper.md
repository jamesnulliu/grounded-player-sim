# When, Not What: An Evolving Per-Individual Latent State Predicts a Person's Future Think-Time, Not Their Choices

*Manuscript draft (markdown, pre-LaTeX). Target: ICLR, primary area
applications to neuroscience & cognitive science. All numbers are taken
verbatim from the frozen artifacts in `results/`; the claim–evidence mapping
lives in `documents/claim_evidence_map.md`. Figure slots are marked TODO.*

*Draft-scaffolding notes (strip at LaTeX translation): (1) appendix-demotion
candidates if the page budget requires: Table 5 (static-vs-evolving over
Allie), the scale-sweep rows of Table 8. (2) Before submission, state
definitively why the E-C3 move gap on 2013-01 (−0.067, 3 seeds, 15 epochs)
exceeds the at-scale tier-1 2013-01 move gap (−0.0027 mlp / −0.0149 conv):
the documented causes (joint-objective suppression + era differences) cover
the clocked cohorts, but the 2013 protocol difference needs one verified
sentence. (3) Title uses "Not Their Choices"; the body claim is calibrated
to "almost no state dependence" — confirm the title's rhetorical compression
is acceptable or soften to "More Than Their Choices."*

## Abstract

People do not behave the same way twice. A chess player who just lost plays
the next game differently, and a student who struggled through ten problems
answers the eleventh differently. Systems that simulate specific humans
represent them statically, as a persona, a rating, or a fixed embedding, so
they cannot track this drift. History-conditioned models could track it, but
they cannot say whether their gains come from accumulated state or from
merely seeing recent context. We model the drift with a per-individual
latent state that evolves over a person's own action-and-timing trajectory,
and we isolate its value with an equal-capacity, same-input **memoryless
twin** evaluated on strict future splits of that person's data. Our central
finding is an asymmetry: the evolving state is far more legible in *when* a
person acts than in *what* they choose. On real Lichess chess the evolving
latent beats the twin at predicting future think-time in all eight
era-by-backbone conditions from 2017 to 2023, every confidence interval
excluding zero, and it improves held-out think-time likelihood over Allie, a
released think-time model, in the direct add-on test on all three cohorts
tested (two of three under the strictest controls). Move choice, by
contrast, shows almost no state dependence. A probe recovers deviation from
Maia-2, a released human-move model, at R² = 0.009. The timing edge
concentrates where behavior is least predictable from the average, 2.7–3.6×
under time pressure after variance control and about 3× for the weakest
players. On eight real knowledge-tracing datasets the evolving latent wins
on responses in 22 of 24 dataset-seed cells and recovers the population's
accuracy distribution at half the Wasserstein distance of an average-person
baseline. Response *times* do not transfer on the two education datasets
tested; we hypothesize the timing channel requires time to be a
strategically managed resource. Finally, which injection channel carries the
state is backbone-dependent: a trained hidden vector beats a verbal note on
a board-native backbone, and that advantage disappears inside an
instruction-tuned LLM. [TODO: code + frozen-protocol release sentence.]

---

## 1 Introduction

Simulated humans are becoming infrastructure. Tutoring systems are tuned
against simulated students, game platforms train and evaluate against
human-like opponents, and interactive agents are stress-tested against user
models. All of these need models of *specific* people rather than an average
person, and the concrete tasks are unambiguous: given a named Lichess
player's games through April, predict how long they will think and what they
will play in May; given a named student's first hundred answers, predict the
hundred-and-first.

Current approaches to modeling a specific person fail at a specific step:
they freeze the person. Rating-conditioned and per-player-embedding chess
models fix each player as a static vector, and persona-conditioned LLM
simulators fix them as a static text profile, so none of them can track the
within-session drift that human behavior visibly exhibits. A player tilts
after a loss, fatigues late in a session, and reprioritizes under a ticking
clock. Models that condition on recent history could in principle track this
drift, but they confound two different capabilities: *accumulating* a state
over a person's trajectory and merely *seeing* recent context. To our
knowledge, no prior line separates the two with a capacity-matched,
same-input control on real human behavior with a timing target (§2), so even
where history-conditioning wins, the win cannot be attributed to accumulated
state.

Our starting point is that if a person carries an evolving behavioral state,
that hypothesis makes three testable predictions. The state should be most
legible in self-paced *timing*, the channel a person controls directly, and
less legible in choices that the task itself largely constrains (§4.2). Its
predictive value should *concentrate* where behavior departs most from the
population average, under time pressure and for less disciplined players,
rather than spreading uniformly (§4.4). And capturing it should require
*accumulation*, so an equal-capacity model fed the same instantaneous inputs
but denied a persistent state should fall measurably short (§4.1). Each
prediction is falsifiable against a memoryless twin on held-out future
behavior.

We test all three with a deliberately simple architecture. A recurrent
injector accumulates a per-individual latent state over the person's own
action-and-timing trajectory and passes it to a swappable backbone, either
as a hidden vector or as a verbal note. The backbone predicts the next
action and its think-time, and every comparison is scored on a strict future
split of that person's data (Figure 1, TODO: architecture figure). The
design lets one injector and one trainer run unchanged on chess and
knowledge tracing (KT), and it makes the control exact: the memoryless twin
is identical in everything but state accumulation.

Our contributions, each tied to the experiment that supports it:

1. **An equal-capacity evolving-vs-memoryless control on strict future
   splits.** Identical parameters, inputs, and optimizer; only state
   accumulation differs. The evolving latent wins future think-time in all 8
   clocked era-by-backbone conditions (Table 3) and survives the twin at
   twice its latent width (single-seed sweep) (§4.1–4.2).
2. **The when-not-what asymmetry on real humans.** Move choice is a
   near-null, small at best and backbone-sensitive, and a probe from the
   latent to deviation-from-Maia-2 reaches only R² = 0.009. The timing edge,
   by contrast, survives baselines at or above the best published think-time
   rank correlation on Lichess, including Allie's released think-time head:
   the add-on is significant on all three cohorts in the direct test and on
   two of three under the strictest co-fit and identity controls (§4.2–4.3,
   Tables 3–5).
3. **A mechanism account.** The timing edge concentrates 2.7–3.6× under time
   pressure after variance control and about 3× for the weakest players; on
   a synthetic player with a known hidden state the latent encodes that
   state (probe R² = 0.93 vs 0.65) and clamping it moves predictions
   monotonically (§4.4).
4. **Generality and utility beyond games.** On eight real knowledge-tracing
   datasets the evolving latent wins responses in 22 of 24 dataset-seed
   cells; the same latent recovers real population heterogeneity at half the
   Wasserstein distance of an average-person baseline and generates diverse,
   plausible synthetic students (recall 1.00 vs the matched average-person's
   0.00). Two real education response-time tests are negative and set the
   timing claim's scope (§4.5–4.6, Tables 6–7).
5. **The injection-channel ordering is backbone-dependent.** The trained
   hidden vector beats the verbal note by 0.07–0.12 nats on a board-native
   backbone; inside Qwen3 the verbal note helps think-time on all seeds
   while the hidden channel adds no advantage over it. A supervised
   fine-tuning (SFT) probe reproduces a timing-over-move advantage at every
   scale from 0.6B to 8B parameters: a clean move-null under low-capacity
   LoRA adapters, a graded advantage of about 1.5× under full fine-tuning
   (§4.7, Table 8).

## 2 Related work

**Human-like and per-individual chess.** A fast-moving line models human
rather than optimal chess. Maia conditions a policy on a rating scalar to
match a population at a given Elo (McIlroy-Young et al., 2020), and Maia-2
unifies the rating ladder in one model (Tang et al., 2024). ChessMimic
(arXiv:2606.04473) sharpens this to per-100-Elo-band transformers for move,
clock, and outcome, reporting per-move think-time correlation r ≈ 0.41, and
Allie (Zhang et al., 2025) adds a dedicated think-time head to a
decoder-only policy. A second cluster is genuinely per-individual,
conditioning a strong base policy on a static per-player style vector
(Matilda, arXiv:2606.25176; Maia4All, arXiv:2507.21488). All of these
freeze the person.
They are static per player or cohort-level on the clock, none validate on a
per-individual *future* split, and none isolate within-session dynamics from
static individualization. We use their strongest released artifacts, Maia-2
and Allie, as the baselines our evolving latent must add value over (§4.3).

**Dynamic psychological state in games.** Closest in spirit is Ailed
(arXiv:2603.05352), a psyche-driven chess engine that modulates move
selection and latency by an evolving emotional state. It shares our premise
that play is state-dependent, but it is a *generative* construct: by its
authors' own account it has no human-subject validation, so its state
dynamics are asserted rather than measured against real players. We make the
corresponding claim falsifiable. The evolving state is fit to, and scored
against, specific players' held-out future games, and its value is
established by an equal-capacity control rather than by construction.

**Timing as a readout of latent state.** That latency reveals latent
cognitive state more richly than accuracy is a settled result in
response-time psychometrics: van der Linden's hierarchical speed-accuracy
model (2006; 2007) combines a *stable* per-person latent speed with item
time-intensity, and later dynamic extensions and change-point models
(e.g. arXiv:2605.29182) report response times out-predicting accuracy-only
item-response models. The bar this literature sets is therefore not "timing
beats choice" but "an *evolving* state beats a *stable* per-person speed
calibration." We test that bar directly. A per-player constant fed the same
features and timing head as the evolving latent, the neural analogue of the
stable-speed term, is beaten on three of five real cohorts, with all five
point estimates favoring the evolving state (§4.4).

**Evolving user state in simulators.** Sequential recommendation and user
simulation model evolving user state and routinely contrast it against
memoryless baselines. LATTE (arXiv:2605.26612) forecasts an evolving
per-user preference state injected as a soft token into a frozen LLM under a
future temporal split, and already runs a matched evolving-vs-static
comparison, so that contrast alone is not ours to claim first. HumanLM
(arXiv:2603.03303) RL-trains an LLM to emit natural-language psychological
states aligned to real users, and history-aware verbal student profiles have
been RL-trained on tutoring dialogues (Duan et al., arXiv:2605.30051).
UniMaia (arXiv:2605.27767) steers a chess policy with natural-language
descriptions of desired play, a controllability result that motivates
comparing injection channels. What none of these run is an *equal-capacity,
same-input* memoryless twin on a per-decision oracle domain with a timing
target, the control that isolates accumulated state from
history-conditioning. And each commits to a single injection channel, verbal
text or a single soft vector; we compare the two head-to-head and find the
ordering backbone-dependent (§4.7).

## 3 Method

**Decision points.** We reduce every domain to one interface. A person's
trajectory is a sequence of decision points; each bundles the observable
context of one decision: the game or item state, the legal action set, a
per-decision reference signal from an external oracle, a time signal (clock
remaining, elapsed time), and the person's recent outcome stream. The chess
policy itself is oracle-free; the reference signal serves evaluation and the
escalating baselines of §4.3, and in KT the item-difficulty reference feeds
the backbone directly. From the decision point we compute a small vector of
history features `h_t` (time pressure, post-loss recency, fatigue, momentum)
which both model arms receive. Chess and KT share this interface unchanged.

**The evolving latent.** A gated recurrent unit `f_φ` accumulates the
history features into a per-individual state, `z_t = f_φ(z_{t−1}, h_t)`. The
recurrence is causal: `z_t` depends only on steps up to `t`. A few latent
dimensions are anchored during training to the interpretable history
features (time pressure, post-loss tilt, fatigue, momentum); the remaining
dimensions are free. The state reaches the backbone through one of two
channels, either as the full hidden vector or rendered as a short verbal
note describing the anchored dimensions, so the channel comparison in §4.7
holds the state fixed and varies only its encoding.

**The memoryless twin.** The control shares the injector's architecture,
parameter count, inputs, and optimizer; the only difference is that the
incoming hidden state is zeroed at every step, so the twin sees the same
instantaneous `h_t` but can accumulate nothing. Because the recurrent
weights `W_hh` then receive no gradient, roughly two-thirds of the
injector's parameters are inert in the twin; we verified the comparison is
not a capacity artifact with a width sweep in §4.1. We chose this twin over
a generic history-window baseline because it removes exactly one capability,
persistence, and nothing else.

**Backbones.** The chess backbone is board-native and oracle-free. FEN board
planes (the standard chess position encoding) feed an MLP or a convolutional
encoder, two backbone variants, that emits factored from/to logits masked to
legal moves. The KT backbone is a logistic head over item difficulty and the
latent. The LLM arm (§4.7) uses Qwen3, an open family of instruction-tuned
LLMs spanning 0.6B–8B parameters, served via the sglang inference engine.
Think-time is modeled by a log-normal head whose parameters read *only* the
latent, never the board encoder; because real clocks quantize to whole
seconds and premoves put mass at zero, the head is zero-inflated on clocked
data. This choice makes the timing comparison structurally independent of
backbone strength, which §4.3 then also verifies empirically against
released models.

**Synthetic control players.** For mechanism experiments we constructed
players whose hidden state is *known*. The synthetic chess player carries a
hidden tilt `u_t`, a leaky integral of its recent losses; high tilt degrades
its move quality and slows its think-time, and `u_t` is recorded at every
decision. The synthetic student is the KT analogue: a hidden frustration
state, again a leaky integral of recent errors, lowers the probability of a
correct response and slows response time. In both, the hidden state is
constructed so it cannot be reconstructed from the instantaneous windowed
features alone; recovering it requires accumulation.

**Training and evaluation.** We fit `φ` by SFT on move negative
log-likelihood (NLL) plus λ times timing NLL over each person's *earlier*
trajectory and scored the held-out *later* trajectory. We use two split
forms: the *fraction* split holds out the last 30% of a player's moves, and
the *future-sessions* split holds out the player's later sittings entirely;
the future-sessions split is the decisive form. The split is leakage-safe by
construction: the loss is masked to training steps, and because the
recurrence is causal, warming the latent over the whole trajectory cannot
peek at the future; `z_t` at a held-out step still sees only that person's
past. Significance is assessed by bootstrap over *players*, the independent
unit; decisions within a player are correlated and are never resampled
directly. We report the 95% bootstrap CI and call a result significant when
the CI excludes zero. Alongside it we report P(D−B<0), the fraction of
player-bootstrap resamples in which the evolving latent wins, as a
directional-consistency summary rather than a frequentist p-value. The
experiments below test the three predictions of §1 in turn.

## 4 Experiments

Throughout, **D** denotes the evolving latent and **B** the memoryless twin.
NLL is held-out and lower is better, so D−B < 0 favors the evolving latent;
all gaps are in nats. Table 1 summarizes the data. Lichess archives carry
per-move clocks only from 2017 onward, so the 2013-01 cohorts support the
move channel only, and "clocked cohorts" below means 2017-04 through
2023-04.

**Table 1 — datasets and cohorts.**

| Domain | Source | Cohorts | Unit counts | Channels |
|---|---|---|---|---|
| Chess | Lichess open DB | 2013-01 (blitz+rapid, move-only), 2017-04, 2019-07, 2021-04, 2021-06, 2023-04 | 100–120 players each, ≥30 games, ≈70k decisions/cohort | move NLL, think-time NLL |
| Knowledge tracing | ASSISTments 09/12/15/17, KDD Algebra + Bridge, Spanish, Statics | 8 datasets, frozen manifest | 150–500 students each, 3 seeds | response NLL |
| KT timing | ASSISTments 2009 `ms_first_response`, EdNet-KT1 singleton bundles | frozen protocols | 500 students × 3 seeds | response + timing |
| Go | OGS (Online Go Server) JSON think-times | 9×9 / 13×13 / 19×19 | up to N=519 trajectories | timing |
| Synthetic (known state) | hidden-tilt player / frustrated student (§3) | chess + KT | 24 players × 30 games; 24 students × 120 items | move/response + timing |
| LLM arm | Lichess 2017 | Qwen3 0.6B–8B | 3 seeds per cell | move + timing completion NLL |

### 4.1 Is the evolving latent more than history-conditioning?

The equal-capacity control answers the first objection any reader should
raise: perhaps the latent wins only because it sees history at all. It does
not. The twin sees the same history features at every step and still loses
on every split tried (Table 2). Against a static per-player embedding, the
gap is larger still, so identity alone does not explain the win either. The
decisive form is the future-sessions split, which scores the model on games
from sittings the model never trained on; the win persists there (−0.067)
and replicates on an independent rapid-time-control cohort (−0.062).

**Table 2 — chess move NLL vs controls (2013-01, 100 players, 3 seeds).
Gaps in nats; negative favors the evolving latent; CIs are 95%
player-bootstrap intervals.**

| Control | Split | D−B | 95% CI | P(D−B<0) |
|---|---|---:|---|---:|
| static per-player embedding | future sessions | −0.120 | [−0.135, −0.106] | 1.00 |
| memoryless twin | fraction | −0.069 | per-seed range [−0.095, −0.039]* | 1.00 |
| memoryless twin | future sessions | −0.067 | [−0.077, −0.057] | 1.00 |
| memoryless twin (rapid replication) | future sessions | −0.062 | [−0.072, −0.052] | 1.00 |

*\*This row reports the range of per-seed CIs rather than a pooled CI.*

A capacity sweep closes the remaining loophole. Giving the twin a latent
twice as wide leaves the evolving win intact; the gap closes only at four
times the width, where the twin carries about 25% more total parameters
(single-seed sweep). The advantage is the persistence, not the parameters.

The large move gaps in Table 2 do not generalize to the clocked cohorts, and
the difference is explainable. Table 2's gaps come from the unclocked 2013
cohorts, which train move-only; on the clocked cohorts of §4.2, which train
jointly with the timing objective, the move gap is an order of magnitude
smaller. Two documented causes contribute: the joint
objective suppresses part of the move advantage because both heads compete
for one latent (a λ = 0 control on 2017-04 moves the gap from −0.0001 to
−0.0066, better but still not significant), and the later eras carry a
genuinely weaker move-dynamics signal. This is why the paper's move-channel
claim is calibrated to §4.2's at-scale numbers, not to Table 2.

### 4.2 Is the state legible in timing or in move choice?

This is the paper's central question, and the answer is asymmetric. At scale
(five seeds, two backbones, four clocked cohorts spanning 2017 to 2023), the
evolving latent beats the twin on future think-time in all eight
era-by-backbone conditions, with the effect tightly clustered at −0.021 to
−0.033 nats and every CI excluding zero (Table 3). The move channel is a
different story. The convolutional backbone shows a small win on every
cohort, but the MLP backbone is null on three of four clocked cohorts, and
multi-seed pooling corrected single-seed swings as large as ±0.03 down to
±0.005. We therefore read the move effect as small and backbone-sensitive.

**Table 3 — timing and move D−B at scale (5 seeds × 2 backbones,
future-sessions split, 120 players/cohort). Gaps in nats with 95%
player-bootstrap CIs; negative favors the evolving latent.**

| Cohort | mlp timing | conv timing | mlp move | conv move |
|---|---:|---:|---:|---:|
| 2017-04 | −0.0262 [−0.0378, −0.0154] | −0.0329 [−0.0417, −0.0247] | +0.0005 [−0.0006, +0.0016] | −0.0046 [−0.0056, −0.0037] |
| 2019-07 | −0.0277 [−0.0401, −0.0159] | −0.0331 [−0.0426, −0.0240] | −0.0009 [−0.0025, +0.0008] | −0.0083 [−0.0100, −0.0067] |
| 2021-04 | −0.0239 [−0.0325, −0.0154] | −0.0319 [−0.0380, −0.0256] | +0.0009 [−0.0007, +0.0025] | −0.0059 [−0.0075, −0.0043] |
| 2023-04 | −0.0211 [−0.0310, −0.0112] | −0.0270 [−0.0342, −0.0198] | −0.0048 [−0.0068, −0.0029] | −0.0175 [−0.0192, −0.0157] |

The timing result is not an artifact of the likelihood head. Real blitz
clocks quantize to whole seconds and put roughly 9% of mass on zero-second
premoves, which a plain log-normal mishandles. A zero-inflated log-normal
head fits about 0.77 nats better in absolute terms, and the evolving
advantage persists under it (D−B = −0.026, CI [−0.030, −0.022]).

The move channel can also be probed against a released model. We define a
player's *deviation from Maia-2* — Maia-2 (Tang et al., 2024) is the
strongest released human-move model — as the log probability Maia-2 assigns
to the played move. A linear probe from the trained latent to this deviation
reaches held-out R² = 0.009 for the evolving latent and −0.001 for the twin,
against R² = 0.93 when the same probe targets the known synthetic hidden
state (§4.4). Move choice is near-stateless even measured against the
strongest released human-move model.

### 4.3 Does the timing edge survive released-model baselines?

A hand-built baseline invites the strawman objection, so we escalated the
baseline until it contained released models. The evolving latent is
evaluated as an *add-on*: we ask whether appending the per-step latent `z`
to a baseline model improves held-out think-time NLL, written
(base+z)−base. Table 4 traces the escalation. An Elo+clock aggregate is
beaten clearly. A position-aware baseline adding the branching factor
reaches Spearman 0.38–0.41, essentially ChessMimic's published 0.41, and is
still beaten. Boosting the baseline with Maia-2's learned position
difficulty (its legal-move entropy) pushes Spearman to 0.414–0.447, at or
above ChessMimic, and the latent still improves held-out NLL on every cohort
and every seed.

**Table 4 — the evolving latent as an add-on over escalating baselines
((base+z)−base, in nats, 100 players, 5 seeds; 95% player-bootstrap CIs.
"Baseline Spearman" is the baseline's per-player rank correlation with
actual think-time, given per cohort (2017-04 / 2019-07 / 2021-06).**

| Baseline | 2017-04 | 2019-07 | 2021-06 | Baseline Spearman |
|---|---:|---:|---:|---|
| Elo+clock* | −0.0430 [−0.0560, −0.0310] | — | — | 0.25 |
| Elo+clock+branching | −0.0247 [−0.0338, −0.0157] | −0.0281 [−0.0407, −0.0167] | — | 0.382 / 0.406 |
| Elo+clock+branching+Maia-2 entropy | −0.0254 [−0.0348, −0.0162] | −0.0291 [−0.0428, −0.0163] | −0.0386 [−0.0500, −0.0284] | 0.414 / 0.445 / 0.447 |
| Allie locked (Allie vs Allie+z) | −0.0231 [−0.0303, −0.0154] | −0.0177 [−0.0297, −0.0053] | −0.0329 [−0.0449, −0.0226] | 0.620 / 0.642 / 0.646 |
| Elo+clock+Allie co-fit | −0.0129 [−0.0176, −0.0081] | −0.0054 [−0.0141, +0.0061] | −0.0133 [−0.0183, −0.0088] | 0.627 / 0.654 / 0.661 |

*\*The Elo+clock row is a 2-seed result on 2017-04 only; all other rows are
5 seeds.*

The airtight test replaces the difficulty proxy with an actual released
think-time model. Allie's checkpoint, run over our per-player games, is a
strong baseline in its own right, with per-player Spearman 0.62–0.65, well
above ChessMimic. Yet in the direct Allie-vs-Allie+z comparison the evolving
latent improves held-out NLL significantly on all three cohorts. Against the
fullest co-fit baseline (Elo+clock+Allie) the add-on is significant on 2017
and 2021 and null on 2019. The effect over Allie is smaller than over weaker
baselines; one explanation is that a stronger model absorbs more of the
predictable signal.

One confound remains: the add-on over Allie could be stable per-player
calibration rather than dynamics. Table 5 locks both Allie's prediction and
a learned per-player constant, and asks whether the *evolving* latent still
beats the *static* one. It does on two of three cohorts and on the pooled
299 unique players (−0.0126, CI [−0.0188, −0.0061]); 2019-07 is again null,
consistently its weakest cohort under strong controls.

**Table 5 — evolving vs static individualization over locked Allie (5
seeds, evolving − static, in nats; 95% player-bootstrap CIs).**

| Cohort | evolving − static | 95% CI | Seed audit |
|---|---:|---|---|
| 2017-04 | −0.0170 | [−0.0254, −0.0082] | 5/5 favor evolving |
| 2019-07 | −0.0014 | [−0.0138, +0.0112] | 3 neg / 2 pos, null |
| 2021-06 | −0.0194 | [−0.0300, −0.0088] | 5/5 favor evolving |
| Pooled unique players (n=299) | −0.0126 | [−0.0188, −0.0061] | P(<0)=1.00 |

### 4.4 Where does the edge live, and what does the latent encode?

If the latent tracks a real behavioral state, its advantage should
concentrate where that state matters most. On real chess it does: bucketing
the held-out timing gap by time pressure, aggregating to one point per
player per bucket and bootstrapping over players, the high-pressure tercile
carries a 4.0–6.1× larger edge than the low tercile. Because high-pressure
decisions are also the noisiest, we normalized each bucket's per-player mean
by that bucket's own decision-level standard deviation; the standardized
ratio is 2.7–3.6×, and every high-pressure CI excludes zero (2 cohorts × 2
seeds; Figure 2, TODO: concentration bars). The same bucketing by post-loss
recency and fatigue is flat, which makes the concentration specific to the
clock. The identical bucketed analysis on the *move* channel is ≈0 in every
bucket on both cohorts, the contrast that anchors the when-not-what reading.

The edge is also monotone in skill. Pooling the four clocked cohorts (480
players), the weakest rating tercile shows a timing gap of −0.0404 against
−0.0139 for the strongest, a roughly 3× ratio, with the high-minus-low
difference significant (+0.0265, CI [+0.015, +0.038]). The edge lives
exactly where clock management is most individual and state-dependent.

An evolving state also beats a *stable* per-person speed calibration, the
bar the psychometric literature of §2 sets. The static-speed control (a
per-player constant fed the same features and timing head) is beaten
significantly on three of five cohorts — 2017-04, 2021-04 (−0.052), and
2021-06 (−0.137) — and is null on 2019-07 and 2023-04, with all five point
estimates favoring the evolving latent. The result is usually significant
but cohort-dependent; the memoryless twin remains the primary control.

On the synthetic player, whose hidden tilt is recorded per decision (§3),
the mechanism is fully checkable. A linear probe recovers the hidden state
from the evolving latent at held-out R² = 0.93 against 0.65 for the twin, so
accumulation is what recovers the ordered integral the instantaneous
features only proxy. Clamping the latent along the probed state direction
moves predictions monotonically in the expected direction: move entropy
rises and predicted think-time lengthens as the clamp pushes toward
"tilted". The policy uses the state rather than merely carrying it.

On real data the evolving-vs-memoryless edge blends genuine state-tracking
with per-person online calibration, and the two must be separated. A
temporal-shuffle control on KT (§4.5) identifies the response-channel edge
there as order-invariant individualization. The dynamics reading for chess
timing rests on the concentration signature above — a uniform
individualization edge would not concentrate under time pressure — together
with the synthetic probe and clamp. The paper's central claim does not
depend on which reading wins: under either interpretation, accumulated
per-individual evidence is legible in when a person acts and near-absent
from what they choose, which is the asymmetry the memoryless twin isolates.

### 4.5 Does it generalize beyond chess?

The same injector, trainer, and evaluation, with only the backbone swapped,
transfer to knowledge tracing. Under a frozen eight-dataset protocol with
recorded source and cohort hashes, the evolving latent wins the response
channel in 22 of 24 dataset-seed cells, spanning four ASSISTments releases
(an online mathematics-tutoring platform), two KDD Cup 2010 tutoring
datasets, a Spanish vocabulary corpus, and an engineering statics course
(Table 6). The two exceptions reproduce bit-for-bit: one Statics seed is
null, and one ASSISTments 2015 seed significantly favors the twin, large
enough to flip that dataset's three-seed mean. Seven of eight dataset means
favor the evolving latent. A temporal-shuffle control, permuting each
student's response order (single seed), leaves the edge undiminished
(−0.0151 shuffled vs −0.0095 unshuffled, both CIs excluding zero), so the
response-channel gain is order-invariant individualization, the reading
§4.4's scope note relies on.

**Table 6 — fixed-loader KT replication (8 datasets, 3 seeds, response D−B
in nats; negative favors the evolving latent). Spread is each dataset's
per-student accuracy standard deviation.**

| Dataset | Spread | Mean D−B | Seed cells favoring D |
|---|---:|---:|---|
| Bridge-to-Algebra 06 | 0.096 | −0.0057 | 3/3 |
| Algebra 05 | 0.123 | −0.0086 | 3/3 |
| Statics | 0.142 | −0.0051 | 2/3 (one null) |
| ASSISTments 17 | 0.147 | −0.0152 | 3/3 |
| ASSISTments 12 | 0.154 | −0.0110 | 3/3 |
| ASSISTments 15 | 0.158 | +0.0063 | 2/3 (one significant reversal) |
| ASSISTments 09 | 0.190 | −0.0128 | 3/3 |
| Spanish | 0.258 | −0.0444 | 3/3 |

Across datasets, the signed advantage is descriptively associated with each
population's per-student accuracy spread (Pearson 0.776, Spearman 0.476;
Figure 3, TODO: scatter with bootstrap band), which is the direction an
individualization account predicts: the latent buys more where individuals
differ more. The association is fragile: the dataset-bootstrap CIs cross
zero, and Pearson falls to 0.138 without the Spanish anchor. We present it
as a suggestive pattern consistent with the within-chess stratification in
§4.4, not as a scaling law.

Do real education response *times* show the chess asymmetry? No, and this
sets the timing claim's scope. On ASSISTments 2009, whose raw release
carries a first-response timer (the standard five-column preprocessing drops
it, so we re-derived it), timing is inconsistent across seeds (+0.0003,
+0.0045, −0.0062) while the response channel stays significant in every
seed. On EdNet-KT1, a large Korean TOEIC-tutoring log, under a protocol
frozen before ingestion (singleton bundles only, to avoid its unresolved
bundle-time ambiguity), response again favors the evolving latent (−0.0159,
CI [−0.0202, −0.0118]) but timing is null (−0.0004, CI [−0.0059, +0.0059]).
We hypothesize the boundary condition is strategic control of time: a chess
clock is a managed resource, while an education platform's timer records
incidental interface time.

Does the timing edge appear in Go? No. On real games from OGS, once board
size is controlled, the edge is absent at every size, and the one residual
9×9 signal collapses on a 2.5× larger cohort; the naive mixed-cohort
positive was a board-size confound, the latent detecting the game regime
rather than any player's state.

### 4.6 Can the latent recover and generate a population?

Predicting individuals is the test; recovering a population is the payoff
for simulation. A simulator built on an average person is plausible
decision-by-decision and wrong as a population, covering none of the
population's diversity. On the same 500 real ASSISTments 2009 students used
in §4.5, the per-individual latent recovers the observed per-student
accuracy distribution at half the Wasserstein distance of the average-person
baseline and ranks students at correlation 0.96, while the average-person
baseline's recall of the population's diversity is 0.00 (Table 7).

**Table 7 — population recovery, real ASSISTments 2009 (500 students).
W1 = Wasserstein-1 distance to the observed per-student accuracy
distribution (lower is better); JS = Jensen–Shannon divergence;
precision/recall follow Kynkäänniemi et al. (2019). Best per column in
bold.**

| Model | W1 to observed | JS | Precision | Recall | corr(pred, obs) |
|---|---:|---:|---:|---:|---:|
| Per-individual latent | **0.074** | **0.17** | 0.86 | **0.75** | 0.96 |
| Average-person | 0.147 | 0.61 | 1.00 | 0.00 | — |

The latent also generates novel individuals. Sampling never-seen players
from a full-covariance Gaussian prior fit to the trained per-player evolving
latents of the synthetic cohort yields a population that is both plausible
and fully diverse: precision 0.93 and recall 1.00, against the matched
average-person baseline's 1.00 and 0.00 on the same cohort. Here precision
measures whether generated players lie within the real behavioral
distribution and recall measures how much of the real diversity the
generated set covers (Kynkäänniemi et al., 2019). Covering the population is
a capability an average-person model lacks at any point-prediction
accuracy.

### 4.7 Which channel carries the state?

The same state can reach a backbone as a trained hidden vector or as a
verbal note, and prior lines commit to one channel each. Comparing them
head-to-head shows the ordering depends on the backbone (Table 8). On the
board-native backbone, which has no language prior, the hidden vector beats
the verbal rendering of the same state by 0.069 nats on synthetic data and
0.117 on real chess. The verbal note is lossy there, dropping the unanchored
dimensions the full vector carries. Inside Qwen3, an instruction-tuned LLM,
that advantage disappears. The verbal note improves think-time prediction on
all three seeds, and the hidden soft-prompt adds no advantage over it: the
LLM reads the note semantically, and a from-scratch soft prompt of the same
numbers cannot compete at this adaptation scale.

**Table 8 — channel and LLM results. All values are held-out NLL
differences in nats; negative favors the first-named condition; Δ denotes
with-state − no-state.**

| Setting | Comparison | Result |
|---|---|---|
| Board-native, synthetic | hidden − verbal | −0.069, P(<0)=1.00 |
| Board-native, real 2013 | hidden − verbal | −0.117, P(<0)=1.00 |
| Qwen3-1.7B LoRA SFT, 3 seeds | verbal − none (timing) | −0.0050, all 3 seeds negative |
| Qwen3-1.7B LoRA SFT, 3 seeds | hidden − verbal (timing) | +0.0034 (hidden advantage absent) |
| Qwen3 0.6B→8B LoRA SFT | timing Δ | −0.0107 to −0.0136 at every scale |
| Qwen3 4B/8B full-param SFT | timing Δ / move Δ | −0.0110/−0.0128 timing; −0.0072/−0.0083 move |

Two negative results shaped the LLM probe. A frozen LLM given the verbal
state is no better than one given irrelevant filler text, so persona-style
prompting alone does not use the state. Sparse-reward reinforcement learning
(GRPO, rewarding a match to the player's actual move) learns the mimicry
task but cannot resolve the small state effect.

A dense behavior-cloning SFT probe can, and it replicates the central
asymmetry. The state helps held-out think-time prediction at every scale
from 0.6B to 8B (Δ = −0.011 to −0.014), roughly three times its move-channel
effect under LoRA, a parameter-efficient adapter method. Under LoRA the move
effect vanishes at 4B and above. Full-parameter fine-tuning at both of those
scales recovers a stable move benefit (−0.0072/−0.0083, all seeds negative),
so the clean move-null is an adapter-capacity artifact rather than a
property of scale. Under full fine-tuning the timing effect exceeds the move
effect by about 1.5× at every seed and both scales; the adaptation-invariant
claim is the think-time benefit.

## 5 Limitations

The headline evolving-vs-memoryless numbers use small from-scratch
backbones. For timing this concern is retired two ways: the timing head
reads only the latent, so the pillar is backbone-independent by
construction, and §4.3 shows the latent adds value over released models.
The move-channel ceiling under a strong move backbone remains open.

Effect sizes are −0.01 to −0.07 nats. The utility case does not rest on the
raw gap: the timing head is a proper likelihood, so the gap is a held-out
log-likelihood improvement, and the population result (§4.6) is a capability
an average-person model lacks at any gap size.

The two channels are also not measured on a common scale. The timing and
move heads have different output spaces and entropies, so the raw
cross-channel comparison (−0.021 to −0.033 nats timing versus ≈0 to −0.018
move) illustrates the asymmetry but cannot by itself quantify its
magnitude. The when-not-what claim therefore rests on scale-free contrasts:
the same linear probe that recovers a known synthetic hidden state at
R² = 0.93 recovers deviation-from-Maia-2 at 0.009 (§4.2), and the timing
edge concentrates 2.7–3.6× under time pressure while the move edge stays ≈0
in the same buckets (§4.4).

The when-not-what asymmetry holds on real chess and synthetic knowledge
tracing; it does not transfer to the two real education timers tested
(§4.5). The boundary condition we propose, and have not yet tested, is that
the timing channel requires time to be a strategically managed resource.

Dynamics and individualization are fully separable only on synthetic data,
where the probe and clamp identify the state directly. On real data the
dynamics reading rests on the concentration signature; the KT response edge
is identified as individualization by the shuffle control.

We ran no live human study. Held-out prediction of 480+ real players'
future games over a six-year span and interactive human judgment answer
different questions, and each catches failure modes the other cannot; we
report the one we ran and regard live evaluation as complementary.

## 6 Conclusion

An evolving per-individual latent state, isolated by an equal-capacity
memoryless twin on strict future splits, shows that a person's current state
is far more legible in *when* they act than in *what* they choose, robustly
across six years of real chess and against released think-time models. The
same latent wins student-response prediction in 22 of 24 dataset-seed cells
across eight real KT datasets and recovers the population heterogeneity that
average-person simulators lose, and the best channel for injecting it
depends on whether the backbone can read language. For human simulation, the
practical lesson is concrete: model when people act, not only what they do,
and carry a state that accumulates.

## Reproducibility Statement

[TODO: code release URL; frozen protocol manifests (KT eight-dataset,
EdNet, stable-speed cohorts); data provenance (Lichess open database
prefixes with byte counts, OGS scrape date, dataset export hashes);
deterministic CPU reproduction commands; compute (2×A100 for the tier-1
sweep and LLM arm); hyperparameters incl. λ, latent width, epochs, seeds.]

## Ethics Statement

[TODO: all human data is from public releases (Lichess open database, OGS
public games, de-identified education datasets released for research);
no new human-subjects data was collected; per-player modeling uses public
pseudonymous handles; discuss simulation-of-individuals dual use briefly.]

---

## References (draft list; verify all four fields before bib freeze)

- Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg, Ashton Anderson.
  *Aligning Superhuman AI with Human Behavior: Chess as a Model System.*
  KDD 2020. arXiv:2006.01855. — corrected & verified in review 2026-07-19
- Zhixuan Tang, Ruoyu Jiao, Reid McIlroy-Young, Jon Kleinberg, Siddhartha
  Sen, Ashton Anderson. *Maia-2: A Unified Model for Human-AI Alignment in
  Chess.* NeurIPS 2024. arXiv:2409.20553. — verified in review 2026-07-19;
  **re-confirm author list at bib time**
- Yiming Zhang, Athul Paul Jacob, Vivian Lai, Daniel Fried, Daphne
  Ippolito. *Human-Aligned Chess With a Bit of Search.* ICLR 2025 (Allie).
  arXiv:2410.03893. — verified in review 2026-07-19
- ChessMimic. arXiv:2606.04473. — verified 2026-07-13
- Matilda: Engine-Agnostic Search with Human Policy Guidance (Carlson).
  arXiv:2606.25176. — re-verified 2026-07-19: this ID resolves to Matilda,
  NOT the previously listed "Elo-Disentangled Player-Style Embeddings"
  (which does not exist under this ID; the mislabel is corrected in §2)
- Maia4All. arXiv:2507.21488. — added 2026-07-19, **verify at bib time**
- Ailed. arXiv:2603.05352. — verified 2026-07-13; **re-check the
  no-human-subject-validation statement at submission; pin to a section or
  quote**
- UniMaia. arXiv:2605.27767. — verified 2026-07-13
- LATTE. arXiv:2605.26612. — verified 2026-07-13
- HumanLM. arXiv:2603.03303. — verified 2026-07-13
- Duan et al. *Who Am I? History-Aware Profiles for Student Simulation in
  Tutoring Dialogues.* arXiv:2605.30051. — verified 2026-07-13
- Latent-variable RT models with individual change-points.
  arXiv:2605.29182. — verified 2026-07-13
- Wim J. van der Linden. *A lognormal model for response times on test
  items.* Journal of Educational and Behavioral Statistics, 31(2), 2006.
  — venue corrected in review 2026-07-19
- Wim J. van der Linden. *A hierarchical framework for modeling speed and
  accuracy on test items.* Psychometrika, 2007. — **verify at bib time**
- Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo
  Aila. *Improved Precision and Recall Metric for Assessing Generative
  Models.* NeurIPS 2019. — **verify at bib time**
- [TODO tooling/method cites: Qwen3; sglang; LoRA (Hu et al., ICLR 2022);
  GRPO; GRU (Cho et al., 2014); TRL (used for the SFT probe — name it in
  the Reproducibility Statement).]
- [TODO dataset cites: ASSISTments (2009/12/15/17); EdNet (Choi et al.);
  KDD Cup 2010 (Algebra, Bridge-to-Algebra); Spanish vocabulary (verify:
  Lindsey et al. 2014); Statics (CMU OLI DataShop); Lichess open database;
  OGS; the five-column KT preprocessing pipeline
  (theophilee/learner-performance-prediction — verify: Gervet et al.,
  JEDM 2020).]
- [TODO re-add with verified citations: Player-Specific chess modeling and
  Mixture-of-Masters (dropped from §2 pending verification — candidate ID
  arXiv:2605.11893, unverified).]
- [TODO: DASKT/DEKT (TKDE'25), Maia4All (TMLR'26); manual
  Psychometrika/JEBS sweep for dynamic latent-speed models.]

## Figure TODOs

- Figure 1: architecture — decision point → history features → evolving
  latent vs memoryless twin → injection channels → backbone heads.
- Figure 2: concentration bars — timing D−B by time-pressure tercile, raw
  and variance-normalized, with player-bootstrap CIs (§4.4). Asset path:
  `figures/timing_concentration/`.
- Figure 3: KT spread-vs-advantage scatter with dataset-bootstrap band
  (§4.5). Asset path: `figures/kt_spread_scatter/`.
