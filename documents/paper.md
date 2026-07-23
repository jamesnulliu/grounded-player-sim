# Beyond Static Personas: Simulated Humans Need an Evolving State — Legible in When They Act, Not What They Choose

*Manuscript draft (markdown, pre-LaTeX). Target: ICLR. Primary-area choice
re-opened by the 2026-07-22 reframe (user-simulation-first): candidates are
"applications to neuroscience & cognitive science" (cognitive finding) vs a
general applications/agents area (user simulation); decide at submission.
All numbers are taken verbatim from the frozen artifacts in `results/`; the
claim–evidence mapping lives in `documents/claim_evidence_map.md`. Figure
slots are marked TODO.*

*Draft-scaffolding notes (strip at LaTeX translation): (1) appendix-demotion
candidates if the page budget requires: the scale-sweep rows of Table 8
(Table 5, static-vs-evolving over Allie, is now headline-load-bearing and
stays in the body). (2) Before submission, state definitively why the E-C3
move gap on 2013-01 (−0.067, 3 seeds, 15 epochs) exceeds the at-scale tier-1
2013-01 move gap (−0.0027 mlp / −0.0149 conv): the documented causes
(joint-objective suppression + era differences) cover the clocked cohorts,
but the 2013 protocol difference needs one verified sentence. (3) Title uses
"Not What They Choose"; the body claim is calibrated to "almost no state
dependence" — confirm the title's rhetorical compression is acceptable or
soften.*

## Abstract

People do not behave the same way twice. A chess player who just lost plays
the next game differently, and a student who struggled through ten problems
answers the eleventh differently. Yet systems that simulate specific humans
freeze the person into a static profile — a persona paragraph, a rating, a
fixed embedding — so the simulated policy reduces to static attributes plus
reasoning over the current context, and within-person drift is
unrepresentable by construction. That this is a defect is by now widely
*asserted*: a wave of recent user simulators equips agents with evolving
emotional states. It has never been *measured*: no controlled comparison
isolates what a dynamic state adds beyond identity, context, and model
capacity. We measure it. We decompose a person's policy into a
static attribute and an evolving behavioral state, and test the
decomposition where it is cleanly measurable: grounded domains with discrete
action spaces and logged per-decision timing. A per-individual latent state
evolves over the person's own action-and-timing trajectory, and its value is
isolated by an equal-capacity, same-input **memoryless twin** evaluated on
strict future splits of that person's data — and, beyond the twin, by the
controls the decomposition demands: a learned *static* per-player embedding
(the profile term alone) and a hand-designed *structured memory* of running
history statistics (dynamics without learning). On real Lichess chess the
evolving state beats the twin at predicting future think-time in all eight
era-by-backbone conditions from 2017 to 2023, every confidence interval
excluding zero; it still adds value on top of Allie, a released think-time
model, on all three cohorts tested, and on top of Allie *plus* the static
embedding on two of three (pooled −0.0126, CI excluding zero) — so the
dynamic term is separately measurable, not identity in disguise.
The memory control answers the "an evolving latent is just a memory"
objection by conceding its premise and testing it: a training-free running
summary of the person's raw history beats the static profile on **all
three** cohorts, at 2–3× the learned arm's margin (pooled −0.0276) —
including the cohort where the learned contrast was null — and an
input-matched learned control ties it, so the dynamic term is
instrument-robust and its value is set by the information the state
carries, not by whether its update rule is learned. Our central empirical
characterization is an asymmetry: the state is far more legible in *when* a person acts than in
*what* they choose. Move choice shows almost no state dependence (a probe
recovers deviation from Maia-2, a released human-move model, at R² = 0.009),
and the timing edge concentrates exactly where an evolving state should
matter, 2.7–3.6× under time pressure after variance control and about 3×
for the weakest players. On eight real knowledge-tracing datasets the same
latent wins on responses in 22 of 24 dataset-seed cells; the training-free
memory instrument reproduces the effect on 7 of 8 (repairing the one
dataset where the learned arm's training reversed); and a fitted *frozen*
per-student profile fails on every dataset — the per-person estimate must
keep updating. The same latent then does what no static persona library
can: it recovers the population's accuracy distribution at half the
Wasserstein distance of an average-person baseline and generates novel,
diverse, plausible individuals from its prior. Response *times* do not transfer on
the two education datasets tested; we hypothesize the timing channel
requires time to be a strategically managed resource. Finally, which
injection channel carries the state is backbone-dependent: a trained hidden
vector beats a verbal note on a board-native backbone, and that advantage
disappears inside an instruction-tuned LLM. [TODO: code + frozen-protocol
release sentence.]

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

Current approaches to modeling a specific person share one structural
assumption: the person can be frozen. Rating-conditioned and
per-player-embedding chess models fix each player as a static vector, and
persona-conditioned LLM simulators fix them as a static text profile ("this
user is impatient") over which the model then reasons. Under that design the
simulated policy decomposes as *static attribute + reasoning about the
current context* — and everything a real person carries *between* decisions
is unrepresentable by construction. A player tilts after a loss, fatigues
late in a session, and reprioritizes under a ticking clock; a static profile
cannot drift, however good the reasoning on top of it. We take the opposite
decomposition as our hypothesis: a person's policy is a static attribute
*plus an evolving behavioral state*, the slow-moving, emotion-like component
that the environment writes into and the next decision reads out of. The
claim is not that we measure emotion — no emotion labels exist in any of our
data — but that the decomposition's dynamic term exists, is separately
measurable, and improves prediction of real individuals' future behavior.
By 2026 the premise itself is widely shared: an LLM-agent line equips
simulated clients, customers, and social-media users with evolving
emotional states precisely because static personas fall short (§2). But
that line *builds the premise in* — the state is prompted, scripted, or
stored as text, and evaluation asks whether the simulator *seems* human.
The premise has, to our knowledge, never been measured: no experiment
isolates what the dynamic state contributes, on real individuals, against
controls that could falsify it. That measurement is this paper.

Testing that hypothesis on open-ended interaction is hopeless: with
free-form text there is no fixed action space over which two models can be
scored likelihood-against-likelihood. So we test it where it is cleanly
measurable — grounded domains with discrete action spaces, logged
per-decision timing, and years of per-individual public data (online chess,
knowledge tracing). If the dynamic term is real anywhere, it must be
demonstrable here; and a decomposition validated here is a design
requirement any user simulator must answer, whatever its backbone.
Attribution is the hard part. Models that condition on recent history
confound two capabilities: *accumulating* a state over a person's trajectory
and merely *seeing* recent context. To our knowledge, no prior line
separates the two with a capacity-matched, same-input control on real human
behavior with a timing target (§2), so even where history-conditioning wins,
the win cannot be attributed to accumulated state.

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

A fair objection is that the premise, once stated, sounds self-proving —
*of course* tracking a user helps. The data disagrees with every
comfortable version of that intuition. The state avoids the channel folk
psychology predicts: tilt barely touches *which* move a player chooses
(probe R² = 0.009) and lives almost entirely in how long they think.
Updating does not help everywhere: it fails outright in Go and on both
real education timers — boundaries a truly obvious claim could not have. A
*fitted* frozen profile, far stronger than any persona sentence, is worse
than no profile at all on six of eight education datasets. The learned
state mechanism the field keeps building adds nothing over hand-designed
running statistics at equal information. And inside the LLM channel
itself, a dynamically-updated persona ties a frozen fitted one on a
low-drift horizon — the gap that matters there is fitted-versus-guessed,
not frozen-versus-updating (§4.7). What survives measurement is not
a slogan but a structure: which channel carries the state, where its value
concentrates, what information it needs, and what only a distributional
state model can do.

Our contributions, each tied to the experiment that supports it:

1. **A controlled test of the decomposition: static attribute + evolving
   state.** The dynamic term is isolated by an equal-capacity, same-input
   **memoryless twin** on strict future splits (identical parameters,
   inputs, and optimizer; only state accumulation differs) and then by the
   two controls the decomposition demands: a learned **static per-player
   embedding** (the attribute term alone) and a training-free **structured
   memory** of running history statistics (dynamics without learning). The
   evolving latent wins future think-time in all 8 clocked era-by-backbone
   conditions (Table 3), survives the twin at twice its latent width, adds
   value over Allie on 3/3 cohorts and over Allie+static on 2/3 (pooled
   −0.0126, CI excluding zero) — and the structured-memory arm beats the
   static profile on **3/3** cohorts at 2–3× that margin, so the dynamic
   term is real under two entirely different instruments, one of which has
   no trained parameters at all (§4.1–4.3, Tables 2–5). An input-matched
   learned control — a GRU over the same memory statistics — ties the
   linear memory readout (pooled +0.0096, CI crossing zero) while still
   beating the static profile (−0.0180, CI excluding zero): the dynamic
   term is instrument-robust, and its value is set by the information the
   state carries rather than by the update mechanism (§4.3).
2. **The when-not-what asymmetry on real humans.** Move choice is a
   near-null, small at best and backbone-sensitive, and a probe from the
   latent to deviation-from-Maia-2 reaches only R² = 0.009. The timing edge,
   by contrast, survives baselines at or above the best published think-time
   rank correlation on Lichess, including Allie's released think-time head
   (§4.2–4.3, Tables 3–5).
3. **A mechanism account.** The timing edge concentrates 2.7–3.6× under time
   pressure after variance control and about 3× for the weakest players; on
   a synthetic player with a known hidden state the latent encodes that
   state (probe R² = 0.93 vs 0.65) and clamping it moves predictions
   monotonically (§4.4).
4. **Generality, and the capability that separates a state model from a
   persona library.** On eight real knowledge-tracing datasets the evolving
   latent wins responses in 22 of 24 dataset-seed cells; the same latent
   recovers real population heterogeneity at half the Wasserstein distance
   of an average-person baseline and generates diverse, plausible synthetic
   students from its prior (recall 1.00 vs the matched average-person's
   0.00) — an operation no static persona set and no text-memory store
   supports, because neither carries a distribution over individuals one can
   sample. Two real education response-time tests are negative and set the
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

**Dynamic psychological state in games and tutoring.** Closest in spirit is
Ailed (arXiv:2603.05352), a psyche-driven chess engine that decomposes an
engine's behavior into a static hand-designed *personality* preset and a
dynamic *psyche* scalar updated move-to-move from positional factors —
structurally the same attribute-plus-state decomposition we test. But Ailed
is a *generative* construct: every modulation parameter is hand-tuned, the
evaluation is engine-vs-engine, and by its authors' own account it has no
human-subject validation, so its state dynamics are asserted rather than
measured against real players; it models no clock and no individual. DASKT
(arXiv:2502.10396) brings "dynamic affect" to knowledge tracing, but its
affect states are a hand-engineered pipeline (feature clustering plus graph
smoothing), with no static-vs-dynamic ablation and no future-split
per-individual validation. We make the shared premise falsifiable: the evolving state is
*learned* from, fit to, and scored against specific real players' held-out
future games, and its value is established by an equal-capacity control and
a static-embedding control rather than by construction.

**Emotion-dynamic LLM user simulators.** A 2025–26 LLM-agent line argues
our premise for us: static personas make user simulators unrealistic
because a real user's emotional state evolves. AnnaAgent
(arXiv:2506.00551) maintains an evolving emotional and complaint state for
simulated counseling clients; Customer-R1 (arXiv:2510.07230) simulates
customer behavior with dynamic state; TWICE (arXiv:2602.22222) models the
temporal evolution of social-media users with an event-driven *textual
memory*; and prompted-appraisal agents refine a persona's emotion turn by
turn (arXiv:2607.07824). We differ on the four axes that turn the shared
premise from a design choice into a finding. *Mechanism:* their state is
prompted, scripted, or stored as text; ours is a latent learned end-to-end
from the person's real behavior. *Evidence:* they evaluate the simulator's
plausibility (LLM-judge or human ratings of realism); we evaluate held-out
predictive likelihood of specific real individuals' future behavior.
*Attribution:* none isolates the dynamic term against controls — an
equal-capacity memoryless twin, a static profile, a structured memory — so
"the evolving state helped" cannot be separated from "the prompt carried
more context"; our §4.3 ladder is exactly that isolation, and its memory
arm quantifies the very mechanism class (a running store of the person's
history) this line deploys. *Grounding:* free-form dialogue has no fixed
action space over which two simulators can be scored
likelihood-against-likelihood, which is why we test the decomposition in
discrete grounded domains. Read together, that line supplies the demand
and we supply the measurement: a validated, sampleable dynamic-state
carrier such simulators could adopt.

**Timing as a readout of latent state.** That latency reveals latent
cognitive state more richly than accuracy is a settled result in
response-time psychometrics: van der Linden's hierarchical speed-accuracy
model (2006; 2007) combines a *stable* per-person latent speed with item
time-intensity, later dynamic extensions and change-point models
(e.g. arXiv:2605.29182) report response times out-predicting accuracy-only
item-response models, and Latency-Response Theory (arXiv:2512.07019)
recently carried the same hierarchical framework to LLM evaluation, proving
latency strictly adds trait information whenever it correlates with the
latent. Every model in this lineage, classical and recent, keeps the
per-person trait *static*. The bar it sets is therefore not "timing beats
choice" but "an *evolving* state beats a *stable* per-person speed
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
comparing injection channels; its auxiliary "move delay" objective is the
nearest published touchpoint to our timing channel, but it serves as a
training signal for move prediction and is never evaluated as calibrated
held-out per-player timing likelihood. What none of these run is an
*equal-capacity, same-input* memoryless twin on a per-decision oracle domain
with a timing target, the control that isolates accumulated state from
history-conditioning. And each commits to a single injection channel, verbal
text or a single soft vector; we compare the two head-to-head and find the
ordering backbone-dependent (§4.7).

**Memory modules.** The nearest architectural alternative to an evolving
latent is a *memory*: agent frameworks append episodic records and retrieve
them as text (MemGPT, arXiv:2310.08560; the memory streams of generative
agents, Park et al., 2023), and recent student and user simulators carry
exactly such modules (e.g. The Imperfect Learner, arXiv:2511.05903;
ContextSim, arXiv:2604.09549). A memory in this sense also updates with the
person's history, so "your evolving latent is just a memory" is a fair
objection — and we answer it with an experiment rather than a definition. We
instantiate the strongest summary-statistic memory this task admits (running
and recency-weighted statistics of the person's raw past think-times, of a
released model's per-decision errors on them, and of their recent outcomes,
optimally linearly read out; §4.3) and compare it, per player, against both
the static profile and the learned evolving state. Two properties still
separate the learned state from any text or statistic store: it is trained
end-to-end for behavior prediction, and it induces a *distribution over
individuals* that can be sampled to generate a population (§4.6), an
operation a memory of one person's events does not define.

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

**The structured-memory control.** If an evolving latent is "just a memory,"
then an explicit memory should do the same work — so we built the strongest
one this task admits and let it compete under the identical protocol (same
locked Allie offset, session split, and linear readout as Table 5). The
memory arm's per-step state is a hand-designed vector of causal running
statistics of the player's history: running, recency-weighted, and lagged
summaries of their raw past think-times, of Allie's per-decision errors on
them, their premove rate, recent outcomes, and session position. It has no
trained parameters at all, and it deliberately reads *richer* inputs than
the learned arms (the GRU never sees a raw think-time; its input is only the
four engineered history features). Three results follow (Table 5b). First,
the memory arm beats the static profile on all three cohorts
(memory − static = −0.0148 / −0.0350 / −0.0327, every CI excluding zero;
pooled −0.0276, CI [−0.0365, −0.0210], n=299) — including 2019-07, the
cohort where the learned evolving arm was null. The dynamic term of the
decomposition is therefore real under a second, training-free instrument,
and larger than the learned instrument measures. Second, the same memory
beats the four-feature learned evolving latent overall (evolving − memory =
+0.0150, CI [+0.0070, +0.0254]): given richer history access, hand-designed
dynamics outrun learned dynamics over poorer inputs. Third, the
input-matched control closes the loop: a GRU whose per-step input is
exactly the same 15 memory statistics (3 seeds, same split and scoring)
lands within noise of the linear memory readout (pooled GRU − memory =
+0.0096, CI [−0.0032, +0.0237]), matches the four-feature evolving arm
(−0.0054, CI crossing zero), and still beats the static profile
(−0.0180, CI [−0.0316, −0.0038]). Given equal information, learned and
hand-designed dynamics coincide.
The reading we take is deliberately modest and story-first: what matters is
*carrying a dynamic state* — any faithful carrier of within-person history
beats the static profile — and the informational content of that state
(raw timing history, a released model's errors) matters more than whether
its update rule is learned. The learned latent remains the only arm that is
trainable end-to-end, works from impoverished inputs, and supports the
population sampling of §4.6.

**Table 5b — the structured-memory arm over locked Allie (paired per
player; learned arms averaged over their 5 seeds; memory arm deterministic;
95% player-bootstrap CIs).**

| Cohort | Allie | +static | +memory | +evolving | memory − static | evolving − memory |
|---|---:|---:|---:|---:|---:|---:|
| 2017-04 | 2.5421 | 2.5361 | 2.5213 | 2.5191 | −0.0148 [−0.0212, −0.0092] | −0.0022 [−0.0113, +0.0097] |
| 2019-07 | 2.3349 | 2.3186 | 2.2836 | 2.3172 | −0.0350 [−0.0577, −0.0201] | +0.0336 [+0.0158, +0.0586] |
| 2021-06 | 2.3054 | 2.2920 | 2.2592 | 2.2726 | −0.0327 [−0.0427, −0.0237] | +0.0133 [+0.0042, +0.0244] |
| Pooled (n=299) | — | — | — | — | −0.0276 [−0.0365, −0.0210] | +0.0150 [+0.0070, +0.0254] |

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

The structured-memory instrument of §4.3 transfers, and repeats its chess
verdict here (Table 6, last two columns). The KT memory arm — running,
recency-weighted, and lagged statistics of the student's own past answers
and of their residuals against the item-difficulty baseline, read out by a
logistic fit with zero other trained parameters — beats the memoryless
twin on 7 of 8 datasets, the same 7-of-8 profile as the learned latent it
mirrors. Head-to-head, memory and the learned latent tie on five datasets,
the latent wins one (ASSISTments 17), and memory wins two — including
ASSISTments 15, the replication's known anomaly: where the learned arm
significantly *loses* to the twin (+0.0063, a training reversal), the
training-free memory still wins (−0.0058, CI excluding zero), so the
anomaly was optimization instability, not an absent effect (the KT
counterpart of chess's 2019-07 cohort). On Spanish, the most heterogeneous
population, memory outruns the latent outright (−0.0646 vs −0.0444). The
cross-domain synthesis is the same sentence as §4.3: what matters is
carrying per-person state and the information it carries — raw answer
history here, raw timing history in chess — not whether the update rule is
learned.

Can the profile at least be *frozen*? The shuffle control's
"order-invariant individualization" reading invites exactly that
objection, so we measured it. A fitted static profile — one frozen number
per student, their accuracy over their own ≥35 training answers, a far
stronger profile than any written persona — is beaten by the updating
memory on **8 of 8** datasets under the identical logistic fit
(memory − static −0.011 to −0.098), and on 6 of 8 it loses even to the
memoryless twin. The failure is largest exactly where students change most
(Spanish vocabulary, −0.098): the last third of a learning sequence sits
at a different level than the first two-thirds, and a confidently frozen
estimate mis-calibrates where an updating one keeps tracking.
Order-invariant is therefore not freezable: a running mean ignores order
yet still updates, and the updating is load-bearing. The KT scoping is
accordingly sharpened — the gain is a *continuously updated* estimate of
who the student is; what this channel does not show is the fast
emotion-like flavor of the state, which remains a chess-timing result
(§4.3–4.4).

**Table 6 — fixed-loader KT replication (8 datasets, 3 seeds, response
NLL gaps in nats; negative favors the first-named arm). Spread is each
dataset's per-student accuracy standard deviation. M is the structured
memory of §4.3 in its KT form (causal running statistics of the student's
answers, logistic readout, zero trained parameters beyond it;
deterministic, paired per student against the seed-averaged learned
arms).**

| Dataset | Spread | Mean D−B | Seed cells favoring D | M−B | D−M |
|---|---:|---:|---|---:|---:|
| Bridge-to-Algebra 06 | 0.096 | −0.0057 | 3/3 | −0.0047 ✓ | −0.0010 ns |
| Algebra 05 | 0.123 | −0.0086 | 3/3 | −0.0075 ✓ | −0.0011 ns |
| Statics | 0.142 | −0.0051 | 2/3 (one null) | −0.0031 ns | −0.0020 ns |
| ASSISTments 17 | 0.147 | −0.0152 | 3/3 | −0.0120 ✓ | −0.0032 ✓ |
| ASSISTments 12 | 0.154 | −0.0110 | 3/3 | −0.0109 ✓ | −0.0001 ns |
| ASSISTments 15 | 0.158 | +0.0063 | 2/3 (one significant reversal) | −0.0058 ✓ | +0.0121 (M wins) |
| ASSISTments 09 | 0.190 | −0.0128 | 3/3 | −0.0126 ✓ | −0.0002 ns |
| Spanish | 0.258 | −0.0444 | 3/3 | −0.0646 ✓ | +0.0202 (M wins) |

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
| Persona ladder, Qwen3-1.7B, 99 players, 3 seeds | static − none / memory − none | −0.0104 / −0.0096, both CIs exclude zero |
| Persona ladder | memory − static (updating vs frozen persona) | +0.0009, CI [−0.0036, +0.0052] — tie |
| Persona ladder | hidden − memory / hidden − static | +0.0045 / +0.0054, text beats vector |

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

Finally, the *persona ladder* runs the paper's field-facing comparison
inside the LLM itself (Table 8, last three rows): the same prompt skeleton
carries either nothing, a *frozen* fitted persona sentence (rating, median
think-time, premove rate, from the training split only), a per-decision
*updating* text scorecard, or the scorecard's numbers as a soft prefix.
Fitted person-information is what the probe resolves: both fitted text
arms beat no-information clearly (≈ −0.010, every CI excluding zero) —
note this is an upper bound on practice, which hand-writes personas from
no data at all. Frozen versus updating is a tie in this channel (+0.0009,
CI spanning zero): the within-player dynamic increment that the
board-native ladder measures at −0.0126 over a static profile sits at or
below this probe's resolution, and a blitz player's typical speed barely
drifts within a one-month cohort — the low-drift regime where §4.5 showed
freezing costs least (in the high-drift regime, education, the frozen
profile failed on every dataset). And the channel ordering replicates G3
on fresh data: the same numbers help significantly more as language than
as a projected vector. The practical reading for LLM user simulators:
*fit the persona from the person's data* (that alone is worth ≈ −0.010
here), keep it updating when the person can drift, and deliver it as
text.

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
where the probe and clamp identify the state directly. On real chess timing
the dynamics reading now rests on two independent instruments — the
static-vs-evolving contrast over locked Allie (2/3 cohorts) and the
training-free structured-memory arm, which beats the static profile on 3/3
cohorts — together with the concentration signature; the KT response edge
is identified as individualization by the shuffle control.

The structured-memory comparison is deliberately input-asymmetric: the
memory arm reads the person's raw past think-times and the released model's
past errors, which the learned arms never see, so its win over the
four-feature evolving latent conflates information access with mechanism
(§4.3 reports the input-matched learned control). Both dynamic arms are
evaluated in the filtering setting, conditioning on the realized past; in a
closed-loop rollout the memory arm would have to consume its own generated
think-times, a deployment gap the latent does not share in the same form
because it never reads the raw target variable.

We ran no live human study. Held-out prediction of 480+ real players'
future games over a six-year span and interactive human judgment answer
different questions, and each catches failure modes the other cannot; we
report the one we ran and regard live evaluation as complementary.

## 6 Conclusion

Simulated humans are built today from static profiles, so the simulated
policy is an attribute plus reasoning, and everything a person carries
between decisions is lost by construction. The field increasingly says so —
and saying is where it has stopped: the dynamic state's contribution had
not been measured. We decomposed the policy into a static attribute and an
evolving behavioral state and tested the decomposition where it is
measurable. The dynamic term is real: it survives
an equal-capacity memoryless twin, a released think-time model, a learned
static per-player profile, and it reappears — larger — under a training-free
structured memory, on every cohort including the one where the learned
instrument was null. It is legible far more in *when* a person acts than in
*what* they choose, it concentrates exactly where an emotion-like state
should (under time pressure, in the least disciplined players), and,
unlike any static persona set or memory store, its learned form carries a
distribution over individuals that can be sampled into a realistic
population. For user simulation the design requirement is concrete: a
simulator of a specific person must carry a per-individual state that
updates as the person acts — a static profile, however good the reasoning
on top of it, structurally cannot represent the person it simulates.

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
- ChessMimic: *Per-Rating Transformer Models for Human Move, Clock, and
  Outcome Prediction in Online Blitz Chess.* arXiv:2606.04473. — verified
  2026-07-13; full title confirmed 2026-07-22
- Jason Carlson. *Matilda: Engine-Agnostic Search with Human Policy
  Guidance.* arXiv:2606.25176. — retitle history verified 2026-07-22: v1
  (2026-06-23) was titled "Elo-Disentangled Player-Style Embeddings for
  Human Chess via Rating-Conditioned Residual Move Model"; v2 (2026-07-13)
  retitled to Matilda with substantially revised numbers. Cite v2; quote v2
  numbers only (+2.53% rel NLL over Maia-3 at 2500+, rising to +21.9% at
  3000+; style embedding is static 32-d per player). Claims code release
  but contains no URL; nothing findable on GitHub/HF as of 2026-07-22.
- Maia4All. arXiv:2507.21488. — added 2026-07-19, **verify at bib time**
- Diego Armando Resendez Prado. *Ailed: A Psyche-Driven Chess Engine with
  Dynamic Emotional Modulation.* arXiv:2603.05352, March 2026. — full-text
  verified 2026-07-22: static personality preset + dynamic psyche scalar
  ψ∈[−100,+100], all modulation hand-tuned; evaluation is 12,414
  engine-vs-engine games vs Maia2-1100; paper explicitly states no
  human-subject validation; no timing model, no per-player modeling.
  Partial code (signal chain only): github.com/chrnx-dev/ailed-chess.
- Sherman Siu, Lesley Istead. *UniMaia: Steering Chess Policies with
  Language for Human-like Play.* arXiv:2605.27767, May 2026. — verified
  2026-07-22 (also Siu's Waterloo Master's thesis); UniMaia-Aux predicts
  "move delay" as an auxiliary objective, never evaluated as held-out
  per-player timing likelihood; no per-player embeddings; no code release.
- LATTE. arXiv:2605.26612. — verified 2026-07-13
- HumanLM. arXiv:2603.03303. — verified 2026-07-13
- Duan et al. *Who Am I? History-Aware Profiles for Student Simulation in
  Tutoring Dialogues.* arXiv:2605.30051. — verified 2026-07-13
- Latent-variable RT models with individual change-points.
  arXiv:2605.29182. — verified 2026-07-13
- Zhiyu Xu, Jia Liu, Yixin Wang, Yuqi Gu. *Latency-Response Theory Model:
  Evaluating Large Language Models via Response Accuracy and
  Chain-of-Thought Length.* arXiv:2512.07019. — verified 2026-07-22; builds
  explicitly on van der Linden (2007); static per-model traits; code
  released (github.com/Toby-X/Latency-Response-Theory-Model)
- DASKT: *Dynamic Affect Simulation for Knowledge Tracing.*
  arXiv:2502.10396. — added 2026-07-22 (novelty sweep: closest single
  competitor); affect states are hand-engineered
  (feature clustering + graph smoothing), no static-vs-dynamic ablation,
  no per-individual future-split validation. **Read in full before
  submission.**
- AnnaAgent. arXiv:2506.00551. — added 2026-07-22 (LLM dynamic-emotion
  user simulation line; prompt/scheduler-based, qualitative evaluation),
  **verify at bib time**
- Customer-R1. arXiv:2510.07230. — added 2026-07-22 (same line), **verify
  at bib time**
- TWICE: *Modeling the Temporal Evolution of Personalized User Behavior
  via Event-Driven Agents.* arXiv:2602.22222. — added 2026-07-22; states
  our exact premise ("static personas cannot capture how behavior
  evolves") and fixes it with an event-driven **textual memory module** —
  the mechanism class our §4.3 memory arm races; qualitative evaluation
  only. **verify at bib time**
- *From Triggers to Emotions: A CPM-Grounded Appraisal Multi-Agent for
  Dynamic Emotional Evolution in Persona-Based Dialogue.* arXiv:2607.07824.
  — added 2026-07-22 (same premise, prompted appraisal mechanism,
  qualitative evaluation), **verify at bib time**
- Charles Packer et al. *MemGPT: Towards LLMs as Operating Systems.*
  arXiv:2310.08560. — memory-module line, **verify at bib time**
- Joon Sung Park et al. *Generative Agents: Interactive Simulacra of Human
  Behavior.* UIST 2023. arXiv:2304.03442. — memory streams, **verify at
  bib time**
- The Imperfect Learner. arXiv:2511.05903. — student simulator with memory
  module, added 2026-07-22, **verify at bib time**
- ContextSim. arXiv:2604.09549. — user simulator with memory module, added
  2026-07-22, **verify at bib time**
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
