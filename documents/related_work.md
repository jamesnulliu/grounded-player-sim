# Related work (positioning, updated 2026-07-22)

*Replaces the internal comparison in `design.md §8` for the paper. Updated
2026-07-22 after (a) the reframe around the static-attribute + evolving-state
decomposition, (b) the structured-memory control landing
(`results/memory_baseline.txt`), and (c) a full-text prior-art digest + a
24-row novelty sweep (`idea-forge` scratchpad: `prior_art_digest.md`,
`novelty_report.md`). Key corrections vs the 2026-07-13 version: the
arXiv:2606.25176 mislabel is resolved (v1 "Elo-Disentangled Player-Style
Embeddings" was retitled to "Matilda" in v2, 2026-07-13, with substantially
revised numbers — quote v2 only); Ailed's content is now verified at
full-text level; DASKT and the LLM dynamic-emotion user-sim line are added;
the memory-module paragraph is new and load-bearing.*

---

**Human-like and per-individual chess.** A fast-moving line models *human*
rather than optimal chess. Maia and Maia-2/-3 condition a policy on a rating
scalar to match a *population* at a given Elo; ChessMimic (arXiv:2606.04473,
"Per-Rating Transformer Models for Human Move, Clock, and Outcome
Prediction") sharpens this to per-100-Elo-band transformers for move, clock,
and outcome, reaching per-move think-time correlation r≈0.41. Allie (Zhang
et al., ICLR 2025) adds a dedicated think-time head to a decoder-only
policy; its released checkpoint is our strongest timing baseline (per-player
Spearman 0.62–0.65 on our cohorts). A second cluster is genuinely
*per-individual* but **static**: Matilda (arXiv:2606.25176; v1 was titled
"Elo-Disentangled Player-Style Embeddings") re-ranks a frozen Maia-3 with an
optional static 32-d per-player style embedding (v2 numbers: +2.53% relative
NLL over Maia-3 at 2500+, +21.9% at 3000+, gains concentrated at elite
ratings and partly attributed to Stockfish features); Maia4All
(arXiv:2507.21488) adapts a strong backbone per player. **All of these
freeze the person**: static per player or cohort-level on the clock, no
per-individual *future*-split validation, no separation of within-session
dynamics from static individualization. We use the strongest released
artifacts — Maia-2 and Allie — as the baselines our evolving latent must add
value *over*, and Matilda's paradigm (static per-player add-on over a
released model) is precisely the *static* arm our evolving arm beats
(G4 static-vs-evolving; `results/g4_allie_static_vs_evolving.txt`). Their
gains concentrate at elite play and ours in weak players under time
pressure — near-complementary regimes.

**Dynamic psychological state in games and tutoring.** Closest in spirit is
Ailed (arXiv:2603.05352), a "psyche-driven" chess engine that decomposes
behavior into a static hand-designed *personality* preset and a dynamic
*psyche* scalar ψ∈[−100,+100] updated move-to-move from positional factors —
structurally the same attribute+state decomposition we test, which makes it
the must-cite conceptual neighbor. The boundary is method and evidence:
every Ailed modulation parameter is hand-tuned (a 7-stage audio-inspired
signal chain reshaping a base policy's distribution), evaluation is 12,414
*engine-vs-engine* games against Maia2-1100, the paper itself states there
is **no human-subject validation**, and it models no clock, no individual,
no population. Our state is *learned* from real players, validated on their
held-out future behavior, and attributed by controls. DASKT
(arXiv:2502.10396) brings "dynamic affect" to knowledge tracing — the
closest single competitor found in a 24-row novelty sweep — but its affect
states come from a hand-engineered pipeline (feature clustering + graph
smoothing), with no static-vs-dynamic ablation and no per-individual
future-split validation. A 2025–26 LLM-agent line now argues our very
premise — static personas are insufficient because users carry evolving
emotional state — for dialogue, mental-health, and shopping simulation
(TWICE, arXiv:2602.22222; AnnaAgent, arXiv:2506.00551; Customer-R1,
arXiv:2510.07230; "From Triggers to Emotions", arXiv:2607.07824), but
delivers the state through prompts or scripted schedulers and evaluates
qualitatively. The premise is therefore *shared territory*; the controlled,
quantitative demonstration — equal-capacity twin, released-SOTA/static/
memory ladder, strict future splits, timing target — exists nowhere in this
line and is our claim.

**Timing as a readout of latent state.** That latency reveals latent
cognitive state beyond accuracy is settled psychometrics: van der Linden's
lognormal RT model (JEBS 2006) and hierarchical speed-accuracy framework
(Psychometrika 2007) combine a **stable** per-person latent speed with item
time-intensity; dynamic extensions and change-point models
(e.g. arXiv:2605.29182) report RTs out-predicting accuracy-only IRT; and
Latency-Response Theory (LaRT, arXiv:2512.07019) carries the same
hierarchical framework to LLM evaluation, proving latency strictly adds
trait information whenever it correlates with the latent. Every model in
this lineage keeps the per-person trait **static**, so the bar it sets is
not "RT beats choice" (decades old) but "an *evolving* state beats a
*stable* per-person speed calibration." We test exactly that: the
static-speed control (a per-player constant fed the same features and
timing head) is significantly beaten on 3/5 real cohorts with all five
point estimates favoring the evolving state
(`results/stable_speed_baseline.txt`), and the stronger G4 form (static
embedding over locked Allie) on 2/3 cohorts — with the structured-memory
instrument confirming the dynamic term on 3/3 (below). LaRT's
information-gain theorem is convergent theory for our when-not-what
asymmetry and worth a connecting sentence in the paper.

**Evolving user state in simulators.** Sequential recommendation and user
simulation model evolving user state and contrast it against
memoryless/Markov baselines. LATTE (arXiv:2605.26612) forecasts an evolving
per-user *preference* state injected as a soft token into a frozen LLM
under a future temporal split, and already runs an evolving-vs-static
matched comparison — so that contrast *alone* is not ours to claim first.
HumanLM (arXiv:2603.03303) RL-trains an LLM to emit natural-language
psychological states; Duan et al. (arXiv:2605.30051) RL-train
history-aware verbal student profiles on tutoring dialogues. UniMaia
(arXiv:2605.27767) steers a chess policy with language; its "move delay"
auxiliary objective is the nearest published touchpoint to our timing
channel but is never evaluated as calibrated held-out per-player timing
likelihood. What none of these run is the *equal-capacity, same-input*
memoryless twin on a per-decision oracle domain with a timing target; and
each commits to one injection channel, where we compare hidden vs verbal
head-to-head and find the ordering backbone-dependent (RQ6 vs G3).

**Memory modules (new, load-bearing).** The nearest architectural
alternative to an evolving latent is a *memory*: MemGPT
(arXiv:2310.08560) and the memory streams of Generative Agents (Park et
al., 2023, arXiv:2304.03442) append episodic records and retrieve them as
text, and recent student/user simulators carry exactly such modules (The
Imperfect Learner, arXiv:2511.05903; ContextSim, arXiv:2604.09549). "Your
evolving latent is just a memory" is a fair objection, and we answer it
with an experiment, not a definition: we instantiate the strongest
summary-statistic memory the task admits (running/EWMA/lagged statistics of
raw past think-times, of Allie's per-decision errors, of recent outcomes;
zero trained parameters; optimal linear readout) and race it under the
identical protocol. Result (`results/memory_baseline.txt`): the memory arm
beats the *static* profile on 3/3 cohorts (pooled −0.0276) — independently
confirming the dynamic term with a training-free instrument — and
matches/beats the 4-feature learned GRU (pooled +0.0150 against the GRU),
which we report as the honest mechanism finding: *carrying dynamic state is
what matters, and the information the state carries (raw timing history, a
released model's errors) matters more than whether its update rule is
learned* (input-matched learned control: `results/memory_gru_arm.txt`).
The verdict transfers to the KT response channel
(`results/kt_memory_arm.txt`): the KT memory arm beats the memoryless twin
on 7/8 datasets (the learned latent's own profile), ties it head-to-head
on 5, and repairs the ASSISTments-15 anomaly where the learned arm's
training reversed — so "memory suffices" is cross-domain, and the LLM-agent
line's memory-based mechanism choice (TWICE et al.) is quantitatively
reasonable; what it lacks is the attribution and the evaluation, not the
mechanism. What still separates the learned state from any memory store:
end-to-end training for behavior prediction, operation from impoverished
inputs, and a *distribution over individuals* that can be sampled to
generate a population — an operation a memory of one person's events does
not define.

**Positioning, stated plainly.** No single axis here is unclaimed —
per-individual chess (Matilda), cohort move+clock (ChessMimic), dynamic
chess emotion as a design (Ailed), dynamic affect in KT (DASKT), the
static-personas-are-insufficient premise (AnnaAgent and kin),
timing-reveals-state (van der Linden through LaRT), evolving latent + future
split (LATTE), memory modules (MemGPT line) each exist. Our contribution is
the *controlled decomposition with the ladder of instruments the individual
lines lack*: on real human data, with released-SOTA baselines, a strict
future split, and a timing target, the dynamic term of "policy = static
attribute + evolving state" is real (twin, static, and memory controls all
agree), legible in *when* rather than *what*, concentrated where an
emotion-like state should be, and — uniquely for the learned form —
sampleable into a realistic population.
