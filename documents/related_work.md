# Related work (draft — corrected positioning, 2026-07-13)

*Replaces the internal comparison in `design.md §8` for the paper. Rewritten
after a 2026-07 prior-art sweep that surfaced three uncited chess papers and the
psychometric response-time literature. Key change: we no longer claim the
evolving-psychological-state-in-chess axis is untouched (Ailed touches it), and
we explicitly place the "timing reveals latent state better than choice" finding
against decades of response-time psychometrics. All arXiv IDs below (ChessMimic
2606.04473, Elo-Disentangled 2606.25176, Ailed 2603.05352, UniMaia 2605.27767,
LATTE 2605.26612, HumanLM 2603.03303, Duan et al. 2605.30051, 2605.29182) were
checked directly against arxiv.org on 2026-07-13 and confirmed real; re-check
Ailed before submission specifically (its "no human-subject validation"
admission could go stale with a future revision — still present as of this
check).*

---

**Human-like and per-individual chess.** A fast-moving line models *human* rather
than optimal chess. Maia and its successor Maia-3 / "Chessformer" condition a
policy on a rating scalar to match a *population* at a given Elo; ChessMimic
(arXiv:2606.04473) sharpens this to *per-100-Elo-band* transformers
for move, clock, and outcome, reaching per-move think-time correlation r≈0.41 and
beating Maia-2 on moves. A second cluster is genuinely *per-individual*:
Elo-Disentangled Player-Style Embeddings (arXiv:2606.25176, single-author,
non-peer-reviewed preprint) improve move NLL 27–37% over Maia-3 with a
rating-conditioned base model **plus** a per-player residual embedding — but
by the authors' own account the *base model* (Maia-3 logits + Stockfish
features) drives that gain and the per-player embedding **adds little** on
top of it, and its *static* move ceiling is itself a **near-null**, which we
read as convergent evidence with our own move-channel near-null (§below), not
as a competing positive result. Player-Specific modeling and Mixture-of-Masters
emulate individual grandmasters by adapting a strong backbone. **All of these
are *static* per player and either
aggregate the clock (ChessMimic, Allie) or ignore timing entirely
(Elo-Disentangled, Player-Specific, Mixture-of-Masters); none validate on a
strict *future*-per-individual split, and none isolate the contribution of
*within-session dynamics* from static individualization.** We build directly on
this line — we use the same real Lichess data and treat these models as the
baselines our per-individual *evolving* latent must add value *over* (§E-C6/G4) —
but our claim is orthogonal to raising the move ceiling: it is that a person's
*evolving behavioral state* predicts their *future think-time* beyond what any
static or cohort model captures.

**Dynamic psychological state in chess.** Closest in spirit is Ailed
(arXiv:2603.05352), a "psyche-driven" engine that modulates move selection and
latency by an evolving emotional state (anxiety/confidence/frustration). It shares
our motivation that play is state-dependent, not purely rational. It differs in
the way that matters for an empirical claim: Ailed is a *generative* engine
evaluated against a Maia opponent with, in the authors' own words, **no
human-subject validation** — its emotional dynamics are asserted, not measured
against real players' behavior. We make the corresponding claim *falsifiable*: the
evolving state is fit to, and scored against, *specific* players' held-out future
games, and its value is established by an equal-capacity control (below) rather
than by construction. UniMaia (arXiv:2605.27767, verified 2026-07-13: "UniMaia:
Steering Chess Policies with Language for Human-like Play") steers a chess policy
with natural-language descriptions of desired play; it is a controllability result,
not a per-individual future-prediction one, and it informs our hidden-vs-verbal
channel comparison (below).

**Timing as a readout of latent state.** That *when* a person acts reveals latent
cognitive state more richly than *what* they choose is not new: response-time
psychometrics has held for decades that latency exposes processing speed, effort,
caution, and engagement not recoverable from accuracy alone. The specific
precedent a psychometrics-literate reviewer will raise is van der Linden's
hierarchical speed-accuracy model (2006, 2007): a *stable* per-person latent
speed combined with item time-intensity predicts response time well without
any within-session dynamics, plus later dynamic extensions (e.g. a per-person
hidden-Markov item-state model fit jointly on responses and RTs, 2016) and
recent formalizations (Latency-Response Theory; latent-variable RT models with
individual-specific change-points, e.g. arXiv:2605.29182) report RT out-predicting
accuracy-only item-response models. So the bar our claim must clear is not "RT
beats choice" (settled for decades) but "an *evolving* state beats a *stable*
per-person speed calibration" — we test this directly: an equal-capacity
static-individual control (a per-player constant fed the same item/position
features and lognormal timing head as the evolving latent — the neural
analogue of van der Linden's stable-speed term) is significantly beaten on
timing on one of two real cohorts (`results/stable_speed_baseline.txt`;
cohort-dependent, not a universal win — reported honestly rather than only on
the cohort where it lands). Our contribution is therefore *not* the bare
"timing > choice" asymmetry but its specific form here: (i) the relevant latent is
an *evolving within-session behavioral state* (tilt/fatigue/time-pressure), not a
stable trait, and we test that distinction against a stable-trait baseline
directly rather than only against a zero-history one; (ii) it is measured in an
interactive game against a *per-decision engine oracle*; (iii) it is validated
on a strict *future* split; and (iv) an equal-capacity *evolving-vs-memoryless*
control shows the asymmetry is a property of the modeled
*dynamics/individualization*, not of the estimator. We also reproduce the
asymmetry in a non-game oracle domain (knowledge tracing), tying it to the ITS
response-time literature rather than to chess alone.

**Evolving user state and the equal-capacity control.** Sequential
recommendation and user-simulation model evolving user state and routinely
contrast it against memoryless/Markov baselines; LATTE (arXiv:2605.26612)
forecasts an evolving per-user *preference* state injected as a single
soft token into a frozen LLM with a future temporal split, and already runs an
**evolving-vs-static matched comparison** (same architecture, optimizer,
training examples, and validation split, differing only in whether the
per-user state accumulates) — so "an evolving-vs-static/memoryless contrast at
fixed capacity" is not, by itself, a claim we can make first. HumanLM
(arXiv:2603.03303) RL-trains an LLM to emit *natural-language* psychological
latent states aligned to real users. Duan et al., "Who Am I? History-Aware
Profiles for Student Simulation in Tutoring Dialogues" (arXiv:2605.30051, May
2026), RL-trains history-conditioned verbal per-student profiles in an LLM on
real tutoring dialogue data with architecture held constant across
history/no-history conditions — occupying the same knowledge-tracing +
verbal-channel territory we touch in RQ6, though without timing or a hidden
latent. These establish that "evolving latent into a policy/LLM,"
"evolving-vs-static at fixed capacity," "natural-language latent state," and
"future temporal-split validation" are each individually *not* ours to claim.
What we add is methodological rigor tailored to the "is it just
history-conditioning?" objection, precisely stated: an *equal-capacity,
same-input* memoryless twin evaluated on the *same* future split over a
*per-decision oracle* domain (not a soft token into a frozen LLM, not RL over
sparse reward) — a control none of these run in this combination — which
isolates the value of accumulating state from the value of merely seeing
recent history. And where HumanLM/LATTE commit to a single channel (verbal text / a soft token,
respectively), we compare the *hidden* and *verbal* channels head-to-head and show
the ordering is **backbone-dependent**: with no language prior a trained hidden
vector is richer, but inside an instruction-tuned LLM the verbal note wins because
the model reads it semantically.

**Positioning, stated plainly.** No single axis here is unclaimed — per-individual
chess (Elo-Disentangled), cohort move+clock (ChessMimic), dynamic chess emotion
(Ailed), timing-reveals-state (RT psychometrics), evolving latent + future split
(LATTE/HumanLM) each exist. Our contribution is an *empirical synthesis with a
control that the individual lines lack*: on real human data, with a per-decision
oracle and a strict future split, an equal-capacity evolving latent beats a
memoryless twin at predicting a *specific person's future think-time* — robustly
across a six-year era span and reproduced in a non-game domain — while move choice
is a near-null, and the same latent recovers population heterogeneity a
"positive-average-person" baseline cannot.
