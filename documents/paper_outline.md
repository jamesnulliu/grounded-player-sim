# Paper outline (concise draft)

*Working draft toward the submission manuscript. Numbers are taken verbatim
from the frozen artifacts in `results/` and `documents/results_ec.md`; every
claim below carries a landing point (mapping table at the end). No figures
yet; figure slots are marked TODO.*

**Working title (pick one):**
1. *When, Not What: An Evolving Per-Individual State Predicts a Person's
   Future Think-Time, Not Their Choices*
2. *Modeling When a Person Acts: Evolving Latent State Beats Memoryless
   History-Conditioning on Future Human Behavior*

Throughout: **D** = evolving latent, **B** = equal-capacity memoryless twin;
lower NLL is better, so D−B < 0 favors the evolving latent. "Significant" =
the 95% player-bootstrap CI excludes zero; P(D−B<0) is a bootstrap
sign-support statistic reported alongside, not a p-value.

---

## Submission metadata (ICLR)

*ICLR is single-track; the choice below is the OpenReview **primary area**
(reviewer routing) + keywords, not an acceptance track. Verify the exact
primary-area list against the live CFP at submission time — the wording is
refreshed yearly, but these choices are stable.*

- **Primary area:** *Applications to neuroscience & cognitive science.* This
  routes to reviewers who evaluate human cognitive/behavioral modeling — the
  audience for the when-not-what finding, the response-time framing, and the
  van der Linden comparison. It is the fit-maximizing choice.
- **Secondary area (if allowed):** *Learning on time series and dynamical
  systems* — the evolving latent, the memoryless-twin control, and the future
  split read directly as sequential-modeling contributions.
- **Do NOT select** an NLP or agent primary area: the language component is
  one secondary, partly-negative result, and there is no agentic loop — the
  LLM is a policy backbone. Either choice misroutes the whole paper to
  reviewers whose fit bar it cannot clear.
- **Keywords:** human behavior modeling; individual differences; latent
  state; sequential/temporal modeling; response time; think-time prediction;
  knowledge tracing; chess; behavior simulation; personalization. (Include one
  LLM/foundation-model keyword to signal relevance to the simulator community
  without misrouting.)
- **One-sentence summary (TL;DR):** An evolving per-individual latent state
  predicts a person's *future think-time* — not their choices — beating an
  equal-capacity memoryless control on real chess and knowledge tracing.

---

## Abstract

People do not behave the same way twice. A chess player who just lost plays
the next game differently; a student who struggled through ten problems
answers the eleventh differently. Models that simulate specific individuals
represent them statically, as a persona, a rating, or a fixed embedding, so
they cannot track this within-session drift, and existing history-conditioned
models cannot say whether their gains come from accumulated state or from
merely seeing recent context. We model the drift with a per-individual latent
state that evolves over a person's own action-and-timing trajectory, and we
isolate its value with an equal-capacity, same-input **memoryless twin**
evaluated on strict future splits of that person's data. Our central finding
is an asymmetry: the evolving state is legible in *when* a person acts, not
*what* they choose. On real Lichess chess the evolving latent beats the twin
at predicting future think-time in all eight era-by-backbone conditions from
2017 to 2023 (gap −0.021 to −0.033 nats, every CI excluding zero) and still
adds value over Allie, a released think-time model that alone reaches Spearman
0.62–0.65, on all three cohorts tested; move choice shows almost no state
dependence (a probe recovers deviation-from-Maia-2 at R² = 0.009). The timing
edge concentrates where behavior is least predictable from the average: 2.7–3.6×
under time pressure after variance control, and about 3× for the weakest
players. On eight real knowledge-tracing datasets the evolving latent wins on
responses in 22 of 24 dataset-seed cells, and it recovers the population's
accuracy distribution twice as closely as an average-person baseline; response
*times* do not transfer on the two education datasets tested, scoping the
timing result to domains where time is strategically managed. Finally, which
injection channel carries the state is backbone-dependent: a trained hidden
vector beats a verbal note on a board-native backbone by 0.07–0.12 nats, and
that advantage disappears inside an instruction-tuned LLM. [TODO: code +
frozen-protocol release sentence.]

---

## 1. Introduction (skeleton)

1. **Problem.** Simulated humans (opponents, students, users) are evaluated
   and trained against models of *specific* people; concrete tasks: predict a
   named Lichess player's next-move think-time, a named student's next
   response.
2. **Failure mechanism of current approaches.** Static per-player embeddings
   and cohort models freeze the person; history-conditioned models confound
   accumulated state with recent context, and no prior line runs a
   capacity-matched control that separates the two (to our knowledge; §2).
3. **Insight.** If a person carries an evolving behavioral state, it should
   be most legible in self-paced *timing*, concentrate where behavior is
   least average (time pressure, weaker players), and require *accumulation*
   to capture — all three are testable against a memoryless twin.
4. **Method overview** + Figure 1 pointer [TODO: architecture figure].
5. **Contribution list** (below), one-to-one with §4.

### Contributions

1. **An equal-capacity evolving-vs-memoryless control on strict future
   splits.** Identical parameters, inputs, and optimizer; only state
   accumulation differs. The evolving latent wins future think-time in all 8
   clocked era×backbone conditions (Table 2), survives the twin at 2× its
   latent width, and adds value even after a released think-time model *and* a
   stable per-player calibration are locked in — isolating the within-player
   *dynamic* contribution from static individualization (§4.1, §4.3).
2. **The when-not-what asymmetry on real humans.** Move choice is a
   near-null (trunk-dependent at best; deviation-from-Maia-2 probe R² 0.009)
   while the timing edge survives baselines at or above published SOTA rank
   correlation, including Allie's released think-time head (§4.2–4.3,
   Tables 3–4).
3. **A mechanism account.** The timing edge concentrates 2.7–3.6× under time
   pressure (variance-controlled) and ≈3× for the weakest players; on a
   synthetic player with known hidden state the latent encodes it (probe R²
   0.93 vs 0.65) and clamping it moves predictions monotonically (§4.4,
   Table 5).
4. **Generality and utility beyond games.** On eight real knowledge-tracing
   datasets the evolving latent wins responses in 22/24 seed cells; the same
   latent recovers real population heterogeneity (Wasserstein 2× closer than
   an average-person baseline) and generates diverse, plausible synthetic
   students (recall 1.00 vs 0.00). Two real education response-time tests are
   negative and set the timing claim's scope (§4.5–4.6, Tables 6–7).
5. **The injection-channel ordering is backbone-dependent.** The trained
   hidden vector beats the verbal note by 0.07–0.12 nats on a board-native
   backbone; inside Qwen3 the verbal note helps think-time on all seeds while
   the hidden channel adds no advantage over it, and an SFT probe reproduces
   timing ≫ move from 0.6B to 8B (§4.7, Table 8).

---

## 2. Related work (themes + differentiation cuts)

- **Human-like and per-individual chess** (Maia-2/3, Maia4All, ChessMimic,
  Elo-Disentangled, Player-Specific). Cut: all static per player or
  cohort-level on the clock; none validate on a per-individual *future* split
  or isolate within-session dynamics from static individualization.
- **Dynamic psychological state in games** (Ailed; esports momentum). Cut:
  generative or forecasting constructs without validation against specific
  real players' held-out future behavior; we make the claim falsifiable.
- **Timing as a readout of latent state** (van der Linden speed-accuracy;
  Latency-Response Theory). Cut: the bar is not "RT beats choice" (settled)
  but "an *evolving* state beats a *stable* per-person speed calibration" —
  tested directly (§4.4, 3/5 cohorts significant, all point estimates favor
  the evolving state).
- **Evolving user state in simulators/recsys** (LATTE, HumanLM, Duan et al.,
  DASKT/DEKT). Cut: none run the equal-capacity same-input memoryless control
  on a future split; affect-KT lines predict correctness only, no
  response-time target; channel is fixed (verbal-only or vector-only) rather
  than compared.
- [TODO before submission: manual Psychometrika/JEBS sweep for dynamic
  latent-speed models — arXiv-invisible lineage.]

---

## 3. Method (one page)

- **DecisionPoint** interface: state, legal actions, per-decision reference
  (engine/IRT difficulty), time signal, recent-outcome stream. Chess and KT
  share it unchanged.
- **Evolving latent.** A GRU `f_φ` accumulates the same engineered history
  features (time-pressure, post-loss, fatigue, momentum) the control sees;
  `z_t` is causal (depends only on steps ≤ t). Rendered to the backbone as a
  hidden vector or a verbal note (one state, two channels).
- **Memoryless twin.** Identical parameters and inputs; the incoming hidden
  state is zeroed each step. Disclose the inert `weight_hh` nuance (≈2/3 of
  injector parameters receive zero gradient by construction).
- **Backbones.** Board-native FEN→from/to head (chess), logistic head (KT),
  Qwen3 via sglang (LLM arm). Timing head is a (zero-inflated) log-normal
  read from the latent only.
- **Training and evaluation.** SFT on move-NLL + λ·timing-NLL over each
  player's earlier trajectory, loss masked to train steps; scored on held-out
  later sessions; bootstrap over players.

---

## 4. Experiments and results

**Setup table (Table 1): datasets and cohorts.**

| Domain | Source | Cohorts | Unit counts | Channels |
|---|---|---|---|---|
| Chess | Lichess open DB | 2013-01 (blitz+rapid), 2017-04, 2019-07, 2021-04, 2021-06, 2023-04 | 100–120 players each, ≥30 games, ≈70k decisions/cohort | move NLL, think-time NLL |
| Knowledge tracing | ASSISTments 09/12/15/17, KDD Algebra + Bridge, Spanish, Statics | 8 datasets, frozen manifest | 150–500 students each, 3 seeds | response NLL (+ real RT where available) |
| KT (timing negatives) | ASSISTments 2009 `ms_first_response`, EdNet-KT1 singleton bundles | frozen protocols | 500 students × 3 seeds | response + timing |
| Go (negative) | OGS JSON think-times | 9×9 / 13×13 / 19×19 | up to N=519 trajectories (9×9) | timing |
| LLM arm | Lichess 2017 | Qwen3 0.6B–8B | 3 seeds per cell | move + timing completion NLL |

### 4.1 Is the evolving latent more than history-conditioning? (RQ1/RQ3)

**Table 2a — chess move NLL vs controls (2013-01, 100 players, 3 seeds).**

| Test | Control | Split | D−B | 95% CI | P(D−B<0) |
|---|---|---|---:|---|---:|
| E-C1 | static per-player embedding | session | −0.120 | [−0.135, −0.106] | 1.00 |
| E-C2 | memoryless twin | fraction | −0.069 | per-seed [−0.095, −0.039] | 1.00 |
| E-C3 | memoryless twin | future sessions | −0.067 | [−0.077, −0.057] | 1.00 |
| E-C3 replication (rapid) | memoryless twin | future sessions | −0.062 | [−0.072, −0.052] | 1.00 |

Capacity sweep: D's win survives B at 2× latent width; the gap closes only at
4× (≈+25% total parameters).

### 4.2 Is the state legible in timing or in move choice? (the headline)

**Table 2b — timing D−B at scale (5 seeds × 2 backbones, session split).**

| Cohort | mlp timing D−B (P) | conv timing D−B (P) | mlp move D−B (P) | conv move D−B (P) |
|---|---:|---:|---:|---:|
| 2017-04 | −0.0262 (1.00) | −0.0329 (1.00) | +0.0005 (0.18, null) | −0.0046 (1.00) |
| 2019-07 | −0.0277 (1.00) | −0.0331 (1.00) | −0.0009 (0.85, null) | −0.0083 (1.00) |
| 2021-04 | −0.0239 (1.00) | −0.0319 (1.00) | +0.0009 (0.13, null) | −0.0059 (1.00) |
| 2023-04 | −0.0211 (1.00) | −0.0270 (1.00) | −0.0048 (1.00) | −0.0175 (1.00) |

Timing wins all 8 conditions across a 6-year span; the move effect is small
(conv) to null (mlp). Zero-inflated timing head (proper model for 1s-quantized
clocks): D−B = −0.026 [−0.030, −0.022], so the gap is not a head artifact.

### 4.3 Does the timing edge survive released-model baselines? (G4)

**Table 3 — evolving latent as an add-on over aggregate → released baselines
(all rows: (B+z)−B, 100 players, 5 seeds).**

| Baseline | 2017-04 | 2019-07 | 2021-06 | Baseline Spearman |
|---|---:|---:|---:|---|
| Elo+clock | −0.043 [−0.056,−0.031] | — | — | 0.25 |
| Elo+clock+branching | −0.0247 | −0.0281 | — | 0.37–0.41 |
| Elo+clock+branching+Maia-2 entropy | −0.0254 | −0.0291 | −0.0386 | 0.414 / 0.445 / 0.447 |
| Allie locked (Allie vs Allie+z) | −0.0231 [−0.0303,−0.0154] | −0.0177 [−0.0297,−0.0053] | −0.0329 [−0.0449,−0.0226] | 0.620 / 0.642 / 0.646 |
| Elo+clock+Allie co-fit | −0.0129 (sig) | −0.0054 (ns, P=0.85) | −0.0133 (sig) | 0.627 / 0.654 / 0.661 |

The Maia-2-boosted baseline meets ChessMimic's published rank correlation
(0.41) and the latent still adds value on every cohort and seed. Against
Allie's released think-time head the direct test is significant on all three
cohorts; the fullest co-fit is significant on two of three (report the 2019
null).

**Table 3b — is the add-on over Allie just individualization?** After locking
Allie's released prediction *and* a stable per-player calibration, the
evolving latent still adds timing value (5 seeds, evolving − static-individual,
same locked-Allie offset and split).

| Cohort | evolving − static | 95% CI | Seed audit |
|---|---:|---|---|
| 2017-04 | −0.0170 | [−0.0254, −0.0082] | 5/5 favor evolving |
| 2019-07 | −0.0014 | [−0.0138, +0.0112] | 3 neg / 2 pos, null |
| 2021-06 | −0.0194 | [−0.0300, −0.0088] | 5/5 favor evolving |
| Unique-player aggregate (n=299) | −0.0126 | [−0.0188, −0.0061] | P(<0)=1.00 |

Subtracting a stable per-player calibration isolates the *within-player
dynamic* contribution: significant on 2 of 3 cohorts, with the 2019-07 null
matching its other strong-control results.

**Table 4 — the move channel against released SOTA.** Probe from latent to
deviation-from-Maia-2 (log P_Maia of the played move), held-out R²: D = 0.009,
B = −0.001 (vs 0.93 for a known synthetic state). Move choice is
near-stateless even relative to a human-move SOTA.

### 4.4 Where does the edge live, and what does the latent encode? (mechanism)

**Table 5 — concentration and controls.**

| Analysis | Result |
|---|---|
| Time-pressure terciles (timing D−B, 2 cohorts × 2 seeds) | high-pressure bucket 2–8× larger raw; 2.7–3.6× after per-bucket variance normalization; every high bucket CI excludes zero |
| Post-loss / fatigue buckets | flat (specificity check) |
| Rating strata (480 players, pooled) | weakest −0.0404 vs strongest −0.0139; high−low +0.0265 [+0.015, +0.038] |
| Stable per-player speed baseline (van der Linden analogue), 5 cohorts | significant D win on 2017-04, 2021-04 (−0.052), 2021-06 (−0.137); null on 2019-07, 2023-04; all 5 point estimates favor D |
| Synthetic known-state probe (held-out R²) | D 0.93 vs B 0.65 |
| Causal clamp ±ασ | monotone dose-response: entropy up, think-time up |
| Move-channel concentration | ≈0 in every bucket, both cohorts (the contrast) |

Scope note (state once): on real data the evolving-vs-memoryless edge blends
state-tracking with per-individual online calibration; the concentration
signature and the synthetic probe/clamp carry the dynamics reading, the
KT temporal-shuffle control shows the response-channel edge is
order-invariant individualization.

### 4.5 Does it generalize beyond chess? (KT + honest negatives)

**Table 6 — fixed-loader KT replication (8 datasets, 3 seeds, response D−B).**

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

22/24 cells significant for D; 7/8 dataset means favor D. Spread-vs-advantage
association: Pearson 0.776 / Spearman 0.476, dataset-bootstrap CIs cross zero,
Pearson 0.138 without Spanish — a suggestive association, not a law.

**Negatives (scope-setting, report plainly).**
- Real response-time transfer fails twice: ASSISTments 2009 timing
  inconsistent across seeds (+0.0003 / +0.0045 / −0.0062); EdNet-KT1
  (preregistered, singleton bundles) response −0.0159 [−0.0202,−0.0118] but
  timing −0.0004 [−0.0059,+0.0059]. Hypothesis (marked as such): education
  timers are incidental UI time, not a strategically managed clock.
- Go: null at every board size once board-size is controlled; the residual
  9×9 signal collapses on a 2.5× larger cohort (mean ≈ 0).

### 4.6 Can the latent recover and generate a population? (utility)

**Table 7 — population heterogeneity, real ASSISTments 2009 (500 students).**

| Model | W1 to observed | JS | Precision | Recall | corr(pred, obs) |
|---|---:|---:|---:|---:|---:|
| Per-individual latent | 0.074 | 0.17 | 0.86 | 0.75 | 0.96 |
| Average-person | 0.147 | 0.61 | 1.00 | 0.00 | — |
| Generated players (sampled latents, synthetic cohort) | 0.024 | 0.124 | 0.93 | 1.00 | — |

The average-person baseline is plausible but covers none of the population's
diversity; the latent both recovers it and generates it.

### 4.7 Which channel carries the state? (hidden vs verbal, by backbone)

**Table 8 — channel comparison.**

| Setting | Comparison | Result |
|---|---|---|
| Board-native, synthetic | hidden − verbal | −0.069, P=1.00 |
| Board-native, real 2013 | hidden − verbal | −0.117, P=1.00 |
| Qwen3-1.7B LoRA SFT, 3 seeds | verbal − none (timing) | −0.0050, all 3 seeds negative |
| Qwen3-1.7B LoRA SFT, 3 seeds | hidden − verbal (timing) | +0.0034 (hidden advantage absent) |
| Qwen3 0.6B→8B LoRA SFT | state-helps-timing Δ | −0.0107 … −0.0136 at every scale |
| Qwen3 4B/8B full-param SFT | timing / move Δ | −0.0110/−0.0128 timing; −0.0072/−0.0083 move (LoRA move-null is an adapter-capacity artifact) |

With no language prior the trained hidden vector is the richer channel; inside
an instruction-tuned LLM the verbal note carries the state and the hidden
channel adds nothing over it. The timing ≫ move asymmetry reproduces in the
LLM under a dense SFT probe and is graded (≈1.5×) under full fine-tuning.
Supporting negatives: frozen verbal injection ≈ irrelevant filler; sparse-reward
RL learns the task but cannot resolve the state effect.

---

## 5. Limitations (each with a boundary condition)

- Headline D-vs-B numbers use small from-scratch backbones; retired for
  timing by construction (head reads only the latent) and by G4, remains open
  for the move ceiling.
- Effect sizes are −0.01 to −0.07 nats; utility rests on calibration (proper
  log-likelihood) and the population demonstration, not the raw gap.
- The when-not-what asymmetry holds on real chess and synthetic KT; it does
  not transfer to the two real education timers tested. Boundary: domains
  where time is strategically self-paced.
- Dynamics vs individualization is only fully separable on synthetic data;
  real-data evidence is the concentration signature.
- No live human study; held-out future-behavior prediction and interactive
  judgment answer different questions.

## 6. Conclusion (3 sentences, TODO)

---

## Claim–evidence mapping (abstract → landing points)

| Abstract claim | Landing point | Status |
|---|---|---|
| Timing win, all 8 conditions, −0.021…−0.033 | Table 2b | ✓ |
| Adds value over Allie, 3/3 cohorts (direct test) | Table 3 rows 4–5 | ✓ (state the 2019 co-fit null in §4.3) |
| Move near-null; Maia-deviation R² 0.009 | Tables 2b, 4 | ✓ |
| Concentration 2.7–3.6× variance-controlled; ≈3× weakest players | Table 5 | ✓ |
| KT 22/24 cells; heterogeneity 2× Wasserstein | Tables 6, 7 | ✓ |
| Education RT negatives scope the timing claim | §4.5 negatives | ✓ |
| Hidden > verbal board-native; advantage disappears in LLM | Table 8 | ✓ (phrase as "disappears", not "verbal wins") |

## TODO before submission

- Figure 1 (architecture), Figure 2 (concentration bars), Figure 3 (KT
  spread-vs-advantage scatter with bootstrap band).
- ~~Run the missing joint control: Allie + static per-player calibration vs
  Allie + evolving latent.~~ **DONE** (Table 3b, `g4_allie_static_vs_evolving`):
  −0.0126 aggregate, 2/3 cohorts significant.
- G3 seeds + CIs, or keep the demoted "advantage disappears" phrasing.
- Cite DASKT/DEKT (TKDE'25), Maia4All (TMLR'26); manual psychometrics sweep.
- Verify every arXiv ID once more at submission time (Ailed's no-human-
  validation statement specifically).
