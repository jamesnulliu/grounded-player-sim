# TODO / status

Work plan for `grounded-player-sim`, consolidated after the core results landed
(2026-07). Detailed results live in the dedicated docs; this file is now a
**status + remaining-ideas** index, not a task checklist.

## Submission levers (2026-07-23 assessment) — the path to ICLR

Frank calibration: borderline at ICLR main (~35–45% with strong
execution); the science is done, acceptance hinges on execution. In
priority order:

1. **Downstream-utility demo (the single biggest lever).** Nothing yet
   shows the −0.01 NLL changing a decision anyone cares about. Candidate:
   use the population generator to stress-test an adaptive tutoring/item-
   selection policy and show the average-person simulator picks a
   measurably worse policy than the state-carrying one. Converts "small
   effect" into "consequential effect."
2. **Figures — all three are still TODO in paper.md.** The ladder diagram,
   concentration bars, and the drift-speed 2×2 ARE the argument.
3. **9-page story discipline.** Spine: decomposition → ladder →
   when-not-what → freezing/drift law → population. Everything else to
   appendix.
4. Read DASKT (2502.10396) in full; fill release/ethics TODO sentences.
5. Optional: median-scorecard G5 rerun (3 cells, bounds the mean-vs-median
   wart); deep full-text dedup on "LLM memory vs latent head-to-head"
   (sweep was abstract-level).
6. Routing: ICLR cog-sci/applications primary area; TMLR is the honest
   high-probability fallback.

## Reframe + memory control (2026-07-22) — status

The project reframed around the founding story: **user policy = static
attribute + evolving state**; static-persona simulation cannot represent the
person (emotion as motivation, "dynamic behavioral state" as the measured
claim). `documents/paper.md` (title/abstract/§1/§2/§4.3/§5/§6),
`summary.md`, `README.md`, `related_work.md`, `paper_outline.md` (header
note), and `claim_evidence_map.md` all updated.

- **DONE — structured-memory control (the "just a memory" objection).**
  New `latent_control="memory"` arm (`_memory_latents`, 15 causal running
  stats incl. raw think-time history + Allie residuals; zero trained
  params) under the exact G4 locked-Allie protocol. Memory beats static
  3/3 cohorts (pooled −0.0276) — the dynamic term confirmed by a
  training-free second instrument, including on 2019-07 where the learned
  arm was null. Richer-input memory matches/beats the 4-feature evolving
  GRU (pooled +0.0150): the claim is *carrying dynamic state*, not the
  GRU. `results/memory_baseline.txt`, `scripts/memory_vs_learned.py`.
- **DONE — input-matched learned control** (GRU over the same 15 memory
  features, 3 seeds × 3 cohorts): pooled GRU−memory = +0.0096
  [−0.0032, +0.0237] (tie, linear nominally ahead), GRU−evolving = −0.0054
  ns, GRU−static = −0.0180 [−0.0316, −0.0038] significant. Given equal
  information, learned and hand-designed dynamics coincide; every dynamic
  arm beats static. `scripts/memory_gru_arm.py`,
  `results/memory_gru_arm.txt`.
- **DONE — prior-art full-text digest** (Ailed = hand-tuned
  personality+psyche, engine-vs-engine, no human validation, no timing —
  conceptual neighbor, not an empirical threat; 2606.25176 v1
  "Elo-Disentangled" retitled v2 "Matilda", quote v2 numbers only; UniMaia
  "move delay" is an aux objective, never evaluated as held-out timing;
  LaRT = arXiv 2512.07019, static traits, van der Linden lineage).
  Scratchpad `prior_art_digest.md`.
- **DONE — novelty sweep (24 rows).** The *premise* (static personas
  insufficient, users carry evolving emotion) is now common in 2025–26
  LLM-agent papers (TWICE, AnnaAgent, Customer-R1); no paper runs the
  controlled quantitative demonstration (twin + released-SOTA/static/
  memory ladder + future splits + timing + population generation).
  Closest single competitor: **DASKT (arXiv 2502.10396)** — hand-engineered
  affect pipeline for KT; cite + read in full before submission.
  Scratchpad `novelty_report.md`.
- **DONE — KT-side memory arm** (2026-07-22, `scripts/kt_memory_arm.py`,
  paired against the frozen fixed-loader cells, fingerprints verified):
  M−B significantly negative 7/8 datasets (same profile as D−B), D−M ties
  5/8 with M winning 2 (incl. REPAIRING the ASSISTments-15 training
  reversal: D−B +0.0063 but M−B −0.0058) and D winning 1 (assist17).
  "Memory suffices" is CROSS-DOMAIN; per the shuffle control the KT
  instrument confirms the individualization term specifically.
  `results/kt_memory_arm.txt`. Docs updated (paper.md Table 6 + §4.5 ¶,
  abstract clause; summary/README/related_work/claim map).
- **DONE — four-axis LLM-agent contrast** (2026-07-22): §1 hook + dedicated
  §2 paragraph "Emotion-dynamic LLM user simulators"
  (mechanism/evidence/attribution/grounding) against AnnaAgent,
  Customer-R1, TWICE, prompted-appraisal agents.
- **DONE — "asserted, never measured" emphasis** (user-directed): abstract,
  §1, §6, summary.md one-line pitch.
- **DONE — KT fitted-static-profile arm** (2026-07-22, user question "does
  a static profile suffice on KT?"): S = frozen per-student train-split
  accuracy in the same logistic as the memory arm. ANSWER: NO — updating
  memory beats the frozen profile 8/8 (−0.011…−0.098, biggest where
  students drift most: Spanish), and S loses even to the memoryless twin
  6/8. "Order-invariant individualization" (shuffle) ≠ freezable — a
  running mean is order-invariant yet still updates, and the updating is
  load-bearing. KT scoping sharpened in paper.md §4.5 + abstract.
  `scripts/kt_static_arm.py`, `results/kt_static_arm.txt`.
- **DONE — G5 persona ladder inside the LLM** (2026-07-23,
  `scripts/g5_persona_ladder.py`, 12 cells, 99 eval players, paired
  player bootstrap): fitted person-info is what the LLM probe resolves —
  static−none −0.0104 and memory−none −0.0096, both significant (largest
  LLM person-effect in the project; upper-bounds hand-written-persona
  practice). Frozen-vs-updating is a TIE in this channel (+0.0009, CI
  spans 0): the board-native dynamic increment (−0.0126 over static) sits
  at/below this probe's resolution, and one-month blitz speed is the
  low-drift regime (contrast KT, where frozen failed 8/8). Channel
  ordering replicates G3 on fresh data: text beats the soft vector
  (hidden−memory +0.0045 sig). Practical line for the paper: fit the
  persona from data, keep it updating when the person drifts, deliver as
  text. Known wart: scorecard uses mean (premove-dragged) vs persona's
  median — a median-scorecard rerun (3 cells) would bound the tie; low
  priority. `results/g5_persona_ladder.txt`, paper.md §4.7 + Table 8.
- **DONE — `paper.md` placeholders filled** with the landed GRU-arm
  numbers (abstract, contribution 1, §4.3 third finding).

**North star — three results that stand on their own** (design.md §8,
`documents/related_work.md`):
1. **when-not-what** — evolving state is legible in *timing*, near-null in *move
   choice*; robust across a 6-year span on real chess + reproduced on
   *synthetic* non-game (KT) timing. (Real non-game *timing* is now attempted
   and is an honest negative — see Paper-readiness pass below; real KT
   *response* generality is a separate, positive, landed result.)
2. **the equal-capacity evolving-vs-memoryless control on a strict future split**
   — isolates dynamics/individualization from habit; settles the #1 objection.
3. **the backbone-dependent hidden-vs-verbal channel ordering** — hidden wins with
   no language prior (board-native RQ6), verbal wins inside an LLM (G3).
Per-individual / evolving / oracle / future-split / cross-domain are *supporting*
territory — build on them, never re-claim as novel.

---

## Paper-readiness pass (2026-07-13) — status

Triggered by an internal reviewer-style audit + independent verification of its
claims against the actual code (not just docs). Two of the claims that
survived verification (KT split leakage; the "equal-capacity" control's
active-vs-declared parameter asymmetry) were **not** in the original audit —
found by re-deriving them from the code directly. Work below, done this pass:

- **FIXED — KT split leakage.** `load_kt_csv` fit per-skill difficulty over
  the whole file, including an evaluated student's own held-out future
  responses. Fixed to training-prefix-only (regression-tested); re-running the
  n=500 headline post-fix leaves the effect intact, marginally *larger*
  (mean D−B ≈ −0.0128 vs the old −0.010, P=1.00 every seed) — not an artifact
  of leakage. `results/real_kt_rt.txt`.
- **DONE (honest negative) — real KT with response times.** The public
  ASSISTments 2009 release *does* carry `ms_first_response` (the standard
  preprocessing recipe just drops it); re-derived it
  (`scripts/prepare_kt_data.py`) and ran the real timing channel for the first
  time. Response stays robustly significant; timing is inconsistent across
  seeds and does not stabilize with more training — **does not replicate the
  when-not-what asymmetry on real response-time data.** Scope that asymmetry
  to real chess + synthetic KT, not a real cross-domain law.
  `results/real_kt_rt.txt`.
- **DONE (survives) — variance-controlled chess concentration.**
  Player-bootstrapped + variance-normalized re-analysis
  (`run_concentration_stratified`, `gps.experiments.ec`): the "timing edge
  concentrates under time pressure" signature survives normalizing by each
  bucket's own decision-level noise (raw ratio 4.0–6.1× shrinks to a still
  clearly >1× 2.7–3.6×, across 2 cohorts × 2 seeds) — firms up the
  state-dependence reading rather than requiring it be softened.
  `results/concentration_variance_controlled.txt`.
- **DONE (usually significant, cohort-dependent) — stable per-individual-speed
  (van der Linden-style) baseline.** A frozen three-cohort extension adds
  significant D wins on 2021-04 (−0.052) and 2021-06 (−0.137), plus a
  negative but null 2023-04 result (−0.037). With the existing 2017-04 win and
  2019-07 null, the pattern is **3 significant / 2 null**, with all five point
  estimates favoring dynamics. `results/stable_speed_baseline.txt` and
  `results/stable_speed_extension.json`.
- **DONE — manuscript corrections.** Elo-Disentangled misattribution fixed
  (base model drives the gain, not the embedding; its move near-null cited as
  convergent evidence); LATTE's evolving-vs-static matched comparison
  acknowledged; Duan et al. (arXiv:2605.30051) added; all previously-unverified
  arXiv IDs (ChessMimic, UniMaia, LATTE) directly checked against arxiv.org
  2026-07-13 and confirmed real, Ailed re-checked (still no human-subject
  validation as of this check); `P(D−B<0)` clarified as a bootstrap
  sign-support statistic, not a p-value, in `paper_draft.md` and
  `results_ec.md`; added explicit Limitations bullets on live-human-study
  framing (complementary, not categorically stronger) and practical
  significance (calibration + Milestone F as the utility demonstration, not
  the raw NLL gap). Abstract's ambiguous "reproduced in a non-game domain"
  phrasing corrected to distinguish real-response vs synthetic-timing vs
  real-timing (negative).
- **DONE (material correction) — fixed-loader KT replication.** The
  missing scratchpad path has been replaced by a frozen eight-dataset manifest,
  strict preparation + provenance hashes, resumable per-seed cells, and an
  aggregate fit with dataset bootstrap and leave-one-out sensitivity:
  `scripts/{prepare,run}_kt_replication*.py` and
  `src/gps/experiments/kt_replication.py`. Exact inputs were recovered from the
  raw ASSISTments 2009 mirror and theophilee commit `a7ae193`; all 24 cells ran.
  D wins significantly in 22/24 cells and on 7/8 dataset means. Statics seed 0
  is null; ASSISTments 2015 seed 2 significantly reverses and makes that dataset
  favor B on average (both anomalies reproduce exactly). Signed spread-vs-edge
  is Pearson 0.776 / Spearman 0.476 with wide sensitivity, so the old Pearson
  0.89 “law” is demoted to a suggestive association. Full results:
  `results/kt_replication_fixed_loader.{txt,json}`. Same provenance gap remains
  for some LLM/SFT results.
- **DISCLOSED — equal-capacity control's active-vs-declared parameters.**
  `persist=False` (arm B) zeroes the incoming hidden state every step, so
  `weight_hh` (≈2/3 of the injector's params) provably receives zero gradient
  and never leaves random init (verified by backprop). D and B still have
  identical *declared* params/inputs/optimizer, and `weight_hh`'s value is
  causally inert when its input is always zero regardless of training — not a
  fixable unfairness, just an undisclosed nuance. Documented directly in
  `src/gps/latent/neural.py`'s `NeuralInjector` docstring rather than left
  implicit.
- **DONE — 3rd clocked cohort for G4.** Downloaded a 370MB prefix of
  2021-06 (same recipe as 2017-04/2019-07), ingested 100 players, cached
  Maia-2 + Allie predictions, ran the full 5-seed G4 sweep. The exact test
  that was ns on 2019 (Elo+clock+Allie co-fit, −0.005, P=0.85) is
  **significant on 2021** (−0.0133, CI [−0.0183,−0.0088], P=1.000, closely
  matching 2017's −0.0129). Resolves the previously-marginal 1-significant/
  1-null co-fit picture to 2-significant/1-null — "usually significant,
  cohort-dependent," not "marginal." The direct Allie-vs-Allie+z test and the
  Maia-complexity baseline test are both significant on all 3 cohorts.
  `results/g4_timing.txt`, `g4_data/ec2021/`.
- **DONE — Allie + static-individual vs Allie + evolving.** With the released
  Allie prediction locked identically in both arms, evolving beats a static
  per-player embedding on 2017-04 (−0.017) and 2021-06 (−0.019), and is null on
  2019-07 (−0.001). Unique-player aggregate: −0.0126
  [−0.0188,−0.0061], n=299. This isolates dynamics beyond stable identity on
  2/3 cohorts. `results/g4_allie_static_vs_evolving.{txt,json}`.

---

## Milestones — all landed or resolved

| M | Status | Where |
|---|--------|-------|
| **A** — is the latent just history-conditioning? | **DONE.** Equal-capacity evolving-vs-memoryless control; E-A1 D−B=−0.006 P=1.00, capacity-robust; E-C on real chess. | `documents/milestone_a.md` |
| **B** — make training real | **DONE.** Real SFT loop (move+λ·timing NLL, temporal-split eval), minibatched; board-native + KT + sglang/API backbones. | code + `documents/training.md` |
| **C** — chess data pipeline | **DONE.** Lichess PGN ingestion (`gps ingest`), E-C1/C2/C3 headline (dynamic > static, > memoryless, future-sessions split), E-C6 timing. | `documents/results_ec.md` |
| **D** — generality | **Go = honest negative.** Fixed-loader KT: 22/24 seed cells and 7/8 dataset means favor D; broad but not universal. | `results_ec.md` (Go + KT) |
| **E** — verbal-vs-hidden (RQ6) | **DONE** (board-native): hidden richer than verbal (−0.069/−0.117, P=1.00). | `results_ec.md` |
| **F** — population recovery/generation | **DONE** on real KT: recovers the accuracy distribution (W1 2× < average-person) + generates novel players (recall 1.00 vs 0.00). chess-F = future work. | `results_ec.md` |
| **G** — LLM + released-SOTA benchmark | **DONE.** LLM is a secondary result (SFT probe; G3 hidden⊁verbal → backbone-dependent). **G4 LANDED:** latent adds think-time value over released models (Maia-2 ≈ ChessMimic Spearman; Allie's actual think-time 0.62–0.65). | `documents/milestone_g.md`, `results/g4_timing.txt` |

Also landed: concentration (timing edge 2–8× under time pressure, ≈3× for
weaker players); RQ2 probe + causal clamp on a synthetic hidden state. The
fixed-loader KT heterogeneity association is suggestive, not a law (signed
Pearson 0.776 / Spearman 0.476; wide bootstrap and Spanish-sensitive).

## Active paper critical path

1. **KT verification — DONE.** The result package and manuscript framing now
   use the fixed-loader summary; the “significant every seed” and “one law”
   claims are retired.
2. **Stable-speed extension — DONE.** Re-ingested and hash-froze 2021-04,
   2023-04, and 2021-06; all 9 static-control cells completed under one frozen
   protocol. Five-cohort result: 3 significant / 2 null, all point estimates
   favor D.
3. **EdNet decision — DONE, honest timing null.** Protocol was frozen before
   ingestion and restricted to singleton bundles. Response replicates strongly
   (−0.0159, CI excludes zero); timing is null (−0.0004, CI crosses zero) with
   one wrong-sign seed. Two real education timing datasets now fail to transfer
   the chess asymmetry.
4. **Manuscript — sole active critical path.** Convert `paper_draft.md` from
   synthesis to a submission structure. The empirical package is now frozen;
   preserve the corrected KT scaling, 3-of-5 stable-speed result, and two real
   education timing nulls.

## Remaining ideas / future work (kept — not scheduled)

- **Real non-game timing, done differently** — ASSISTments 2009 and the frozen
  singleton-bundle EdNet test are both clean timing negatives. A future domain
  would need response time under genuine strategic/self-paced control rather
  than another incidental UI timer. This is future work, not on the current
  paper critical path.
- **chess-F** — needs a *calibrated* per-individual timing head (the evolving
  injector is a state model, not an identity model; static embedding correlates
  0.61 but the log-think-time head is mis-scaled).
- **Go, done right** — a true byo-yomi clock (not a proxy) and/or a Go move
  backbone; the current negative used an oracle-free timing head only.
- **Controllable generation / valid stochastic sampling** — direct intervention on
  anchored latent dims; a variational/KL latent or fitting the empirical
  per-individual latent distribution (E-F1 did full-covariance prior sampling).
- **Design lever:** per-individual *parameter* vs. amortized state — a free
  per-user vector sharpens the distinction from LATTE but fights the 20-game
  data-efficiency goal; pick per experiment.

## Deprecated / not pursued (record, do not resurrect without reason)

- **B1–B9 baseline reconstruction** — superseded by the equal-capacity memoryless
  twin (the crucial control) + the B4 aggregate + the **G4 released-model**
  benchmark (Maia-2, Allie), which is stronger than re-implementing each baseline.
- **Stockfish / KataGo engine oracle** — the landed chess results are **oracle-free**
  (board-native FEN→from/to); KT uses an IRT oracle. Engine grading was not the
  moat that held (the future-split + equal-capacity control were).
- **Maia move-backbone (G1)** — not run by design (a strong trunk absorbs the move
  signal; timing is backbone-independent), answered architecturally + by G4.
- **LLM-as-headline / hidden≫verbal-*in-LLM*** — refuted by G3; the LLM is a
  secondary result and the channel ordering is backbone-dependent.

## Housekeeping (ongoing)

- `ruff format . && ruff check .` (line-length 79) + CPU-green `pytest` before each
  commit. W&B entity `jamesnulliu-university-of-southern-california`.

## Critical path (how it was built — record)

A (equal-capacity control, kills the #1 objection) → B (real SFT) → C (chess data
+ E-C headline: dynamic > static & > memoryless on the future split) → E
(verbal-vs-hidden, RQ6) → D (KT generality; Go negative) → F (population demo) →
G (LLM secondary + G4 released-SOTA add-on). Lead with **timing**; move + the LLM
are honest secondaries.
