# TODO / status

Work plan for `grounded-player-sim`, consolidated after the core results landed
(2026-07). Detailed results live in the dedicated docs; this file is now a
**status + remaining-ideas** index, not a task checklist.

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
- **DONE (cohort-dependent) — stable per-individual-speed (van der Linden-style)
  baseline.** The existing static-individual control (B2), scored on timing
  for the first time: D beats it significantly on 2017-04 (3/3 seeds) but is
  null on 2019-07 (3/3 seeds) — a genuine, honestly-reported partial result,
  not a universal win. `results/stable_speed_baseline.txt`.
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
- **PARTIAL — reproducibility.** `scripts/prepare_kt_data.py` (new, committed)
  + the `gps kt --data ... --response-time-col` CLI is now itself the
  reproducible entry point for the real-KT-with-timing result. The 7
  cross-dataset/platform KT replications (KDD-Cup, Spanish, Statics, etc.) in
  `results_ec.md` still depend on scripts that were never committed
  (`scratchpad/*.py`, not in this repo) — **not re-verified this pass**, only
  the primary ASSISTments-2009 cohort was. Same gap for some LLM/SFT results.
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

---

## Milestones — all landed or resolved

| M | Status | Where |
|---|--------|-------|
| **A** — is the latent just history-conditioning? | **DONE.** Equal-capacity evolving-vs-memoryless control; E-A1 D−B=−0.006 P=1.00, capacity-robust; E-C on real chess. | `documents/milestone_a.md` |
| **B** — make training real | **DONE.** Real SFT loop (move+λ·timing NLL, temporal-split eval), minibatched; board-native + KT + sglang/API backbones. | code + `documents/training.md` |
| **C** — chess data pipeline | **DONE.** Lichess PGN ingestion (`gps ingest`), E-C1/C2/C3 headline (dynamic > static, > memoryless, future-sessions split), E-C6 timing. | `documents/results_ec.md` |
| **D** — generality | **Go = honest negative** (no robust effect under a board-size control + power check). **KT generality DONE** on real students (8 datasets, 3 subjects). | `results_ec.md` (Go + KT) |
| **E** — verbal-vs-hidden (RQ6) | **DONE** (board-native): hidden richer than verbal (−0.069/−0.117, P=1.00). | `results_ec.md` |
| **F** — population recovery/generation | **DONE** on real KT: recovers the accuracy distribution (W1 2× < average-person) + generates novel players (recall 1.00 vs 0.00). chess-F = future work. | `results_ec.md` |
| **G** — LLM + released-SOTA benchmark | **DONE.** LLM is a secondary result (SFT probe; G3 hidden⊁verbal → backbone-dependent). **G4 LANDED:** latent adds think-time value over released models (Maia-2 ≈ ChessMimic Spearman; Allie's actual think-time 0.62–0.65). | `documents/milestone_g.md`, `results/g4_timing.txt` |

Also landed: the **heterogeneity scaling law** (|D−B| vs population spread, Pearson
0.89 across 8 KT datasets); concentration (timing edge 2–8× under time pressure,
≈3× for weaker players); RQ2 probe + causal clamp on a synthetic hidden state.

## Remaining ideas / future work (kept — not scheduled)

- **Real non-game timing, done differently** — the ASSISTments 2009
  `ms_first_response` attempt was a clean negative (see Paper-readiness pass);
  a domain where response time is under genuine strategic/self-paced control
  (unlike an incidental UI timer) may be needed to reproduce the timing side
  of when-not-what outside chess. EdNet (also has RT) is one candidate to try
  before concluding the asymmetry is chess-specific.
- **Commit the remaining KT-replication reproducibility scripts** — the 7
  cross-dataset/platform replications (KDD-Cup, Spanish, Statics, etc.) in
  `results_ec.md` still depend on uncommitted `scratchpad/*.py` scripts; only
  the primary ASSISTments-2009 cohort was re-verified against the leakage fix
  this pass. Re-derive + commit, mirroring `scripts/prepare_kt_data.py`.
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
