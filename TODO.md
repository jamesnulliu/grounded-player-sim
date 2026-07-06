# TODO / status

Work plan for `grounded-player-sim`, consolidated after the core results landed
(2026-07). Detailed results live in the dedicated docs; this file is now a
**status + remaining-ideas** index, not a task checklist.

**North star — three results that stand on their own** (design.md §8,
`documents/related_work.md`):
1. **when-not-what** — evolving state is legible in *timing*, near-null in *move
   choice*; robust across a 6-year span + reproduced in a non-game domain.
2. **the equal-capacity evolving-vs-memoryless control on a strict future split**
   — isolates dynamics/individualization from habit; settles the #1 objection.
3. **the backbone-dependent hidden-vs-verbal channel ordering** — hidden wins with
   no language prior (board-native RQ6), verbal wins inside an LLM (G3).
Per-individual / evolving / oracle / future-split / cross-domain are *supporting*
territory — build on them, never re-claim as novel.

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

- **A 3rd clocked cohort for G4** — the 2019 Elo+clock+Allie co-fit was ns
  (−0.005, P=0.85); a third cohort would firm the direct-test-significant /
  co-fit-marginal picture.
- **Real KT *with response-times*** (ASSISTments 2012 `ms_first_response`, EdNet)
  — the public 2009 release has no timing, so the *when-not-what* asymmetry is
  shown on real *chess* + synthetic KT only; real non-game timing would make it a
  cross-domain law.
- **Dynamics-vs-individualization de-confound on real chess** — the shuffle
  control shows the real-data edge is order-invariant *individualization*, and the
  concentration signal is variance-confounded; a within-game, variance-controlled
  test would strengthen the "dynamics" reading (currently clean only on synthetic).
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
