# TODO

Work plan for `grounded-player-sim`, ordered by dependency. Grounded in
`documents/design.md` (decisions) and `documents/priorart.md`-equivalent
positioning (§8 of design.md). Legend: `[ ]` todo, `[~]` partial/stub exists,
`[x]` done. **P0/P1/P2** = priority (do P0 first).

The north star (from the prior-art deep-read): the contribution is the
**conjunction** — per-individual + temporally-evolving + behavioral state
(tilt/fatigue/time-pressure) + drives moves & timing + validated on the
person's *future* games + chess **and** Go. Never claim a single axis as
novel (see design.md §8 for what is shared territory).

---

## Milestone A — Close the most dangerous reviewer objection (P0)

> "Isn't the dynamic latent just an expressive history-conditioned policy?"
> This is the #1 desk-reject risk. Must be answerable before any real run.

> Full runbook + resources: **`documents/milestone_a.md`**.

### Code
- [x] **New baseline: history-conditioned, no-latent policy.**
  `src/gps/policy/history_conditioned.py` — a `PolicyBackbone` that takes the
  *same* engineered history features (via the shared
  `gps.latent.structured.history_features`) the injector sees, but with **no
  structured/evolving latent** (feeds features to the head). The
  capacity-matched GPU control; `feature_vector`/`param_report` done +
  CPU-tested, `predict` is a documented torch stub.
- [x] **CPU memoryless control: `HistoryConditionedInjector`** (in
  `gps/latent/structured.py`) — same `history_features`, no accumulation —
  wired as the Phase-0 `history` arm so E-A1's direction is exercisable today.
- [ ] **New module `src/gps/baselines/` (the proposal's B1–B8).** One thin
  factory per baseline so experiments name them uniformly:
  - [ ] B1 population / no-personalization (no-latent backbone, no per-player signal) — use **Maia-3 / "Chessformer"** (ICLR 2026, current SOTA) as the move baseline, not the stale Maia-2
  - [ ] B2 static individual (per-player embedding, NO dynamic state)
  - [ ] B3 static-skill state-space (skill drifts across games, fixed within) — the "is it just rating drift?" foil
  - [ ] B4 aggregate timing (Allie-style: Elo-conditioned think-time, not per-individual)
  - [ ] B5 LLM persona prompt ("play as player X" from a profile) → `APIBackbone` + a static verbal `Injection`
  - [ ] B6 LLM persona + last-K-games
  - [ ] B7 static-covariate tilt (recent win-ratio as a fixed covariate; the mostly-null aggregate foil)
  - [ ] B8 = our proposed dynamic latent-state model
  - [ ] B9 ChessMimic-style cohort move+timing (per-100-Elo-band transformer;
    the sharpest move+timing-in-chess competitor — static, no oracle, no future
    split) → the head-to-head for E-C6 and a strong static foil for E-C1
- [x] **Trainable neural injector** (the real `f_phi`, replacing the
  parameter-free `StructuredInjector` for training).
  `src/gps/latent/neural.py` — `NeuralInjector`: a GRU recurrence over the
  shared `history_features` with real torch `parameters()` (so `SFTTrainer`
  no longer hits its no-op guard — guarded by
  `tests/test_neural_injector.py::test_neural_injector_unblocks_sft_guard`).
  One learned state reads out to the anchored `DIMENSIONS` and renders to
  **both** `VERBAL` and `HIDDEN` (channel-only contrast for RQ6). Lazy torch,
  CPU-importable; `param_report()` for the equal-capacity claim. Differentiable
  end-to-end. The training path (`latent_trajectory` + `persist` flag) and the
  real SFT loop are now wired (E-A1 below). The capacity-matched
  `HistoryConditionedBackbone` (board-native trunk) is still a stub — E-A1
  currently uses the `persist=False` twin as the equal-capacity control.

### Experiments
- [x] **E-A1 (Phase 0, extended) — first positive signal.** Two stages:
  - **CPU (untrained):** the history-conditioned baseline is a fourth arm in
    `experiments/phase0.py` (`history`); the *untrained* EMA heuristic does
    **not** beat it (`>history` False on all mechanisms incl. hysteresis,
    1.354 vs 1.350) — expected, the fixed EMA uses the wrong time constant.
  - **GPU (trained), the real test:** `experiments/ea1.py` (`gps train-ea1`)
    trains arm **D** (`NeuralInjector(persist=True)`) vs arm **B**
    (`persist=False`, the memoryless twin at *exactly* equal capacity — 1159
    params each) by SFT, on a **strict temporal split** (earlier games train,
    move-NLL scored on held-out later games), on the `HysteresisTiltPlayer`.
  - **Hardened verdict (the §5 decision rule, applied):** per-player bootstrap
    pooled over **240 distinct players** (5 seeds × 48), equal capacity →
    **mean D−B = −0.0060, 95% CI [−0.0081, −0.0040], P(D−B<0) = 1.000, D wins
    69%**. Significant (5/5 seeds negative, 4/5 individually sig). **Capacity
    (§6):** still significant at 3× B params; only at **12× B params** does it
    go marginal (−0.0025, CI just includes 0) while D still wins 65% — a
    memoryless control needs ~10× the capacity merely to approach parity.
    Magnitude small (~0.4% rel.) but capacity-robust; gate cleared. *Caveat:*
    per-run GPU nondeterminism (cudnn) makes single-seed CIs wobble, so the
    pooled-over-players number is the trustworthy one. Tooling:
    `gps.eval.bootstrap`, `run_ea1_capacity_sweep`, `--capacity-sweep`.
  - **§5 concentration check: done** (`run_concentration`) — the latent's
    held-out edge is monotone-concentrated in high-tilt decisions (move dD−B
    +0.002→−0.154, timing +0.142→−0.118), no better than memoryless at the calm
    baseline. The latent helps *where the dynamics are*; also explains why
    pooled effects are modest. E-C2 on real chess: also done (see Milestone C).
- `(a)` the **hysteretic** mechanism: done — `HysteresisTiltPlayer` (hidden
  leaky-loss integral; `test_hysteresis_is_not_reconstructable_from_history_features`
  proves the state is *not* a function of `history_features`).

---

## Milestone B — Make training real (P0)

### Code
- [x] **`src/gps/train/sft.py`:** real tensor loop done. Teacher-forces the
  latent along each trajectory; loss = move-NLL + λ·timing-NLL; AdamW; grad
  clip; per-epoch metrics streamed to W&B + `metrics.jsonl`; checkpoint saved;
  strict-temporal-split eval (`EvalSpec`). Generic over
  `injector.latent_trajectory` + `backbone.encode_batch/trajectory_loss`, so a
  board-native/LLM backbone drops in unchanged. Differentiable backbone for
  Phase-0/E-A1: `src/gps/policy/diff_policy.py` (`DiffMovePolicy`). Runs on the
  RTX 4060 (CUDA) or CPU. **Minibatching: done** (`_train_masked_minibatched`,
  `config.batch_size`): length-sorted, pre-encoded minibatches over players on
  the masked (E-C) path → 100 players train in ~97s (full-batch had walled
  >9 min); the E-A1 window path is unchanged. This unblocked the significant
  real E-C2 headline. **Still TODO:** the `HIDDEN`-soft-prompt LLM path, and
  **val-NLL early stopping** (the from-scratch board head overfits past ~30
  epochs on real data — for now pass `epochs≈15`).
- [x] **`src/gps/policy/sglang_backbone.py`: DONE + RUN.** `_engine()` launches
  `sgl.Engine` + an `AutoTokenizer`; `move_logprobs(dp, injection)` scores each
  legal move as a continuation (logprob from sglang `input_token_logprobs`);
  `predict()` softmaxes → `MoveDistribution`. **First LLM result**
  (`experiments/llm_inject.py`, `python -m gps.experiments.llm_inject`): a frozen
  LLM + verbal state note on real chess move-NLL is **unreliable** — helps
  Qwen3-1.7B (−0.031) but HURTS Qwen3-8B (+0.076). The frozen note never learned
  the player-specific mapping → motivates the *trained* latent (board-native is
  the validated win) and shows the persona-prompt baseline (B5/B6) is weak.
  Gotcha: sglang spawns subprocs that re-exec the main script → run from a FILE,
  not `-c`/heredoc. **Timing from the LLM still TODO.**
- [ ] **`src/gps/policy/sglang_backbone.py` HIDDEN path:** soft-prompt /
  prefix-embedding injection for `InjectionKind.HIDDEN` (`enable_hidden=True`).
  **This + a trained injector is the real LLM test** (the frozen-verbal result
  above shows the naive path is insufficient).
- [~] **`src/gps/policy/board_native.py`:** **minimal version landed** — an
  oracle-free, CPU-runnable move model: FEN → 12×64 piece planes (+ side-to-
  move) → factored from/to logits masked to the position's legal moves, latent
  conditioning via concat, plus a latent-driven log-normal timing head. Crucial
  unblock: unlike `DiffMovePolicy` (which scores a softmax over **engine
  value-advantages** and so *needs* a per-move oracle), this scores real Lichess
  positions from `fen_before` + `legal_actions` alone — so E-C1/2/3 next-move-NLL
  needs **no Stockfish/eval-set**. Variable-length + masked `BoardBatch`
  (`step_mask`/`action_mask`), per-player temporal-split helpers
  (`split_indices`/`train_eval_masks`), and `encode_batch`/`trajectory_loss`/
  `per_traj_move_nll` matching the trainer protocol. Tested CPU/offline incl. a
  smoke-train that drops move-NLL (`tests/test_board_native.py`). **Conv trunk
  (`trunk="conv"`) LANDED + tested:** Maia-style spatial 12×8×8; lowers move-NLL
  (2.96 vs 3.07) but shrinks the latent's move D−B (−0.027 → −0.003) — a
  stronger board model absorbs the move signal, so the move-win is partly a
  weak-backbone artifact (timing head reads only the latent → unaffected). Honest
  caveat in `results_ec.md`. **Still documented-future:** mover-perspective
  orientation, promotion head (today two promos sharing from/to split evenly).
- [ ] **`src/gps/train/sft.py::save` + a `load`:** checkpoint injector
  (+ optionally backbone) state_dicts.

### Experiments
- [ ] **E-B1:** smoke-train the neural injector on Phase-0 synthetic data on
  GPU; confirm it recovers injected mechanisms better than the hand-specified
  `StructuredInjector` (closes the "untrained heuristic isn't asserted to
  win" gap from design.md §5).

---

## Milestone C — Chess data pipeline (P0, gates all real results)

### Code
- [x] **`src/gps/data/lichess.py`:** parse Lichess PGN + `[%clk]` (python-chess
  `GameNode.clock()`), emit `train.base.Trajectory` per player. Exclude bot
  accounts. Focus blitz/rapid for within-session density. **Landed + validated
  on real data (2026-06-28):** ran the *real* `open_pgn`/`iter_game_records`
  over 60k games of the 2017-04 archive (streamed from a 120MB HTTP-range
  prefix) — parses clean (99.4% clocked, 100% UTC, 0 bots, no large-`--long`
  zstd bug), `bucket→stats→select→build_trajectory` all run, a sample player
  rendered 186 games → 5,582 DecisionPoints. Two data facts folded in below
  (parser throughput; 1s-quantized, zero-inflated clocks).
- [ ] **`src/gps/games/chess.py`:** concrete `Game` — board encoding (FEN for
  LLM backbone, tensor for board-native), `legal_moves` (UCI), `apply_move`.
  *Note: logged trajectories already carry `fen_before` + `legal_actions`, so
  next-move-NLL scoring (E-C1/2) does not strictly need this; it is the
  board-native tensor-encoding path that does.*
- [~] **`src/gps/games/oracles/stockfish.py`:** `EngineOracle` wrapping
  Stockfish (or the published Lichess eval set) → per-move centipawn loss.
  **Record depth** (centipawn-loss is settings-dependent — must report it).
  *Both oracle sketches landed* (`StockfishOracle` with precise `move_loss`;
  `LichessEvalOracle.from_subset` + `eval_set_coverage`). **Not yet exercised
  on real data:** no Stockfish binary on PATH, and the coverage call needs the
  multi-GB `lichess_db_eval.jsonl.zst`. Deferred — only gates E-C4/5/6, not the
  E-C1/2/3 next-move-NLL headline.
- [x] **Player selection util:** filter players by volume AND multi-game
  session count (need within-session dynamics). *`select_players` (volume +
  session gates) landed in `lichess.py` and validated — even a 60k-game prefix
  yields 13 players at ≥50 games/≥3 sessions, so a full month is not data-
  limited.* **Single time-control class:** done — `speed_class()` (Lichess
  estimated-duration buckets: ultrabullet/bullet/blitz/rapid/classical) +
  `--speed` filter applied in *both* ingest passes, so timing analysis is never
  contaminated by a mixed-clock cohort (the 4430s outlier).
- [x] **Production ingest driver (`gps ingest` CLI).** Landed:
  `src/gps/data/ingest.py::run_ingest` + the `gps ingest` subcommand. **2-pass:**
  pass-1 is a cheap *header-only* stats scan (`iter_game_summaries` via
  `chess.pgn.read_headers`, no board/legal-move gen) → `select_players` cohort;
  pass-2 does the full parse for that cohort only → `build_dataset` → persist.
  **Parallelized:** `iter_game_records_parallel` ships whole-game text chunks
  (`split_pgn_games`, cheap producer) to a worker pool that does the python-chess
  parse (the ~119 games/s bottleneck), so a full month drops from ~26h to
  ~26h/workers across the 12 cores. **Persistence:** new `src/gps/data/store.py`
  — gz-aware JSONL, one trajectory/line, with deduped (re-aliased on load)
  per-game `OutcomeStream`s so an active player's history is not re-serialized
  O(moves×games). Writes `dataset.jsonl.gz` + a reproducibility `manifest.json`.
  Engine oracle intentionally left off (gates only E-C4/5/6). **Still TODO:**
  pass-1 buckets all players in memory (lightweight summaries; cap with
  `--max-games` or shard for a full uncapped month).
- [ ] **`src/gps/data/sessions.py`:** already done — but **sweep the gap
  threshold as an ablation** (it is an unlabeled construct, see design.md §6).

### Experiments
- [x] **E-C1 (RQ1) — LANDED on real data.** Dynamic (B8/arm D) vs. **static
  individual** (B2 = per-player `nn.Embedding`, no within-trajectory dynamics;
  `gps.latent.static_individual.StaticIndividualInjector`,
  `run_ec(control="static")` / `gps train-ec --control static`). On the
  100-player `2013-01` blitz cohort (session split, near-equal capacity — B2 is
  *slightly bigger* at 64402p vs D 63926p, so D winning is conservative), D
  beats B2 across 3 seeds (D−B2 −0.089/−0.108/−0.164, all P=1.00, D wins
  78–90%); **pooled 300: D−B2=−0.120, CI [−0.135, −0.106], P=1.000** — a larger
  gap than the memoryless control, as expected (a fixed style is a weaker foil
  than instantaneous history-conditioning). So the evolving latent beats *both*
  a fixed per-player style **and** memoryless history-conditioning. **Remaining:**
  calibration + timing metrics; replicate on more months/cohorts.
- [x] **E-C2 (RQ1, the dangerous one) — LANDED on real data.** On Lichess
  `2013-01` blitz (100-player cohort, strict per-player future split, equal
  capacity, equal inputs — only the persist bit differs), the evolving latent
  **significantly** beats the memoryless twin: **D−B ≈ −0.069, 95% CIs exclude
  0 across 3 seeds, P(D−B<0)=1.00, D wins 72–84% of players** (both beat
  uniform-over-legal 3.18). This answers the #1 reviewer objection on *real*
  chess, not just synthetic E-A1. Reproduce: `gps ingest … --max-players 100
  --max-games-per-player 20` then `gps train-ec … --epochs 15 --batch-size 16`.
  Caveats to state: weak from-scratch backbone; **move-NLL win is
  cohort-dependent** — strong on `2013-01` (blitz −0.067, rapid −0.062, P=1.00)
  but **null on `2017-04`** (−0.0001, P=0.52; see `documents/results_ec.md` and
  E-C6). Lead with the robust **timing** result; present the move-NLL win as
  significant-but-cohort-dependent, not universal. **Next:** more months,
  a stronger/engine-informed move model to test if the 2017 move null is a weak-
  backbone artifact. *(Driver details below.)*
- [~] **E-C2 driver:** `experiments/ec.py::run_ec`
  (`gps train-ec <dataset>`) trains arm D (`NeuralInjector` persist=True) vs
  arm B (persist=False, parameter-identical) on the oracle-free
  `BoardNativeBackbone`, on a **per-player** temporal split (variable-length,
  masked — `SFTTrainer` generalized: `EvalSpec.splits` + `train_eval_masks`,
  E-A1's global-window path untouched), bootstrapping D−B over players exactly
  as E-A1 so the numbers are comparable. CPU/offline-validated end-to-end
  (`tests/test_ec_experiment.py`, incl. store→reload→run_ec and a no-leakage
  check); on board-determined synthetic targets it correctly reports D=B=0 (no
  hidden dynamics to exploit → unbiased). **Positive control validated:**
  `HiddenTiltChessPlayer` (`synthetic/chess_players.py`, hysteresis hidden tilt,
  dual-use engine ref) → D beats B; pooled over 64 players (4 seeds × 16)
  **D−B=−0.0177, 95% CI [−0.034,−0.002] (excludes 0), P(D−B<0)=0.984** (single
  seeds wobble — pool, as in E-A1). Two findings folded into design: (1)
  `base_beta` must be low (~3) so the player loses enough to give the hidden
  integral `h` real range — base_beta=6 gives a spurious null; (2) the latent's
  edge is *cleaner with an engine-value move model* (`DiffMovePolicy`,
  significant at a single seed P=1.00 on the same data) than a from-scratch
  board model (needs pooling) — so E-C move-NLL benefits from engine-informed
  sharpness or a strong/pretrained trunk. Also fixed: `BoardNativeBackbone`
  now seeds its init (forked RNG) so D/B share identical init and runs are
  reproducible. **First REAL run done** (Lichess `2013-01` blitz via
  `gps ingest`, 40-player cohort, ≤20 games each): at early-stop (epochs 10–30)
  the model **beats uniform-over-legal (3.18)** and **D beats B with
  P(D−B<0)=0.94–0.96** (D−B≈−0.003 to −0.012) — the headline *direction*,
  near-significant at 40 players. Two operational findings: early stopping is
  mandatory (overfits past ~30 epochs → NLL above uniform); a *significant* CI
  needs more players, which the full-batch trainer can't fit → **minibatching**
  is the blocker. Added `--max-games-per-player` to bound/equalize lengths.
  **Remaining:** minibatch → larger cohort → significant CI; then a 2017+
  archive (for clocks → E-C6 timing).
- [x] **E-C3 (RQ3, the decisive test) — LANDED on real data.** Train on a
  player's earlier **sessions**, predict their **later** sessions
  (`run_ec(split_mode="session")` / `gps train-ec --split session`;
  `session_split_indices` recovers sessions from the persisted
  `recent_outcomes`). On the 100-player `2013-01` blitz cohort (98/100
  multi-session, median 6), D significantly beats the memoryless twin across
  3 seeds (D−B −0.053/−0.061/−0.087, all P(D−B<0)=1.00, D wins 72–88%);
  **pooled 300: D−B=−0.067, 95% CI [−0.077, −0.057], P=1.000**. The win
  survives the harder future-sessions split, so it is real evolving dynamics,
  not habit memorized within a sitting. **Remaining:** replicate on more
  months/cohorts and a 2017+ archive; report calibration + timing too.
- [x] **E-C4 (RQ2):** state-recovery probe **landed (presence + CAUSAL use:**
  `run_causal_intervention` clamps the latent ±α·σ along the probed direction →
  monotone dose-response in the expected direction, so the state is *used*, not
  just present). Presence detail: —
  `experiments/ec.py::run_state_recovery` linearly probes each arm's latent for
  the ground-truth hidden state (`hidden_h`, available on the synthetic
  `HiddenTiltChessPlayer`), held-out R². The evolving latent recovers it far
  better than the memoryless twin: **held-out R² D=0.93 vs B=0.65 (ΔR²=+0.27)**
  — the latent *encodes* the behavioural state, which explains why D predicts
  behaviour better (links E-C2→RQ2). **Causal intervention also landed**
  (`run_causal_intervention`): clamping the latent ±α·σ along the probed
  hidden-state direction changes held-out predictions **monotonically in α and
  in the expected direction** — tilt flattens moves (entropy↑) and slows timing
  (μ↑); move KL grows ~quadratically. So the latent is *used*, not just present
  (presence-vs-use gap closed, design.md §6). Move response modest (matches the
  small move-NLL effect), timing response larger (matches the robust timing
  pillar). **Still TODO:** probe against engineered indicators on *real* data
  (no ground-truth state there, so probe observable proxies).
- [ ] **E-C5 (RQ4):** B8 vs. LLM persona prompt (B5/B6) — move + timing
  distribution match with the engine-graded yardstick.
- [x] **E-C6 (timing) — LANDED AT SCALE (the robust pillar).** *(5-seed ×
  5-cohort × 2-backbone Tier-1 sweep on 2×A100: timing D−B significant P=1.00 in
  all 8 clocked conditions across 2017–2023; adds value over an Elo+clock+
  position-complexity baseline, Spearman 0.39 ≈ ChessMimic 0.41 — see §At scale.)*
  First-signal detail:
  `run_ec` now also scores per-player **think-time NLL** (D vs memoryless;
  `BoardNativeBackbone.per_traj_timing_nll`). On `2017-04` blitz (clocked, 100
  players, session split, 3 seeds) the evolving latent **significantly** beats
  the memoryless twin on timing: **pooled D−B=−0.069, 95% CI [−0.082, −0.057],
  P(D−B<0)=1.000** — while move-NLL is *null* on the same cohort (see
  `documents/results_ec.md`). So **timing is where the evolving state robustly
  helps**; lead with it. **Zero-inflated head landed**
  (`timing_model="zi_lognormal"`): a learned latent-driven mass on 0s premoves
  + a log-normal on the rest — fits ~0.77 nats better (NLL 2.46 vs 3.23) and D
  still significantly beats B (pooled D−B=−0.026, CI [−0.030,−0.022], P=1.00).
  So the timing pillar survives a credible discrete model. **B4 aggregate
  baseline landed** (`run_timing_vs_aggregate`): comparing latent-only to an
  Elo+clock aggregate is unfair (clock too strong; B4 2.69 < latent-only 3.23),
  so the fair test is **B4 vs B4+z** (aggregate vs aggregate+evolving latent):
  **(B4+z)−B4=−0.043, CI [−0.056,−0.031], P=1.00** — the evolving latent **adds
  significant value over** the aggregate (the defensible claim: augments
  Elo+clock, doesn't replace it). Per-player Pearson 0.143→0.15 (below
  ChessMimic's r=0.41 — a weak-backbone gap, B9). **Still TODO:** a stronger
  backbone to close the Pearson gap; B9 head-to-head on matched data.

---

## Milestone D — Generality: Go and/or an oracle-preserving non-game domain (P1)

> Decision (design.md §11): chess is primary and proven *first*. Generality is
> polish on a proven mechanism — do **not** start this before the chess
> headline (E-C1/C2/C3) lands. Go gives empty-frontier novelty cover but is
> still a board game; a **non-game oracle domain** (knowledge tracing — EdNet /
> ASSISTments; or competitive programming — Codeforces) gives *real*
> cross-modal generality and is the stronger main-track upgrade. Whatever is
> added must keep the per-decision oracle (the moat); never pivot to an
> oracle-less domain (rec/dialogue/social).

### Code
- [ ] **`src/gps/games/go.py`:** concrete `Game` for Go (SGF positions, GTP
  moves, byo-yomi in `TimeSignal`).
- [ ] **`src/gps/data/sgf.py`:** KGS/OGS SGF parser with per-move time +
  byo-yomi; GoGoD for deep careers. **Confirm per-move timing availability
  before committing scope** (proposal Risk 1).
- [ ] **`src/gps/games/oracles/katago.py`:** `EngineOracle` wrapping KataGo →
  per-move points-lost (winrate/score drop). Record visit count.
- [x] **Non-game oracle domain (knowledge tracing) — LANDED (synthetic).**
  Ported the `DecisionPoint`/`EngineReference` interface to KT with **zero
  changes to the injector, trainer, eval, or bootstrap** — only a new backbone
  (`gps/policy/kt_backbone.py::KTBackbone`, logistic correct/incorrect over
  item-difficulty + latent) and `Game.KNOWLEDGE_TRACING`. `gps/experiments/kt.py`
  (`build_kt_dataset` + `run_kt`). The "game-agnostic core" claim is now
  *demonstrated*, not asserted. **Next:** real KT data (EdNet/ASSISTments).

### Experiments
- [x] **E-D1 (RQ5) — LANDED on synthetic KT + REAL data.** The *same* framework,
  only the backbone swapped, **reproduces the chess pattern** in knowledge
  tracing (a non-game domain): pooled over 96 students, **response-TIME
  D−B=−0.050, CI [−0.055,−0.045], P=1.00, D wins 100%** — timing is the robust
  channel just as in chess, while the discrete correct/incorrect outcome is weak
  (mirrors the chess move-NLL). Same mechanism, same signature, non-game domain →
  the contribution is the dynamic-state mechanism, not a chess artifact.
  **NON-SYNTHETIC (2026-06-30, ASSISTments 2009, future/temporal split):** on
  real student RESPONSE prediction the evolving latent beats the memoryless twin,
  and the win is **robust across a cohort sweep** (n∈{150,300,500} × min_resp∈
  {30,50}): D−B negative in all 5 configs, P(D−B<0)≥0.96 everywhere, CI excludes
  0 for n≥300, significance strengthening monotonically with sample size.
  Headline (n=500), **seed-stable across 3 seeds**: D−B=−0.0095/−0.0116/−0.0090
  (mean ≈−0.010), **P=1.00 in every seed**, all CIs exclude 0, D wins 64–73%.
  **Replication across 8 datasets / multiple platforms / 3 subject domains:**
  ASSISTments 2009/2012/2015/2017, KDD-Cup Algebra + Bridge-to-Algebra, Spanish
  (language), and Statics (engineering); D−B −0.004…−0.03, **significant every
  seed** — not dataset/platform/subject-specific. Synthesis: effect size
  **scales with population heterogeneity** (Pearson 0.89, n=8; strong linear
  trend anchored by the extremes, noisier middle band Spearman 0.74; links
  RQ5↔F). All runs in W&B `gps-kt-scaling`. On
  real students the *discrete-response* channel — weak on synthetic — carries
  genuine signal (real ability/learning/forgetting). Builders
  `scratchpad/real_kt{,_sensitivity,_f500}.py`, raw `results/real_kt.txt`.
  Milestone-F on the *same* 500-student cohort recovers the real accuracy
  distribution too (W1 2.0× < average-person, corr 0.96, recall 0.75 vs 0.00 —
  see Milestone F), so the same 500 students both predict responses (RQ5) and
  recover the population distribution (F).
- [ ] **E-D2:** Go-specific dynamics — lean on within-game byo-yomi time
  pressure where cross-game session data is thinner.
- [ ] **E-D3 (RQ5 stretch, OPTIONAL):** cross-game latent correlation for
  players who play both. **Caveat heavily** — no reliable Lichess↔KGS/OGS
  identity map; the chess∩Go high-volume population is tiny. May be cut.

---

## Milestone E — The verbal-vs-hidden headline (P1, cheap & unclaimed)

> Promoted from ablation to a first-class RQ6 (design.md §9): nobody compares
> text-memory vs. soft-vector as interchangeable injection channels.

### Code
- [x] Both channels exist + the **same learned state** renders either way.
  `NeuralInjector(readout=True)` emits the anchored `DIMENSIONS` (the verbal
  text proxy) instead of the full hidden vector — so the RQ6 comparison is
  channel-only (same seed, same recurrence), not state-only. `run_rq6`.

### Experiments
- [~] **E-E1 (RQ6) — LANDED (move; CPU, no LLM).** Same recurrence delivered
  as the full **hidden** vector vs the few interpretable **verbal** anchored
  dims, near-equal capacity. **Hidden is significantly richer** on both
  synthetic (hidden−verbal=−0.069) and real `2013-01` blitz (−0.117), P=1.00 —
  the verbal channel (portable to closed APIs) is lossy by ~0.07–0.12 nats, and
  the gap is *larger on real data* (richer un-anchored dynamics). Confirms the
  hypothesis: hidden richer but open-weight-only; verbal portable but lossy.
  **Still TODO:** the timing channel; a real verbal LLM backbone (the closed-API
  ceiling / RQ4 link).

---

## Milestone F — Population generation (P2, DEMONSTRATION not a pillar)

> Per design.md §10: build it, instrument it with the full distributional
> eval, decide demo-vs-pillar **last** based on the numbers. Keep fidelity and
> diversity experiments separate.

### Code
- [ ] **Controllable generation:** `OracleInjector`-style **direct
  intervention on the anchored latent dims** (set tilt/fatigue/time-pressure)
  — NOT history editing (design.md §10). Likely a small `LatentIntervention`
  wrapper in `gps/latent/`.
- [ ] **Valid stochastic sampling** (one or more):
  - [ ] variational / KL-regularized latent → sample the prior;
  - [ ] fit + sample the **empirical per-individual latent distribution** over
    the real training population;
  - [ ] interpolate between real players' latents.
  - [ ] Distinguish noise on latent *output* (per-step jitter) vs. on injector
    *weights* (systematic "different coherent player").
- [x] **`src/gps/eval/distributional.py`:** built — `wasserstein_1d`,
  `js_divergence`, and **generative `precision_recall`** (Kynkäänniemi 2019,
  any-dimension k-NN: precision=plausibility, recall=diversity/coverage).
  Wired into `run_population`. **Still TODO:** more behavioral stats
  (time-allocation, blunder-by-phase) + an explicit off-manifold/realism rate.

### Experiments
- [x] **E-F1 — LANDED (synthetic KT).** `run_generation`: sample 200 novel
  players from a **full-covariance** Gaussian prior over the real style latents
  (a *diagonal* prior under-disperses — a real methodological finding), score
  the generated population against the real one (distributional, no pointwise
  ground truth). Generated matches the real accuracy spread (0.155 vs 0.165),
  W1=0.024, JS=0.124, **precision/recall=0.93/1.00** vs average-person
  **1.00/0.00** — plausible AND fully diverse, beating the "positive average
  person". So the latent is a per-individual *generator*, not just a predictor.
  **Next:** real data; richer behavioral stats (timing, blunder-by-phase).
- [x] **E-F2 — LANDED (synthetic + REAL KT).** *(Real ASSISTments 2009, same
  500-student cohort as RQ5: recovers the real accuracy dist, Wasserstein 2× <
  average-person, corr 0.96, recall 0.75 vs 0.00 — see `results_ec.md` §Population
  and `results/real_kt.txt`.)* Does the per-individual latent
  **recover real heterogeneity** and beat the "positive average person"? **Yes,
  strongly.** `run_population` (`build_kt_dataset(skill_spread=)`): with distinct
  per-student skills, the latent reproduces the population accuracy distribution
  — **4–5× lower Wasserstein** to the observed dist than the average-person
  point-mass baseline (W1 0.032 vs 0.174 at spread 1.5), model spread 0.184 vs
  observed 0.206, **corr(pred,obs)=0.96**. Generative **precision/recall** makes
  it crisp: model 0.97/0.95 (plausible AND diverse) vs average-person
  **1.00/0.00** (perfectly plausible, ZERO coverage = the textbook average-
  person trap); JS 0.12 vs 0.68. Scales monotonically with the true skill
  spread. **Strong → candidate second pillar.** **Next:** real data; true
  generation (E-F1, sample latents).

---

## Cross-cutting / housekeeping (ongoing)

- [ ] **Tests** for every new module (match the existing CPU, no-GPU,
  stdlib-only pattern where possible; mark GPU paths to fail with an
  informative error, as the backbones already do).
- [ ] **`ruff format . && ruff check .`** (line-length 79) before each commit.
- [x] **Verify 2026 preprint metadata** — done 2026-06-25 (/research sweep).
  All resolve: HumanLM 2603.03303 (real, Stanford), LATTE 2605.26612 (real),
  **ChessMimic 2606.04473 (real — the earlier "verify it exists / likely drop"
  was wrong; it just hadn't propagated to general web search. It is now the
  primary chess competitor).** New competitors found and folded into
  design.md §8: Player-Specific Behaviors (2605.11893), Mixture of Masters
  (2602.04447), Maia-3 / Chessformer (ICLR 2026), BGU "Blunder prediction in
  chess" (Springer 2026), DASKT (2405.16799). **Still to do:** read ChessMimic
  + HumanLM + LATTE experimental sections in full before drafting related work.
- [ ] **Related-work section** writeup: the comparison table + the
  one-sentence framing from design.md §8. Lead with the conjunction; never a
  single shared axis.
- [ ] **Decide per-individual *parameter* vs. amortized state** (design-level):
  a free per-user vector is a sharper distinction from LATTE's amortized
  predictor, but conflicts with the 20-game data-efficiency goal. Pick per
  experiment.

---

## Suggested critical path (minimal first paper)

1. Milestone A (history-conditioned baseline + neural injector) — kills the
   #1 objection.
2. Milestone B (real SFT + sglang predict) — makes training real.
3. Milestone C (chess data + E-C1/2/3) — first real headline:
   **dynamic > static individual on the future-behavior split**, and
   **dynamic > history-conditioned at equal capacity**.
4. Milestone E (verbal-vs-hidden, RQ6) — cheap, unclaimed, differentiating.
5. Milestone D (generality, RQ5) — only after the chess headline lands. Go for
   empty-frontier novelty cover, or a non-game oracle domain (knowledge
   tracing / Codeforces) for stronger cross-modal generality (design.md §11).
6. Milestone F (population demo) — only if A–E land; decide demo-vs-pillar
   on the numbers.
