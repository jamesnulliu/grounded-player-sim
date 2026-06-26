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
- [ ] **Trainable neural injector** (the real `f_phi`, replacing the
  parameter-free `StructuredInjector` for training).
  `src/gps/latent/neural.py` — a recurrent/state-space latent implementing
  `LatentStateInjector` with `parameters()` so `SFTTrainer` stops hitting its
  no-op guard. Keep `produces` honoring both `VERBAL` and `HIDDEN`.

### Experiments
- [~] **E-A1 (Phase 0, extended):** the history-conditioned baseline is now a
  fourth arm in `experiments/phase0.py` (`history`), and
  `Phase0Result.dynamic_beats_history` reports the direction. **CPU status:**
  the *untrained* EMA heuristic does **not** beat the memoryless control on
  the current near-instantaneous mechanisms — expected, and exactly why the
  *trained* injector must earn it (see `milestone_a.md` §3). Remaining: (a) a
  **hysteretic** synthetic mechanism a memoryless model provably cannot
  reconstruct, (b) the *trained* neural injector beating `history` on it at
  **equal capacity** (needs Milestone B). If trained-D still ties B, the
  structured latent does not earn its keep — surface honestly (reshapes paper).

---

## Milestone B — Make training real (P0)

### Code
- [ ] **`src/gps/train/sft.py`:** implement the tensor loop (currently a
  documented stub with a no-op guard). Teacher-force the latent along each
  trajectory; loss = move-NLL + λ·timing-NLL. Needs the neural injector
  (Milestone A) + a differentiable backbone.
- [ ] **`src/gps/policy/sglang_backbone.py`:** finish `_engine()` (launch
  sglang Engine, logprobs on) and `predict()` (constrained decoding over
  `dp.legal_actions`, read move logprobs → `MoveDistribution`; derive timing).
  `build_prompt` already done + tested.
- [ ] **`src/gps/policy/sglang_backbone.py` HIDDEN path:** soft-prompt /
  prefix-embedding injection for `InjectionKind.HIDDEN` (`enable_hidden=True`).
- [ ] **`src/gps/policy/board_native.py`:** build/load the Maia/KataGo-style
  trunk + move head + timing head; latent conditioning via FiLM/concat
  (`accepts = (HIDDEN,)`). This is the backbone for the "latent helps
  independent of LLM" control.
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
- [ ] **`src/gps/data/lichess.py`:** parse Lichess PGN + `[%clk]` (python-chess
  `GameNode.clock()`), emit `train.base.Trajectory` per player. Exclude bot
  accounts. Focus blitz/rapid for within-session density.
- [ ] **`src/gps/games/chess.py`:** concrete `Game` — board encoding (FEN for
  LLM backbone, tensor for board-native), `legal_moves` (UCI), `apply_move`.
- [ ] **`src/gps/games/oracles/stockfish.py`:** `EngineOracle` wrapping
  Stockfish (or the published Lichess eval set) → per-move centipawn loss.
  **Record depth** (centipawn-loss is settings-dependent — must report it).
- [ ] **Player selection util:** filter players by volume AND multi-game
  session count (need within-session dynamics).
- [ ] **`src/gps/data/sessions.py`:** already done — but **sweep the gap
  threshold as an ablation** (it is an unlabeled construct, see design.md §6).

### Experiments
- [ ] **E-C1 (RQ1):** dynamic vs. static individual (B8 vs B2) on next-move
  NLL/calibration + timing, chess, **strict temporal split**.
- [ ] **E-C2 (RQ1, the dangerous one):** B8 vs. history-conditioned-no-latent
  at equal inputs/capacity.
- [ ] **E-C3 (RQ3, the decisive test):** train on a player's earlier sessions,
  predict their **later** sessions. This is what separates real dynamics from
  memorized habit. Report all metrics on the held-out later sessions.
- [ ] **E-C4 (RQ2):** state-recovery probes (presence) **and** a causal
  intervention check (clamp a latent dim, measure prediction change) — the
  probe-presence-vs-use gap from design.md §6.
- [ ] **E-C5 (RQ4):** B8 vs. LLM persona prompt (B5/B6) — move + timing
  distribution match with the engine-graded yardstick.
- [ ] **E-C6 (timing):** per-individual think-time NLL + per-player Spearman
  vs. Allie-style aggregate (B4) **and vs. ChessMimic's per-move clock model
  (Pearson r=0.41 — the concrete number to beat; B9)**. Differentiator: our
  timing is conditioned on the *evolving* state and is *per-individual*, not
  Elo-band-aggregate.

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
- [ ] **OR (stronger generality) non-game oracle domain.** Port the
  `DecisionPoint` / `EngineReference` interface to **knowledge tracing**
  (EdNet/ASSISTments: `state`=student+item context, `legal_actions`=response
  options, `EngineReference`=IRT difficulty/correct-prob oracle, `TimeSignal`
  =response time, `recent_outcomes`=prior items) or **Codeforces** (verdict +
  problem-rating oracle). This tests the "game-agnostic core" claim for real —
  the cheapest first step is just writing the interface adapter and seeing how
  much of `gps/interface.py` survives unchanged.

### Experiments
- [ ] **E-D1 (RQ5):** the *same* framework, only the encoder + oracle swapped,
  reproduces the chess pattern in a second domain (Go, or — preferred for
  cross-modal generality — a non-game oracle domain). Establishes the
  contribution is the dynamic-state mechanism, not a chess artifact.
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
- [ ] Both channels already exist (`InjectionKind.VERBAL/HIDDEN`,
  `StructuredInjector(kind=...)`). Ensure the **neural injector** supports
  both and that the **same learned state** can be rendered either way (so the
  comparison is channel-only, not state-only).

### Experiments
- [ ] **E-E1 (RQ6):** same task, same backbone family — verbal vs. hidden
  injection — on move + timing fidelity. Which channel wins, and where?
  (Hypothesis: hidden richer but open-weight-only; verbal portable to closed
  APIs but lossy. The closed-API verbal result is also the RQ4 ceiling.)

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
- [ ] **`src/gps/eval/distributional.py`:** population-level metrics —
  KL/JS/Wasserstein on behavioral stats (move-quality dist, time-allocation,
  blunder-by-phase) + **precision/recall for generative models** (coverage of
  real diversity vs. plausibility) + an **off-manifold / realism rate**.

### Experiments
- [ ] **E-F1:** generated population vs. real held-out population on the
  distributional metrics. Explicitly note this is **distributional, not
  pointwise** ground truth (invented players have none).
- [ ] **E-F2:** does the generated population **recover real heterogeneity**
  and beat the "positive average person" baseline (the field's named-but-
  unsolved problem)? Strong result → promote to a second pillar; weak →
  keep as a downstream demo.

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
