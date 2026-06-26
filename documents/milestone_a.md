# Milestone A — "Is the dynamic latent just an expressive history-conditioned policy?"

This is the **#1 desk-reject objection** for the whole project (design.md §6,
TODO.md Milestone A). If a no-latent policy fed the *same* engineered history
features matches our evolving latent at equal capacity, the contribution
evaporates. This doc is the runbook for the experiment that settles it.

The experiment is cheap, falsifiable, and gates everything downstream:
**run it before investing in the full chess pipeline.** A negative result
here does not kill the project — it reshapes the paper toward "engineered
history features suffice" — but you must know which world you are in first.

---

## 1. The claim under test

> An **evolving, per-individual latent state** `z_t` predicts a player's
> moves and timing better than a **memoryless policy fed the identical
> history features**, at **equal inputs and equal capacity**.

The contrast is exactly one bit: presence vs. absence of an *evolving* latent
over identical inputs. Everything else — backbone, features, training data,
eval — is held constant.

Two distinct controls implement "no evolving latent," at two cost tiers:

| Arm | What it removes | Inputs | Capacity | Where it runs |
|-----|-----------------|--------|----------|---------------|
| **`history` (CPU)** | memory/recurrence | same `history_features` | parameter-free | laptop, today |
| **history-conditioned backbone (GPU)** | the latent's inductive bias | same `history_features` | matched to the latent arm | training host |

The CPU arm proves the *mechanism* matters in a world with known ground
truth; the GPU arm proves it survives at *matched capacity* on real chess.
You need both — the CPU result alone is rejectable as "your heuristic was
just weak," and the GPU result alone has no ground-truth anchor.

---

## 2. The 2×2 that makes "equal inputs / equal capacity" airtight

The framework already factors a model into `Simulator(injector, backbone)`.
That gives a clean 2×2 — vary *evolving latent* against *history features*
independently:

```
                      │ no history features        │ same history features
──────────────────────┼────────────────────────────┼───────────────────────────
  no evolving latent   │ A. static / population      │ B. history-conditioned
                      │    Simulator(None, BB)      │    (THE CONTROL)
                      │                             │    Simulator(None, HCBackbone)
                      │                             │    or Simulator(HistoryInj, BB)
──────────────────────┼────────────────────────────┼───────────────────────────
  evolving latent      │ (ill-posed: a latent needs  │ D. PROPOSED
                      │  history to evolve from)    │    Simulator(NeuralInj, BB)
```

- **A vs. D** answers "does modeling the session help at all?" (the easy win).
- **B vs. D** is the hard one — the desk-reject defense. Same features in
  both; the *only* difference is whether they are accumulated into an
  evolving state (D) or consumed instantaneously (B).
- **The headline number is D − B**, reported with capacity (parameter counts)
  for both arms printed side by side.

"Equal inputs" is enforced *in code*, not promised in prose:
`gps.latent.structured.history_features(dp)` is the single source of truth,
and **both** the structured/neural injector and the history-conditioned
control read it. There is no path for one arm to see a feature the other
cannot. (`tests/test_phase0.py::test_phase0_history_uses_same_features_as_structured`
guards this.)

---

## 3. What is implemented now (CPU, runnable today)

```
src/gps/latent/structured.py
  history_features(dp)            # single source of truth for "what history is visible"
  StructuredInjector              # EMA -> evolving z_t (the latent arm, untrained reference)
  HistoryConditionedInjector      # NEW: same features, memoryless (CPU control, arm B)
  OracleInjector                  # reads true degradation (upper bound, P0.2)

src/gps/policy/history_conditioned.py
  HistoryConditionedBackbone      # NEW: no-latent head fed raw history features
                                  #      (capacity-matched GPU control; predict() is a stub)

src/gps/experiments/phase0.py
  run_phase0(...)                 # NOW 4 arms: static | history | heuristic | oracle
  Phase0Result.dynamic_beats_history   # the E-A1 direction, reported not asserted
```

### Run it

```bash
# CPU, no GPU, ~1s. (This repo targets Python >=3.10; if your only
# interpreter is 3.9, run against src directly — the type hints are lazy:
#   PYTHONPATH=src python -m gps.cli phase0 --games 30
# otherwise, after `pip install -e .`:)
gps phase0 --games 30                  # hidden-vector injection channel
gps phase0 --games 30 --injection verbal
gps phase0 --player tilt --games 60 --seed 1   # one mechanism, longer session
```

### Reading the output

```
[tilt] static NLL=1.9698 | history NLL=1.4723 | heuristic NLL=1.4819 (>history: False) | oracle NLL=1.5044 (helps: True) | recovery R^2=0.930 | mechanism fired 60% of plies
```

- `static` — no latent, the floor.
- `history` — memoryless control (arm B). Sees the instantaneous features.
- `heuristic` — the untrained EMA structured injector (arm D's *reference*,
  not its trained form).
- `(>history: ...)` — **the E-A1 direction**: does the evolving heuristic beat
  the memoryless control?
- `oracle` — reads true degradation; upper-bounds achievable gain (P0.2).

**Current CPU result, stated honestly:** `>history` is **False** on all three
synthetic mechanisms — the untrained EMA heuristic does *not* beat the
memoryless control. This is expected and is *why Milestone A exists*:

1. The Phase-0 mechanisms (tilt/time-pressure/fatigue) are near-step-functions
   of the instantaneous features, so a memoryless reader captures most of the
   signal and EMA smoothing only *lags* the true state.
2. The heuristic injector is **parameter-free** — its EMA `alpha`, decay
   windows, and dimension weights are hand-set, not fit. The "accumulate
   history" hypothesis is not that *any* accumulation wins; it is that a
   *trained* `f_phi` learns the right time constants.

So the CPU experiment does its job: it shows the evolving-latent claim is
**not free** — it must be *earned by training* (E-B1) and *on data with
genuine cross-step dynamics* (real chess sessions, E-C2), where the state at
step *t* depends on history a memoryless model cannot reconstruct from the
current `dp` alone. If the *trained* neural injector still cannot beat the
memoryless control on real chess, that is the reshape-the-paper finding.

---

## 4. What must be built to finish Milestone A (GPU)

Ordered; each step is small and unblocks the next.

1. **Neural injector** — `src/gps/latent/neural.py` (TODO Milestone A). A
   recurrent/state-space `LatentStateInjector` with real `parameters()` so
   `SFTTrainer` stops hitting its no-op guard. Must honor both `VERBAL` and
   `HIDDEN` `produces`. This is arm D's trainable form.
2. **SFT loop** — `src/gps/train/sft.py` (TODO Milestone B). The tensor loop
   is sketched in-source; bind it to the neural injector + a differentiable
   backbone. Loss = move-NLL + λ·timing-NLL, teacher-forcing `z_t` along each
   trajectory.
3. **`HistoryConditionedBackbone.predict`** — `src/gps/policy/history_conditioned.py`.
   The feature contract (`feature_vector`, `param_report`) and the wiring are
   done and CPU-tested; finish the torch forward (trunk + feature-fusion MLP +
   move/timing heads). **Size the fusion MLP to match the latent injector's
   added parameter budget** and assert it via `param_report()`.
4. **E-A1 (extended Phase 0, GPU):** smoke-train the neural injector on
   Phase-0 synthetic data; confirm trained-D > B where the memoryless model
   provably cannot reconstruct the state (use a *delayed* / *hysteretic*
   synthetic mechanism — see §6 — not just the near-instantaneous ones).
5. **E-C2 (the real test, chess):** trained-D vs. history-conditioned-backbone
   on Lichess, strict temporal split, equal capacity. This is the number that
   goes in the paper.

---

## 5. The decision rule (write this down before you look at results)

Pre-registering the rule prevents post-hoc rationalization.

- **D − B > 0, significant, concentrated in high-dynamics moments**
  (time scrambles, post-loss games): the latent earns its keep. Proceed to
  the full chess headline (E-C1/C3), then Go / verbal-vs-hidden.
- **D − B ≈ 0** (within noise): the structured latent does *not* beat
  engineered history features. **Reshape the paper**: the contribution
  becomes the *benchmark + the finding that history features suffice* — still
  publishable, but a different abstract. Do **not** proceed to Go/population.
- **D − B > 0 but tiny and diffuse**: bounded phenomenon. Report it as such;
  consider a workshop venue rather than a main-track gamble.

Significance: bootstrap over players (not over moves — moves within a player
are correlated). Report per-player D − B distribution, not just the pooled
mean, because the whole thesis is *per-individual* heterogeneity.

---

## 6. Methodological cautions (do not skip)

- **Make the synthetic mechanism genuinely sequential for E-A1.** The current
  tilt/fatigue mechanisms are mostly reconstructable from the current `dp`
  (loss flag, session position), so a memoryless model is a strong control by
  construction. Add a mechanism with **hysteresis / a hidden carry-over**
  (e.g., tilt whose depth depends on *how many* of the last K games were
  losses *and* decays at a rate the model must integrate over time) so that
  no instantaneous feature set is sufficient. That is where an evolving latent
  *should* win, and if it doesn't there, it never will.
- **Capacity, not just inputs.** A win for D that disappears when you size B's
  MLP up is a capacity artifact, not a latent contribution. Always print both
  parameter counts (`param_report()`).
- **The history features themselves are a design choice.** If you hand B a
  *rich* feature set (last-K results, clock trace, engine-swing history), B
  gets stronger and the test gets harder — which is the honest version. Don't
  cripple B with a thin feature set to manufacture a D win.
- **Probe presence ≠ use.** A high `recovery R^2` (P0.1) shows the latent
  *carries* the mechanism, not that the policy *uses* it. The causal version
  (clamp a latent dim, measure prediction change) is E-C4; don't conflate.

---

## 7. Resources required

### Runs today — zero cost

| Resource | Need |
|---|---|
| Hardware | any laptop CPU; no GPU |
| Python | repo targets ≥3.10 (`pyproject.toml`). On a 3.9-only box use `PYTHONPATH=src` (type hints are lazy via `from __future__ import annotations`) |
| Deps | stdlib only for `gps phase0`; `pip install -e '.[dev]'` for `pytest`/`ruff` |
| Wall-clock | `gps phase0` ≈ 1 s; full `pytest` ≈ a few s |
| Data | none (synthetic) |
| API keys | none |

### Finishing Milestone A — GPU

The CPU result is suggestive; the *publishable* Milestone A number (E-C2)
needs the trained arms on real chess. Estimated resources:

| Stage | GPU | Notes |
|---|---|---|
| **E-A1** neural injector smoke-train on Phase-0 synthetic | **1× consumer GPU** (e.g. 1×24 GB, RTX 4090 / A10 / L4) | Tiny model (recurrent injector + small board/feature trunk). Minutes–hours. No LLM needed. |
| **E-C2** trained-D vs. history-backbone on Lichess, board-native trunk | **1× 24–48 GB GPU** (A10/A100-40GB/L40S); 1 GPU sufficient | Maia-scale CNN (≈ tens of M params) + injector. Hours–day per full run incl. temporal-split eval. |
| **E-C2 with an LLM backbone** (if you keep the LLM-as-policy branch, design.md §2) | **1× 80 GB (A100/H100)** to serve Qwen3-8B via sglang + train the injector; **2× 80 GB** if you also fine-tune the backbone | LLM logprob inference over legal moves dominates cost. Constrained decoding. This is the expensive branch — the board-native trunk is the cheaper, stronger-move-NLL control and is recommended for the *Milestone A number specifically*. |

**Which backbone for Milestone A?** Use the **board-native trunk** as the
capacity-matched control even if your headline model is the LLM. The Maia
objection ("an LLM is a weak move predictor") is irrelevant *to the latent
question* — Milestone A asks only whether the evolving latent beats memoryless
at equal capacity, and the board-native trunk makes that comparison cheap,
strong, and reviewer-proof. Run the LLM version later (it doubles as the
RQ4/RQ6 material), not as the gating Milestone-A experiment.

### Data for E-C2 (chess)

| Item | Source | Cost / notes |
|---|---|---|
| Games + per-move clocks | Lichess open database (CC0) | Free download. One month of blitz/rapid is plenty to start; filter to high-volume, multi-game-session players. Exclude bot accounts. |
| Engine oracle | Stockfish per-position eval, **or** the published Lichess Stockfish eval set | Stockfish self-run is CPU-heavy (per-position depth-N eval over millions of plies) — budget a multi-core CPU box or reuse the Lichess eval set to skip it. **Record depth** (centipawn-loss is settings-dependent). |
| Storage | — | ~tens of GB for a working chess subset + cached engine evals; ~hundreds of GB if you cache evals for a large player pool. |
| Compute for eval | CPU | Engine eval + metrics are CPU-bound; can run alongside GPU training. |

**No external API keys are required for Milestone A.** OpenAI/Anthropic keys
are only for the RQ4 persona-prompting baselines (B5/B6), which are *not* part
of Milestone A.

### Rough end-to-end estimate to a publishable Milestone A number

- **People-time:** neural injector + SFT loop + history-backbone forward ≈ the
  bulk of the work; the data pipeline (Lichess parse + Stockfish/eval-set
  wiring) is the other half. Both are TODO Milestones A–C.
- **GPU:** 1 GPU (24–48 GB) is enough for the board-native version end to end.
  Add an 80 GB GPU only if you run the LLM-backbone variant.
- **$ if renting cloud:** a single mid-range GPU for a few days of iteration —
  small. The LLM branch is the only thing that escalates cost.

---

## 8. Checklist

- [x] `HistoryConditionedInjector` — CPU memoryless control (arm B), shares
  `history_features` + renderer with the structured injector.
- [x] `HistoryConditionedBackbone` — capacity-matched GPU control; feature
  contract + `param_report` done and CPU-tested, `predict` is a documented
  stub.
- [x] Phase-0 wired to 4 arms; `dynamic_beats_history` reported (not asserted);
  CLI + tests updated; `ruff` + `pytest` green.
- [ ] Neural injector (`gps/latent/neural.py`) with real `parameters()`.
- [ ] SFT tensor loop (`gps/train/sft.py`).
- [ ] `HistoryConditionedBackbone.predict` torch forward + capacity match.
- [ ] A *hysteretic* synthetic mechanism for a fair E-A1.
- [ ] E-A1 (trained D > B on Phase-0, GPU).
- [ ] E-C2 (trained D vs. B on Lichess, equal capacity, temporal split).
- [ ] Pre-registered decision rule (§5) recorded before looking at E-C2.
