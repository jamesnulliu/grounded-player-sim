# grounded-player-sim (`gps`)

Personalized human game-play simulation built around an **LLM policy
augmented with a trainable, per-individual dynamic latent state**.

**Thesis.** An LLM conditioned on a static verbal persona/emotion label is
not enough to reproduce how a specific person plays *right now*. You need a
learned latent state `z_t` that evolves over the player's own action+timing
trajectory, is injected into the policy, and is validated against the
player's *future* behavior. Designed around games with an engine-graded
decision interface (chess / Go via Stockfish / KataGo); **demonstrated on real
chess** and, for cross-domain generality, on a real non-game oracle domain
(knowledge tracing). Go was attempted on real OGS games but shows **no robust
timing effect under a board-size control** (an apparent effect was a confound —
honest negative), so it remains future work.

**Positioning.** The contribution is the *conjunction* — per-individual +
temporally-evolving + a behavioral state (tilt/fatigue/time-pressure) that
drives moves & timing + validated on the person's *future* games + across
domains (delivered on real **chess** and a real non-game oracle domain,
**knowledge tracing**; Go was attempted on real OGS games but has no robust
effect under controls — future work).
No single axis is claimed as novel: "evolving latent in an LLM,"
"natural-language latent," and "future temporal-split validation" are each
already owned by a 2026 competitor (LATTE / HumanLM). See `documents/design.md`
§8 for the head-to-head differentiation vs. Allie, HumanLM, and LATTE.

See **`documents/paper_draft.md`** for the landed-results synthesis (abstract +
contributions + headline table), `documents/results_ec.md` for the detailed
results + exact reproduction, `documents/proposal_v2.md` for the full research
proposal, `documents/design.md` for how the code maps onto it (and the key
decisions + prior-art positioning), `documents/training.md` for the GPU/data
wiring, `documents/milestone_a.md` for the make-or-break "is the latent just
history-conditioning?" runbook, and **`TODO.md`** for the work plan.

**Status (landed).** Every research question and both milestones have results
on **real** data — real Lichess chess **and** real student knowledge tracing
(ASSISTments 2009) — plus synthetic controls. The evolving latent
beats a static-individual style (E-C1) and an equal-capacity **memoryless**
control (E-C2/E-C3, future-sessions split, capacity sweep). The **headline is
think-time**: at scale (5 seeds × 5 cohorts × 2 backbones on 2×A100) the
per-individual evolving latent wins on think-time in **all 8 clocked conditions
across a 6-year era span (2017–2023), P=1.00**, and **adds value over a near-SOTA
Elo+clock+position-complexity baseline** (Spearman ≈ ChessMimic's 0.41). An
interpretability story with a clean **channel asymmetry**: the latent **encodes +
causally uses** the hidden state (E-C4, probe R²=0.93 + clamp dose-response), and
its think-time edge **concentrates** under time pressure (2–8×) and for weaker
players (≈3×) — while the **move** edge is a flat near-null (state is legible in
*when* you act, not *what* you play). Plus **generality** to a non-game domain
(E-D1/RQ5) — confirmed on **real students across 8 datasets, multiple platforms,
and 3 subject domains** (ASSISTments 2009/2012/2015/2017, KDD-Cup Algebra +
Bridge-to-Algebra, Spanish, and Statics; D beats the memoryless twin, significant
every seed) — a **unifying scaling law** (the latent's edge **scales with
population heterogeneity**, Pearson 0.89 across the 8 datasets, tying the KT and
chess results together),
the **hidden > verbal** channel (E-E1/RQ6), and **population heterogeneity
recovery + generation** beating the "positive average person" (E-F1/E-F2, also on
real KT: Wasserstein 2× better than average-person). And an **actual LLM policy**
(Qwen3): frozen verbal/persona-prompt injection is a *negative control* (≈
irrelevant filler) and RL's sparse reward can't resolve the effect, but a dense
**behavior-cloning SFT probe reproduces the board-native asymmetry** — state
helps **think-time ≫ moves**. The **think-time benefit is robust and
adaptation-invariant**: Δ ≈ −0.013 across 0.6B→8B *and* across LoRA vs **full
fine-tuning** (full-param −0.0110/−0.0128 at 4B/8B ≈ LoRA −0.0114/−0.0136, 3 seeds
each), run here via a single-GPU 8-bit paged optimizer (no FSDP). The move
channel's apparent collapse to a clean null at ≥4B is a **LoRA-capacity artifact** —
full-param fine-tuning recovers a stable move benefit at *both* 4B (−0.0072) and 8B
(−0.0083, all seeds), so the timing≫move asymmetry is robust but
*graded* (~1.5×), not a clean null in a full-capacity LLM. Honest
caveat: the headline D-vs-B results still use small from-scratch backbones (the
LLM is a *probe*, not the headline); larger/instruction-tuned LLM policies remain
future work. CLI: `gps ingest`,
`gps train-ec`, `gps phase0`, `gps kt` (RQ5/F on knowledge tracing, synthetic
or real via `--data`), `gps info`; at-scale sweep in `scripts/`.

## Architecture at a glance

```
DecisionPoint  (shared chess/Go schema)
      │
      ▼
LatentStateInjector ──renders──▶ Injection (verbal text | hidden vector)
   (the contribution)                   │
      ▲ update(z_{t-1} → z_t)           ▼
      └──────────────────────── PolicyBackbone ──▶ Prediction (moves + timing)
                                   (swappable)
```

* **`PolicyBackbone`** is swappable — open-weight LLM via **sglang**
  (Qwen3-8B), closed-source LLM via **API**, or a board-native (Maia/KataGo
  style) baseline. Keeping the backbone a controlled variable lets us prove
  "does the dynamic latent help?" independent of backbone, which is the
  defense against the "an LLM is a weak move predictor vs. Maia" objection.
* **`LatentStateInjector`** has two interchangeable realizations behind one
  interface — `verbal` (memory in words, works with any backbone) and
  `hidden` (soft-prompt / prefix vectors, open-weight backbones only).
* **Training** (`gps.train`): **SFT** (default — imitate observed
  moves/timing by max-likelihood) and **slime RL** (behavior-matching
  reward, or fine-tuning the agent to fit the latent).

## What runs *today*, with no GPU

The shared interface, the Phase-0 synthetic players (known dynamics), the
mock backbone, the latent injectors, and the full eval harness are
**pure-stdlib** and run on CPU. Everything GPU/network/engine-bound
(sglang, slime, torch, OpenAI/Anthropic, python-chess, Stockfish, KataGo)
is behind lazy imports and optional extras, so the package imports and the
Phase-0 experiment runs on a laptop.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # base (numpy only)
gps info                    # show which backends are importable
gps phase0 --games 30       # run the CPU Phase-0 experiment
```

Example Phase-0 output (three arms over the same synthetic trajectory):

```
[tilt] static NLL=1.9698 | heuristic NLL=1.4819 (helps: True) |
       oracle NLL=1.5044 (helps: True) | recovery R^2=0.930 |
       mechanism fired 60% of plies
```

* `static` — no latent (foil). `oracle` — knows the true injected state
  (proves dynamics carry signal; **P0.2**). `heuristic` — the untrained
  structured injector (reported, *not* asserted to win — that gain is what
  the trained injector must earn). `recovery R^2` — a linear probe
  recovering the injected mechanism from the latent (**P0.1 / RQ2**).

## Standing rules for runs

Enforced in code (`documents/training.md` has the detail):

* **Results** — every run writes `runs/<experiment>/<run_id>/` with
  `run.json` / `config.json` / `env.json` / `metrics.json` /
  `metrics.jsonl` / `artifacts/` (`gps.results`).
* **Tracking** — every *training* run must log to Weights & Biases;
  `WANDB_API_KEY` is read from the environment and a missing key aborts the
  run (`gps.tracking`). No opt-out.
* **Backends** — LLM inference uses **sglang**, LLM training uses **slime**
  (+ sglang rollouts); the board-native CNN control is plain-torch exempt
  (`gps.backends`).

## GPU / API install

```bash
pip install -e '.[serve]'   # sglang + transformers + torch (local LLM)
pip install -e '.[api]'     # openai + anthropic (closed-source baseline)
pip install -e '.[train]'   # torch (+ install slime from source; see docs)
pip install -e '.[chess]'   # python-chess (Lichess PGN + clocks)
pip install -e '.[dev]'     # pytest + ruff
```

## Development

```bash
pip install -e '.[dev]'
ruff format . && ruff check .   # line-length 79
pytest                          # 34 CPU tests, no GPU needed
```

## Layout

| Path | Purpose |
|------|---------|
| `src/gps/interface.py` | shared `DecisionPoint` schema (chess + Go) |
| `src/gps/prediction.py` | move + timing prediction objects |
| `src/gps/latent/` | the dynamic latent injector (verbal + hidden) |
| `src/gps/policy/` | swappable backbones (sglang / API / board-native / mock) |
| `src/gps/simulator.py` | composes injector + backbone over a trajectory |
| `src/gps/synthetic/` | Phase-0 players with known dynamics + toy game |
| `src/gps/eval/` | metrics, state-recovery probes, temporal splits |
| `src/gps/train/` | SFT + slime-RL trainers (enforce the standing rules) |
| `src/gps/data/` | Lichess/SGF ingestion + session segmentation |
| `src/gps/results.py` | result-store rule: `runs/<exp>/<run_id>/` + JSON format |
| `src/gps/tracking.py` | mandatory W&B (reads `WANDB_API_KEY`, errors if unset) |
| `src/gps/backends.py` | backend policy: sglang (LLM inference) / slime (LLM train) |
| `src/gps/experiments/` | runnable experiments (Phase 0, E-A1, E-C, KT) |
| `documents/` | proposal, design notes, training notes |
| `tests/` | CPU test suite |
