# grounded-player-sim (`gps`)

Personalized human game-play simulation built around an **LLM policy
augmented with a trainable, per-individual dynamic latent state**.

**Thesis — model *when* a person acts, not only *what* they choose.** A learned
latent `z_t` that evolves over a player's own action+timing trajectory predicts
that specific person's **think-time** far better than their **move choice** —
robustly, concentrating under time pressure and for weaker players, and scaling
with population heterogeneity. Validated against the player's *future* behavior
on a strict temporal split, on real Lichess chess and (for cross-domain
generality) real knowledge tracing. The state's value lives in a *trained
hidden* latent, not a verbal persona prompt (RQ6) — and it adds think-time value
even over a **released SOTA think-time model** (Allie, ICLR'25; Milestone G4).

**Three things we lead with** (each axis has prior art — dynamic emotional chess,
per-individual style, timing>choice in psychometrics; our contribution is the
*controlled synthesis on real humans*, not standalone-axis novelty. Full cited
positioning in `documents/related_work.md`; superseded framing in `design.md §8`):
1. **The when-not-what finding.** Evolving state is legible in *timing*,
   near-null in *move choice* — robust across a 6-year era span and reproduced
   in a non-game domain. That timing reveals latent state better than choice is
   old in response-time psychometrics; what is new is its form here — an
   *evolving within-session behavioral state* against a *per-decision oracle* on
   a *strict future split*, with move choice a near-null.
2. **The equal-capacity evolving-vs-memoryless control on a strict future
   split.** Isolates *dynamics* from *habit* and raw *individualization* — the
   #1 reviewer objection ("isn't this just history-conditioning?"), settled.
   Evolving-vs-memoryless is a routine seqrec ablation; the differentiator is
   the *equal-capacity, same-input* form on a per-decision **oracle** domain
   with a real future split — which no behavior-simulation competitor runs.
3. **The hidden-vs-verbal channel ordering is backbone-dependent** (measured,
   not assumed). With *no* language prior (board-native, RQ6) the trained
   *hidden* latent beats the verbal note (−0.069/−0.117). But *inside an actual
   LLM* (G3, Qwen3-1.7B, 3 seeds) the ordering **flips**: the verbal note wins
   (hidden−verbal ≈ +0.003), because the LLM reads "tilt"/"time pressure"
   *semantically* — while injected state still **helps** think-time (verbal−none
   ≈ −0.005, all 3 seeds). So we show *when* the verbal channel today's LLM
   simulators (HumanLM) use is right, and when the hidden vector is. `results/g3_llm.txt`.

See **`documents/paper_draft.md`** for the landed-results synthesis (abstract +
contributions + headline table), `documents/results_ec.md` for the detailed
results + exact reproduction, `documents/proposal_v2.md` for the full research
proposal, `documents/design.md` for how the code maps onto it (and the key
decisions + prior-art positioning), `documents/training.md` for the GPU/data
wiring, `documents/milestone_a.md` for the make-or-break "is the latent just
history-conditioning?" runbook, `documents/milestone_g.md` for the **LLM-agent
deployment** result + why a strong Maia backbone was *deprioritized*, and
**`TODO.md`** for the work plan.

**Status.** Strong, landed results are **board-native** (small from-scratch
head). A Maia-2/3 backbone was **considered and deprioritized** (a strong trunk
*absorbs* the move signal, and the timing headline is backbone-independent by
construction — the timing head reads only the latent — so a strong backbone
can't change it; `TODO.md` Milestone G). The headline is **timing**:

- **Timing (headline).** The evolving latent beats an equal-capacity memoryless
  twin on think-time in **all 8 clocked conditions across 2017–2023** (5 seeds ×
  2 backbones, 2×A100), P=1.00, and **adds value over a near-SOTA
  Elo+clock+complexity baseline** (Spearman ≈ ChessMimic's 0.41). It also beats a
  static-individual style (E-C1) and survives a capacity sweep and a
  future-*sessions* split (E-C2/E-C3).
- **Move choice is a near-null** on real data (flat by post-loss / time-pressure)
  — the when-not-what asymmetry.
- **Mechanism.** On a synthetic player with a *known* hidden state the latent
  encodes it (probe R²=0.93 vs 0.65) and causally uses it (clamp dose-response);
  the real timing edge **concentrates** under time pressure (2–8×) and for weaker
  players (≈3×), and **scales with population heterogeneity** (Pearson 0.89
  across 8 real KT datasets) — one law across populations, players, and contexts.
- **Generality.** Reproduced in knowledge tracing on **real students across 8
  datasets / multiple platforms / 3 subjects**; population-heterogeneity recovery
  and generation beat the "positive average person" (Wasserstein 2× better). Go:
  **honest negative** (no robust effect under a board-size control) — future work.

**LLM arm (an honest secondary result — RESOLVED, not the headline).** Qwen3 via
sglang is implemented and runs. Frozen verbal injection is a *negative control*;
RL (slime GRPO) learns the task but its sparse reward can't resolve the state
effect; a dense **SFT probe reproduces the board-native asymmetry** — state helps
think-time (Δ ≈ −0.011, robust across 0.6B→8B and LoRA→full fine-tuning) more than
moves. These effects are **small**. Milestone G settled the LLM's role: *hidden
does **not** beat verbal inside the LLM* (G3 — the LLM reads the note
semantically), so the channel ordering is **backbone-dependent**, and the LLM is a
deployment/secondary result — board-native timing stays the headline. CLI: `gps
ingest`, `gps train-ec`, `gps phase0`, `gps kt` (RQ5/F, synthetic or real via
`--data`), `gps info`; at-scale sweep in `scripts/`.

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
