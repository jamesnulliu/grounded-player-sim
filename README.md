# grounded-player-sim (`gps`)

Grounded evidence that **simulated humans need an evolving state, not just a
static profile** — built as a per-individual dynamic latent injected into a
swappable (board-native or LLM) policy.

**Thesis — user policy = static attribute + evolving state.** Today's user
simulators freeze the person (a persona prompt, a rating, a fixed
embedding), so the simulated policy is *attribute + reasoning* and
within-person drift (tilt, fatigue, clock panic — the emotion-like state) is
unrepresentable by construction. We test the opposite decomposition where it
is cleanly measurable — chess and knowledge tracing, discrete actions,
logged per-decision timing, strict per-player *future* splits — and find the
dynamic term is real, separately measurable, and concentrated exactly where
an emotion-like state should matter. (We never claim to measure emotion; no
emotion labels exist. The claim is the decomposition.)

**What we lead with** (the *premise* — users drift, static personas can't —
is now common in 2025–26 LLM-agent papers; the *controlled quantitative
demonstration* is ours. Full cited positioning in
`documents/related_work.md`):
1. **The dynamic term survives every control.** Beats an equal-capacity
   memoryless twin in all 8 clocked era×backbone conditions (2017–2023);
   still adds think-time value over **Allie** (released ICLR'25 think-time
   model, Spearman 0.62–0.65) on 3/3 cohorts and over Allie + a **static
   per-player embedding** on 2/3 (pooled −0.0126). A training-free
   **structured-memory arm** (running stats of raw history, linear readout)
   beats the static profile on **3/3** cohorts at 2–3× that margin
   (pooled −0.0276) — dynamics confirmed under a second instrument with
   zero trained parameters. The same memory arm matches/beats the
   4-feature learned GRU, and an input-matched GRU over the same 15
   statistics ties the linear readout while still beating static → the
   claim is *carrying dynamic state* and the information it carries, not
   the specific mechanism. The KT memory arm transfers the verdict:
   beats the twin 7/8 datasets (same profile as the learned latent), ties
   D on 5, repairs the ASSISTments-15 training reversal
   (`results/memory_baseline.txt`, `results/memory_gru_arm.txt`,
   `results/kt_memory_arm.txt`).
2. **When, not what.** The state is legible in *timing* (think-time),
   near-null in *move choice* (probe to deviation-from-Maia-2 R² = 0.009);
   the edge concentrates 2.7–3.6× under time pressure (variance-controlled)
   and ≈3× for the weakest players. Real education response *times* are
   honest negatives (ASSISTments, EdNet); real KT *responses* replicate
   (22/24 seed cells over 8 datasets).
3. **Only a state model generates populations.** Sampling the latent prior
   recovers a real population's accuracy distribution (W1 2× closer than
   average-person; generated recall 1.00 vs 0.00) — an operation no static
   persona library and no memory store defines.
4. **The hidden-vs-verbal channel ordering is backbone-dependent.**
   Board-native: hidden beats verbal (−0.069/−0.117). Inside Qwen3 the
   ordering flips (the LLM reads "tilt" semantically) while injected state
   still helps think-time on all seeds. `results/g3_llm.txt`.
5. **The persona ladder inside the LLM (G5).** none / frozen fitted
   persona / updating text scorecard / soft vector, one loop, 99 players:
   fitted person-info is worth ≈ −0.010 (both text arms significant);
   updating-vs-frozen is a **tie** on the one-month low-drift horizon
   (contrast KT, where the frozen fitted profile fails 8/8 — the freezing
   law: *frozen descriptions fail in proportion to how fast the person
   drifts*); text beats vector, replicating G3. Practical rule: fit the
   persona from data, keep it updating when the person drifts, deliver as
   text. `results/g5_persona_ladder.txt`, `results/kt_static_arm.txt`.

See **`documents/paper_draft.md`** for the landed-results synthesis (abstract +
contributions + headline table), `documents/results_ec.md` for the detailed
results + exact reproduction, `documents/proposal_v2.md` for the full research
proposal, `documents/design.md` for how the code maps onto it (and the key
decisions + prior-art positioning), `documents/training.md` for the GPU/data
wiring, `documents/milestone_a.md` for the make-or-break "is the latent just
history-conditioning?" runbook, `documents/milestone_g.md` for the **LLM-agent
deployment** result + why a strong Maia backbone was *deprioritized*, and
**`TODO.md`** for the status index. The active verification and writing gates
are tracked in **`documents/paper_readiness_plan.md`**.

**Status.** Strong, landed results are **board-native** (small from-scratch
head). A Maia-2/3 backbone was **considered and deprioritized** (a strong trunk
*absorbs* the move signal, and the timing headline is backbone-independent by
construction — the timing head reads only the latent — so a strong backbone
can't change it; `TODO.md` Milestone G). The headline is **timing**:

- **Timing (headline).** The evolving latent beats an equal-capacity memoryless
  twin on think-time in **all 8 clocked conditions across 2017–2023** (5 seeds ×
  2 backbones, 2×A100), P=1.00, and **adds value over a near-SOTA
  Elo+clock+complexity baseline** (Spearman ≈ ChessMimic's 0.41). It also beats a
  static-individual style (E-C1), and over the same locked Allie prediction the
  evolving latent beats a static per-player embedding on 2 of 3 cohorts. It
  also survives a capacity sweep and a future-*sessions* split (E-C2/E-C3).
- **Move choice is a near-null** on real data (flat by post-loss / time-pressure)
  — the when-not-what asymmetry.
- **Mechanism.** On a synthetic player with a *known* hidden state the latent
  encodes it (probe R²=0.93 vs 0.65) and causally uses it (clamp dose-response);
  the real timing edge **concentrates** under time pressure (2–8×) and for weaker
  players (≈3×). Across KT datasets the analogous signed association is
  suggestive (Pearson 0.78) but rank-weak (Spearman 0.48), bootstrap-uncertain,
  and Spanish-sensitive — evidence for a hypothesis, not a scaling law.
- **Generality.** On real KT responses, the fixed-loader rerun significantly
  favors the evolving latent in **22/24 seed cells** across 8 datasets / 3
  subjects; one Statics seed is null and one ASSISTments 2015 seed significantly
  reverses, leaving 7/8 dataset means in D's favor. Population-heterogeneity
  recovery and generation beat the "positive average person" (Wasserstein 2×
  better). Go: **honest negative** (no robust effect under a board-size control)
  — future work.

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
