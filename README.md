# grounded-player-sim (`gps`)

Personalized human game-play simulation built around an **LLM policy
augmented with a trainable, per-individual dynamic latent state**.

**Thesis.** An LLM conditioned on a static verbal persona/emotion label is
not enough to reproduce how a specific person plays *right now*. You need a
learned latent state `z_t` that evolves over the player's own action+timing
trajectory, is injected into the policy, and is validated against the
player's *future* behavior. Demonstrated on chess and Go, which share an
engine-graded decision interface (Stockfish / KataGo).

**Positioning.** The contribution is the *conjunction* — per-individual +
temporally-evolving + a behavioral state (tilt/fatigue/time-pressure) that
drives moves & timing + validated on the person's *future* games + chess
**and** Go. No single axis is claimed as novel: "evolving latent in an LLM,"
"natural-language latent," and "future temporal-split validation" are each
already owned by a 2026 competitor (LATTE / HumanLM). See `documents/design.md`
§8 for the head-to-head differentiation vs. Allie, HumanLM, and LATTE.

See `documents/proposal_v2.md` for the full research proposal,
`documents/design.md` for how the code maps onto it (and the key decisions),
`documents/training.md` for the GPU/data wiring,
`documents/milestone_a.md` for the runbook + resources for the make-or-break
"is the latent just history-conditioning?" experiment, and **`TODO.md`** for
the prioritized work plan (code to write + experiments to run).

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
| `src/gps/experiments/` | runnable experiments (Phase 0 today) |
| `documents/` | proposal, design notes, training notes |
| `tests/` | CPU test suite |
