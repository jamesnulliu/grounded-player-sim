# Scaling the E-C results across GPUs

The per-player D-vs-B comparison is **embarrassingly parallel** — every
(cohort, seed, trunk) cell is an independent run, so N GPUs give ~N× throughput.
This directory turns the single-seed / n=100 results in `documents/results_ec.md`
into a pooled, multi-seed, at-scale, multi-cohort table — the single biggest
credibility upgrade for the paper.

## Pipeline

1. **Ingest** the cohorts (one-time, CPU/network): `gps ingest ...` → one
   `*.jsonl.gz` per cohort. Point `gen_manifest.py:COHORTS` at them.
2. **Generate the manifest** (one cell per line):
   `python scripts/gen_manifest.py > runs/manifest.txt`
3. **Launch** one GPU per cell via a SLURM array (`%K` caps concurrency to your
   K GPUs): `sbatch --array=0-$(($(wc -l < runs/manifest.txt)-1))%8 scripts/sweep.sbatch`
   — or without SLURM: `xargs -P 8 -I{} bash -c '{}' < runs/manifest.txt`.
4. **Pool**: `python scripts/sweep_pool.py runs/*.json` → per-(cohort, trunk)
   pooled D−B (move + timing), averaged per-player across seeds, bootstrapped
   over players. Validated end-to-end on CPU.

## What each resource tier unlocks (priority order)

**Tier 1 — kill the single-seed / small-n caveats (highest ROI, pure parallel).**
The default grid: 5 cohorts (2013→2023, era-generality) × 5 seeds × {mlp, conv}
× 1000 players = **50 cells**. Each ~10–20 min on one GPU at 1000 players, 30
epochs. On **8 GPUs ≈ 1–2 h wall**. Output: pooled CIs across seeds and a
cohort-generality table → directly answers "is the timing win robust across
seeds, cohorts, *and* backbones?" (the 2×2 in `results_ec.md`, but solid).

**Tier 2 — the trained-LLM-injection pillar (the one positive-LLM result left).**
`HIDDEN` soft-prompt + a trained injector on a frozen LLM via **slime+sglang**
(both installed). A single A100-80GB does a first version on ≤32B; **2–4 GPUs**
make slime RL rollout+train efficient and allow a stronger base model. This is
the heaviest build (soft-prompt wiring + slime config), not just compute.

**Tier 3 — stronger backbone / bigger base model.** A Maia-scale conv (or a
chess-pretrained trunk) trained on millions of positions lifts absolute move
quality; a **70B** base LLM needs **2–4× 80 GB** (sharding) for chess strength.

## What I need from you to launch

- **GPU count + hours + SLURM partition/account** (this looks like CARC — I'll
  adapt the `#SBATCH` lines). With that I generate the manifest and submit.
- Confirmation to **ingest the extra cohorts** (downloads to `/home1`, 99 TB
  free) — or point me at datasets you already have.

Until then everything here is **validated on CPU** and launch-ready; a single
idle A100 is already available, so I can start Tier 1 (or Tier 2) immediately on
one GPU and simply widen the array when more arrive.
