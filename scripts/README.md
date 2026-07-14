# Reproducible experiment scripts

## Fixed-loader KT replication

The eight-dataset response replication is frozen in
`kt_replication_manifest.json`: seeds 0–2, 60 epochs, `train_frac=0.7`,
`min_responses=50`, `max_len=200`, and the original cohort caps (500 except
Spanish 150 and Statics 200). Data is not redistributed.

Place the exact source exports at the manifest paths under
`data/kt/source/`. ASSISTments 2009 starts from the corrected raw skill-builder
CSV; the other seven inputs are the standard five-column exports used by the
original replication. Then run:

```bash
PYTHONPATH=src python scripts/prepare_kt_replications.py
PYTHONPATH=src python scripts/run_kt_replication.py --validate-only
PYTHONPATH=src python scripts/run_kt_replication.py
```

Preparation writes canonical TSVs plus provenance receipts containing source,
output, and frozen-manifest SHA-256 hashes. The run command writes one resumable
cell per dataset/seed under `runs/kt-replication-fixed-loader/`; it refuses to
fit the headline until all 24 cells exist and share the expected cohort hashes.
Training still follows the project tracking policy and requires
`WANDB_API_KEY`. To fan cells out, pass one `--dataset ID --seed N` pair per
job, then aggregate only after all jobs finish:

```bash
PYTHONPATH=src python scripts/run_kt_replication.py --aggregate-only \
  --summary-out results/kt_replication_fixed_loader.json
```

The summary records both signed latent advantage and the historical absolute
effect definition, with Pearson/Spearman, dataset-bootstrap intervals, a sign
audit, and leave-one-dataset-out ranges. Absolute value must not be described as
latent advantage when a dataset mean reverses sign.

## Stable-speed extension

`stable_speed_manifest.json` freezes exact 370 MB Lichess source prefixes,
source hashes, cohort selection, and the full static-control training protocol
for 2021-04, 2023-04, and 2021-06. Reproduce with:

```bash
PYTHONPATH=src python scripts/prepare_stable_speed_cohorts.py --download
PYTHONPATH=src python scripts/run_stable_speed_replication.py --validate-only
PYTHONPATH=src python scripts/run_stable_speed_replication.py
PYTHONPATH=src python scripts/run_stable_speed_replication.py \
  --aggregate-only --summary-out results/stable_speed_extension.json
```

The runner writes one resumable cell per cohort/seed under
`runs/stable-speed-extension/`. Pooling first averages each player's paired
`D-B` across seeds, then bootstraps players within each cohort. Training follows
the project tracking policy and requires `WANDB_API_KEY`.

## EdNet real-timing test

The primary EdNet-KT1 test is frozen in `ednet_manifest.json` and documented in
`documents/ednet_protocol.md`. It restricts timing to singleton-question
bundles because upstream issue #5 leaves bundle timing semantics unresolved.

```bash
PYTHONPATH=src python scripts/prepare_ednet.py --download
PYTHONPATH=src python scripts/run_ednet_replication.py --validate-only
PYTHONPATH=src python scripts/run_ednet_replication.py
PYTHONPATH=src python scripts/run_ednet_replication.py --aggregate-only \
  --summary-out results/ednet_replication.json
```

Cells are resumable under `runs/ednet-kt1-singleton/`. Pooling averages each
student's paired `D-B` over seeds before bootstrapping students.

## Allie static-vs-evolving control

The strongest G4 identity control compares the same locked Allie prediction
plus a static per-player embedding against Allie plus the evolving latent. The
runner uses the three existing Allie-cached cohorts, five seeds, 15 epochs, and
a held-out session split:

```bash
PYTHONPATH=src python scripts/g4_allie_static_vs_evolving.py
PYTHONPATH=src python scripts/g4_allie_static_vs_evolving.py --aggregate-only
```

Pass repeated `--cell COHORT:SEED` arguments to partition the 15 cells across
GPUs. Cells are resumable under `runs/g4-allie-static-vs-evolving/`; the pooled
artifact is `results/g4_allie_static_vs_evolving.json`.

## Scaling the E-C results across GPUs

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
