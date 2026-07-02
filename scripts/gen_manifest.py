#!/usr/bin/env python
"""Emit a manifest (one ``sweep_cell.py`` invocation per line) for the grid.

Edit COHORTS to point at your ingested datasets, then::

    python scripts/gen_manifest.py > runs/manifest.txt
    sbatch --array=0-$(($(wc -l < runs/manifest.txt)-1)) scripts/sweep.sbatch

Each line is one (cohort, seed, trunk) cell -> one GPU. The full default grid
is len(COHORTS) x len(SEEDS) x len(TRUNKS) cells.
"""

from __future__ import annotations

# (label, path-to-ingested-dataset.jsonl.gz). Point these at real ingests.
COHORTS = [
    ("2013-01-blitz", "data/2013-01-blitz.jsonl.gz"),
    ("2017-04-blitz", "data/2017-04-blitz.jsonl.gz"),
    ("2019-12-blitz", "data/2019-12-blitz.jsonl.gz"),
    ("2021-06-blitz", "data/2021-06-blitz.jsonl.gz"),
    ("2023-06-blitz", "data/2023-06-blitz.jsonl.gz"),
]
SEEDS = [0, 1, 2, 3, 4]
TRUNKS = ["mlp", "conv"]
N_PLAYERS = 1000
EPOCHS = 30


def main() -> None:
    for label, path in COHORTS:
        for trunk in TRUNKS:
            for seed in SEEDS:
                out = f"runs/{label}_{trunk}_s{seed}.json"
                print(
                    "python scripts/sweep_cell.py "
                    f"--dataset {path} --cohort {label} --seed {seed} "
                    f"--trunk {trunk} --n-players {N_PLAYERS} "
                    f"--epochs {EPOCHS} --split-mode session "
                    f"--timing-model zi_lognormal --out {out}"
                )


if __name__ == "__main__":
    main()
