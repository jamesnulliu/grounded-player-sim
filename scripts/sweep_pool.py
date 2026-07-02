#!/usr/bin/env python
"""Pool E-C sweep cells across seeds -> a per-(cohort, trunk) headline table.

Reads the per-cell JSONs written by ``sweep_cell.py``. For each (cohort, trunk)
it averages every player's ``D - B`` across seeds (reducing seed noise), then
bootstraps over **players** (the independent unit) for a pooled CI. This is the
multi-seed, at-scale version of the single-seed numbers in ``results_ec.md`` --
exactly what kills the "single seed / n=100" reviewer objection.

Example::

    python scripts/sweep_pool.py runs/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

from gps.eval.bootstrap import bootstrap_ci


def _pool(cells, d_key, b_key):
    """Average per-player (D-B) across seeds, keyed by player id."""
    per_player = defaultdict(list)
    for c in cells:
        d, b = c.get(d_key), c.get(b_key)
        if not d or not b:
            continue
        for pid, dv, bv in zip(c["players"], d, b):
            per_player[pid].append(dv - bv)
    return [sum(v) / len(v) for v in per_player.values()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="cell JSONs (globs ok)")
    ap.add_argument("--bootstrap-n", type=int, default=5000)
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        files.extend(glob.glob(p))
    cells = [json.load(open(f)) for f in files]

    groups = defaultdict(list)
    for c in cells:
        groups[(c["cohort"], c["trunk"])].append(c)

    print(
        f"{'cohort':<22}{'trunk':<6}{'seeds':<6}{'players':<8}"
        f"{'channel':<8}{'D-B':>9}  95% CI            P(<0)"
    )
    print("-" * 86)
    for (cohort, trunk), cs in sorted(groups.items()):
        seeds = sorted({c["seed"] for c in cs})
        for chan, dk, bk in (
            ("move", "move_d", "move_b"),
            ("timing", "timing_d", "timing_b"),
        ):
            diffs = _pool(cs, dk, bk)
            if not diffs:
                continue
            ci = bootstrap_ci(diffs, n_resamples=args.bootstrap_n, seed=0)
            print(
                f"{cohort:<22}{trunk:<6}{len(seeds):<6}{len(diffs):<8}"
                f"{chan:<8}{ci.point:>+9.4f}  "
                f"[{ci.low:+.4f},{ci.high:+.4f}]  {ci.p_below_zero:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
