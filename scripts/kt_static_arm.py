#!/usr/bin/env python
"""KT fitted-STATIC-profile arm: does a frozen per-student profile suffice?

The KT shuffle control says the response-channel gain is order-invariant
individualization. The sharpest version of the resulting objection: "then a
STATIC profile also works — the dynamic state is unnecessary on KT." This
arm measures that. S = logistic regression over [1, item_difficulty,
student_train_accuracy], where student_train_accuracy is one FROZEN number
per student — their accuracy over their own training split (leakage-safe:
the held-out steps never contribute). This is the strongest static profile
available: a fitted summary of >=35 observed answers, i.e. far better than
any hand-written persona. It never updates during evaluation.

Contrasts (paired per student against the cached arms in
runs/kt-memory-arm/<id>.json and the frozen replication cells):
  static_minus_b      -- does even a frozen fitted profile beat no-profile?
  memory_minus_static -- does UPDATING through the stream add anything the
                         frozen profile misses (drift / learning)?
  d_minus_static      -- learned latent vs the frozen profile.

Reading guide: if memory_minus_static ~ 0, then on this channel a
well-fitted static profile captures the whole individualization effect and
the honest claim is "you must ESTIMATE the person from their data (B still
loses), but once estimated the profile can be frozen." If
memory_minus_static < 0, updating tracks real within-student drift (e.g.
learning) beyond the frozen profile.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gps.data.kt_csv import load_kt_csv
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.kt_replication import (
    cohort_fingerprint,
    load_replication_manifest,
)
from gps.policy.board_native import BoardNativeBackbone

DEFAULT_MANIFEST = Path("scripts/kt_replication_manifest.json")
MEM_DIR = Path("runs/kt-memory-arm")
DEFAULT_OUT_DIR = Path("runs/kt-static-arm")
DEFAULT_SUMMARY = Path("results/kt_static_arm.json")


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (y - p) - 1e-6 * w
        hess = (x * (p * (1.0 - p) + 1e-9)[:, None]).T @ x
        hess[np.diag_indices_from(hess)] += 1e-6
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def _ci_dict(ci) -> dict:
    return {
        "point": float(ci.point),
        "low": float(ci.low),
        "high": float(ci.high),
        "p_below_zero": float(ci.p_below_zero),
        "n_units": int(ci.n_units),
        "confidence": float(ci.confidence),
    }


def _run_dataset(spec, protocol) -> dict:
    mem_cell = json.loads((MEM_DIR / f"{spec.dataset_id}.json").read_text())
    dataset = load_kt_csv(
        str(spec.prepared_path),
        n_students=spec.n_students,
        min_responses=protocol.min_responses,
        max_len=protocol.max_len,
        train_frac=protocol.train_frac,
    )
    if cohort_fingerprint(dataset) != mem_cell["cohort_fingerprint"]:
        raise ValueError(f"{spec.dataset_id}: cohort fingerprint mismatch")
    students = [t.player_id for t in dataset.trajectories]
    if students != mem_cell["students"]:
        raise ValueError(f"{spec.dataset_id}: student order mismatch")
    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=protocol.train_frac
    )

    train_x, train_y = [], []
    eval_rows: list[tuple[int, list[float], float]] = []
    for b, (traj, sp) in enumerate(zip(dataset.trajectories, splits)):
        # The FROZEN per-student profile: training-split accuracy only.
        train_correct = [
            1.0 if traj.observations[t].move == "correct" else 0.0
            for t in range(sp)
        ]
        profile = (
            sum(train_correct) / len(train_correct) if train_correct else 0.5
        )
        for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
            x = [1.0, float(dp.state[0]), profile]
            y = 1.0 if obs.move == "correct" else 0.0
            if t < sp:
                train_x.append(x)
                train_y.append(y)
            else:
                eval_rows.append((b, x, y))
    w = _fit_logistic(np.asarray(train_x), np.asarray(train_y))

    per_student: dict[int, list[float]] = {}
    for b, x, y in eval_rows:
        z = float(np.dot(w, x))
        nll = math.log1p(math.exp(-abs(z))) + max(z, 0.0) - y * z
        per_student.setdefault(b, []).append(nll)
    s_nll = [
        statistics.mean(per_student[b])
        for b in range(len(dataset.trajectories))
    ]

    m_nll = mem_cell["memory_nll"]
    d_nll = mem_cell["d_nll_seed_mean"]
    b_nll = mem_cell["b_nll_seed_mean"]
    contrasts = {
        "static_minus_b": [s - b for s, b in zip(s_nll, b_nll)],
        "memory_minus_static": [m - s for m, s in zip(m_nll, s_nll)],
        "d_minus_static": [d - s for d, s in zip(d_nll, s_nll)],
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec.dataset_id,
        "label": spec.label,
        "cohort_fingerprint": mem_cell["cohort_fingerprint"],
        "n_students": len(students),
        "protocol": {
            "static_profile": (
                "per-student training-split accuracy (frozen scalar)"
            ),
            "base_features": ["intercept", "item_difficulty"],
            "model": "logistic IRLS on train steps (deterministic)",
            "train_frac": protocol.train_frac,
        },
        "students": students,
        "static_nll": s_nll,
        "means": {
            "b_memoryless": statistics.mean(b_nll),
            "static_profile": statistics.mean(s_nll),
            "memory": statistics.mean(m_nll),
            "d_evolving": statistics.mean(d_nll),
        },
        "contrasts": contrasts,
        "cis": {
            key: _ci_dict(bootstrap_ci(vals, n_resamples=5000, seed=0))
            for key, vals in contrasts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_replication_manifest(args.manifest)
    results = []
    for spec in manifest.select(args.dataset):
        target = args.out_dir / f"{spec.dataset_id}.json"
        if target.exists() and not args.force:
            result = json.loads(target.read_text())
            print(f"[skip] {target}")
        else:
            result = _run_dataset(spec, manifest.protocol)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            tmp.replace(target)
        results.append(result)
        means = result["means"]
        print(
            f"[kt-static] {result['dataset_id']} "
            f"B={means['b_memoryless']:.4f} "
            f"S={means['static_profile']:.4f} M={means['memory']:.4f} "
            f"D={means['d_evolving']:.4f}"
        )
        for key, ci in result["cis"].items():
            print(
                f"[kt-static] {result['dataset_id']} {key}="
                f"{ci['point']:+.4f} [{ci['low']:+.4f},{ci['high']:+.4f}] "
                f"P(<0)={ci['p_below_zero']:.3f}",
                flush=True,
            )

    if args.dataset is None:
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "comparison": (
                "frozen fitted per-student profile vs memoryless / memory /"
                " evolving arms, fixed-loader KT replication"
            ),
            "datasets": {
                r["dataset_id"]: {"means": r["means"], "cis": r["cis"]}
                for r in results
            },
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.summary_out.with_name(args.summary_out.name + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.summary_out)
        for key in (
            "static_minus_b",
            "memory_minus_static",
            "d_minus_static",
        ):
            sig_neg = sum(1 for r in results if r["cis"][key]["high"] < 0)
            sig_pos = sum(1 for r in results if r["cis"][key]["low"] > 0)
            print(
                f"[kt-static] summary {key}: {sig_neg}/8 significantly "
                f"negative, {sig_pos}/8 significantly positive"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
