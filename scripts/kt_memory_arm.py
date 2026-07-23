#!/usr/bin/env python
"""KT-side structured-memory arm over the frozen 8-dataset replication.

Chess timing found (results/memory_baseline.txt) that a training-free
structured memory carries the dynamic term as well as the learned latent.
This script asks whether that transfers to the knowledge-tracing *response*
channel: race a hand-designed causal running-statistics memory against the
cached evolving (D) and memoryless (B) arms of the fixed-loader replication
(runs/kt-replication-fixed-loader/, which stores per-student NLLs).

Arm M: logistic regression P(correct) = sigmoid(w . [1, item_difficulty,
memory features]) fit by exact IRLS on each student's training steps (same
fraction split as run_kt), evaluated on held-out steps. The memory features
are strictly causal running statistics of the student's own past answers:
running/EWMA/recent-10 accuracy, lag-1 correctness, a signed streak, the
running/EWMA/lag-1 residual against the item-difficulty prediction (does
this student beat the difficulty baseline?), a memory-fill indicator, plus
the four engineered history features (GRU input parity). Zero trained
parameters beyond the logistic readout; deterministic given the dataset.
Like the chess memory arm it is deliberately stronger-input than the
learned arms: it reads raw past correctness and the difficulty-baseline
errors, while the GRU reads only the four engineered features.

Pairing: the frozen cells store per-student d_nll/b_nll for 3 seeds; the
learned arms are seed-averaged per student and paired with the
deterministic M per student. Cohort identity is verified against each
cell's cohort_fingerprint and prepared_sha256.
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
from gps.latent.structured import DIMENSIONS, history_features
from gps.policy.board_native import BoardNativeBackbone

DEFAULT_MANIFEST = Path("scripts/kt_replication_manifest.json")
CELL_DIR = Path("runs/kt-replication-fixed-loader")
DEFAULT_OUT_DIR = Path("runs/kt-memory-arm")
DEFAULT_SUMMARY = Path("results/kt_memory_arm.json")

KT_MEMORY_FEATURE_NAMES = (
    "mem_fill",  # log-saturating count of remembered answers
    "running_acc",  # running accuracy over all past answers
    "ewma_acc",  # recency-weighted accuracy (alpha=0.1)
    "recent10_acc",  # accuracy over the last 10 answers
    "lag1_correct",  # previous answer correct?
    "streak",  # tanh(signed consecutive-correct streak / 5)
    "mean_resid",  # running mean of (correct - difficulty-baseline pred)
    "ewma_resid",  # recency-weighted residual (alpha=0.1)
    "lag1_resid",  # previous answer's residual
    "has_memory",  # 1 once at least one answer is remembered
) + DIMENSIONS  # current-step engineered features (GRU input parity)


def _memory_rows(traj) -> list[list[float]]:
    """Strictly causal memory features, one row per decision."""
    rows = []
    n = 0
    sum_c = 0.0
    ewma_c = 0.0
    recent: list[float] = []
    lag1_c = 0.0
    streak = 0
    sum_r = 0.0
    ewma_r = 0.0
    lag1_r = 0.0
    for dp, obs in zip(traj.decisions, traj.observations):
        feats = history_features(dp)
        rows.append(
            [
                math.log1p(n) / 6.0,
                (sum_c / n) if n else 0.5,
                ewma_c if n else 0.5,
                (sum(recent) / len(recent)) if recent else 0.5,
                lag1_c,
                math.tanh(streak / 5.0),
                (sum_r / n) if n else 0.0,
                ewma_r,
                lag1_r,
                1.0 if n else 0.0,
            ]
            + [feats[d] for d in DIMENSIONS]
        )
        # Fold the current answer into memory only after emitting z_t.
        correct = 1.0 if obs.move == "correct" else 0.0
        p_item = 1.0 - float(dp.state[0])  # difficulty-baseline prediction
        resid = correct - p_item
        n += 1
        sum_c += correct
        ewma_c = correct if n == 1 else 0.9 * ewma_c + 0.1 * correct
        recent.append(correct)
        if len(recent) > 10:
            recent.pop(0)
        lag1_c = correct
        streak = streak + 1 if correct else min(streak, 0) - 1
        if correct and streak < 0:
            streak = 1
        sum_r += resid
        ewma_r = resid if n == 1 else 0.9 * ewma_r + 0.1 * resid
        lag1_r = resid
    return rows


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact IRLS with a tiny ridge; deterministic."""
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
    cells = []
    for seed in protocol.seeds:
        path = CELL_DIR / spec.dataset_id / f"seed-{seed}.json"
        cells.append(json.loads(path.read_text()))
    shas = {cell["prepared_sha256"] for cell in cells}
    prints = {cell["cohort_fingerprint"] for cell in cells}
    if len(shas) != 1 or len(prints) != 1:
        raise ValueError(f"{spec.dataset_id}: seed cells disagree on cohort")
    orders = {tuple(p["player_id"] for p in cell["players"]) for cell in cells}
    if len(orders) != 1:
        raise ValueError(f"{spec.dataset_id}: player order differs")

    dataset = load_kt_csv(
        str(spec.prepared_path),
        n_students=spec.n_students,
        min_responses=protocol.min_responses,
        max_len=protocol.max_len,
        train_frac=protocol.train_frac,
    )
    if cohort_fingerprint(dataset) != next(iter(prints)):
        raise ValueError(f"{spec.dataset_id}: cohort fingerprint mismatch")
    students = [t.player_id for t in dataset.trajectories]
    if students != list(next(iter(orders))):
        raise ValueError(f"{spec.dataset_id}: student order mismatch")

    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=protocol.train_frac
    )

    # Design matrix: [1, item_difficulty] + memory features.
    train_x, train_y = [], []
    eval_rows: list[tuple[int, list[float], float]] = []  # (student, x, y)
    for b, (traj, sp) in enumerate(zip(dataset.trajectories, splits)):
        mem = _memory_rows(traj)
        for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
            x = [1.0, float(dp.state[0])] + mem[t]
            y = 1.0 if obs.move == "correct" else 0.0
            if t < sp:
                train_x.append(x)
                train_y.append(y)
            else:
                eval_rows.append((b, x, y))
    w = _fit_logistic(np.asarray(train_x), np.asarray(train_y))

    per_student_nlls: dict[int, list[float]] = {}
    for b, x, y in eval_rows:
        z = float(np.dot(w, x))
        # Numerically stable binary NLL in nats.
        nll = math.log1p(math.exp(-abs(z))) + max(z, 0.0) - y * z
        per_student_nlls.setdefault(b, []).append(nll)
    m_nll = [
        statistics.mean(per_student_nlls[b])
        for b in range(len(dataset.trajectories))
    ]

    d_nll = [
        statistics.mean(cell["players"][i]["d_nll"] for cell in cells)
        for i in range(len(students))
    ]
    b_nll = [
        statistics.mean(cell["players"][i]["b_nll"] for cell in cells)
        for i in range(len(students))
    ]
    contrasts = {
        "d_minus_memory": [d - m for d, m in zip(d_nll, m_nll)],
        "memory_minus_b": [m - b for m, b in zip(m_nll, b_nll)],
        "d_minus_b": [d - b for d, b in zip(d_nll, b_nll)],
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec.dataset_id,
        "label": spec.label,
        "prepared_sha256": next(iter(shas)),
        "cohort_fingerprint": next(iter(prints)),
        "n_students": len(students),
        "protocol": {
            "memory_features": list(KT_MEMORY_FEATURE_NAMES),
            "base_features": ["intercept", "item_difficulty"],
            "model": "logistic IRLS on train steps (deterministic)",
            "learned_arm_seeds": list(protocol.seeds),
            "train_frac": protocol.train_frac,
        },
        "students": students,
        "memory_nll": m_nll,
        "d_nll_seed_mean": d_nll,
        "b_nll_seed_mean": b_nll,
        "means": {
            "b_memoryless": statistics.mean(b_nll),
            "memory": statistics.mean(m_nll),
            "d_evolving": statistics.mean(d_nll),
        },
        "contrasts": {k: v for k, v in contrasts.items()},
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
    specs = manifest.select(args.dataset)
    results = []
    for spec in specs:
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
            f"[kt-mem] {result['dataset_id']} "
            f"B={means['b_memoryless']:.4f} "
            f"M={means['memory']:.4f} D={means['d_evolving']:.4f}"
        )
        for key, ci in result["cis"].items():
            print(
                f"[kt-mem] {result['dataset_id']} {key}={ci['point']:+.4f} "
                f"[{ci['low']:+.4f},{ci['high']:+.4f}] "
                f"P(<0)={ci['p_below_zero']:.3f}",
                flush=True,
            )

    if args.dataset is None:
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "comparison": (
                "structured-memory logistic arm vs cached evolving (D) and"
                " memoryless (B) arms, fixed-loader KT replication"
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
        for key in ("d_minus_memory", "memory_minus_b", "d_minus_b"):
            sig_neg = sum(1 for r in results if r["cis"][key]["high"] < 0)
            sig_pos = sum(1 for r in results if r["cis"][key]["low"] > 0)
            print(
                f"[kt-mem] summary {key}: {sig_neg}/8 significantly "
                f"negative, {sig_pos}/8 significantly positive "
                f"-> {args.summary_out}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
