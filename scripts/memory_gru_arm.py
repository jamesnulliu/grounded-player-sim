#!/usr/bin/env python
"""Equal-input learned-dynamics arm: a GRU over the SAME memory features.

`scripts/memory_vs_learned.py` found that the hand-designed structured-memory
control beats the 4-feature evolving GRU on chess timing. That comparison is
deliberately input-asymmetric (the memory reads raw past think-times and
Allie's past errors; the GRU reads only the four engineered history
features), so it cannot say whether the memory wins because of its *richer
information* or because learning adds nothing. This arm resolves that: train
a GRU whose per-step input is exactly the 15 memory features of
`gps.experiments.ec._memory_latents`, with a linear head predicting the
residual of log think-time over the locked Allie offset, on each player's
training steps (same session split); score held-out steps with the identical
zero-mean-Gaussian NLL treatment (homoscedastic sigma from train residuals,
matching the lstsq arms). Learned-vs-linear over the same information is
then a clean mechanism contrast:

  gru(mem) - linear(mem)  : does learned recurrence add value at equal input?
  gru(mem) - evolving(4f) : how much was the old arm's input poverty costing?

Pairs per player against `runs/memory-vs-learned/<cohort>.json` (memory,
static, evolving NLLs already seed-averaged there for the learned arms).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gps.data.store import load_dataset
from gps.eval.bootstrap import bootstrap_ci
from gps.experiments.ec import (
    MEMORY_FEATURE_NAMES,
    _memory_latents,
    session_split_indices,
)

MEM_DIR = Path("runs/memory-vs-learned")
DEFAULT_OUT_DIR = Path("runs/memory-gru-arm")
DEFAULT_SUMMARY = Path("results/memory_gru_arm.json")
DEFAULT_COHORTS = {
    "2017-04": Path(
        "/project2/xiangren_1715/liuyanch/g4_data/"
        "ec2017/dataset_allie.jsonl.gz"
    ),
    "2019-07": Path(
        "/project2/xiangren_1715/liuyanch/g4_data/"
        "ec2019/dataset_allie.jsonl.gz"
    ),
    "2021-06": Path(
        "/project2/xiangren_1715/liuyanch/g4_data/"
        "ec2021/dataset_allie.jsonl.gz"
    ),
}
SEEDS = (0, 1, 2)
HIDDEN_DIM = 16  # matches the learned arms' latent_dim
EPOCHS = 300
LR = 1e-2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _ci_dict(ci) -> dict:
    return {
        "point": float(ci.point),
        "low": float(ci.low),
        "high": float(ci.high),
        "p_below_zero": float(ci.p_below_zero),
        "n_units": int(ci.n_units),
        "confidence": float(ci.confidence),
    }


def _run_cohort(cohort: str, path: Path) -> dict:
    import torch
    from torch import nn

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    dataset_sha = _sha256(path)
    mem_cell_path = MEM_DIR / f"{cohort}.json"
    mem_cell = json.loads(mem_cell_path.read_text())
    if mem_cell["dataset_sha256"] != dataset_sha:
        raise ValueError(f"{mem_cell_path}: dataset hash mismatch")

    dataset = load_dataset(str(path))
    trajectories = dataset.trajectories
    if [t.player_id for t in trajectories] != mem_cell["players"]:
        raise ValueError(f"{cohort}: player order differs from memory cell")
    splits = session_split_indices(trajectories, 0.7)
    lat = _memory_latents(trajectories)  # [T][B][15]
    n_feat = len(MEMORY_FEATURE_NAMES)
    n_players = len(trajectories)
    t_max = max(len(t.decisions) for t in trajectories)
    half_log2pi = 0.5 * math.log(2 * math.pi)

    # Padded tensors: features [B,T,F], residual target [B,T], masks [B,T].
    feats = torch.zeros(n_players, t_max, n_feat)
    target = torch.zeros(n_players, t_max)
    y_all = torch.zeros(n_players, t_max)
    train_mask = torch.zeros(n_players, t_max, dtype=torch.bool)
    eval_mask = torch.zeros(n_players, t_max, dtype=torch.bool)
    for b, (traj, sp) in enumerate(zip(trajectories, splits)):
        for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
            feats[b, t] = torch.tensor(lat[t][b])
            y = math.log(max(obs.time_spent or 1e-3, 1e-3))
            pred = dp.context.get("external_time_pred")
            if pred is None:
                raise ValueError("dataset lacks cached Allie predictions")
            off = math.log(max(float(pred), 1e-3))
            y_all[b, t] = y
            target[b, t] = y - off
            (train_mask if t < sp else eval_mask)[b, t] = True

    # Standardize inputs with train-step statistics (lstsq is scale-free;
    # the GRU is not -- this only levels conditioning, adds no information).
    flat = feats[train_mask]
    mu_f = flat.mean(dim=0)
    sd_f = flat.std(dim=0).clamp_min(1e-6)
    feats_n = (feats - mu_f) / sd_f

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        gru = nn.GRU(n_feat, HIDDEN_DIM, batch_first=True)
        head = nn.Linear(HIDDEN_DIM, 1)
        params = list(gru.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=LR)
        for _epoch in range(EPOCHS):
            opt.zero_grad()
            h, _ = gru(feats_n)
            pred = head(h).squeeze(-1)
            loss = ((pred - target)[train_mask] ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            h, _ = gru(feats_n)
            pred = head(h).squeeze(-1)
            sigma = float((pred - target)[train_mask].std()) or 1.0
        # Per-player held-out NLL, identical formula to the lstsq arms.
        per_player = []
        for b in range(n_players):
            idx = eval_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                per_player.append(float("nan"))
                continue
            resid = (target[b, idx] - pred[b, idx]) / sigma
            nll = (
                0.5 * resid**2 + math.log(sigma) + half_log2pi + y_all[b, idx]
            )
            per_player.append(float(nll.mean()))
        seed_results.append(
            {
                "seed": seed,
                "train_mse": float(loss),
                "sigma": sigma,
                "per_player_nll": per_player,
            }
        )
        print(
            f"[gru] {cohort} seed={seed} train_mse={float(loss):.4f}",
            flush=True,
        )

    gru_nll = [
        statistics.mean(sr["per_player_nll"][i] for sr in seed_results)
        for i in range(n_players)
    ]
    mem_nll = mem_cell["memory_nll"]
    evolving_nll = mem_cell["evolving_nll_seed_mean"]
    static_nll = mem_cell["static_nll_seed_mean"]
    contrasts = {
        "gru_minus_memory": [g - m for g, m in zip(gru_nll, mem_nll)],
        "gru_minus_evolving": [g - e for g, e in zip(gru_nll, evolving_nll)],
        "gru_minus_static": [g - s for g, s in zip(gru_nll, static_nll)],
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "dataset_path": str(path),
        "dataset_sha256": dataset_sha,
        "protocol": {
            "inputs": list(MEMORY_FEATURE_NAMES),
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "lr": LR,
            "seeds": list(SEEDS),
            "target": "log think-time residual over locked Allie offset",
            "sigma": "homoscedastic from train residuals (lstsq parity)",
        },
        "players": mem_cell["players"],
        "gru_nll_seed_mean": gru_nll,
        "per_seed": seed_results,
        "means": {
            "allie_plus_gru_mem": statistics.mean(gru_nll),
            "allie_plus_memory": statistics.mean(mem_nll),
            "allie_plus_evolving": statistics.mean(evolving_nll),
            "allie_plus_static": statistics.mean(static_nll),
        },
        "contrasts": contrasts,
        "cis": {
            key: _ci_dict(bootstrap_ci(vals, n_resamples=5000, seed=0))
            for key, vals in contrasts.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", action="append", choices=list(DEFAULT_COHORTS)
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cohorts = args.cohort or list(DEFAULT_COHORTS)
    results = []
    for cohort in cohorts:
        target = args.out_dir / f"{cohort}.json"
        if target.exists() and not args.force:
            result = json.loads(target.read_text())
            print(f"[skip] {target}")
        else:
            result = _run_cohort(cohort, DEFAULT_COHORTS[cohort])
            _write_json_atomic(target, result)
        results.append(result)
        means = result["means"]
        print(
            f"[gru] {cohort} +gru(mem)={means['allie_plus_gru_mem']:.4f} "
            f"+mem={means['allie_plus_memory']:.4f} "
            f"+evolving={means['allie_plus_evolving']:.4f} "
            f"+static={means['allie_plus_static']:.4f}"
        )
        for key, ci in result["cis"].items():
            print(
                f"[gru] {cohort} {key}={ci['point']:+.4f} "
                f"[{ci['low']:+.4f},{ci['high']:+.4f}] "
                f"P(<0)={ci['p_below_zero']:.3f}",
                flush=True,
            )

    if set(cohorts) == set(DEFAULT_COHORTS):
        pooled: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for result in results:
            for key, vals in result["contrasts"].items():
                for player, diff in zip(result["players"], vals):
                    pooled[key][player].append(diff)
        overall = {}
        for key, per_player in pooled.items():
            unique = [statistics.mean(v) for v in per_player.values()]
            overall[key] = {
                "n_unique_players": len(unique),
                "ci": _ci_dict(bootstrap_ci(unique, n_resamples=5000, seed=0)),
            }
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "comparison": "GRU over memory features vs linear memory readout",
            "cohorts": {
                r["cohort"]: {"means": r["means"], "cis": r["cis"]}
                for r in results
            },
            "overall": overall,
        }
        _write_json_atomic(args.summary_out, summary)
        for key, entry in overall.items():
            ci = entry["ci"]
            print(
                f"[gru] overall {key}={ci['point']:+.4f} "
                f"[{ci['low']:+.4f},{ci['high']:+.4f}] "
                f"P(<0)={ci['p_below_zero']:.3f} "
                f"n={entry['n_unique_players']} -> {args.summary_out}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
