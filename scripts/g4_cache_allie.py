"""G4: cache Allie's per-decision think-time prediction onto a dataset.

Allie (Zhang et al., ICLR'25; github.com/ippolito-cmu/allie) is a RELEASED
decoder-only model with a dedicated think-time head -- an actual released
human-timing model, the cleanest external predictor for G4 (vs Maia-2, which is
move-only and gives only a difficulty PROXY). We feed Allie each game's move
history up to a decision and read its predicted think-time for the move about
to be played (decode.py: undo_time_normalization(time_logits[0,-1])), cached as
``context["external_time_pred"]`` (seconds). run_timing_vs_aggregate then tests
whether the evolving latent adds value OVER Allie (external_pred / locked).

The dataset stores one player's decisions (FENs) per game; Allie needs the FULL
move sequence (both players). We reconstruct it by FEN-diffing: the single
opponent ply between consecutive player decisions is the legal move that yields
the next decision's placement+turn (opponent times unknown -> -1; the player's
own past move times are known and packed).

Usage: python scripts/g4_cache_allie.py IN.jsonl.gz OUT.jsonl.gz
Requires the cloned allie repo + medium checkpoint (see documents/g4_plan.md).
"""

from __future__ import annotations

import sys
import time

import chess

from gps.data.store import load_dataset, save_dataset

_BASE = "/project2/xiangren_1715/liuyanch/g4_data"
ALLIE_SRC = f"{_BASE}/allie_repo/src"
ALLIE_CKPT = f"{_BASE}/allie_models/medium/best.pt"
ALLIE_CFG = f"{_BASE}/allie_repo/pretrain_config/medium.yaml"


def _placement_turn(fen: str) -> tuple[str, str]:
    parts = fen.split()
    return parts[0], parts[1]


def reconstruct_prefixes(traj):
    """Per-decision (moves, secs) prefix, or None where unreconstructable.

    A trajectory concatenates the player's ~20 games; ``move_number`` resets to
    a lower value at each new game, so we reset the board there. Within a game,
    ``moves`` is the full UCI ply list (both players) up to that decision (the
    lone opponent ply between the player's decisions is recovered by FEN-diff);
    ``secs`` match (player's own moves = known int seconds, opponent = -1). A
    per-decision None means that game failed to reconstruct (skip just it).
    """
    board = chess.Board()
    moves: list[str] = []
    secs: list[int] = []
    out = []
    prev_mn = None
    game_broken = False
    for dp, obs in zip(traj.decisions, traj.observations):
        mn = dp.time_signal.move_number
        if prev_mn is None or mn <= prev_mn:  # new game -> reset
            board = chess.Board()
            moves, secs, game_broken = [], [], False
        prev_mn = mn
        if game_broken:
            out.append(None)
            continue
        tgt_pl, tgt_turn = _placement_turn(dp.state)
        guard = 0
        while board.board_fen() != tgt_pl or (
            "w" if board.turn else "b"
        ) != tgt_turn:
            found = None
            for mv in board.legal_moves:
                board.push(mv)
                ok = board.board_fen() == tgt_pl and (
                    "w" if board.turn else "b"
                ) == tgt_turn
                board.pop()
                if ok:
                    found = mv
                    break
            if found is None or guard > 4:
                game_broken = True
                break
            moves.append(found.uci())
            secs.append(-1)
            board.push(found)
            guard += 1
        if game_broken:
            out.append(None)
            continue
        out.append((list(moves), list(secs)))
        pm = obs.move
        try:
            board.push_uci(pm)
        except Exception:
            game_broken = True
            out[-1] = None
            continue
        moves.append(pm)
        t = obs.time_spent
        secs.append(int(t) if (t is not None and t > 0) else 0)
    return out


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, ALLIE_SRC)
    import torch
    from modeling.data import Game, UCITokenizer, undo_time_normalization
    from modeling.model import initialize_model
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ALLIE_CFG)
    tok = UCITokenizer(**cfg.data_config.tokenizer_config)
    model = initialize_model(tok, **cfg.model_config).cuda().eval()
    ck = torch.load(ALLIE_CKPT, map_location="cuda")
    model.load_state_dict(ck["model"], strict=False)
    print("[allie] model loaded", flush=True)

    ds = load_dataset(in_path)
    n_dec = sum(len(t.decisions) for t in ds.trajectories)
    done = skipped = 0
    t0 = time.time()

    @torch.inference_mode()
    def predict(moves, secs, tc, we, be):
        g = Game(
            time_control=tc, white_elo=int(we), black_elo=int(be),
            outcome=None, normal_termination=False, moves=moves,
            moves_seconds=secs,
        )
        b = tok.pad_and_collate([g])
        b = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in b.items()}
        out = model(**b)
        return float(undo_time_normalization(out["time_logits"][0, -1].item()))

    for traj in ds.trajectories:
        prefixes = reconstruct_prefixes(traj)
        for dp, pf in zip(traj.decisions, prefixes):
            if pf is None:
                dp.context["external_time_pred"] = None
                skipped += 1
                continue
            moves, secs = pf
            color = dp.context.get("color")
            pe = dp.context.get("player_elo") or 1500
            oe = dp.context.get("opponent_elo") or pe
            we, be = (pe, oe) if color == "white" else (oe, pe)
            tc = dp.context.get("time_control") or "300+0"
            pred = predict(moves, secs, tc, we, be)
            dp.context["external_time_pred"] = max(0.01, pred)
            done += 1
            if done % 5000 == 0:
                print(
                    f"[allie] {done}/{n_dec} ({done/(time.time()-t0):.0f}/s, "
                    f"skipped={skipped})",
                    flush=True,
                )

    # drop unreconstructable decisions so the G4 run sees only valid preds
    kept = []
    for traj in ds.trajectories:
        keep_d, keep_o = [], []
        for dp, obs in zip(traj.decisions, traj.observations):
            if dp.context.get("external_time_pred") is not None:
                keep_d.append(dp)
                keep_o.append(obs)
        if keep_d:
            traj.decisions = keep_d
            traj.observations = keep_o
            kept.append(traj)
    ds.trajectories = kept
    save_dataset(ds, out_path)
    print(
        f"[allie] wrote {out_path}: {done} preds, {skipped} skipped "
        f"({100*skipped/max(done+skipped,1):.1f}%), {len(kept)} players, "
        f"{time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
