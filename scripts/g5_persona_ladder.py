#!/usr/bin/env python
"""G5: the persona ladder INSIDE the LLM — static vs memory vs latent.

The paper's field-facing question, asked in the LLM's own channel: today's
LLM user simulators condition on a STATIC persona prompt. Does a
dynamically-updated state beat it, at matched information access?

Four conditions, ONE loop (same data, optimizer, LoRA, chat template):
  none    -- position + clock only.
  static  -- + a per-player PERSONA sentence, constant over the trajectory,
             computed from the player's TRAINING split only (leakage-safe):
             rating, typical think-time, premove rate. The field's status
             quo ("this user is impatient"), done as well as it can be.
  memory  -- + a per-decision UPDATED scorecard line (strictly causal
             running stats: moves seen, avg/last think-time, last-game
             result, recent win rate, session position, premove rate).
             The dynamic-text arm; what a memory module would write.
  hidden  -- the none prompt + the SAME scorecard numbers as a trained
             soft-prompt prefix (near-zero init). The latent arm at input
             parity with `memory`, so memory-vs-hidden is a pure channel
             contrast and static-vs-memory is a pure static-vs-dynamic
             contrast.

Target: think-time completion (" 2.0" after "\\nThink time (s):"), held-out
mean completion NLL per condition; per-example NLLs + player ids are saved
so significance can be bootstrapped over players afterwards.

Data: built directly from the real Lichess 2017-04 G4 cohort (100 players);
per-player 70/30 future split; one fixed subsample (data seed 0) shared by
every condition and seed.

Usage:
  g5_persona_ladder.py smoke [model]
  g5_persona_ladder.py full [model] [--cond none,static] [--seeds 0,1,2]
"""

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home1/liuyanch/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
)

from gps.data.store import load_dataset  # noqa: E402
from gps.experiments.llm_hidden import (  # noqa: E402
    assemble_completion_step,
    split_boundary,
)
from gps.policy.hidden_prefix import HiddenPrefixProjector  # noqa: E402

DATASET = "/project2/xiangren_1715/liuyanch/g4_data/ec2017/dataset.jsonl.gz"
OUT_DIR = Path("runs/g5-persona-ladder")
SUMMARY = Path("results/g5_persona_ladder.json")
CONDS = ("none", "static", "memory", "hidden")
LATENT_DIM = 8  # numeric twin of the scorecard facts (input parity)
N_TRAIN, N_EVAL = 5000, 1000


def build_records():
    """Per-decision records with static persona, causal scorecard, latent."""
    ds = load_dataset(DATASET)
    train, evals = [], []
    for traj in ds.trajectories:
        sp = split_boundary(traj)
        # Leakage-safe static persona from the TRAIN split only.
        train_times = [
            traj.observations[t].time_spent or 0.0 for t in range(sp)
        ]
        med_t = statistics.median(train_times) if train_times else 1.0
        premove_pct = (
            100.0
            * sum(1 for x in train_times if x <= 0.05)
            / max(len(train_times), 1)
        )
        elos = [
            traj.decisions[t].context.get("player_elo") or 1500
            for t in range(sp)
        ]
        elo = int(statistics.median(elos)) if elos else 1500
        persona = (
            f"Player profile: blitz rating {elo}; typically thinks about "
            f"{med_t:.1f}s per move; premoves {premove_pct:.0f}% of moves."
        )
        # Causal running stats for the scorecard + latent.
        n = 0
        sum_t = 0.0
        last_t = 0.0
        n_premove = 0
        for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
            ts = dp.time_signal
            clock = ts.time_remaining or 0.0
            stream = dp.recent_outcomes
            last = stream.last()
            lost_last = 1.0 if (last and last.won is False) else 0.0
            wr = stream.recent_win_rate(k=5)
            winrate = 0.5 if wr is None else wr
            sess = stream.session_position
            mean_t = (sum_t / n) if n else 0.0
            premove_frac = (n_premove / n) if n else 0.0
            scorecard = (
                f"Player state: {n} moves watched; avg think {mean_t:.1f}s,"
                f" last move {last_t:.1f}s; "
                f"{'lost' if lost_last else 'did not lose'} the last game;"
                f" recent win rate {winrate:.1f};"
                f" game {sess + 1} of this session;"
                f" premoves {100 * premove_frac:.0f}% so far."
            )
            latent = [
                math.log1p(n) / 6.0,
                min(mean_t, 60.0) / 60.0,
                min(last_t, 60.0) / 60.0,
                lost_last,
                winrate,
                min(sess / 20.0, 1.0),
                min(clock, 600.0) / 600.0,
                premove_frac,
            ]
            base = (
                "Predict how long this chess player will think before "
                "their next move.\n"
                f"Position (FEN): {dp.state}\n"
                f"Clock: {clock:.0f}s left, move {ts.move_number}."
            )
            spent = obs.time_spent or 0.0
            rec = {
                "player": traj.player_id,
                "base": base,
                "persona": persona,
                "scorecard": scorecard,
                "latent": latent,
                "label": f"{spent:.1f}",
            }
            (train if t < sp else evals).append(rec)
            # Update running stats AFTER emitting the record (causal).
            n += 1
            sum_t += spent
            last_t = spent
            if spent <= 0.05:
                n_premove += 1
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(evals)
    return train[:N_TRAIN], evals[:N_EVAL]


def prompt_for(cond, rec):
    if cond == "static":
        return rec["base"] + "\n" + rec["persona"]
    if cond == "memory":
        return rec["base"] + "\n" + rec["scorecard"]
    return rec["base"]  # none, hidden


def run_condition(
    cond,
    seed,
    model_name,
    train,
    ev,
    *,
    n_prefix=4,
    epochs=2,
    lr=1e-4,
    accum=16,
    max_len=384,
    device="cuda",
):
    torch.manual_seed(seed)
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.train()
    embed = model.get_input_embeddings()
    hidden_size = model.config.hidden_size

    projector = None
    extra = []
    if cond == "hidden":
        projector = HiddenPrefixProjector(
            LATENT_DIM, hidden_size, n_prefix=n_prefix, seed=seed
        ).to(device)
        net = projector._build()
        with torch.no_grad():
            net.proj.weight.mul_(0.02)
            net.proj.bias.zero_()
        extra = list(projector.parameters())
    params = [p for p in model.parameters() if p.requires_grad] + extra
    opt = torch.optim.AdamW(params, lr=lr)

    def _prompt_ids(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        return tok(text, add_special_tokens=False)["input_ids"]

    def _loss(rec):
        prompt = prompt_for(cond, rec) + "\nThink time (s):"
        pid = _prompt_ids(prompt)
        cid = tok(" " + rec["label"], add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > max_len:
            pid = pid[-(max_len - len(cid)) :]
        prefix = None
        if cond == "hidden":
            z = torch.tensor(rec["latent"], dtype=torch.float32, device=device)
            prefix = projector.project(z).to(torch.bfloat16)
        emb, attn, labels = assemble_completion_step(
            embed,
            torch.tensor(pid, device=device),
            torch.tensor(cid, device=device),
            prefix,
        )
        out = model(
            inputs_embeds=emb.to(torch.bfloat16),
            attention_mask=attn.to(device),
            labels=labels.to(device),
        )
        return out.loss

    for ep in range(epochs):
        random.shuffle(train)
        opt.zero_grad()
        for i, rec in enumerate(train):
            (_loss(rec) / accum).backward()
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            if (i + 1) % 1000 == 0:
                print(
                    f"  [{cond} s{seed}] ep{ep} {i + 1}/{len(train)}",
                    flush=True,
                )
        opt.step()
        opt.zero_grad()

    model.eval()
    per_example = []
    with torch.no_grad():
        for rec in ev:
            per_example.append(float(_loss(rec)))
    del model, projector
    torch.cuda.empty_cache()
    return per_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "full"])
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--cond", default=",".join(CONDS))
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    if args.mode == "smoke":
        model_name = args.model or "Qwen/Qwen3-0.6B"
        seeds, epochs, ntr, nev = [0], 1, 40, 20
    else:
        model_name = args.model or "Qwen/Qwen3-1.7B"
        seeds = (
            [int(x) for x in args.seeds.split(",")]
            if args.seeds
            else [0, 1, 2]
        )
        epochs, ntr, nev = 2, N_TRAIN, N_EVAL
    conds = [c for c in args.cond.split(",") if c]
    for c in conds:
        if c not in CONDS:
            raise SystemExit(f"unknown condition {c!r}")

    train_all, ev_all = build_records()
    train_all, ev_all = train_all[:ntr], ev_all[:nev]
    print(
        f"records train={len(train_all)} eval={len(ev_all)} "
        f"model={model_name} conds={conds} seeds={seeds}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for cond in conds:
            target = OUT_DIR / f"{cond}-seed{seed}.json"
            if target.exists() and args.mode == "full":
                print(f"[skip] {target}", flush=True)
                continue
            per_example = run_condition(
                cond,
                seed,
                model_name,
                list(train_all),
                ev_all,
                epochs=epochs,
            )
            nll = sum(per_example) / max(len(per_example), 1)
            payload = {
                "cond": cond,
                "seed": seed,
                "model": model_name,
                "mode": args.mode,
                "mean_nll": nll,
                "per_example_nll": per_example,
                "eval_players": [r["player"] for r in ev_all],
            }
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(json.dumps(payload) + "\n")
            tmp.replace(target)
            print(f"RESULT cond={cond} seed={seed} nll={nll:.4f}", flush=True)

    done = {
        (c, s): json.loads((OUT_DIR / f"{c}-seed{s}.json").read_text())
        for c in CONDS
        for s in ([0, 1, 2] if args.mode == "full" else seeds)
        if (OUT_DIR / f"{c}-seed{s}.json").exists()
    }
    if args.mode == "full" and len(done) == 12:
        summary = {f"{c}-seed{s}": done[(c, s)]["mean_nll"] for c, s in done}
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
        print("\n==== G5 PERSONA LADDER (mean held-out NLL) ====")
        for s in (0, 1, 2):
            row = {c: done[(c, s)]["mean_nll"] for c in CONDS}
            print(
                f"seed {s}: none={row['none']:.4f} "
                f"static={row['static']:.4f} memory={row['memory']:.4f} "
                f"hidden={row['hidden']:.4f} | "
                f"static-none={row['static'] - row['none']:+.4f} "
                f"memory-static={row['memory'] - row['static']:+.4f} "
                f"hidden-memory={row['hidden'] - row['memory']:+.4f}"
            )


if __name__ == "__main__":
    main()
