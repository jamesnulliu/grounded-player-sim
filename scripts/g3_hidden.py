"""G3: hidden-vs-verbal state injection INSIDE the LLM (Milestone G headline).

Three conditions, ONE custom completion-NLL loop (fair: same optim, same LoRA):
  none    -- position only.
  verbal  -- position + the state as an English note (what HumanLM et al. do).
  hidden  -- position + the SAME state as a trained soft-prompt PREFIX
             (gps.policy.hidden_prefix.HiddenPrefixProjector), text identical
             to `none`.

All three are derived from the existing chess_twithstate_*.jsonl (which carry
the FEN + the verbal state line), so they are perfectly aligned: `none`/`hidden`
strip the state line, `verbal` keeps it, and the hidden latent is parsed FROM
that same line -- so hidden vs verbal is a pure channel contrast on identical
state. (v1: state-features-as-soft-prompt. v2 = the full trained evolving latent
via joint injector training -- the richer board-native RQ6 form -- is follow-up.)

Metric: held-out mean completion NLL. hidden < verbal => the trained hidden
channel beats the verbal note in the LLM (the direct comparison vs HumanLM).

Usage: g3_hidden.py [smoke|full] [model]
"""

import json
import os
import random
import re
import sys

os.environ.setdefault("HF_HOME", "/home1/liuyanch/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gps.experiments.llm_hidden import assemble_completion_step  # noqa: E402
from gps.policy.hidden_prefix import HiddenPrefixProjector  # noqa: E402

GPS = "/tmp/gps_slime"
LATENT_DIM = 5  # [post_loss, fatigue, losing_streak, low_clock, clock_norm]


def parse_state(prompt_text: str) -> list[float]:
    """The verbal 'Player state: ...' line -> the same info as a vector."""
    m = re.search(r"Player state: (.*)", prompt_text)
    s = m.group(1) if m else ""
    low = 1.0 if "LOW ON CLOCK" in s else 0.0
    mclk = re.search(r"about (\d+)s left", s)
    clock_norm = (
        min(int(mclk.group(1)), 600) / 600.0
        if mclk
        else (0.02 if low else 0.5)
    )
    return [
        1.0 if "recently lost" in s else 0.0,
        1.0 if "deep in a long session" in s else 0.0,
        1.0 if "on a losing streak" in s else 0.0,
        low,
        clock_norm,
    ]


def strip_state(prompt_text: str) -> str:
    """Drop the 'Player state: ...' line -> the none/hidden prompt text."""
    return "\n".join(
        ln for ln in prompt_text.split("\n") if not ln.startswith("Player st")
    )


def load_records(split: str) -> list[dict]:
    recs = [
        json.loads(ln)
        for ln in open(f"{GPS}/chess_twithstate_{split}.jsonl")
    ]
    out = []
    for r in recs:
        vp = r["prompt"]
        out.append(
            {
                "base": strip_state(vp),
                "verbal": vp,
                "latent": parse_state(vp),
                "label": r["label"],
            }
        )
    return out


def run_condition(
    cond, seed, model_name, train, ev, *,
    n_prefix=4, epochs=2, lr=1e-4, accum=16, max_len=384, device="cuda",
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
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
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
        # Near-zero init: the soft prefix starts ~0 (a mild perturbation, not
        # random garbage before the chat tokens) and LEARNS to inject signal.
        net = projector._build()
        with torch.no_grad():
            net.proj.weight.mul_(0.02)
            net.proj.bias.zero_()
        extra = list(projector.parameters())
    params = [p for p in model.parameters() if p.requires_grad] + extra
    opt = torch.optim.AdamW(params, lr=lr)

    def _prompt_ids(prompt):
        # Qwen3 is instruction-tuned: use the chat template (as the published
        # TRL probe did) so the model actually engages the natural-language
        # state note. Raw text under-uses it -> the verbal effect vanishes.
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        return tok(text, add_special_tokens=False)["input_ids"]

    def _loss(r):
        prompt = r["base"] if cond in ("none", "hidden") else r["verbal"]
        pid = _prompt_ids(prompt)
        cid = tok(" " + r["label"], add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > max_len:
            pid = pid[-(max_len - len(cid)):]
        prefix = None
        if cond == "hidden":
            z = torch.tensor(r["latent"], dtype=torch.float32, device=device)
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
        for i, r in enumerate(train):
            (_loss(r) / accum).backward()
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            if (i + 1) % 1000 == 0:
                print(f"  [{cond} s{seed}] ep{ep} {i + 1}/{len(train)}",
                      flush=True)
        opt.step()
        opt.zero_grad()

    model.eval()
    tot = 0.0
    with torch.no_grad():
        for r in ev:
            tot += float(_loss(r))
    nll = tot / max(len(ev), 1)
    del model, projector
    torch.cuda.empty_cache()
    return nll


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-0.6B"
        seeds, epochs, ntr, nev = [0], 1, 40, 20
    else:
        model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
        seeds, epochs, ntr, nev = [0, 1, 2], 2, 5000, 1000
        if len(sys.argv) > 3:  # e.g. "full <model> 0" runs seed 0 only
            seeds = [int(x) for x in sys.argv[3].split(",")]

    train_all = load_records("train")
    ev_all = load_records("eval")
    print(f"loaded train={len(train_all)} eval={len(ev_all)} | model={model_name}"
          f" mode={mode}", flush=True)

    rows = {}
    for seed in seeds:
        for cond in ("none", "verbal", "hidden"):
            tr = train_all[:ntr]
            ev = ev_all[:nev]
            nll = run_condition(
                cond, seed, model_name, list(tr), ev, epochs=epochs
            )
            rows[(seed, cond)] = nll
            print(f"RESULT seed={seed} cond={cond} nll={nll:.4f}", flush=True)

    print("\n==== G3 SUMMARY (held-out completion NLL; lower=better) ====")
    for seed in seeds:
        n, v, h = (rows[(seed, c)] for c in ("none", "verbal", "hidden"))
        print(
            f"seed {seed}: none={n:.4f} verbal={v:.4f} hidden={h:.4f} | "
            f"verbal-none={v - n:+.4f} hidden-none={h - n:+.4f} "
            f"hidden-verbal={h - v:+.4f}"
        )
    print("HEADLINE: hidden-verbal < 0 across seeds => hidden channel wins.",
          flush=True)


if __name__ == "__main__":
    main()
