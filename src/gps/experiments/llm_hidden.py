"""G2/G3: SFT an LLM with the player's state as a VERBAL note or HIDDEN prefix.

The LLM-native port of RQ6 (Milestone G). We fine-tune an open-weight LLM to
imitate a player's move / think-time by **completion NLL**, delivering that
player's *evolving* state through one of three channels and comparing held-out
NLL:

* ``none``   -- no state (the baseline).
* ``verbal`` -- the state as a natural-language note spliced into the prompt
  (what HumanLM / generative-agent simulators do).
* ``hidden`` -- the *same* trained latent as a soft-prompt **prefix** prepended
  to the input embeddings (:mod:`gps.policy.hidden_prefix`).

The headline (G3) is ``hidden`` < ``verbal`` *inside* the LLM -- the direct
comparison against a verbal latent. This dense completion-NLL objective is the
sharp probe (RL's match-reward was too sparse; ``results/slime_rl_llm.txt``).

What is CPU-testable here (and tested): the per-decision example assembly
(:func:`build_examples`) and the completion-NLL step assembly
(:func:`assemble_completion_step` -- input embeds, attention mask, and the
label mask that scores *only* the completion, prefix + prompt ignored). The
model forward + LoRA fit (:func:`run_hidden_sft_condition`) is the GPU-host
wiring; it composes the tested pieces and is run from a file
(``python -m gps.experiments.llm_hidden``) with the ``serve``/``train`` extras.
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.policy.board_native import BoardNativeBackbone
from gps.policy.hidden_prefix import prepend_prefix
from gps.policy.sglang_backbone import SGLangBackbone
from gps.train.base import Trajectory

#: The three injection conditions compared (G3 headline: hidden < verbal).
CHANNELS = ("none", "verbal", "hidden")


@dataclass
class HiddenSFTExample:
    """One (prompt, completion, latent) SFT example for a player's decision.

    ``latent`` is set only for the ``hidden`` channel (the soft-prompt vector);
    ``verbal`` folds the state into ``prompt`` and ``none`` omits it entirely.
    ``completion`` is what the LLM must produce (the played move, or the
    think-time), scored by NLL.
    """

    prompt: str
    completion: str
    latent: list[float] | None
    channel: str


def _as_float_list(v) -> list[float]:
    """A latent payload (torch tensor or sequence) -> a flat list of floats."""
    if hasattr(v, "detach"):  # torch tensor
        return [float(x) for x in v.detach().reshape(-1).tolist()]
    return [float(x) for x in v]


def build_examples(
    trajectory: Trajectory,
    injector,
    backbone: SGLangBackbone,
    *,
    channel: str,
    target: str = "move",
) -> list[HiddenSFTExample]:
    """Assemble per-decision SFT examples over one player's trajectory.

    Threads the injector's lifecycle (``initial_state`` -> ``render`` /
    ``update``) so the state at each step is *causal* (depends only on the
    player's past). For ``hidden`` we take the full evolving latent vector
    (``z.payload`` -- the rich channel RQ6 shows beats the anchored verbal
    dims); ``verbal`` renders the note into the prompt; ``none`` omits it.
    The prompt/completion formatting mirrors the scoring path
    (:meth:`SGLangBackbone.move_logprobs`: ``... "\\nMove:" + " <move>"``).
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    suffix, fmt = (
        ("\nMove:", lambda o: " " + o.move)
        if target == "move"
        else ("\nThink time (s):", lambda o: f" {(o.time_spent or 0.0):.1f}")
    )

    examples: list[HiddenSFTExample] = []
    z = injector.initial_state(trajectory.player_id)
    for dp, obs in zip(trajectory.decisions, trajectory.observations):
        latent = None
        if channel == "verbal":
            prompt = backbone.build_prompt(dp, injector.render(z, dp))
        else:
            prompt = backbone.build_prompt(dp, None)
            if channel == "hidden":
                latent = _as_float_list(z.payload)
        examples.append(
            HiddenSFTExample(
                prompt=prompt + suffix,
                completion=fmt(obs),
                latent=latent,
                channel=channel,
            )
        )
        z = injector.update(z, dp, obs)
    return examples


def assemble_completion_step(
    embed,
    prompt_ids,
    completion_ids,
    prefix_rows=None,
    ignore_index: int = -100,
):
    """Build ``(inputs_embeds, attention_mask, labels)`` for one NLL step.

    ``embed`` is the LLM's input-embedding lookup (``model.get_input_embeddings
    ()``); ``prompt_ids`` / ``completion_ids`` are 1-D token-id sequences;
    ``prefix_rows`` is an optional ``[n_prefix, hidden]`` soft prompt (the
    HIDDEN channel).

    Returns tensors ``[1, L, hidden]`` / ``[1, L]`` / ``[1, L]`` with
    ``L = n_prefix + len(prompt) + len(completion)``. **Only the completion
    positions carry labels**; the prefix and prompt are ``ignore_index``, so
    the loss is exactly the player's move/time completion NLL (the model's
    internal causal shift lines each label up with the position that predicts
    it). All positions are attended.
    """
    import torch

    p = torch.as_tensor(prompt_ids, dtype=torch.long)
    c = torch.as_tensor(completion_ids, dtype=torch.long)
    tok_embeds = embed(torch.cat([p, c]))  # [P+C, hidden]

    if prefix_rows is not None:
        seq = prepend_prefix(prefix_rows, tok_embeds)  # [n_prefix+P+C, hidden]
        n_prefix = int(prefix_rows.shape[0])
    else:
        seq = tok_embeds
        n_prefix = 0

    length = seq.shape[0]
    labels = torch.full((length,), ignore_index, dtype=torch.long)
    labels[n_prefix + p.shape[0] :] = c  # score only the completion tokens
    attention_mask = torch.ones(length, dtype=torch.long)
    return seq.unsqueeze(0), attention_mask.unsqueeze(0), labels.unsqueeze(0)


def run_hidden_sft_condition(  # pragma: no cover - GPU host
    model,
    tokenizer,
    examples: list[HiddenSFTExample],
    *,
    projector=None,
    train_frac: float = 0.7,
    epochs: int = 2,
    lr: float = 1e-4,
    device: str = "cuda",
) -> float:
    """Fit one condition by completion NLL; return held-out mean NLL (GPU).

    Composes the tested pieces: each example tokenizes prompt/completion,
    projects the latent to a soft prefix (``hidden`` only, via ``projector``),
    calls :func:`assemble_completion_step`, and backprops the model's
    ``labels`` loss into the projector (+ the model's LoRA/trainable params).
    Held-out examples (the per-player temporal tail) are scored, not trained --
    the strict future split. This is the ``none``/``verbal``/``hidden`` inner
    loop; the caller runs all three and reports ``hidden - verbal`` (G3).
    """
    import torch

    embed = model.get_input_embeddings()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if projector is not None:
        projector.to(device)
        trainable = list(projector.parameters()) + trainable
    opt = torch.optim.AdamW(trainable, lr=lr) if trainable else None

    n_train = max(1, int(round(train_frac * len(examples))))
    train, held = examples[:n_train], examples[n_train:]

    def _step(ex: HiddenSFTExample):
        pi = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        ci = tokenizer(ex.completion, add_special_tokens=False)["input_ids"]
        prefix = None
        if ex.channel == "hidden" and projector is not None:
            prefix = projector.project(
                torch.tensor(ex.latent, dtype=torch.float32, device=device)
            )
        emb, attn, labels = assemble_completion_step(embed, pi, ci, prefix)
        out = model(
            inputs_embeds=emb.to(device),
            attention_mask=attn.to(device),
            labels=labels.to(device),
        )
        return out.loss

    for _ in range(epochs):
        for ex in train:
            loss = _step(ex)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

    model.eval()
    with torch.no_grad():
        held_nll = [float(_step(ex)) for ex in held] if held else [0.0]
    model.train()
    return sum(held_nll) / len(held_nll)


def split_boundary(trajectory: Trajectory, train_frac: float = 0.7) -> int:
    """The per-player train/held boundary (reuses the E-C split contract)."""
    return BoardNativeBackbone.split_indices([trajectory], train_frac)[0]


def main() -> None:  # pragma: no cover - GPU entry point
    import argparse
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ap = argparse.ArgumentParser(description="G2/G3 hidden-vs-verbal LLM SFT")
    ap.add_argument("dataset")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--target", choices=("move", "time"), default="time")
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.parse_args()
    raise SystemExit(
        "GPU host: load the model + tokenizer (transformers, optional PEFT "
        "LoRA), build a HiddenPrefixProjector(latent_dim, model hidden_size), "
        "then per channel ('none','verbal','hidden') build_examples(...) and "
        "run_hidden_sft_condition(...); report hidden - verbal (G3). The "
        "example + completion-NLL assembly is unit-tested; wire the loop here."
    )


if __name__ == "__main__":
    main()
