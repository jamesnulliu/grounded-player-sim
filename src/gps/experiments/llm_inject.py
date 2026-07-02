"""Frozen-LLM verbal-injection on real chess (the first *LLM-policy* test).

Does a verbal note about a player's current state (tilt / fatigue / momentum,
rendered from ``history_features``) help a **frozen** open-weight LLM predict
that player's *next* move? This is the LLM analog of the board-native E-C2 --
and the persona-prompt-style baseline (B5/B6): no training, just the state
spliced into the prompt.

Run from a **file** (``python -m gps.experiments.llm_inject``) -- sglang spawns
subprocesses that re-exec the main script, so ``python -c``/heredoc fails (the
child dies with ``FileNotFoundError: '<stdin>'``). Needs a GPU + the ``serve``
extra; set ``HF_HUB_OFFLINE=1`` for a cached model.

Finding (2026-06-29, 100 held-out decisions, Lichess 2013 blitz): the frozen
note is **unreliable** -- it helps a weak model (Qwen3-1.7B, ΔNLL −0.031) but
hurts a strong one (Qwen3-8B, +0.076). A frozen LLM never learned how the state
maps to *this player's* moves; the *trained* dynamic latent (the board-native
results) is what works. Motivates the `HIDDEN` soft-prompt + trained-injector
path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gps.latent.base import Injection, InjectionKind
from gps.latent.structured import history_features
from gps.policy.board_native import BoardNativeBackbone
from gps.train.base import TrajectoryDataset


def verbal_state_note(dp) -> str:
    """A short natural-language note on the player's behavioural state."""
    h = history_features(dp)
    bits = [
        "tilted after a recent loss"
        if h["post_loss"] > 0.5
        else "composed, no recent loss",
        "fatigued late in a long session"
        if h["fatigue"] > 0.5
        else "fresh, early in the session",
    ]
    if h["momentum"] > 0.3:
        bits.append("on a winning streak")
    elif h["momentum"] < -0.3:
        bits.append("on a losing streak")
    tr = dp.time_signal.time_remaining
    if tr is not None and tr < 30:
        bits.append("low on clock / under time pressure")
    return "Player state right now: " + "; ".join(bits) + "."


#: Irrelevant text of similar length to the state note -- the control that
#: separates "the LLM uses the state content" from "any extra tokens shift the
#: distribution".
CONTROL_NOTE = (
    "Background: the weather outside is mild today and the room is quiet."
)


@dataclass
class LLMInjectResult:
    n: int
    nll_no_injection: float
    nll_with_injection: float
    nll_control: float
    frac_helped: float
    model: str

    def summary(self) -> str:
        d = self.nll_with_injection - self.nll_no_injection
        dc = self.nll_control - self.nll_no_injection
        # The state note helps *beyond just adding tokens* iff it beats both
        # the no-injection and the irrelevant-control conditions.
        content = (
            "state content helps beyond token-count"
            if self.nll_with_injection
            < min(self.nll_no_injection, self.nll_control)
            else "no content effect (≈ control / no-injection)"
        )
        return (
            f"[LLM verbal-injection] {self.model}, n={self.n} held-out moves\n"
            f"        move-NLL: none={self.nll_no_injection:.4f} | "
            f"state-note={self.nll_with_injection:.4f} | "
            f"control(irrelevant)={self.nll_control:.4f}\n"
            f"        state effect={d:+.4f} (helps {self.frac_helped:.0%}) | "
            f"control effect={dc:+.4f}\n"
            f"        VERDICT: {content}"
        )


def run_llm_inject(
    dataset: TrajectoryDataset,
    *,
    model_path: str = "Qwen/Qwen3-8B",
    train_frac: float = 0.7,
    n_players: int = 4,
    max_decisions_per_player: int = 25,
) -> LLMInjectResult:
    """Score held-out move-NLL with vs without the verbal state note."""
    from gps.policy.sglang_backbone import SGLangBackbone

    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    bb = SGLangBackbone(model_path=model_path)
    no_inj, inj, ctrl = [], [], []

    def _nll(dp, actual, injection):
        p = bb.predict(dp, injection).moves.probs
        return -math.log(max(p.get(actual, 1e-9), 1e-9))

    try:
        for traj, sp in list(zip(dataset.trajectories, splits))[:n_players]:
            held = list(range(sp, len(traj.decisions)))[
                :max_decisions_per_player
            ]
            for t in held:
                dp = traj.decisions[t]
                actual = traj.observations[t].move
                if actual not in dp.legal_actions:
                    continue
                state = Injection(
                    kind=InjectionKind.VERBAL, text=verbal_state_note(dp)
                )
                control = Injection(
                    kind=InjectionKind.VERBAL, text=CONTROL_NOTE
                )
                no_inj.append(_nll(dp, actual, None))
                inj.append(_nll(dp, actual, state))
                ctrl.append(_nll(dp, actual, control))
    finally:
        bb.close()

    n = len(no_inj)
    helped = sum(1 for a, b in zip(inj, no_inj) if a < b) / max(n, 1)
    mean = lambda xs: sum(xs) / max(len(xs), 1)  # noqa: E731
    return LLMInjectResult(
        n=n,
        nll_no_injection=mean(no_inj),
        nll_with_injection=mean(inj),
        nll_control=mean(ctrl),
        frac_helped=helped,
        model=model_path,
    )


def main() -> None:  # pragma: no cover - GPU/sglang entry point
    import argparse
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from gps.data.store import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-players", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=25)
    args = ap.parse_args()
    res = run_llm_inject(
        load_dataset(args.dataset),
        model_path=args.model,
        n_players=args.n_players,
        max_decisions_per_player=args.max_decisions,
    )
    print(res.summary())


if __name__ == "__main__":
    main()
