"""Open-weight LLM policy backbone served by sglang (GPU).

This is the *core* policy in the headline configuration: an open-weight LLM
(e.g. Qwen3-8B) is the agent, and the dynamic latent is injected either as
verbal memory in the prompt (``VERBAL``) or as soft-prompt/prefix embeddings
(``HIDDEN``). The move distribution is read from constrained-decoding token
logprobs over the legal-move strings; think-time is read from a small timing
head or a structured field the model emits.

GPU-only. Everything heavy (sglang, torch, the served engine) is imported
lazily inside ``__init__`` / ``_engine`` so that importing this module on a
CPU box is free. The body below is intentionally a thin, documented stub:
the interface is final, but the on-GPU wiring (engine launch, constrained
decoding grammar, soft-prompt injection) is filled in on the training host.

References
----------
* sglang server + ``Engine`` API for offline batched logprob scoring.
* Constrained / regex decoding to restrict generation to legal moves.
"""

from __future__ import annotations

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.policy.base import PolicyBackbone
from gps.prediction import Prediction


class SGLangBackbone(PolicyBackbone):
    """LLM policy backed by a local sglang engine."""

    # Verbal works out of the box; hidden requires soft-prompt support in the
    # serving path (prefix embeddings), enabled per-deployment.
    accepts = (InjectionKind.VERBAL, InjectionKind.HIDDEN)

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-8B",
        *,
        max_legal_tokens: int = 8,
        enable_hidden: bool = False,
        engine_kwargs: dict | None = None,
    ) -> None:
        self.model_path = model_path
        self.max_legal_tokens = max_legal_tokens
        self.enable_hidden = enable_hidden
        self.engine_kwargs = engine_kwargs or {}
        self._eng = None  # lazily launched
        if not enable_hidden:
            self.accepts = (InjectionKind.VERBAL,)

    def _engine(self):
        """Launch (once) and return the sglang engine."""
        if self._eng is None:
            try:
                import sglang as sgl  # noqa: F401  (lazy, GPU-only)
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "sglang is required for SGLangBackbone; install the "
                    "'serve' extra on a GPU host: pip install '.[serve]'"
                ) from e
            # TODO(gpu): launch sgl.Engine(model_path=self.model_path, ...)
            # with logprob return enabled and a constrained-decoding grammar
            # restricting output to dp.legal_actions.
            raise NotImplementedError(
                "sglang engine launch is wired on the GPU host; the "
                "interface is final, the body is a documented stub."
            )
        return self._eng

    def predict(
        self, dp: DecisionPoint, injection: Injection | None = None
    ) -> Prediction:
        # On GPU this will:
        #   1. Build the prompt from dp.state (FEN/SGF + move list + context).
        #   2. Splice injection.text (VERBAL) or attach injection.vector as a
        #      prefix embedding (HIDDEN).
        #   3. Score each legal move's token(s) -> logprobs -> MoveDistribution
        #      normalised over dp.legal_actions.
        #   4. Read/derive think-time -> TimingPrediction.
        self._engine()  # raises the informative NotImplementedError for now
        raise NotImplementedError

    def build_prompt(
        self, dp: DecisionPoint, injection: Injection | None
    ) -> str:
        """Construct the model prompt. Pure-Python; unit-testable on CPU.

        Kept separate from :meth:`predict` precisely so prompt construction
        (the part with no GPU dependency) can be tested without a served
        model.
        """
        lines = [
            "You are simulating a specific human player's next move.",
            f"Game: {dp.game.value}",
            f"Player: {dp.player_id}",
            f"Position: {dp.state}",
            f"Legal moves: {', '.join(dp.legal_actions)}",
        ]
        if dp.time_signal.time_remaining is not None:
            lines.append(
                f"Time remaining: {dp.time_signal.time_remaining:.1f}s"
            )
        if injection is not None and injection.kind is InjectionKind.VERBAL:
            lines.append(injection.text or "")
        lines.append("Respond with exactly one legal move.")
        return "\n".join(lines)

    def close(self) -> None:
        if self._eng is not None:
            # TODO(gpu): shut down the served engine.
            self._eng = None

    @property
    def name(self) -> str:
        return f"SGLangBackbone({self.model_path})"
