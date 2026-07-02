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
        self._tok = None
        if not enable_hidden:
            self.accepts = (InjectionKind.VERBAL,)

    def _engine(self):
        """Launch (once) and return the sglang engine + tokenizer.

        NOTE: sglang spawns subprocesses that re-exec the *main* script, so the
        process that launches the engine must be a real ``.py`` file, not a
        ``-c``/stdin heredoc (the child fails with ``FileNotFoundError:
        '<stdin>'``).
        """
        if self._eng is None:
            try:
                import sglang as sgl
                from transformers import AutoTokenizer
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "sglang + transformers are required for SGLangBackbone; "
                    "install the 'serve' extra on a GPU host: "
                    "pip install '.[serve]'"
                ) from e
            kw = dict(
                model_path=self.model_path,
                dtype="bfloat16",
                mem_fraction_static=0.7,
                disable_cuda_graph=True,
                log_level="error",
            )
            kw.update(self.engine_kwargs)
            self._eng = sgl.Engine(**kw)
            self._tok = AutoTokenizer.from_pretrained(self.model_path)
        return self._eng

    def _n_tokens(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def move_logprobs(
        self, dp: DecisionPoint, injection: Injection | None = None
    ) -> dict[str, float]:
        """Per-legal-move total token log-prob under the LLM (un-normalised).

        Scores each legal move as a continuation of the prompt and sums its
        token log-probs (read from sglang ``input_token_logprobs``). The base
        prompt is identical across a position's moves, so its token length
        marks where the move tokens begin.
        """
        eng = self._engine()
        base = self.build_prompt(dp, injection) + "\nMove: "
        base_len = self._n_tokens(base)
        prompts = [base + m for m in dp.legal_actions]
        outs = eng.generate(
            prompts,
            sampling_params={"max_new_tokens": 0, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=0,
        )
        scores: dict[str, float] = {}
        for move, out in zip(dp.legal_actions, outs):
            itl = out["meta_info"]["input_token_logprobs"]
            move_lps = [lp for lp, _, _ in itl[base_len:] if lp is not None]
            scores[move] = float(sum(move_lps))
        return scores

    def predict(
        self, dp: DecisionPoint, injection: Injection | None = None
    ) -> Prediction:
        """Move distribution = softmax over legal-move log-probs.

        (Timing from an LLM needs a structured field / timing head; left as a
        flat default here -- the move distribution is what E-C2/RQ6 score.)
        """
        import math

        from gps.prediction import MoveDistribution, TimingPrediction

        scores = self.move_logprobs(dp, injection)
        mx = max(scores.values())
        exp = {m: math.exp(s - mx) for m, s in scores.items()}
        z = sum(exp.values()) or 1.0
        probs = {m: v / z for m, v in exp.items()}
        return Prediction(
            moves=MoveDistribution(probs=probs),
            timing=TimingPrediction(mu=0.0, sigma=1.0),
        )

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
            try:
                self._eng.shutdown()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._eng = None
            self._tok = None

    @property
    def name(self) -> str:
        return f"SGLangBackbone({self.model_path})"
