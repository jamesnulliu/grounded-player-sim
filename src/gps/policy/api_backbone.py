"""Closed-source LLM policy backbone over an HTTP API.

This is the RQ4 foil: a hosted model prompted to "play as player X" from a
static user profile / persona. It can only consume a *verbal* injection (we
cannot attach soft-prompt vectors to a closed API), which is exactly the
point of the contrast -- the strongest a closed model gets is a verbal
profile, while the open-weight core can also receive a learned hidden latent.

The move distribution is approximated from the API's per-token logprobs when
the provider exposes them; otherwise it falls back to multi-sample frequency
estimation (documented, and flagged in the prediction meta). Network-only;
the provider SDKs are imported lazily.
"""

from __future__ import annotations

import os

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.policy.base import PolicyBackbone
from gps.prediction import Prediction


class APIBackbone(PolicyBackbone):
    """Hosted closed-source LLM, verbal injection only."""

    accepts = (InjectionKind.VERBAL,)

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        *,
        logprobs: bool = True,
        n_samples: int = 16,
        api_key_env: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.logprobs = logprobs
        self.n_samples = n_samples
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "openai SDK required; install the 'api' extra: "
                    "pip install '.[api]'"
                ) from e
            key = os.environ.get(self.api_key_env or "OPENAI_API_KEY")
            self._client = OpenAI(api_key=key)
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "anthropic SDK required; install the 'api' extra: "
                    "pip install '.[api]'"
                ) from e
            key = os.environ.get(self.api_key_env or "ANTHROPIC_API_KEY")
            self._client = Anthropic(api_key=key)
        else:
            raise ValueError(f"unknown provider: {self.provider}")
        return self._client

    def predict(
        self, dp: DecisionPoint, injection: Injection | None = None
    ) -> Prediction:
        # On a networked host this will call the provider, parse a legal
        # move + (optional) think-time, and build a MoveDistribution from
        # logprobs or sampled frequencies. Stubbed until run with creds.
        self._get_client()
        raise NotImplementedError(
            "APIBackbone.predict runs against a live provider; the request/"
            "parse path is finalised on a networked host."
        )

    def build_messages(
        self, dp: DecisionPoint, injection: Injection | None
    ) -> list[dict]:
        """Build chat messages. Pure-Python; testable without network."""
        profile = ""
        if injection is not None and injection.kind is InjectionKind.VERBAL:
            profile = injection.text or ""
        system = (
            "You role-play as a specific human game player and predict the "
            "single move they would make next. " + profile
        )
        user = (
            f"Game: {dp.game.value}\nPlayer: {dp.player_id}\n"
            f"Position: {dp.state}\n"
            f"Legal moves: {', '.join(dp.legal_actions)}\n"
            "Reply with one legal move only."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @property
    def name(self) -> str:
        return f"APIBackbone({self.provider}:{self.model})"
