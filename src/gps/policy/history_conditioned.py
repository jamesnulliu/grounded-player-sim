"""History-conditioned, no-latent backbone -- the Milestone-A control.

This is the single most important baseline in the project (TODO.md Milestone
A; design.md section 6): it answers the #1 desk-reject objection,

    "Isn't the dynamic latent just an expressive history-conditioned policy?"

The construction holds *inputs and capacity equal* and removes exactly one
thing -- the evolving latent's inductive bias:

* **Same inputs.** It ingests the *same* engineered history features the
  latent injector sees, via the shared
  :func:`gps.latent.structured.history_features` (recent results, clock /
  time-pressure, session depth, momentum). No information advantage either
  way.
* **No evolving latent.** Those features are concatenated to the board
  encoding and fed straight to the move + timing heads. There is no recurrent
  / state-space state carried across the trajectory, no ``z_t`` update -- the
  policy is a feed-forward function of (position, instantaneous history
  features).
* **Matched capacity.** The feature-fusion MLP is sized to roughly match the
  parameter budget the latent injector adds on top of the backbone, so a win
  for the latent model cannot be explained by "more parameters." Record the
  parameter counts of both arms and report them (see :meth:`param_report`).

If the proposed dynamic latent model does **not** beat this control at equal
inputs and capacity, the structured/evolving latent does not earn its keep,
and that is a finding to surface honestly rather than bury -- it reshapes the
paper toward "engineered history features suffice."

Because it consumes the latent *as features* rather than as an injected
state, this backbone advertises ``accepts = ()`` and is driven with
``injection=None``; the history features are pulled from ``dp`` inside
``predict``. That keeps the comparison clean: the *proposed* arm is
``Simulator(neural_injector, backbone)`` and this *control* arm is
``Simulator(None, HistoryConditionedBackbone(...))``.

GPU/torch-bound, lazily imported, mirroring the other concrete backbones so
this module stays importable on a CPU-only laptop.
"""

from __future__ import annotations

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.latent.structured import DIMENSIONS, history_features
from gps.policy.base import PolicyBackbone
from gps.prediction import Prediction


class HistoryConditionedBackbone(PolicyBackbone):
    """No-latent policy whose head ingests raw engineered history features.

    Parameters
    ----------
    backbone_kind:
        Which trunk to fuse the history features onto -- ``"board_native"``
        (a Maia/KataGo-style CNN trunk, the cleanest capacity match to the
        proposed board-native + latent arm) or ``"llm"`` (an open-weight LLM
        whose prompt carries the verbalized features). The control should use
        the *same* trunk family as the proposed model it is the foil for.
    feature_names:
        The engineered history feature keys to consume, in fixed order.
        Defaults to the anchored ``DIMENSIONS`` so the inputs are byte-for-
        byte the ones the structured/neural injector smooths into ``z_t``.
    hidden_dim:
        Width of the feature-fusion MLP, tuned to match the latent injector's
        added parameter budget (see :meth:`param_report`).
    """

    # Consumes history as *features pulled from dp*, not as an injected
    # latent. Empty `accepts` => the simulator hands it `injection=None`.
    accepts: tuple[InjectionKind, ...] = ()

    def __init__(
        self,
        backbone_kind: str = "board_native",
        feature_names: tuple[str, ...] = DIMENSIONS,
        hidden_dim: int = 64,
    ) -> None:
        if backbone_kind not in ("board_native", "llm"):
            raise ValueError(
                "backbone_kind must be 'board_native' or 'llm', "
                f"got {backbone_kind!r}"
            )
        self.backbone_kind = backbone_kind
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self._net = None  # lazily built on the GPU host

    def feature_vector(self, dp: DecisionPoint) -> list[float]:
        """The history feature row this backbone conditions on at ``dp``.

        Exposed (and unit-tested on CPU) so experiments can assert the
        control truly sees the same inputs as the latent injector. This is
        the only thing the control knows about session history.
        """
        feats = history_features(dp)
        return [feats[name] for name in self.feature_names]

    def predict(
        self,
        dp: DecisionPoint,
        injection: Injection | None = None,
    ) -> Prediction:
        # The control never receives an injected latent; if one is passed it
        # is ignored on purpose (that is the whole point of the ablation).
        _ = injection
        feats = self.feature_vector(dp)  # noqa: F841 (used by GPU forward)
        self._network()
        # GPU forward, finished on the training host:
        #   x = trunk(encode(dp.state))                 # board / LLM trunk
        #   h = relu(W1 @ feats + b1)                   # feature-fusion MLP
        #   move_logits = move_head(concat(x, h))       # FiLM or concat
        #   move_logits = mask_to_legal(move_logits, dp.legal_actions)
        #   mu, sigma   = timing_head(concat(x, h))
        #   return Prediction(MoveDistribution(softmax(move_logits)),
        #                     TimingPrediction(mu, sigma))
        raise NotImplementedError(
            "HistoryConditionedBackbone.predict binds to a torch trunk + "
            "feature-fusion head on the GPU host; the feature contract "
            "(feature_vector) and capacity report are implemented and "
            "CPU-testable. See documents/milestone_a.md."
        )

    def _network(self):
        if self._net is not None:  # pragma: no cover - GPU-only path
            return self._net
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torch required for HistoryConditionedBackbone; install on "
                "a GPU host: pip install '.[serve]' (board_native) or "
                "'.[serve]' for the LLM trunk."
            ) from e
        # TODO(gpu): build trunk(backbone_kind) + feature-fusion MLP
        #            (hidden_dim) + move head + log-normal timing head.
        raise NotImplementedError(
            "network construction binds to the chosen trunk on the GPU host."
        )

    def param_report(self) -> dict[str, object]:
        """Capacity bookkeeping for the equal-capacity claim.

        Returns the configuration that determines this control's parameter
        budget so an experiment can assert it is matched to the proposed
        latent arm and *report both counts in the paper*. The concrete int
        counts are filled once the network is built on GPU.
        """
        return {
            "backbone_kind": self.backbone_kind,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "hidden_dim": self.hidden_dim,
            "fusion_params_formula": (
                "n_features*hidden_dim + hidden_dim "
                "(+ trunk and head params, shared with the proposed arm)"
            ),
        }

    @property
    def name(self) -> str:
        return f"HistoryConditionedBackbone(trunk={self.backbone_kind})"
