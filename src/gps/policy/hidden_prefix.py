"""Latent vector -> LLM soft-prompt (prefix) embeddings: the HIDDEN channel.

The open-weight analogue of splicing a verbal note into the prompt. A trained
projector turns the injector's evolving latent ``z_t`` into ``n_prefix``
soft-prompt embedding rows that are prepended to the LLM's *input* embeddings,
so the **same** per-individual state can reach an LLM either as ``VERBAL`` text
or as a ``HIDDEN`` prefix. That is the channel-only contrast RQ6 runs
board-native (``documents/results_ec.md``) and that **Milestone G** ports
*into* the LLM -- the direct comparison against HumanLM's *verbal* latent.

Design (mirrors :class:`gps.latent.neural.NeuralInjector` so the two train the
same way):

* **Trainable end-to-end** with the injector under the LLM SFT objective (dense
  held-out completion NLL). ``parameters()`` exposes the projection weights so
  the trainer fits ``phi`` (injector) and the projector jointly, exactly as the
  board-native hidden readout is fit.
* **Lazy torch.** Nothing is imported until a method needs it, so this module
  is importable on a CPU-only box; the network is built on first use. CPU is
  enough to construct, shape-check, and unit-test it -- a GPU only speeds up
  training.
* **Deterministic construction.** Parameters seed off a *forked* RNG, so
  building a projector does not perturb a surrounding training run's global
  torch RNG (again matching ``NeuralInjector``).
"""

from __future__ import annotations


class HiddenPrefixProjector:
    """Projects a latent ``z_t`` to ``[n_prefix, hidden_size]`` prefix rows.

    Parameters
    ----------
    latent_dim:
        Width of the injector's hidden vector (the ``HIDDEN`` injection's
        ``vector`` length) -- i.e. ``NeuralInjector.latent_dim``.
    hidden_size:
        The served LLM's embedding width (e.g. 4096 for Qwen3-8B). On a GPU
        host this is read from the model config; passed explicitly here so the
        projector is constructible and testable on CPU.
    n_prefix:
        Number of soft-prompt rows to prepend (the prefix length). ``1`` is the
        cheapest LATTE-style single-soft-token injection; a handful gives the
        state more room without touching the prompt text.
    seed:
        Seeds parameter initialisation via a forked RNG (reproducible,
        side-effect free w.r.t. the global torch RNG).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        n_prefix: int = 1,
        seed: int = 0,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.n_prefix = int(n_prefix)
        self.seed = int(seed)
        self._net = None  # built lazily on the first call needing torch

    # --- network construction ------------------------------------------
    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for HiddenPrefixProjector; install the "
                "'train' extra: pip install '.[train]' (CPU is fine for "
                "construction and unit tests; a GPU only speeds up training)."
            ) from e

        class _Net(nn.Module):
            def __init__(self, latent_dim, hidden_size, n_prefix):
                super().__init__()
                self.n_prefix = n_prefix
                self.hidden_size = hidden_size
                self.proj = nn.Linear(latent_dim, n_prefix * hidden_size)

            def forward(self, z):
                # [..., latent_dim] -> [..., n_prefix, hidden_size]
                out = self.proj(z)
                return out.reshape(
                    *z.shape[:-1], self.n_prefix, self.hidden_size
                )

        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = _Net(self.latent_dim, self.hidden_size, self.n_prefix)
        self._net = net
        return net

    def parameters(self):
        """torch parameters of the projector (so the SFT trainer fits them)."""
        return self._build().parameters()

    def to(self, device):
        """Move the projector to a torch device (returns self, to chain)."""
        self._build().to(device)
        return self

    # --- projection -----------------------------------------------------
    def project(self, z):
        """Latent ``z`` -> soft-prompt embedding rows.

        ``z`` is ``[B, latent_dim]`` (a batch) or ``[latent_dim]`` (one
        vector), a tensor or a plain list. Returns ``[B, n_prefix,
        hidden_size]`` (or ``[n_prefix, hidden_size]`` for a single vector) --
        the row block prepended to the LLM's token embeddings before the
        position's move/time continuation.
        """
        import torch

        net = self._build()
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float32)
        z = z.to(dtype=torch.float32)
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent width {z.shape[-1]} != projector latent_dim "
                f"{self.latent_dim}"
            )
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        out = net(z)
        return out.squeeze(0) if single else out

    # --- capacity bookkeeping ------------------------------------------
    def param_report(self) -> dict[str, object]:
        """Parameter budget, for the equal-capacity claim vs the verbal note.

        RQ6 in the LLM (Milestone G) compares the *same* trained state given
        as this hidden prefix vs a verbal note; reporting both channels'
        parameter counts keeps that a channel contrast, not a capacity one.
        """
        net = self._build()
        return {
            "projector": type(self).__name__,
            "latent_dim": self.latent_dim,
            "hidden_size": self.hidden_size,
            "n_prefix": self.n_prefix,
            "n_parameters": int(sum(p.numel() for p in net.parameters())),
        }

    @property
    def name(self) -> str:
        return (
            f"HiddenPrefixProjector(latent_dim={self.latent_dim}, "
            f"hidden_size={self.hidden_size}, n_prefix={self.n_prefix})"
        )
