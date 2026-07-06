# Milestone A — "Is the dynamic latent just history-conditioning?" (RESOLVED)

The **#1 desk-reject objection**: if a no-latent policy fed the *same* engineered
history features matches our evolving latent at equal capacity, the contribution
evaporates. Milestone A settles it. **Result: the evolving latent wins at equal
capacity, and the win is capacity-robust** (E-A1 on synthetic; E-C2/E-C3 on real
chess — see `documents/results_ec.md`).

## The claim + the equal-capacity 2×2

> An **evolving, per-individual latent** `z_t` predicts a player's moves/timing
> better than a **memoryless policy fed the identical history features**, at
> **equal inputs and equal capacity**.

`Simulator(injector, backbone)` factors cleanly, so we vary *evolving latent* ×
*history features* independently:

```
                     │ no history features   │ same history features
─────────────────────┼───────────────────────┼──────────────────────────
 no evolving latent  │ A. static/population   │ B. history-conditioned (CONTROL)
 evolving latent     │ (ill-posed)            │ D. PROPOSED (evolving latent)
```

- **A vs D** — "does modeling the session help at all?" (the easy win).
- **B vs D** — the desk-reject defense: identical features; the *only* difference
  is whether they are accumulated into an evolving state (D) or consumed
  instantaneously (B). **The headline number is D − B**, reported with parameter
  counts side by side.

"Equal inputs" is enforced *in code*: `gps.latent.structured.history_features(dp)`
is the single source of truth read by **both** arms (guarded by
`tests/test_phase0.py::test_phase0_history_uses_same_features_as_structured`).

## What the synthetic Phase-0 established (CPU)

`gps phase0` runs 4 arms (static | history | heuristic | oracle) over
tilt/time-pressure/fatigue/hysteresis players. **An *untrained* EMA heuristic
does NOT beat the memoryless control** (`>history` False on all mechanisms) — and
that is the point: the evolving-latent claim is **not free**, it must be *earned
by training* and *on data with genuine cross-step dynamics*. The
`HysteresisTiltPlayer` (a hidden leaky-loss integral, provably not reconstructable
from the instantaneous `history_features`) is the mechanism where a memoryless
reader is insufficient — so the D-vs-B test there is non-vacuous.

## E-A1 — the trained result (GPU, DONE, positive & hardened)

`gps train-ea1` trains arm D (`persist=True`) vs the memoryless twin arm B
(`persist=False`, *exactly* capacity-matched, 1159 params each) on the hysteretic
player, strict temporal split. Pooled bootstrap over **240 distinct players**
(5 seeds × 48):

- mean **D−B = −0.0060**, 95% CI **[−0.0081, −0.0040]**, P(D−B<0) = **1.000**,
  D wins **69%** of players (5/5 seeds negative, 4/5 individually significant).
- **Capacity-robust:** still significant at 3× B's params; only at **12× B's
  params** does it go marginal (−0.0025) while D *still* wins 65%. A memoryless
  control needs an order of magnitude more capacity merely to approach parity —
  not an artifact.
- Magnitude small (~0.4% rel. on ~1.49-nat move-NLL), expected on synthetic data
  where the memoryless control is strong by construction. *Caveat:* single-seed
  GPU (cudnn) CIs wobble — the pooled-over-players number is the trustworthy one.

**On real chess (E-C2/E-C3)** the same D-vs-B lands the headline (D−B ≈ −0.069 on
2013 blitz, P=1.00; survives B at 2× latent width; survives the future-*sessions*
split) — see `documents/results_ec.md`. Gate cleared.

## The decision rule (pre-registered) + cautions

Decision rule, recorded before looking at results:
- **D−B < 0, significant, concentrated in high-dynamics moments** → the latent
  earns its keep; proceed to the full headline. *(This is what happened.)*
- **D−B ≈ 0** → reshape the paper to "engineered history features suffice" (still
  publishable, different abstract); do not proceed to Go/population.
- **D−B < 0 but tiny and diffuse** → bounded phenomenon; workshop venue.
Significance is **bootstrap over players** (moves within a player are correlated);
report the per-player D−B distribution, since the thesis is per-individual.

Cautions that shaped the design: (1) the synthetic mechanism must be genuinely
*sequential* (hidden carry-over) or a memoryless control is trivially strong;
(2) match **capacity**, not just inputs — always print both param counts; (3)
don't cripple B with a thin feature set; (4) probe *presence* ≠ *use* (the causal
clamp is E-C4, not the recovery R²).

Two Phase-0 bring-up bugs, recorded so they don't recur (also in design.md §5):
aliased mutable session history (every decision saw the final history) and a
degenerate win model where nobody ever lost (post-loss tilt never fired). Both
fixed; guarded by tests.
