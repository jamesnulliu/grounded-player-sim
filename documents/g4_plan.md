# G4 — add-on value over a *released* SOTA (experiment spec)

*The highest-ROI experiment before a competitive submission (paper-readiness
audit, 2026-07-05). Converts the "weak from-scratch backbone" liability into the
headline: even a released SOTA human-chess model's own timing/move prediction is
improved by adding our per-individual evolving latent. Builds on the existing
E-C6 harness (`run_timing_vs_aggregate`, `TimingVsAggregate`, `_b4_features` in
`src/gps/experiments/ec.py`).*

## Claim under test

> On held-out future decisions, `(SOTA + z) − SOTA < 0`: the per-individual
> evolving latent `z_t` significantly reduces think-time NLL (and, secondarily,
> move NLL) **over a released, near-SOTA human-chess model's own prediction** —
> not over a hand-built proxy.

This is the exact question E-C6 already answers for a *reconstructed* Elo+clock+
complexity baseline (`(B4+z)−B4 = −0.0315 / −0.0266`, both P=1.00, B4 Spearman
0.371 / 0.398 ≈ ChessMimic's 0.41; `results/posaware_pooled.txt`). G4 replaces the
proxy with a *released* model to preempt the "your baseline is a strawman"
rebuttal — the strongest single move for main-track credibility.

## Why this is the right experiment (and what it defends)

- Kills reviewer objection #1: "results are on a weak from-scratch backbone."
  If a released model that *beats Maia* still gains from `z`, backbone strength is
  no longer a confound — it becomes the point.
- The E-C6 add-on machinery is done; the only new work is producing the released
  model's per-move prediction as a feature/offset. Low marginal cost, high payoff.
- Structurally aligned with our timing head, which reads *only* the latent — so
  "add value over an external predictor" is the natural framing, not "replace it."

## Release status — VERIFIED (2026-07-05 web sweep)

Availability is now resolved, so G4 is **unblocked without ChessMimic**:

| Tier | External model | Channel | Release status | Notes |
|------|----------------|---------|----------------|-------|
| **A (primary, timing)** | **Allie** (arXiv:2410.03893, ICLR'25) | timing | **PUBLIC** — github.com/ippolito-cmu/allie + HF `novachess/novachess-engine` | Decoder-only transformer that explicitly models **pondering times** (think-time) + resignations; a real near-SOTA human model, not a proxy |
| **B (move)** | **Maia-2 / Maia-3** (CSSLab) | move | **PUBLIC** — `maia2` / `maia3`, `from_pretrained` (HF checkpoints) | Human-move SOTA (Maia-3); Maia-2 trained on Lichess rapid *with clock info* |
| **C (move, optional)** | Elo-Disentangled (arXiv:2606.25176) | move | **unverified** — check for a repo | Beats Maia-3 on move NLL; use only if released, else Maia-3 is the anchor |
| ~~ChessMimic clock head~~ | ChessMimic (arXiv:2606.04473) | timing | **NOT CONFIRMED** — two sweeps found no public release | Do **not** depend on it; its described 3-transformer clock head could be reimplemented if a reviewer insists, but Allie removes the need |

**Decision:** anchor **G4-timing on Allie** (public, models pondering-time) and
**G4-move on Maia-3** (public). Keep the reproduced Elo+clock+complexity proxy
(E-C6, Spearman 0.39) as the belt-and-braces baseline. Do **not** silently fall
back to the linear proxy and call it SOTA. Before caching, confirm each model's
exact I/O (Allie: per-position pondering-time output; Maia: move logits/vocab).

## Design (drops into the existing harness)

`run_timing_vs_aggregate` already: trains arm D (log-normal timing head on the
evolving latent), fits `B4` (log-normal whose `mu` is least-squares over
`_b4_features(dp, position_aware)`), forms `B4+z` (same features **plus** the
per-step latent), and bootstraps `(B4+z) − B4` over players. G4 changes exactly
one thing: **what goes into the baseline `mu`.**

1. **Precompute the released model's per-decision think-time prediction**
   `t_hat_ext(dp)` for every decision in each cohort (offline, cached to the
   dataset like `EngineReference`). For a model that emits a distribution, take
   its predicted `mu` (log-seconds); for a point estimate, use `log(t_hat+1)`.
2. **Add a hook to `_b4_features`** (e.g. `external_pred: float | None`) so the
   baseline `mu` regresses on `[Elo, move#, log(time_remaining), complexity,
   t_hat_ext]`. This makes the baseline *at least as strong as* the released model
   plus our aggregate features — a conservative, hard-to-beat B.
   - Also run a **pure-external** variant: `mu = t_hat_ext` directly (no
     re-fit), so the comparison is literally "released model" vs "released model
     + z", with no chance we weakened it by re-fitting.
3. **`B4+z`** appends the per-step latent to whichever baseline feature set (2)
   uses. Unchanged from E-C6.
4. **Score** on the same strict session/future split; bootstrap `(B4+z) − B4`
   over players (the independent unit). Report per-player Pearson/Spearman for
   B, B+z, and the released model alone (sanity: our reproduced/loaded B should
   land near the paper's reported r≈0.41).

Run: 4 clocked cohorts (2017/2019/2021/2023) × 5 seeds, exactly like the Tier-1
sweep, so the result inherits the same era-generality and power. Log to W&B
`gps-g4-sota-addon` (`WANDB_ENTITY=jamesnulliu-university-of-southern-california`).

## Move version (secondary, honest)

Repeat with a released *move* model (Maia-3) as the baseline logits and test
`(SOTA_move + z) − SOTA_move` on held-out move NLL. **Pre-register the likely
outcome: small or null**, consistent with the board-native move near-null and the
full-param LLM's graded ~−0.007. Reporting a near-null here is *good* — it's the
"when-not-what" asymmetry surviving against a genuinely strong move model, which
is a stronger statement than a null against our weak trunk. Frame it that way.

## Decision rule (pre-registered)

- **Primary (timing, Allie):** `(SOTA+z) − SOTA < 0`, 95% bootstrap CI excludes
  0, in ≥ all-but-one cohort × the pooled result → **claim: the evolving latent
  adds significant think-time value over released SOTA.** This becomes the headline
  table row and retires the weak-backbone caveat.
- **If null:** report honestly. A null would mean the released clock model already
  captures the per-individual timing residual — a real, publishable finding that
  *bounds* our contribution (and would redirect the paper toward the
  heterogeneity-recovery / individualization framing, which does not depend on G4).
- **Move (Maia-3):** near-null expected and acceptable; report as the asymmetry
  holding against a strong move model.

## Effort / resources

- Model loading + offline `t_hat_ext` caching: the only real new code. Fits on 1
  GPU (these are ≤ hundreds of M params); inference-only, no training of the
  external model.
- `_b4_features` hook + a pure-external variant: ~a day.
- Sweep reuses `scripts/tier1_sweep.sh` structure; ~the same wall-clock as the
  existing posaware sweep (10 runs ≈ minutes–hours on 2×A100).

## Risks / caveats to write down

- **Release may not exist / be loadable** — mitigated by the tier ladder; Allie is
  a guaranteed-public timing floor.
- **Encoding/vocabulary mismatch** (their board/clock featurization ≠ ours) — the
  offline-prediction-as-feature design sidesteps this: we consume only their
  *output* `t_hat_ext`, never their internals.
- **Fairness of re-fit** — the pure-external variant (no re-fit) removes any doubt
  that we handicapped B.
- **Cohort mismatch** — score the released model on *our* held-out players/decisions
  so B and B+z see identical data; do not compare to their paper's numbers directly
  (only use r≈0.41 as a sanity band).
