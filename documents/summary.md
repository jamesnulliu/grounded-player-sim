# Project Summary: When, Not What

*One-page explainer. Numbers match the frozen artifacts in `results/` and
the manuscript `documents/paper.md` (2026-07-19). Target venue: ICLR,
cognitive-science area.*

## The idea in one sentence

A person's current mental state (tilt, fatigue, clock panic) shows up in
**when** they act — how long they think — and almost not at all in **what**
they choose.

## The idea, plainly

- People do not behave the same way twice. A chess player who just lost
  plays the next game differently. A student who struggled through ten
  problems answers the eleventh differently.
- Today's models of specific people **freeze** the person: a rating, a
  fixed embedding, a persona paragraph. They cannot track this drift.
- We carry a small **state** that updates as the person acts (a recurrent
  latent, fed by simple history features: time pressure, post-loss,
  fatigue, momentum).
- The honest test almost nobody runs: compare against a **twin model that
  is identical in every way** — same size, same inputs, same optimizer —
  except it is forbidden to remember anything across steps. Score both on
  the person's **future** games only.
- If the evolving model wins, the win is attributable to **accumulating a
  state**, not to seeing recent context, and not to extra capacity.

## What we found

1. **The state is legible in timing.** The evolving model beats its
   memoryless twin at predicting future think-time everywhere we tested —
   and still adds value on top of Allie, a released state-of-the-art
   think-time model.
2. **The state is nearly invisible in choices.** Move prediction shows
   almost no gain, and a probe from the state to "how the player deviates
   from a strong move model" recovers essentially nothing (R² = 0.009).
3. **The edge lives where behavior is least average.** It is ~3× larger
   under time pressure (after variance control) and ~3× larger for the
   weakest players.
4. **It transfers to education — for responses, not for timing.** The same
   model wins student-response prediction on 8 real datasets, and it can
   reconstruct (and generate) the diversity of a real student population,
   which an "average person" model cannot.
5. **How you inject the state depends on the backbone.** A hidden vector
   beats a text note in a from-scratch model; inside an LLM, that advantage
   disappears (the LLM reads the note semantically).

## Contributions

1. **The control:** an equal-capacity, same-input evolving-vs-memoryless
   twin on strict future splits — isolates *accumulated state* from
   *history-conditioning*. No prior work runs this on real human behavior
   with a timing target.
2. **The finding:** the when-not-what asymmetry on real humans, robust
   across 6 years of chess and against released baselines.
3. **The mechanism account:** the edge concentrates where state matters
   (time pressure, weak players); on synthetic players with a *known*
   hidden state, the latent provably encodes it (probe R² 0.93 vs 0.65)
   and causally uses it (clamp → monotone response).
4. **Generality + utility:** response-channel wins on 8 real education
   datasets; population recovery and generation an average-person model
   structurally cannot do.
5. **The channel result:** hidden vs verbal injection ordering flips with
   the backbone's language prior.

## Difference from previous work

| Previous work | What it does | What it lacks (that we add) |
|---|---|---|
| Maia / Maia-2 / Allie | Human-like chess: rating-conditioned moves; Allie adds a think-time head | Static — no per-individual evolving state; no per-player future split |
| ChessMimic | Per-Elo-band move + clock transformers | Cohort-level, not per-individual; no memoryless-twin control |
| Matilda, Maia4All | Per-player style conditioning of a strong policy | The style vector is frozen per player; no timing target |
| Ailed | Chess engine with an evolving "emotional state" | Dynamics are asserted, never validated against real players; we make it falsifiable |
| LATTE | Evolving user state injected into a frozen LLM (recsys) | No timing target, no capacity-matched twin; baselines are static profiles |
| HumanLM, Duan et al. | LLM emits verbal psychological state / student profiles | Verbal-only channel; no timing; no equal-capacity control |
| van der Linden (psychometrics) | Stable per-person speed + item difficulty predicts response time | Speed is *stable*; we beat a stable-speed control on 3/5 cohorts (all 5 point estimates favor evolving) |

One line: **each axis exists somewhere; nobody combines the
capacity-matched twin + strict future split + timing target on real human
data — and nobody reports the when-not-what asymmetry.**

## Results (headline numbers)

D = evolving latent, B = memoryless twin. All gaps are held-out NLL in
nats; **negative favors the evolving model**; significant = 95%
player-bootstrap CI excludes zero.

### Chess: timing wins everywhere, moves almost nowhere

| Test | Result | Significant? |
|---|---|---|
| Think-time, 4 eras (2017–2023) × 2 backbones, 5 seeds | D−B = −0.021 to −0.033 | Yes, all 8/8 conditions |
| Move choice, same sweep | −0.0005 to −0.018 | Small (conv backbone) or null (MLP) |
| Probe: state → deviation-from-Maia-2 | R² = 0.009 (vs 0.93 on known synthetic state) | Near-null |
| Capacity check: twin at 2× latent width | D still wins | Yes (single-seed sweep) |

### Chess: the edge survives released state-of-the-art baselines

| Baseline (its own strength) | Latent add-on gain | Significant? |
|---|---|---|
| Elo + clock + branching (Spearman 0.38–0.41 ≈ ChessMimic's 0.41) | −0.025 / −0.028 | Yes, both cohorts |
| + Maia-2 position difficulty (Spearman 0.414–0.447) | −0.025 to −0.039 | Yes, 3/3 cohorts, every seed |
| Allie's released think-time head, locked (Spearman 0.62–0.65) | −0.018 to −0.033 | Yes, 3/3 cohorts (direct test) |
| Strictest: Allie + stable per-player calibration | −0.017 / −0.001 / −0.019 | 2/3 cohorts (2019 null); pooled −0.0126, significant |

### Where the edge concentrates

| Analysis | Result |
|---|---|
| Time-pressure terciles (variance-controlled) | High-pressure edge 2.7–3.6× larger; every CI excludes zero |
| Post-loss / fatigue buckets | Flat (the concentration is specific to the clock) |
| Rating terciles (480 players) | Weakest −0.0404 vs strongest −0.0139 (~3×), difference significant |
| Move channel, same buckets | ≈0 everywhere (the contrast) |

### Education (knowledge tracing) and population recovery

| Test | Result |
|---|---|
| Response prediction, 8 real datasets × 3 seeds | D wins 22/24 cells; 7/8 dataset means favor D |
| Population recovery (500 real students) | Wasserstein 0.074 vs average-person 0.147 (2× closer); rank corr 0.96 |
| Diversity coverage (recall) | Latent 0.75–1.00 vs average-person 0.00 |
| Generation (sampled new students, synthetic cohort) | Precision 0.93, recall 1.00 |

### LLM arm (Qwen3, SFT probe)

| Test | Result |
|---|---|
| State helps think-time, 0.6B → 8B | Δ = −0.011 to −0.014 at every scale |
| State helps moves | Null under LoRA at ≥4B; small (−0.007/−0.008) under full fine-tuning → timing ≈ 1.5× move |
| Hidden vs verbal channel | Board-native: hidden wins by 0.07–0.12 nats; inside the LLM: advantage disappears |

### Honest negatives (we report these, and they set the scope)

| Negative | What it means |
|---|---|
| Real education response *times*: ASSISTments inconsistent, EdNet null | The timing result does not transfer to incidental UI timers; we hypothesize it needs a strategically managed clock |
| Go: null at every board size once board size is controlled | The naive mixed-cohort "positive" was a regime confound |
| KT temporal shuffle: edge undiminished | The education *response* edge is individualization, not order-tracking |

## Status and the one open experiment

- Empirical package frozen; manuscript drafted (`documents/paper.md`);
  five-dimension review passed and applied.
- Independent novelty check (2026-07-19): **no head-on collision**; closest
  work carved at full-text level.
- **One experiment pending** (approved unfreeze): re-run the state→choice
  probe against a strong *released* move backbone (Maia-3/Chessformer). If
  choice stays null, the asymmetry is backbone-robust — the last open flank
  before submission.
