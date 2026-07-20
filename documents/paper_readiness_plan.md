# Paper readiness plan

Status as of 2026-07-19 (was 2026-07-13). This is the active gate list;
`TODO.md` remains the broader project index.

## 0. Idea-forge validation verdict (2026-07-19) — one contained unfreeze

An independent dedup + adversarial-critic run
(`idea-forge/grounded-player-sim/`, idea I1) ruled: novelty gate **pass**
(no head-on collision; ChessMimic/LATTE close-range, carved at full-text
level), bands S2/F4/V3 → overall **C (reshape)**, ICLR cog-sci venue fit
defensible with TMLR as fallback. The named lethal flank: the choice
near-null (probe R² = 0.009, deviation-from-Maia-2) is measured against
small from-scratch move backbones — a strong released move backbone could
make choice state-legible and dissolve the when-not-what headline.

**User-approved disposition (full reshape): the empirical freeze is lifted
for exactly ONE experiment** — the state→choice probe / move D−B re-run
against a strong *released* move backbone (Maia-3/Chessformer,
arXiv:2605.19091; verify weight availability first; Maia-2 fallback is
already in hand). If choice stays near-null, the asymmetry is
backbone-robust and the limitations sentence "move-channel ceiling remains
open" is replaced by this result (expected post-reshape banding S3/F4/V4 →
B). If choice becomes legible, the headline must be recut before
submission — surface that immediately. The freeze stays in force for
everything else. Also completed under this disposition: the
arXiv:2606.25176 mis-citation (actual paper: Matilda) is fixed in
`paper.md` §2 + references, flagged in `related_work.md`; §4.4's scope note
now states the asymmetry survives either mechanism reading.

## 1. Fixed-loader KT replication — complete

**Implemented:** frozen eight-dataset protocol, canonical preparation,
source/output/manifest hashes, strict cohort preflight, resumable 24-cell
runner, and Pearson/Spearman fitting with dataset-bootstrap and
leave-one-dataset-out sensitivity.

**Inputs recovered:** raw corrected ASSISTments 2009 from the USTC mirror; the
other seven five-column exports from
`theophilee/learner-performance-prediction` commit `a7ae193`. Source, prepared,
manifest, and cohort hashes are recorded. The staged data remains git-ignored.

**Outcome:** all 24 cells completed. D wins significantly in 22 cells; Statics
seed 0 is null and ASSISTments 2015 seed 2 significantly favors B. Both repeat
exactly. Seven of eight dataset means favor D. Signed latent advantage vs
spread is Pearson 0.776 / Spearman 0.476; its bootstrap crosses zero and
leave-one-out Pearson reaches 0.138 without Spanish. The old 0.89 law framing
is retired. Artifacts: `results/kt_replication_fixed_loader.{txt,json}`.

## 2. Stable-speed control — complete

**Implemented:** frozen source/ingest/training manifest, exact source-prefix
hashes, three reproducible 100-player cohorts, resumable per-seed cells, and
seed-averaged per-player pooling. All nine extension cells completed.

**Outcome:** dynamic significantly beats the static per-individual control on
2021-04 (D−B −0.052, CI [−0.107,−0.001]) and 2021-06 (−0.137,
[−0.198,−0.076]); 2023-04 is negative but null (−0.037,
[−0.116,+0.051]). With the existing 2017-04 significant result and 2019-07
null, the five-cohort pattern is 3 significant / 2 null, and all point
estimates favor dynamics. Report as usually significant but cohort-dependent.
Artifacts: `results/stable_speed_baseline.txt` and
`results/stable_speed_extension.json`.

## 3. EdNet response time — complete, honest negative

Protocol, field semantics, clipping/exclusions, cohort rule, split, and success
criteria were frozen before ingestion. To resolve the upstream ambiguity about
bundle-averaged elapsed time, the primary cohort uses singleton-question
bundles only. On 500 students × 3 seeds, response significantly favors D
(pooled −0.0159, CI [−0.0202,−0.0118]) but timing is null (−0.0004,
[−0.0059,+0.0059]) with one wrong-sign seed. The timing-transfer and full
when-not-what criteria fail. Artifacts: `documents/ednet_protocol.md`,
`results/ednet_rt.txt`, and `results/ednet_replication.json`.

## 4. Manuscript — active

First full paper-structure draft landed as `documents/paper.md`
(2026-07-19, from the outline + frozen artifacts; claim–evidence mapping in
`documents/claim_evidence_map.md`). `paper_draft.md` remains the internal
synthesis. Five-dimension review completed and revision applied same day:
citation corrections (Maia author list, van der Linden venue; Allie/Maia-2
metadata verified), §4.1↔§4.2 move-gap reconciliation added, synthetic
players now constructed in §3 + Table 1, claim calibration aligned across
abstract/contributions/conclusion, tables renumbered sequentially 1–8 with
CIs added to Table 3, cadence de-telling pass, Reproducibility + Ethics
statement stubs added. Remaining before submission: figures 1–3; resolve
all reference-list TODOs (tooling + dataset cites, re-add Player-Specific /
Mixture-of-Masters with verified IDs); the four verification items in
`claim_evidence_map.md` (rating-tercile balance, static-arm params, re-run
shuffle on fixed loader, 2013 protocol check); page-budget trim at LaTeX
time (~0.5–1 page over; demotion candidates noted in the draft preamble).
The
empirical package is frozen: preserve the corrected KT association, the
3-of-5 stable-speed result, the two real education timing nulls, and the
Allie+static-vs-evolving result (significant on 2017/2021, null on 2019). Do
not start chess-F, another Go experiment, or controllable generation before
the manuscript is stable.
