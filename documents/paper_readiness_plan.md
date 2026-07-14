# Paper readiness plan

Status as of 2026-07-13. This is the active gate list; `TODO.md` remains the
broader project index.

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

Convert `paper_draft.md` from a synthesis into a paper structure now. The
empirical package is frozen: preserve the corrected KT association, the
3-of-5 stable-speed result, the two real education timing nulls, and the
Allie+static-vs-evolving result (significant on 2017/2021, null on 2019). Do
not start chess-F, another Go experiment, or controllable generation before
the manuscript is stable.
