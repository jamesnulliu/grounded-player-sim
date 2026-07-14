# EdNet response-time protocol

Frozen before downloading or ingesting EdNet-KT1 student logs on 2026-07-13.
The executable protocol is `scripts/ednet_manifest.json`.

## Field semantics and scope

The official EdNet README defines KT1 `elapsed_time` as the milliseconds a
student spends on the current question. It also defines `timestamp` as the
shifted Unix-millisecond time when the question was given and `solving_id` as
the bundle-solving session. Correctness is derived by comparing `user_answer`
with `contents/questions.csv:correct_answer`.

There is a material unresolved ambiguity: EdNet issue #5 asks whether
`elapsed_time` is attached to the prior/current question or averaged over a
bundle, and has no maintainer response. The primary test therefore keeps only
questions whose metadata bundle contains exactly one question. This avoids
assigning one bundle duration to several question rows or inventing a split of
that duration. The result concerns EdNet's reported UI elapsed time; it does
not by itself establish strategically managed time.

## Frozen cohort

- EdNet-KT1 plus the official contents archive, both verified by SHA-256.
- Singleton-question bundles only; metadata and final answer must be present.
- `elapsed_time` must be numeric, finite, and positive. Valid times are
  clipped to 0.5--300 seconds by the shared KT loader.
- Users are considered in numeric user-ID order. Select the first 500 with at
  least 50 valid rows and retain their earliest 200 timestamp-ordered rows.
- The scalar item feature is leakage-safe empirical question difficulty. The
  evaluated student's held-out future responses never enter that estimate.

The 13,169-question metadata archive contains 8,006 singleton questions, so
this restriction retains a majority of the item bank rather than selecting a
small special case.

## Frozen comparison

Use the same equal-declared-capacity evolving (`persist=True`) versus
memoryless (`persist=False`) KT arms as ASSISTments, a strict per-student 70/30
future split, 60 epochs, timing weight 0.5, and seeds 0--2. Average each
student's paired `D-B` over seeds, then bootstrap the 500 students.

Timing transfer succeeds only if the pooled timing 95% CI excludes zero below
and every seed point estimate is negative. The stronger when-not-what pattern
requires timing transfer plus a pooled response CI that includes zero. Any
other outcome is still reported: in particular, a timing null scopes the real
when-not-what evidence to chess, while a win on both channels is cross-domain
dynamic-state transfer but not the chess-like asymmetry.

## Outcome

All three seeds completed on the frozen 500-student cohort. Response prediction
significantly favors the evolving model (pooled D-B -0.0159, 95% CI
[-0.0202,-0.0118]), while elapsed-time prediction is null (-0.0004,
[-0.0059,+0.0059]) and seed 2 is wrong-sign. Both timing-transfer and full
when-not-what criteria fail. See `results/ednet_rt.txt` and
`results/ednet_replication.json`.
