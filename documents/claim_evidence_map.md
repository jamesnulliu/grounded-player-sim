# Claim–evidence mapping for `paper.md`

Every abstract and contribution-list claim, its landing point, and the frozen
artifact behind the number. Checked against `results/` on 2026-07-19; updated
after the five-dimension review revision (table numbers are the new
sequential scheme: 1–8). **2026-07-22 reframe update:** abstract/intro now
lead with the decomposition (static attribute + evolving state, user-sim
framing); new §4.3 memory-control block (Table 5b) and §2 memory-modules
paragraph added; rows below extended accordingly.

## Abstract

| Claim (paraphrase) | Landing point | Artifact | Status |
|---|---|---|---|
| Timing win in all 8 era×backbone conditions, every CI excludes zero | Table 3, §4.2 | `results/tier1_pooled.txt` (CIs now shown in-table) | ✓ |
| Improves over Allie in the direct add-on test on all 3 cohorts (2/3 under strictest controls) | Tables 4–5, §4.3 | `results/g4_timing.txt`, `results/g4_allie_static_vs_evolving.txt` | ✓ — scoping now in the abstract itself |
| Structured memory (training-free dynamics) beats the static profile 3/3 cohorts, pooled −0.0276 | Table 5b, §4.3 | `results/memory_baseline.txt` (+ `.json`) | ✓ |
| Richer-input memory matches/beats the 4-feature evolving latent (pooled +0.0150) — mechanism claim scoped to "carrying dynamic state" | Table 5b, §4.3 + §5 | `results/memory_baseline.txt` | ✓ — stated as honest finding, not spun |
| Input-matched learned control: GRU over the same memory features ties linear memory (+0.0096, CI crosses 0), beats static (−0.0180) | §4.3 third finding | `results/memory_gru_arm.txt` (+ `.json`) | ✓ landed 2026-07-22 |
| KT memory arm: M−B significant 7/8 datasets (same profile as D−B); D−M ties 5/8, M repairs ASSISTments-15 reversal; abstract "7 of 8" clause | Table 6 (last two cols), §4.5 ¶ after shuffle control | `results/kt_memory_arm.txt` (+ `.json`) | ✓ landed 2026-07-22; cohort fingerprints + reproduced D−B verified against frozen cells |
| Four-axis contrast vs emotion-dynamic LLM user simulators (mechanism/evidence/attribution/grounding) | §1 ¶2 end, §2 "Emotion-dynamic LLM user simulators" | prior-art digest + novelty sweep (scratchpad), refs flagged verify-at-bib | ✓ positioning claim, no numbers |
| KT frozen fitted profile fails: M−S 8/8 (−0.011…−0.098), S−B worse 6/8; "order-invariant ≠ freezable"; abstract "fails on every dataset" clause | §4.5 "Can the profile at least be frozen?" ¶ | `results/kt_static_arm.txt` (+ `.json`) | ✓ landed 2026-07-22; fingerprint-verified vs frozen cohorts |
| G5 persona ladder in the LLM: static−none −0.0104 / memory−none −0.0096 (sig); memory−static +0.0009 tie; hidden−memory +0.0045 (text > vector) | §4.7 final ¶, Table 8 last 3 rows | `results/g5_persona_ladder.txt` (+ `.json`, per-example NLLs in runs/g5-persona-ladder/) | ✓ landed 2026-07-23; 99-player paired bootstrap |
| Population generation is the capability memory/persona stores lack | §4.6, §2 memory ¶ | `results/real_kt.txt`, `results_ec.md` E-F1 | ✓ — structural argument + measured recall |
| Move: almost no state dependence; deviation-from-Maia-2 probe R²=0.009 | Table 3, §4.2 probe paragraph | `results/g4_timing.txt`, `results/tier1_pooled.txt` | ✓ — "almost no", not absolute |
| Concentration 2.7–3.6× under time pressure (variance-controlled; raw player-level 4.0–6.1×) | §4.4, Figure 2 | `results/concentration_variance_controlled.txt` | ✓ — single pipeline now (player-bootstrap raw → normalized) |
| ≈3× for weakest players | §4.4 | `results/rating_stratification.txt` | ✓ — "to confirm" from review: check per-player decision counts balanced across terciles |
| KT: 22/24 dataset-seed cells favor D | Table 6, §4.5 | `results/kt_replication_fixed_loader.txt` | ✓ |
| Population recovery at half the Wasserstein distance | Table 7, §4.6 | `results/real_kt.txt` (0.074 vs 0.147) | ✓ |
| Education response times do not transfer; strategic-time boundary **hypothesized** | §4.5 negatives, §5 | `results/real_kt_rt.txt`, `results/ednet_rt.txt` | ✓ — abstract now says "we hypothesize" |
| Hidden beats verbal board-native; advantage disappears in LLM | Table 8, §4.7 | `results_ec.md` E-E1, `results/g3_llm.txt` | ✓ |

## Contributions

| # | Claim | Landing point | Status |
|---|---|---|---|
| 1 | Equal-capacity control; wins all 8 conditions; survives 2× width (single-seed, disclosed in-bullet) | Tables 2–3, §4.1–4.2 | ✓ — Allie/static add-on moved out of C1 into C2 |
| 2 | When-not-what: move near-null (R²=0.009); timing survives ≥best-published-rank baselines; direct test 3/3, strictest controls 2/3 | Tables 3–5 | ✓ |
| 3 | Mechanism: concentration 2.7–3.6× / ≈3×; synthetic probe R²=0.93 vs 0.65; monotone clamp | §4.4 | ✓ |
| 4 | Generality: 22/24 KT cells; half the Wasserstein; generated recall 1.00 vs matched average-person 0.00; 2 RT negatives scope | Tables 6–7, §4.5–4.6 | ✓ — "matched" baseline now stated (synthetic-cohort average-person, results_ec.md E-F1) |
| 5 | Channel ordering backbone-dependent; SFT probe: timing-over-move at every scale — clean move-null under LoRA, graded ~1.5× under full FT | Table 8, §4.7 | ✓ — "≫" removed; graded claim matches slime_rl_llm.txt (1.53×/1.54×) |

## Number provenance notes

- Table 4 row "Elo+clock+branching": 5-seed G4 numbers with Spearman from the
  same artifact (−0.0247/−0.0281, 0.382/0.406, `g4_timing.txt`). Do not mix
  with the separate `posaware_pooled.txt` arm (−0.0315/−0.0266, 0.371/0.398).
- Table 4 row "Elo+clock" (−0.0430): 2-seed 2017-04 result (`results_ec.md`
  E-C6); footnoted in-table.
- Table 3 CIs: from `tier1_pooled.txt` verbatim (120 players/cohort for the
  clocked cohorts; Table 2's 2013-01 rows are 100 players).
- Stable-speed control (§4.4 prose): 3-of-5 from
  `stable_speed_baseline.txt`; 2021-04 −0.052, 2021-06 −0.137; nulls
  2019-07, 2023-04.
- **Shuffle control caveat (§4.5)**: the −0.0151 shuffled vs −0.0095
  unshuffled pair is the pre-leakage-fix seed-0 run
  (`scratchpad/real_kt_shuffle.py`, `results_ec.md`); the post-fix
  ASSISTments 09 mean is −0.0128. The qualitative order-invariance reading
  stands (leaked feature cancels in the paired difference), but re-run the
  shuffle on the fixed loader before submission or annotate the run in the
  appendix. **Open verification item.**
- Concentration: paper now uses only the player-bootstrap pipeline
  (raw 4.0–6.1× → normalized 2.7–3.6×,
  `concentration_variance_controlled.txt`); the old decision-level "2–8×"
  is no longer quoted.
- Synthetic probe R²: standardized to 0.93 vs 0.65 (two decimals)
  everywhere; artifact values 0.929/0.654.
- λ = 0 move control (§4.1 reconciliation): −0.0001 → −0.0066 ns, from
  `results_ec.md` (2017-04, single-condition test).

## Post-check edits (2026-07-22)

- §5 gains a cross-channel comparability limitation: raw timing-vs-move nat
  gaps are not scale-comparable across heads, so the asymmetry claim anchors
  to the probe contrast (R² 0.009 vs 0.93, §4.2/§4.4) and the concentration
  pattern (timing 2.7–3.6× under pressure, move ≈0 in the same buckets), not
  the raw nat comparison. No claim-row artifacts change.
- `summary.md` finding 4 and contribution 4 reworded: the education response
  edge is stated as order-invariant individualization (shuffle control), not
  state dynamics — closes the mirror-image misreading that state shows up in
  education *actions*. `summary.pdf` is now stale relative to `summary.md`.

## Review to-confirm items (checks against existing data, no new experiments)

1. Rating-tercile decision-count balance (possible data-quantity confound on
   the ≈3× weakest-players result) — re-bucket `tier1_runs/` per-player
   counts by Elo tercile.
2. Residual extra-parameters explanation for (base+z)−base gains beyond what
   Table 5's static arm rules out — confirm the static arm's parameter count
   matches the evolving arm's.
3. Re-run (or annotate) the KT shuffle control on the leakage-fixed loader
   (see provenance note above).
4. 2013-vs-clocked move-gap reconciliation (§4.1): the λ = 0 test covers the
   joint-objective cause on 2017-04; verify no protocol difference (epochs,
   player caps) also contributes before finalizing the sentence.
