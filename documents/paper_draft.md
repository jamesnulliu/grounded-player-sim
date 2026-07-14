# Grounded player simulation with a dynamic per-individual latent state

*Paper-skeleton synthesis of the landed results. Detailed tables +
reproduction in `results_ec.md`; prior-art positioning in `design.md §8`;
active verification gates in `paper_readiness_plan.md`.*

> **Verification status (2026-07-13).** The leakage-fixed 24-cell KT replication
> is complete. D wins significantly in 22/24 seed cells and on 7/8 dataset
> means; one Statics seed is null and ASSISTments 2015 seed 2 significantly
> favors B. The signed spread-vs-advantage fit is Pearson 0.776 / Spearman 0.476
> with wide bootstrap and leave-one-out sensitivity. The old 0.89 “law” framing
> is superseded. The stable-speed extension is 3 significant / 2 null cohorts,
> all directionally favoring dynamics. EdNet independently reproduces the real
> response win but not timing (pooled timing CI crosses zero), so two real
> education timing datasets are now negative. See the frozen result artifacts.

## Abstract

People do not play a game — or answer a quiz — the same way twice: a recent loss
breeds tilt, a long session brings fatigue, a ticking clock changes everything.
We model *how a specific person behaves right now* with a learned latent state
`z_t` that is **per-individual**, **evolves** over the person's own action+timing
trajectory, and is validated against their **future** behavior on a strict
temporal split. The injector is decoupled from a swappable backbone, so "does the
dynamic latent help?" is provable independent of backbone.

Our central finding is an asymmetry: **the state is legible in *when* a person
acts, not *what* they play.** On real Lichess chess the evolving latent beats an
equal-capacity **memoryless** twin at predicting future think-time across a
6-year era span (2017–2023), two backbones, and five seeds (P=1.00), and adds
value over a near-SOTA Elo+clock+complexity baseline — while **move choice
carries essentially no extra state-dependence.** The timing edge **concentrates**
where behavior is least predictable from the average: under time pressure
(2–8×) and for weaker players (≈3×). Across eight knowledge-tracing datasets,
population spread is descriptively associated with signed latent advantage
(Pearson 0.78) but the rank relationship is weak (Spearman 0.48) and sensitive
to the Spanish anchor, so we treat cross-population scaling as a hypothesis,
not a law. The same latent **generates** a population that recovers real
heterogeneity a "positive average person" cannot.

The state's value lives in a **trained hidden latent, not a verbal persona
prompt** (RQ6) — the bridge to LLM agents, where today's simulators (HumanLM,
generative agents) condition on verbal personas. In an actual LLM (Qwen3) an SFT
probe reproduces the timing-≫-move asymmetry (robust across 0.6B→8B and
LoRA→full fine-tuning, Δ ≈ −0.013 on timing); and delivering the state as a
**hidden soft-prompt** inside the LLM (G3, 3 seeds) shows the hidden-vs-verbal
ordering **flips** — the verbal note wins there (the LLM reads it semantically),
even as injected state still helps think-time. We do not claim novelty by
conjunction, and we are explicit that each axis has prior art — dynamic
emotional chess (Ailed), per-individual chess style (Elo-Disentangled,
ChessMimic), timing-reveals-latent-state (response-time psychometrics), and an
evolving latent under a future split (LATTE, HumanLM). Our contribution is the
**controlled empirical synthesis those lines lack**: on real human data, with a
per-decision oracle and a strict future split, an equal-capacity evolving latent
beats a memoryless twin at predicting a specific person's *future think-time* —
robust across a six-year era span, and the same timing-over-response asymmetry
reproduces in a non-game domain on synthetic response-time data (real
knowledge-tracing datasets support the *response* side in 22/24 fixed-loader
seed cells, but not the *timing* side: frozen real-response-time runs on both
ASSISTments and EdNet are negative, §Limitations) —
while move choice is a near-null; the same latent recovers population
heterogeneity a "positive-average-person" cannot; and the hidden-vs-verbal
channel ordering is **backbone-dependent** (hidden richer with no language
prior, verbal richer inside an LLM).

## Contributions

1. A game-agnostic **dynamic latent-state injector** decoupled from a swappable
   backbone; the same injector + trainer run on chess and knowledge tracing
   unchanged.
2. An **equal-capacity, same-input evolving-vs-memoryless control** on a
   per-decision oracle domain and a strict **future** split — the form the
   human-chess and user-simulation lines do not run, which isolates accumulated
   *dynamics/individualization* from mere history-conditioning and settles the
   #1 reviewer objection on **real** data (RQ1, E-C2/E-C3).
3. A **per-individual evolving timing** result robust across a 6-year era span,
   two backbones, and five seeds, that adds significant value *over* a near-SOTA
   Elo+clock+**position-complexity** baseline (E-C6) — the differentiator vs
   aggregate clock models (Allie / ChessMimic).
4. An interpretability account with a clean **channel asymmetry**: on a
   **synthetic** player with a *known* hidden state, the latent **encodes** it
   (probe held-out R²=0.93 vs 0.65) and **causally uses** it (clamp → monotone
   dose-response); on **real** chess the think-time edge **concentrates** under
   time pressure (2–8×) and for weaker players (≈3×) — while the **move** edge is
   a flat near-null. The evolving state is legible in *when* a person acts, not
   *what* they play.
5. A **multi-granularity heterogeneity analysis**: the chess timing advantage is
   ≈3× larger for weaker/more-variable players and 2–8× larger under time
   pressure. Across eight KT populations the signed advantage has Pearson 0.78
   with accuracy spread, but Spearman is only 0.48, the dataset-bootstrap CI
   crosses zero, and omitting Spanish drops Pearson to 0.14. This is suggestive
   cross-population evidence for *personalization helps more where individuals
   differ*, not a robust scaling law.
6. **Generality** (RQ5) in a non-game oracle domain (KT) and a **heterogeneity-
   recovery** result (Milestone F) beating the "positive average person" — both
   confirmed on **real data** (ASSISTments 2009): the evolving latent beats the
   memoryless twin on real student responses (500 students, mean D−B=−0.0128,
   significant in all 3 seeds). Across 8 datasets / multiple platforms / 3
   subject domains, 22/24 fixed-loader seed cells significantly favor D; one
   Statics seed is null and one ASSISTments 2015 seed significantly favors B,
   leaving 7/8 dataset means in D's favor. On the same 500 ASSISTments 2009
   students, the latent recovers the real accuracy distribution (Wasserstein
   2× better than average-person, corr 0.96, recall 0.75 vs 0.00).
7. A study of the same injection in an **actual LLM policy** (Qwen3, 2×A100):
   a *frozen* verbal state note is a negative control (≈ irrelevant filler), and
   *RL* (GRPO via slime) learns the task but its sparse reward can't resolve the
   state effect; a dense **behavior-cloning SFT** probe, however, reproduces the
   channel asymmetry in the LLM — state helps **think-time** (Δ = −0.011,
   non-overlapping across seeds, holds LoRA→full-param) ≫ **moves** (−0.004). And
   the effect is **not a small-model artifact** — a **backbone-scaling trend**
   (0.6B → 1.7B → 4B → 8B LoRA, 3 seeds each) keeps the think-time help robust
   (−0.011 to −0.014 throughout). Under LoRA the already-small move effect drops
   to ≈0 at ≥4B, but **full-param runs (all weights, 3 seeds each) at both scales
   where LoRA collapses the move channel — 4B and 8B** — enabled on this old-kernel
   node by a **single-GPU 8-bit paged optimizer** (no FSDP/NCCL) — show that
   move-collapse is a **LoRA-capacity artifact**, not a property of scale: full
   fine-tuning holds **timing Δ = −0.0110 / −0.0128** (4B/8B, ≈ LoRA −0.0114 /
   −0.0136, robust to adaptation method) *and* **recovers a stable move Δ = −0.0072 /
   −0.0083** (all 3 seeds < 0 at each scale, vs LoRA's −0.0004 / −0.0008 nulls). So
   the **timing ≫ move asymmetry is robust but graded** (timing ~1.5× move under full
   FT), and the clean move-**null** is specific to the low-capacity LoRA probe and the
   board-native policy — the robust, adaptation-invariant claim is the **think-time**
   benefit. A practical lesson:
   for small latent-injection effects, **SFT is a sharper probe than RL**; the
   *hidden* latent stays the stronger channel than a verbal prompt (RQ6).

## Method (one paragraph)

Every decision — a chess move, a quiz answer — is one game-agnostic
`DecisionPoint` (`state`, `legal_actions`, an engine/IRT `EngineReference`, a
`TimeSignal`, and a `recent_outcomes` stream). A recurrent injector `f_phi`
accumulates engineered `history_features` (time-pressure, post-loss, fatigue,
momentum) into `z_t`, rendered to the backbone as a hidden vector or as verbal
text. A swappable backbone (oracle-free board-native CNN-style head for chess;
a logistic head for KT) predicts a move distribution and a think-time. We fit
`phi` by SFT (move-NLL + λ·timing-NLL) on each player's earlier trajectory and
score the held-out later trajectory; significance is a bootstrap over players.
The split is **leakage-safe by construction**: the training loss is masked to
train-only steps (`[0, b)`; the held-out tail contributes no gradient), and the
latent is a **causal** recurrence (`z_t` depends only on steps `≤ t`), so warming
`z` over the whole trajectory cannot peek at the future — `z_t` at a held-out
step still sees only that player's past.

## Results (headline)

Lower NLL is better; D = evolving latent, B = equal-capacity memoryless twin.
All "P" below are **P(D−B<0)** — the fraction of player-bootstrap resamples
with D<B (`gps.eval.bootstrap.BootstrapCI.p_below_zero`), a bootstrap
sign-support statistic, **not a frequentist p-value**; we lead with the 95%
CI (bootstrap over players, never over decisions/moves, which are
within-player correlated) excluding zero as the significance criterion, and
report P alongside as a directional-consistency summary.

| RQ | Experiment | Result |
|----|-----------|--------|
| RQ1 | **E-C2** dynamic > memoryless (real chess, 100 players) | D−B ≈ −0.069, P=1.00; survives B at 2× latent width |
| RQ1 | **E-C1** dynamic > static-individual embedding (move) | pooled −0.120, CI [−0.135,−0.106], P=1.00 |
| RQ3 | **E-C3** future-*sessions* split | pooled −0.067, CI [−0.077,−0.057], P=1.00 |
| — | replication (rapid, diff. time control) | −0.062, P=1.00 |
| RQ2 | **E-C4** state-recovery probe (presence) | held-out R²: D=0.93 vs B=0.65 |
| RQ2 | **E-C4** causal clamp (use) | monotone dose-response, expected direction |
| §5 | concentration (synthetic) | edge monotone-concentrated in high-tilt decisions |
| §5 | concentration (**real timing**) | think-time edge **2–8× larger under time pressure** (robust 2 cohorts/2 seeds); flat for post-loss/fatigue |
| §5 | concentration, **variance-controlled** (paper-readiness fix) | player-bootstrapped, normalized by within-bucket decision-level stdev: raw ratio 4.0–6.1× shrinks to **2.7–3.6×, still survives** (2 cohorts × 2 seeds) — not a pure variance artifact |
| — | rating stratification | timing edge **≈3× larger for weakest vs strongest players** (high−low +0.027, P(high<low)=0.00, 480 players) |
| — | dynamic **beats a stable per-individual speed baseline** (van der Linden-style; paper-readiness addition) | five-cohort result: significant on **2017-04, 2021-04, and 2021-06**, null on **2019-07 and 2023-04**; all point estimates favor D — usually significant (3/5), still cohort-dependent |
| E-C6 | **timing** vs memoryless (zero-inflated head) | −0.026, P=1.00 (−0.069 log-normal) |
| E-C6 | **timing** adds value over Elo+clock aggregate | (B4+z)−B4 = −0.043, P=1.00 |
| E-C6 | ... and over a **position-aware** baseline (+ branching factor) | (B4+z)−B4 = −0.035, P=1.00; baseline Spearman 0.39 ≈ ChessMimic 0.41 |
| **G4** | ... and over a baseline with a **released SOTA's** difficulty (**Maia-2** move-entropy), 3 cohorts × 5 seeds | (B4+z)−B4 = **−0.025 / −0.029 / −0.039**, P=1.00; baseline **Spearman 0.414 / 0.445 / 0.447 ≥ ChessMimic 0.41** (2017/2019/2021) |
| **G4** | ... and over **Allie**'s (ICLR'25) *actual released think-time* — the airtight test (Spearman **0.62/0.64/0.65** ≫ 0.41) | direct **Allie-vs-Allie+z** is significant on all 3 cohorts; against **Allie+static-individual**, evolving is significant on **2017 and 2021** (−0.017/−0.019) and null on 2019 (−0.001), isolating dynamics beyond identity on 2/3 cohorts |
| G4 | **move** channel vs Maia-2 (does state encode move-deviation?) | latent recovers Maia-deviation at **R²≈0.009** (≈null, vs 0.93 synthetic state) → move near-stateless even vs human-move SOTA |
| RQ5 | **E-D1** knowledge tracing (non-game) | timing D−B=−0.050, P=1.00, D wins 100% |
| RQ5 | **E-D1 real** (ASSISTments 2009, 500 students) | response D−B≈−0.010, P=1.00 in all 3 seeds, D wins 64–73% (robust across 150–500 cohort sweep) |
| RQ5 | **EdNet-KT1 real timing**, singleton bundles (500 students, frozen protocol) | response **−0.0159** [−0.0202,−0.0118]; timing **−0.0004** [−0.0059,+0.0059] — response replicates, timing null |
| RQ5 | **E-D1 replication** (8 datasets, multi-platform, 3 subjects) | Fixed loader: 22/24 seed cells significantly favor D; Statics has 1 null, ASSISTments 2015 has 1 significant reversal and favors B on the 3-seed mean; 7/8 dataset means favor D |
| E-D2 | **Go** timing (real OGS, oracle-free) | **no effect at any board size**: null on 13×13 & 19×19 (well-powered); mixed-cohort "positive" was a board-size confound; the one residual weak 9×9 signal (n=209, 2/3 seeds) **collapses to null on a 2.5× larger 9×9 cohort** (N=519, 1/3 seeds, mean ≈0) — small-cohort noise; Go = future work |
| RQ5↔F | effect vs population heterogeneity | Fixed-loader signed advantage: Pearson 0.78 / Spearman 0.48 (n=8), bootstrap CI crosses zero and leave-one-out Pearson falls to 0.14 without Spanish — suggestive, not a law |
| RQ6 | **E-E1** hidden vs verbal channel | hidden richer: −0.069 (synth) / −0.117 (real), P=1.00 |
| F | **E-F2** population heterogeneity | W1 to observed 4–5× lower than average-person; corr 0.96 |
| F | **E-F2 real** (ASSISTments 2009, 500 students) | W1 2.0× lower than average-person; corr 0.96; recall 0.75 vs 0.00 |
| F | **E-F1** generate novel players (sample latents) | precision/recall 0.93/1.00 vs average-person 1.00/0.00 |
| LLM | **SFT probe** (Qwen3, 2×A100) | state helps timing Δ=−0.011 (3-seed; LoRA=full) ≫ moves −0.004; RL too sparse; frozen = null |
| LLM | **backbone-scaling trend** (0.6B→1.7B→4B→8B, LoRA) | timing robust at every scale (−0.011…−0.014); under LoRA move drops to ≈0 at ≥4B — but this is a LoRA-capacity artifact (see next row), not a small-model artifact |
| LLM | **full-param 4B & 8B** (all weights, 3 seeds each) | timing Δ=−0.0110/−0.0128 (≈LoRA, adaptation+scale-invariant) **but move Δ=−0.0072/−0.0083 at 4B/8B (not the LoRA nulls −0.0004/−0.0008)** → move-collapse is a **LoRA-capacity artifact at both scales**; timing≫move robust but **graded** (~1.5×); single-GPU 8-bit optim (no FSDP) |

A recurring, honest signature: **timing is the channel where the evolving state
robustly helps** (chess and KT alike). The discrete move/response advantage is
smaller and channel-dependent: on **synthetic** data it concentrates in
high-tilt decisions, but on **real** chess it is a flat near-null (no
concentration by post-loss or time-pressure) — so we lead with timing and treat
the real move channel as a genuine near-null, not a faint signal to amplify.

The chess results support a clear principle about *where* the latent helps:
the timing edge is ≈3× larger for weaker/more-variable players and 2–8× larger
under time pressure. The fixed-loader KT synthesis points in the same direction
in Pearson terms (0.78), but its weak rank correlation (0.48), wide bootstrap,
one reversed dataset mean, and Spanish sensitivity prevent elevating that
cross-population pattern to a law.

## Limitations (state these)

- **The quantitative headline runs on a small from-scratch backbone, not an
  LLM.** Every D-vs-B number above uses the board-native MLP / KT logistic head.
  The LLM policy (`SGLangBackbone`) is now **implemented and runs** on real
  chess (sglang + Qwen3-8B), and the first **frozen-LLM verbal-injection** test
  is a clean **negative control for the persona-prompt baseline (B5)**: a state
  note (NLL +0.081 vs none) is **no better than irrelevant filler text** (+0.069)
  — the frozen model does *not use* the state content, having no learned mapping
  from "tilted" to *this player's* moves. So naive verbal/persona-prompt
  injection does not work, which **motivates** the trained dynamic latent (what
  the board-native results validate). We then **RL-trained** an LLM (slime GRPO,
  Qwen3-1.7B, 2×A100) to mimic a player's moves: RL **does** improve held-out
  move-match (+0.025–0.030 over the frozen model, both conditions — the trained
  injector genuinely learns), but the player's verbal **state does not help move
  prediction** (with−no = −0.0013, a clean null over 3 seeds) — move choice is
  stateless. RL-training the LLM to predict **think-time** likewise learns
  strongly (~60% bucket-accuracy) yet the verbal state *still* doesn't help
  (no-state 0.627 > with-state 0.594). So across **both** channels a verbal state
  note fails to help even an RL-trained LLM — coherent with (a) position
  complexity dominating think-time and (b) RQ6's **hidden ≫ verbal**: the
  state's value lives in the *trained hidden latent* (board-native), not a text
  prompt. **However, RL's match-reward is too sparse to resolve small effects; a
  dense behavior-cloning SFT probe (TRL, held-out completion NLL) does surface
  the asymmetry in the LLM**: across 3 seeds, the state helps **think-time**
  prediction (Δ = −0.011, non-overlapping across seeds) ≈3.6× more than **moves**
  (Δ = −0.003) — the same channel signature as board-native, now a positive LLM
  result. (A `HIDDEN` soft-prompt injection is the natural next step; slime RL is
  text-native.)
- **Timing robust across a 6-year era span; move small — settled at scale
  (5 seeds × 5 cohorts × 2 backbones = 50 runs, 2×A100).** Pooling over seeds
  (averaging each player's D−B, then bootstrapping over players) shows **timing
  D−B significant (P=1.00) in all 8 clocked conditions** — 4 eras (2017, 2019,
  2021, 2023) × {mlp, conv}, effect tightly clustered at −0.021 to −0.033. The
  era-generality is striking: the think-time win is essentially constant from
  2017 to 2023. The **move** effect is small and backbone-dependent:
  the conv trunk wins on every cohort (−0.005 to −0.015) but the mlp trunk is
  **null** on both clocked cohorts (P=0.18, 0.85). Multi-seed pooling *corrected*
  noisy single-seed estimates (single-seed 2017 swung mlp +0.032 / conv −0.030;
  pooled they are +0.0005 / −0.0046) — so we do not over-read the move channel.
  A context-bucketed analysis confirms the move edge **does not concentrate**
  anywhere (flat ≈0 by post-loss and time-pressure, both cohorts), whereas the
  timing edge concentrates **2–8× under time pressure** and **≈3× for weaker
  players** — so the evolving state is legible in *how long* a person thinks,
  not *which move* they pick. Lead with timing; treat move as a genuine
  near-null in move choice, not a faint signal to amplify.
- **Weak from-scratch backbone — retired for timing (G4, released SOTA).** The
  MLP trunk gets ~3.0 move-NLL (beats uniform 3.18). The low absolute think-time
  Pearson (~0.14) was a *missing feature*: adding **position complexity**
  (branching factor) to the baseline lifts it to **Spearman 0.39 ≈ ChessMimic's
  0.41** — and the evolving latent *still* adds significant value (P=1.00) on top.
  **G4 settles the residual "is the baseline a strawman?" doubt with actual
  released weights:** we boost the baseline with **Maia-2**'s (CSSLab, released,
  human-move SOTA) learned position difficulty (its legal-move-distribution
  entropy) and re-run on **three** independent real cohorts (2017-04, 2019-07,
  2021-06; 100 players, 5 seeds each). The strongest baseline
  (Elo+clock+branching+Maia-2) reaches **Spearman 0.414 / 0.445 / 0.447 — at or
  above ChessMimic's 0.41** — and the evolving latent *still* adds significant
  think-time value **(B4+z)−B4 = −0.025 / −0.029 / −0.039, P=1.00, CI excludes
  0, every seed**. **The airtight version replaces the Maia difficulty *proxy*
  with an *actual released think-time model* — Allie (Zhang et al., ICLR'25):
  we load its checkpoint, reconstruct each game's full move sequence from our
  per-player FENs, and read its per-decision think-time prediction as the
  external baseline.** Allie alone is strong — per-player Spearman **0.62 /
  0.64 / 0.65, well above ChessMimic's 0.41** — yet in the direct
  *Allie-vs-Allie+z* comparison the evolving latent **still adds significant
  value on all three cohorts (−0.023, P=1.00 / −0.018, P=0.998 / −0.033,
  P=1.00)**. Honestly, the effect is *smaller* than over weaker baselines (a
  strong released model captures more of the timing signal) and, against the
  fullest co-fit baseline (Elo+clock+Allie), is significant on 2017 (−0.013)
  and **2021 (−0.013, P=1.00)** but **non-significant on 2019 (−0.005,
  P=0.85)** — we report that null and do not average it away. A **third
  cohort was added specifically to resolve this**: with 2017 and 2019 alone
  the co-fit result was a 1-significant/1-null tie a reviewer could read
  either way; 2021 breaks the tie 2-significant/1-null, so we now describe the
  co-fit test as "usually significant, cohort-dependent" rather than
  "marginal." So the latent's timing value is not an artifact of a weak
  baseline; it survives even an actual released think-time head on every
  cohort tried. A stricter identity control holds Allie fixed and compares a
  static per-player embedding with the evolving latent: evolving wins on 2017
  and 2021 (−0.017/−0.019, CIs exclude zero) and is null on 2019 (−0.001), so
  evolution beyond stable calibration is significant on 2/3 cohorts.
  Crucially, the
  timing head reads only the latent, never the board trunk, so the robust pillar
  is *structurally* invariant to backbone strength (also confirmed: timing D−B
  stays P=1.00 under the conv trunk) — which is why we lead with timing. On the
  **move** channel, a released-SOTA check tells the when-not-what story from the
  other side: the evolving latent recovers a player's *deviation from Maia-2*
  (log P_maia of the played move) at only **R²≈0.009** (vs 0.93 for a synthetic
  hidden state) — move choice is near-stateless even relative to a human-move
  SOTA. (`results/g4_timing.txt`; `scripts/g4_{cache_maia,cache_allie,run_timing,run_allie}.py`.)
- **Generality (RQ5) and population recovery (F) now hold on real data**
  (ASSISTments 2009), not just synthetic KT: on the *same* 500 real students the
  evolving latent beats the memoryless twin at predicting responses (D−B=−0.0095
  originally; −0.0128 mean on the leakage-fixed re-run below, both P=1.00 every
  seed, robust across a 150–500-student cohort sweep) and recovers the accuracy
  distribution (Wasserstein 2× < average-person, corr 0.96). **We since found
  the standard preprocessing recipe (theophilee) drops a real response-time
  column the raw file has (`ms_first_response`); we re-derived it and ran the
  timing channel on real students for the first time.** The result is a clean
  negative, not the hoped-for cross-domain confirmation: response stays
  significant every seed (P=1.00), but timing is inconsistent across seeds (one
  null, one wrong-sign-significant, one right-sign-significant) and doesn't
  stabilize with more training (`results/real_kt_rt.txt`). So the timing-vs-
  response asymmetry does **not** transfer to real ASSISTments response times —
  and a frozen EdNet-KT1 test reaches the same conclusion on 500 students:
  response significantly favors D (pooled −0.0159, CI [−0.0202,−0.0118]) but
  timing is null (−0.0004, [−0.0059,+0.0059]). The EdNet primary analysis uses
  singleton bundles to avoid its unresolved bundle-time ambiguity. Thus the
  asymmetry stays supported by real chess + synthetic KT only, not claimed as
  a cross-domain law on real data. A plausible, unproven reason is that these
  education fields are incidental UI timers rather than a strategically
  managed self-paced clock.
- **What the edge *is*, mechanistically (dynamics vs. individualization).** On
  synthetic players with a *known* hidden state the latent provably encodes and
  causally uses it (RQ2, probe + clamp). On **real** data the evolving-vs-
  memoryless edge blends genuine state-response with **per-individual online
  calibration**: temporal-shuffle controls on KT leave the edge intact —
  because the equal-capacity control already receives the instantaneous state
  features, the latent's *marginal* gain there is individualization, not
  order-tracking — and a clean chess shuffle is confounded by within-game clock
  structure. We therefore support the real-data *dynamics* reading via the
  **concentration** signature (the edge concentrates under time pressure, which a
  uniform individualization edge would not) and reserve the strongest
  state-*tracking* claim for the synthetic probe/clamp. **This concentration
  signature could itself be a variance artifact (high-time-pressure decisions
  are also the noisiest), so we re-ran it player-bootstrapped and normalized by
  each bucket's own decision-level stdev: the standardized effect survives
  (2.7–3.6× high-vs-low, down from a raw 4.0–6.1×, across 2 cohorts × 2 seeds;
  `results/concentration_variance_controlled.txt`)** — the dynamics reading is
  not merely a noise artifact. The headline empirical claim (evolving >
  memoryless on future behavior) is unaffected either way.
- **No live human study — future-split real-behavior prediction is a
  complementary, not categorically stronger, form of evidence.** HumanLM ran a
  111-participant live study; a reviewer may hold us to that bar. We do not
  claim held-out prediction on 480+ real players' *future* games over a 6-year
  span *substitutes* for interactive human evaluation — the two answer
  different questions (does the model predict what a specific real person will
  do next, vs. does an evaluator judge an interaction as human-like) and each
  has failure modes the other cannot catch (a live study can't detect
  future-behavior drift over months; held-out prediction can't detect
  interaction-level authenticity). We report the one we ran, at a scale (480+
  players, real Lichess/ASSISTments history) a lab session could not match, and
  regard live evaluation as complementary future work, not a gap the current
  numbers already fill.
- **Practical significance of the headline NLL gap.** A −0.01 to −0.07 nats
  improvement invites "does it matter?" We do not rest that answer on the raw
  NLL number. Two more direct utility demonstrations: (i) **calibration** — the
  timing head is a proper log-normal likelihood, so the NLL gap is directly a
  held-out log-likelihood/deviance improvement, not an arbitrary loss units
  comparison; (ii) **Milestone F** (population heterogeneity recovery and
  generation) is the standalone practical payoff — the per-individual latent
  reconstructs the *distribution* of real players' behavior (Wasserstein 2–5×
  closer than an average-person point-mass, recall 0.75–1.00 vs. 0.00) and
  *generates* novel, plausible, diverse players by sampling from a fitted
  latent prior. That is a capability an average-person model structurally
  cannot have regardless of how small its own point-prediction gap looks, and
  we present it as a separate demonstrated utility — not as an argument that
  the point-prediction gap itself is large.
- **Go — attempted, no effect at any board size (honest negative).** We built a
  real-Go pipeline (OGS: per-move think-times from the game JSON; oracle-free) and
  ran the timing D-vs-B. A naive mixed-cohort run *looked* positive (D−B ≈ −0.002,
  P=1.00 in 2/3 seeds) — but a **homogeneity control overturned it**: split by
  board size, the effect is **absent on 19×19** (592 trajectories, well-powered;
  D−B ≈ +0.001, ns, one seed significantly *worse*) and **13×13** (null), leaving
  only a weak, seed-unstable **9×9** signal (n=209, 2/3 seeds P=1.00). So the
  mixed-cohort "positive" was a **board-size/speed confound** (the latent
  detecting the game *regime* — fast 9×9 vs slow 19×19 — i.e. per-trajectory
  think-time *level*). We then **chased that one residual 9×9 signal to a 2.5×
  larger 9×9-only cohort** (a fresh scan → 1554 games → N=519 trajectories): it
  **collapses to null** (D−B −0.0015 / +0.0011 / +0.0001, mean ≈ 0, only 1/3 seeds
  significant, two wrong-signed) — the weak 9×9 effect was **small-cohort noise**,
  not a real signal. A **cross-game** framing (game W/L → post-loss/momentum) was
  also null. **We therefore make no Go claim at any board size:** under a
  homogeneity control *and* a power check, the evolving latent does not beat the
  memoryless twin on real-Go think-time.
  Go remains **future work** — likely needing richer within-game signal (the
  actual byo-yomi clock, not a proxy) and/or the move channel (a Go board
  backbone). The generality claim rests on **chess + knowledge tracing** (both
  real). This is the kind of control a reviewer would run; better we ran it, and
  then powered it up.
- 2013 archives lack `[%clk]` (move-NLL only); timing uses a 2017 prefix.

## Positioning (vs 2026 prior art; full section in `documents/related_work.md`)

**No single axis here is unclaimed** — after a 2026-07 prior-art sweep we say this
plainly. Per-individual chess style (Elo-Disentangled, arXiv:2606.25176, which
beats Maia-3 on move NLL; Player-Specific; Mixture-of-Masters), cohort move+clock
(ChessMimic), dynamic emotional chess (Ailed, arXiv:2603.05352), the principle
that *timing reveals latent state better than choice* (decades of response-time
psychometrics; Latency-Response Theory), and an evolving latent validated on a
future split (LATTE, HumanLM) all already exist. We build on them and must not
re-claim them.

Our contribution is the **controlled empirical synthesis those lines lack**,
which we frame as three findings, each stated in its *honest, differentiated*
form:
1. **The when-not-what asymmetry.** That latency exposes latent cognitive state
   more richly than discrete choice is old news in psychometrics; what is new is
   its form here — an *evolving within-session behavioral state*
   (tilt/fatigue/time-pressure), measured against a *per-decision engine oracle*,
   validated on a *strict future split*, robust across a six-year era span, and
   reproduced on synthetic KT — with move choice a near-null. Frozen real-time
   tests on ASSISTments and EdNet are null, so this is not presented as a real
   cross-domain timing law.
2. **The equal-capacity evolving-vs-memoryless control on a real future split.**
   Evolving-vs-memoryless is a routine seqrec ablation; our differentiator is
   running it as an *equal-capacity, same-input* control on a per-decision oracle
   domain with a strict future split, which isolates the value of accumulating
   state from merely seeing recent history — the control the human-chess and
   user-simulation lines do not run, and the answer to the #1 "isn't this just
   history-conditioning?" objection.
3. **The backbone-dependent hidden-vs-verbal channel ordering.** HumanLM/LATTE
   each commit to one channel (verbal text / a single soft token); we compare
   them head-to-head and show the ordering *flips* with the backbone's language
   prior — hidden richer with no language prior (board-native RQ6, −0.069/−0.117),
   verbal richer inside an instruction-tuned LLM (G3, Qwen3, 3 seeds).

Closest in spirit is **Ailed** (dynamic emotional modulation of chess move + latency),
but it is a *generative* engine with, in its authors' words, **no human-subject
validation** — its state dynamics are asserted, not measured against real players.
We make the corresponding claim falsifiable: the evolving state is fit to and
scored against specific players' held-out future games, its value established by
the equal-capacity control rather than by construction. (`documents/related_work.md`
carries the full comparison and citations; verify the arXiv IDs marked unverified
there before submission.)
