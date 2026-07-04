# Grounded player simulation with a dynamic per-individual latent state

*Paper-skeleton synthesis of the landed results. Detailed tables +
reproduction in `results_ec.md`; prior-art positioning in `design.md §8`;
work plan in `TODO.md`.*

## Abstract

People do not play a game (or answer a quiz) the same way twice: a recent loss
breeds tilt, a long session brings fatigue, a ticking clock changes everything.
We model *how a specific person behaves right now* with a learned latent state
`z_t` that (i) is **per-individual**, (ii) **evolves** over the person's own
action+timing trajectory, (iii) carries a behavioral state (tilt / fatigue /
time-pressure), and (iv) is validated against the person's **future** behavior
on a strict temporal split. The injector is decoupled from a swappable policy
backbone, so "does the dynamic latent help?" is provable *independent of
backbone*. On real Lichess chess the evolving latent significantly beats both a
fixed per-individual style and an equal-capacity **memoryless** control at
predicting future behavior — a result that holds **across a 6-year era span
(2017–2023), two backbones, and five seeds**, and that **adds value over a
near-SOTA Elo+clock+complexity baseline**. The edge is concentrated where it
should be: the latent's advantage lives in **think-time** (how long a person
deliberates) and **concentrates** under time pressure (2–8×) and for weaker
players (≈3×), while **move choice** carries essentially no extra
state-dependence — the state is legible in *when* a person acts, not *what* they
play. The same framework, with only the encoder/oracle swapped, reproduces the
pattern in **knowledge tracing** (a non-game domain, replicated across five real
datasets, two platforms, and three subjects) and **generates** a population that
recovers real heterogeneity a "positive average person" baseline cannot. Across
both domains one law emerges: the latent's advantage **scales with behavioral
heterogeneity** — across students, players, and moments alike, it helps most
precisely where individuals differ most (Pearson 0.89 across eight real
datasets — a strong trend anchored by the extremes, noisier in the middle). In
an **actual LLM policy** (Qwen3), a behavior-cloning SFT probe
reproduces the same asymmetry — the state helps the LLM's think-time prediction
(Δ ≈ −0.013, **robust across LoRA and full fine-tuning, and across
0.6B → 8B**) much more than its move choice — while
confirming that a *verbal* prompt
is a weaker channel than the trained *hidden* latent. (The move channel's
apparent collapse to a clean null at scale is a **LoRA-capacity artifact**:
full-param fine-tuning at *both* 4B and 8B recovers a stable move benefit, so the
timing ≫ move asymmetry is robust but *graded*, not a clean null in a
full-capacity LLM.) The contribution
is the **conjunction**: no single axis is novel alone.

## Contributions

1. A game-agnostic **dynamic latent-state injector** decoupled from a swappable
   backbone; the same injector + trainer run on chess and knowledge tracing
   unchanged.
2. The first test, on **real** data and a strict **future** split, that an
   evolving latent beats an equal-capacity **memoryless history-conditioned**
   control — the #1 reviewer objection, settled (RQ1, E-C2/E-C3).
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
5. A **unifying scaling law**: the latent's advantage grows with **behavioral
   heterogeneity** at three granularities — across *populations* (the KT effect
   size correlates **Pearson 0.89 across eight real datasets** with per-student
   accuracy spread — a strong trend anchored by the extremes, noisier in the
   middle band), across *players* (chess timing edge ≈3× larger for the
   most-variable players), and across *contexts* (2–8× under time pressure). One
   statement — *personalization pays off in proportion to how much individuals
   differ* — ties the chess and KT results together and explains why the same
   latent recovers population heterogeneity (Milestone F).
6. **Generality** (RQ5) in a non-game oracle domain (KT) and a **heterogeneity-
   recovery** result (Milestone F) beating the "positive average person" — both
   confirmed on **real data** (ASSISTments 2009): the evolving latent beats the
   memoryless twin on real student responses (500 students, D−B≈−0.010, P=1.00
   in all 3 seeds, D wins 64–73%; robust across a 150–500-student cohort sweep,
   **and replicated across 8 datasets / multiple platforms / 3 subject domains —
   ASSISTments 2009/2012/2015/2017, KDD-Cup Algebra + Bridge-to-Algebra, Spanish
   (language) and Statics (engineering), significant every seed; the effect size
   **scales with population heterogeneity**, Pearson 0.89, n=8**), and — on the
   same
   500 students — recovers the real accuracy distribution (Wasserstein 2× better
   than average-person, corr 0.96, recall 0.75 vs 0.00).
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
All P below are P(D−B<0); CIs are 95% bootstrap over players.

| RQ | Experiment | Result |
|----|-----------|--------|
| RQ1 | **E-C2** dynamic > memoryless (real chess, 100 players) | D−B ≈ −0.069, P=1.00; survives B at 2× latent width |
| RQ1 | **E-C1** dynamic > static-individual embedding | pooled −0.120, CI [−0.135,−0.106], P=1.00 |
| RQ3 | **E-C3** future-*sessions* split | pooled −0.067, CI [−0.077,−0.057], P=1.00 |
| — | replication (rapid, diff. time control) | −0.062, P=1.00 |
| RQ2 | **E-C4** state-recovery probe (presence) | held-out R²: D=0.93 vs B=0.65 |
| RQ2 | **E-C4** causal clamp (use) | monotone dose-response, expected direction |
| §5 | concentration (synthetic) | edge monotone-concentrated in high-tilt decisions |
| §5 | concentration (**real timing**) | think-time edge **2–8× larger under time pressure** (robust 2 cohorts/2 seeds); flat for post-loss/fatigue |
| — | rating stratification | timing edge **≈3× larger for weakest vs strongest players** (high−low +0.027, P(high<low)=0.00, 480 players) |
| E-C6 | **timing** vs memoryless (zero-inflated head) | −0.026, P=1.00 (−0.069 log-normal) |
| E-C6 | **timing** adds value over Elo+clock aggregate | (B4+z)−B4 = −0.043, P=1.00 |
| E-C6 | ... and over a **position-aware** baseline (+ branching factor) | (B4+z)−B4 = −0.035, P=1.00; baseline Spearman 0.39 ≈ ChessMimic 0.41 |
| RQ5 | **E-D1** knowledge tracing (non-game) | timing D−B=−0.050, P=1.00, D wins 100% |
| RQ5 | **E-D1 real** (ASSISTments 2009, 500 students) | response D−B≈−0.010, P=1.00 in all 3 seeds, D wins 64–73% (robust across 150–500 cohort sweep) |
| RQ5 | **E-D1 replication** (8 datasets, multi-platform, 3 subjects) | ASSISTments 09/12/15/17 + KDD Algebra/Bridge + Spanish (language) + Statics (engineering); D−B −0.004…−0.03, **significant every seed** — not dataset/platform/subject-specific |
| E-D2 | **Go** timing (real OGS, oracle-free) | **no effect at any board size**: null on 13×13 & 19×19 (well-powered); mixed-cohort "positive" was a board-size confound; the one residual weak 9×9 signal (n=209, 2/3 seeds) **collapses to null on a 2.5× larger 9×9 cohort** (N=519, 1/3 seeds, mean ≈0) — small-cohort noise; Go = future work |
| RQ5↔F | effect size **scales with population heterogeneity** | \|D−B\| vs accuracy-spread **Pearson 0.89** (n=8); strong linear trend anchored by the extremes, noisier middle (Spearman 0.74) — latent helps most where individuals differ most |
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

And a single principle governs *where* it helps: **the latent's edge scales with
behavioral heterogeneity**. This shows up at three granularities — across
*populations* (the KT effect size correlates Pearson 0.89 with per-student
accuracy spread across 8 real datasets), across *players* (the chess timing edge
is ≈3× larger for the weakest, most-variable players), and across *contexts* (2–8×
larger under time pressure). The evolving latent buys the most exactly where
behaviour is least predictable from the average — which is also why it recovers
population heterogeneity (Milestone F).

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
- **Weak from-scratch backbone (largely mitigated for timing).** The MLP trunk
  gets ~3.0 move-NLL (beats uniform 3.18). The low absolute think-time Pearson
  (~0.14) was a *missing feature*: adding **position complexity** (branching
  factor) to the baseline lifts it to **Spearman 0.39 ≈ ChessMimic's 0.41** —
  and the evolving latent *still* adds significant value (P=1.00) on top,
  replicated in **all 10 runs** of a 5-seed × 2-cohort sweep. Crucially, the
  timing head reads only the latent, never the board trunk, so the robust timing
  pillar is *structurally* invariant to backbone strength (confirmed: timing D−B
  stays P=1.00 under the conv trunk) — which is precisely why we lead with
  timing. A stronger/pretrained trunk raises the absolute *move* ceiling without
  touching the timing claim.
- **Generality (RQ5) and population recovery (F) now hold on real data**
  (ASSISTments 2009), not just synthetic KT: on the *same* 500 real students the
  evolving latent beats the memoryless twin at predicting responses (D−B=−0.0095,
  P=1.00, robust across a 150–500-student cohort sweep) and recovers the accuracy
  distribution (Wasserstein 2× < average-person, corr 0.96). Residual limitation:
  this public KT release has **no response-time column**, so the real-KT result
  uses the *correctness* channel only — we cannot check the timing-vs-response
  asymmetry on real students the way we do on chess (the synthetic KT, which has
  timing, shows the same timing-robust / response-weak signature).
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
  state-*tracking* claim for the synthetic probe/clamp. The headline empirical
  claim (evolving > memoryless on future behavior) is unaffected either way.
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

## Positioning (vs 2026 prior art; design.md §8)

Each single axis is already owned by a competitor — "evolving latent in an LLM"
(LATTE), "natural-language latent" / "future temporal-split validation"
(HumanLM), aggregate think-time (Allie), per-move clock (ChessMimic), strong
move prediction (Maia-3). We claim **none** of these alone. The contribution is
their **conjunction** on a per-decision engine-graded interface, plus the two
results no competitor reports: the equal-capacity *evolving-vs-memoryless*
control on a real future split, and the *same framework reproducing the channel
signature in a non-game domain* (knowledge tracing).
