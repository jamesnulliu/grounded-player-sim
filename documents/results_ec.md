# E-C results: the dynamic latent on real chess (RQ1 + RQ3)

Status: **landed** on real Lichess data. This file is the running results table
for the chess headline; numbers are deterministic (CPU, fixed seeds) and
reproducible with the commands at the bottom.

> **Backbone caveat (read first).** All results below validate the *dynamic
> latent mechanism* on small **from-scratch** backbones (board-native MLP; KT
> logistic head) — **not on an LLM**. The `sglang`/API LLM backbones are stubs;
> no experiment has run one. The D-vs-B comparison is valid for any shared
> backbone, but the project's "LLM policy + dynamic latent" thesis is **not yet
> empirically demonstrated**. Embedding the method in an LLM agent is the #1
> open item — see `TODO.md` Milestone B.

## Claim

A per-individual, **temporally evolving** latent state injected into a move
policy predicts a player's *future* moves better than (a) a fixed per-player
style and (b) an equal-capacity memoryless control that sees the same history
features — and the win survives a strict *future-sessions* split. This answers
the #1 reviewer objection ("isn't the latent just history-conditioning?") on
real chess, not only the synthetic Milestone-A toy.

## Setup

- **Data:** Lichess open database `2013-01`, ingested with `gps ingest`
  (header-only cohort pass → full sharded parse of the cohort only). One speed
  class at a time. Bot accounts and bot-opponent games excluded.
- **Cohort:** 100 players with ≥30 games and ≥3 sessions, each capped to their
  earliest 20 games (`--max-games-per-player`, so lengths are bounded and
  comparable for the trainer). ~70k decisions / cohort.
- **Backbone:** `BoardNativeBackbone` — oracle-free (FEN → 12×64 planes →
  factored from/to logits masked to legal moves). Weak but shared by both arms,
  so the **comparison** is valid regardless of absolute strength.
- **Arms (equal/near-equal capacity, equal inputs; only the latent differs):**
  - **D** — `NeuralInjector(persist=True)`: the evolving latent (GRU over the
    shared `history_features`).
  - **B (memoryless)** — `NeuralInjector(persist=False)`: same params/inputs,
    no accumulation (E-C2 / E-C3 control).
  - **B2 (static individual)** — `StaticIndividualInjector`: a per-player
    `nn.Embedding`, constant over the trajectory (E-C1 control).
- **Split:** strict per-player temporal split. `fraction` = last 30% of moves
  (E-C2); `session` = hold out the player's later **sessions** (E-C3 / the
  decisive RQ3 form). Latent warmed over the whole history, scored on held-out.
- **Metric / test:** held-out next-move NLL; significance by **bootstrap over
  players** (the independent unit, design.md §5). Early stopping at 15 epochs
  (the from-scratch head overfits beyond ~30). Lower NLL is better; **D−B < 0
  means the evolving latent wins.**

## Results — `2013-01` blitz (100 players, uniform-over-legal = 3.18)

| Exp | Control | Split | D−B (pooled, 3 seeds) | 95% CI | P(D−B<0) | D wins | seeds |
|-----|---------|-------|----------------------:|--------|---------:|-------:|-------|
| **E-C1** | static individual (B2) | session | **−0.120** | [−0.135, −0.106] | 1.000 | 78–90% | all neg, all P=1.00 |
| **E-C2** | memoryless (B) | fraction | **≈−0.069** | per-seed [−0.095, −0.039] | 1.00 | 72–84% | all neg, all P=1.00 |
| **E-C3** | memoryless (B) | session | **−0.067** | [−0.077, −0.057] | 1.000 | 72–88% | all neg, all P=1.00 |

Both arms beat uniform (D ≈ 2.97–3.04, B/B2 ≈ 3.06–3.15 < 3.18), so the model
is sane and the gap is real signal, not noise above chance. B2 is *slightly
larger* than D (64402p vs 63926p), so D winning E-C1 is conservative.

## Replication — `2013-01` rapid (100 players, different time control)

| Exp | Control | Split | D−B (pooled, 3 seeds) | 95% CI | P(D−B<0) | D wins |
|-----|---------|-------|----------------------:|--------|---------:|-------:|
| **E-C3** | memoryless (B) | session | **−0.062** | [−0.072, −0.052] | 1.000 | 73–85% |

Nearly identical to blitz (−0.067) on an independent player population and game
length → the effect is not a blitz/`2013-01` artifact.

## Second month + timing — `2017-04` blitz (clocked, 100 players)

A genuinely different month (ingested from a clocked ~180MB HTTP-range prefix;
100% of moves have `[%clk]`, median think-time 2s, ~9% zero). E-C3 session
split, 3 seeds, **move-NLL and think-time NLL** (the timing head's log-normal
``mu`` is a function of the evolving latent; the same head is used by both arms,
so the D−B comparison is valid even though the absolute log-normal is
mis-specified for 1s-quantized, zero-inflated clocks).

| Metric | D−B (pooled, 3 seeds) | 95% CI | P(D−B<0) | verdict |
|--------|----------------------:|--------|---------:|---------|
| **think-time NLL (E-C6)** | **−0.069** | [−0.082, −0.057] | **1.000** | **significant win** |
| move NLL | −0.0001 | [−0.010, +0.010] | 0.52 | **null (tied)** |

**Zero-inflated timing head (proper model).** The log-normal is mis-specified
for 1s-quantized, zero-inflated clocks (~9% are 0s premoves). With a
**zero-inflated log-normal** head (`timing_model="zi_lognormal"`: a learned
latent-driven mass `pi` on a 0s premove + a log-normal on the rest), absolute
NLL drops sharply and **D still significantly beats B**:

| timing head | D | B | D−B (pooled) | 95% CI | P(D−B<0) |
|-------------|--:|--:|-------------:|--------|---------:|
| log-normal | 3.228 | 3.297 | −0.069 | [−0.082, −0.057] | 1.000 |
| **zero-inflated** | **2.462** | **2.488** | **−0.026** | [−0.030, −0.022] | **1.000** |

The proper head fits ~0.77 nats better (it stops mangling the 0s) and the
per-individual evolving advantage persists, significant — a smaller but cleaner
gap on a credible model.

**Vs the aggregate baseline (B4 / Allie) — the *fair* framing.** Comparing our
latent-only timing head to an Elo+clock aggregate is unfair: the clock is too
strong a predictor, and the latent head doesn't see it. So **B4 (aggregate)
beats the latent-only model** (NLL 2.69 vs 3.23). The right question is whether
the **evolving latent adds value *over* the aggregate** — test B4 vs **B4+z**
(B4 = log-normal on Elo, move-number, log time-remaining; B4+z = the same plus
the per-step evolving latent), held-out, bootstrapped over players:

| model | held-out think-time NLL |
|-------|------------------------:|
| latent-only (our head) | 3.23 |
| **B4 (Elo+clock aggregate)** | **2.69** |
| **B4+z (aggregate + evolving latent)** | **2.65** |

**(B4+z) − B4 = −0.043, 95% CI [−0.056, −0.031], P(<0)=1.00** (2 seeds). So the
evolving per-individual latent **adds significant predictive value over a strong
Elo+clock aggregate** — the defensible E-C6 claim (it *augments* the aggregate,
it does not replace the clock). Per-player Pearson(pred, actual time) rises
0.143 → 0.15 with the latent.

**Closing the Pearson gap — a position-aware baseline (`position_aware=True`).**
The low absolute Pearson (0.14 vs ChessMimic's 0.41) was a *missing feature*, not
a fundamental limit: real think-time tracks **position complexity**. Append the
**branching factor** (legal-move count, oracle-free) to B4 and re-run (2017,
60 players):

| baseline | NLL | (B4+z)−B4 | Pearson | Spearman |
|----------|----:|----------:|--------:|---------:|
| B4 = Elo+clock | 2.677 | −0.043 (P=1.00) | 0.132 | 0.248 |
| **B4 + complexity** | **2.635** | **−0.0345 (P=1.00)** | **0.226** | **0.386** |

The complexity feature lifts rank-correlation from 0.25 to **Spearman 0.386 —
essentially ChessMimic's 0.41** — *and the evolving latent still adds significant
value (P=1.00) over this much stronger, position-aware baseline.* So the timing
claim survives a near-SOTA-strength competitor: the per-individual evolving state
is predictive **beyond** Elo + clock + position complexity.

**At scale (5 seeds × 2 clocked cohorts, 2×A100).** The position-aware test
replicates in **all 10 runs** (`results/posaware_runs/`):

| cohort | (B4+z)−B4 (5-seed mean) | all P=1.00? | B4 Spearman |
|--------|------------------------:|:-----------:|------------:|
| 2017-04 | −0.0315 | **yes** | 0.371 |
| 2019-07 | −0.0266 | **yes** | 0.398 |

Every seed/cohort: the evolving latent significantly augments the
Elo+clock+complexity baseline (P=1.00), and that baseline holds Spearman
≈ 0.37–0.40 ≈ ChessMimic's 0.41. The strongest timing claim is robust, not a
single-seed artifact.

**Two findings, stated honestly:**
1. **Timing is where the evolving state robustly helps.** The per-individual
   evolving advantage holds three ways on 2017-04 (all P=1.00): vs the
   memoryless twin (log-normal −0.069, zero-inflated −0.026), and as a
   significant **add-on over an Elo+clock aggregate** (−0.043). It is the E-C6
   differentiator.
2. **The move-NLL advantage is cohort-dependent** — strong on `2013-01`
   (blitz −0.067, rapid −0.062, P=1.00), **null on `2017-04`** (−0.0001, P=0.52).
   Two hypotheses tested:
   - *Rating* (weaker players tilt more in move quality): **refuted.** Splitting
     2017 by Elo, neither band shows a move win (low band mean 1496: D−B=+0.014,
     P=0.07; high band mean 1923: +0.021, P=0.00) — yet **timing wins in both**
     (D−B −0.100 / −0.142, P=1.00).
   - *Shared-latent timing tradeoff* (clocked 2017 trains with λ=0.5 timing
     loss, which 2013's degenerate clocks make a no-op): **partially confirmed.**
     Training 2017 **move-only (λ=0)** shifts move from −0.0001 (P=0.52) to
     −0.0066 (P=0.88) — better, but **still not significant** and far from
     2013's −0.067. So the joint timing objective suppresses *some* of the move
     advantage (the heads compete for one latent), but 2017 also has a genuinely
     weaker move-dynamics signal than 2013.

**Bottom line:** **timing is the robust, large, cohort- and rating-universal
pillar; the move-dynamics advantage is real on 2013 but era-dependent and
smaller, and is further suppressed by the shared-latent timing objective.** For
the paper: lead with timing; report move as significant-on-2013, cohort-
dependent; note the latent-capacity tradeoff (separate move/timing read-outs or
λ-tuning is a design lever). Pending: more months, a stronger move model.

## State recovery — does the latent *encode* the hidden state? (E-C4 / RQ2)

`run_state_recovery` linearly probes each trained arm's latent for the
**ground-truth hidden state** (only available on the synthetic
`HiddenTiltChessPlayer`, whose `hidden_h` leaky-loss integral is recorded per
decision). **Held-out** R² (probe fit on train steps, scored on the temporal-
split tail), so a high number means the state is present *and generalises*.

| Arm | held-out R²(latent → hidden_h) | train R² |
|-----|-------------------------------:|---------:|
| **D (evolving)** | **0.929** | 0.980 |
| B (memoryless) | 0.654 | 0.884 |

The evolving latent recovers the hidden behavioural integral far better
(ΔR² = +0.27). The memoryless arm gets 0.65 (the instantaneous `momentum` /
`post_loss` features are partial proxies for `h`), but only accumulation
recovers the full ordered integral. This is RQ2 (the latent *encodes* the
state) and it **explains why D predicts behaviour better** (links E-C2 → RQ2).

**Causal intervention (presence → use).** `run_causal_intervention` clamps the
latent ±α·σ along the probed hidden-state direction and measures the change in
held-out predictions. The effect is **monotone in α and in the expected
direction** — clamping toward "tilted" flattens the move distribution (entropy
up → more blunders) and slows think-time:

| clamp α (σ) | move entropy Δ | move KL(tilt‖calm) | timing μ Δ |
|------------:|---------------:|-------------------:|-----------:|
| 1 | +0.0034 | 0.0007 | +0.0086 |
| 2 | +0.0068 | 0.0028 | +0.0171 |
| 4 | +0.0136 | 0.0109 | +0.0343 |

So the policy **uses** the latent, not just carries it (presence *and* use).
Magnitudes: the move response is modest (consistent with the small move-NLL
effect) and the timing response is relatively larger (consistent with timing
being the robust pillar) — the clamp moves the right knobs in the right
directions.

## Concentration — does the latent help *where the dynamics are*? (design.md §5)

`run_concentration` buckets the **held-out** per-decision D−B gap by the
ground-truth hidden-state level (synthetic hidden-tilt player; 24 players ×
30 games). If the latent is genuinely modelling dynamics, its edge should
**localize to high-tilt decisions**, not spread uniformly:

| bucket | move dD−B | timing dD−B |
|--------|----------:|------------:|
| low-tilt (calm) | +0.002 | +0.142 |
| mid-tilt | −0.089 | +0.026 |
| **high-tilt** | **−0.154** | **−0.118** |

**Monotone, both channels.** At the calm baseline the evolving latent is no
better than the memoryless reader (the easy regime); its advantage appears, and
grows, exactly as the hidden tilt rises. This is the §5 concentration check
confirmed — and it explains *why* the pooled effect is modest: the advantage
lives in the high-dynamics tail, diluted across the calm majority. The latent
helps **where it should**.

### On REAL chess: the timing edge concentrates under **time pressure**

The same analysis on real clocked data (`run_concentration(bucket_feature=...)`,
2017-04/2019-07 blitz, 80 players) buckets the held-out timing D−B by an
**observable** anchored dimension. The standout is **time pressure** (low clock):

| time-pressure bucket | 2017 s0 | 2017 s1 | 2019 s0 |
|----------------------|--------:|--------:|--------:|
| low  | −0.046 | −0.051 | −0.026 |
| mid  | −0.029 | −0.006 | +0.024 |
| **high** | **−0.108** | **−0.107** | **−0.211** |

The latent's think-time advantage is **2–8× larger under time pressure** —
robust across seeds and cohorts. Human-meaningful: when the clock is low,
time-allocation becomes most individual and state-dependent (panic, flagging,
premoves vs. composure), and only the *evolving* latent tracks it; a memoryless
twin cannot. The concentration is **specific** — bucketing by `post_loss`
(≈flat: −0.063/−0.068/−0.053) or `fatigue` (non-monotone) shows no such pattern.
So the timing edge is not uniform: it lives exactly where a person's clock
management turns on their current state. *(Caveat: high-time-pressure is also the
highest-variance regime, which amplifies NLL gaps.)*

### ...and for **weaker players** (rating stratification)

Stratifying the Tier-1 per-player timing D−B by player **Elo** (480 players,
4 clocked cohorts, pooled seeds+trunks — no retraining, just re-bucketing the
sweep cells):

| Elo band | timing D−B |
|----------|-----------:|
| low (769–1634, weakest) | **−0.0404** |
| mid (1635–1892) | −0.0296 |
| high (1893–2516, strongest) | **−0.0139** |

Monotone and significant: **high-Elo − low-Elo = +0.0265, 95% CI
[+0.015, +0.038], P(high<low)=0.00** — the latent helps the **weakest players
≈3× more** than the strongest. Same story as time pressure: weaker players have
more erratic, state-dependent clock management, so the evolving latent captures
more of it; disciplined strong players are closer to memoryless-predictable.
Together these say the per-individual evolving state pays off **exactly where
human behaviour is most variable** — high-pressure moments and lower-rated
players.

### The contrast: the *move* edge does NOT concentrate (it's flat)

Running the identical bucketed analysis on the **move** channel
(`channel="move"`, 2017/2019, by `post_loss` and `time_pressure`) gives a clean
**null**: the move D−B is ≈0 in *every* bucket and both cohorts (2019:
−0.000±0.0004 across buckets; 2017: +0.003±0.001, i.e. D marginally *worse*) —
no concentration anywhere. So the asymmetry is real and interpretable: the
evolving state is legible in **how long a person thinks** (timing concentrates
sharply under pressure) but **not in which move they pick** (flat everywhere).
Think-time is a more direct readout of cognitive/emotional state than move
choice, which the position largely constrains. This is *why* we lead with
timing — and it reframes the weak move channel honestly: not a faint signal to
amplify, but a genuine absence of state-dependence in move choice (at this
backbone scale).

## Capacity robustness — is D just "more effective capacity"?

Give the memoryless control B a **wider** latent than D and re-run E-C3
(blitz, session split, seed 0):

| B width | D params | B params | D−B | 95% CI | P(D−B<0) | D wins |
|---------|---------:|---------:|----:|--------|---------:|-------:|
| 1× (equal) | 63926 | 63926 | −0.053 | [−0.072, −0.035] | 1.00 | 72% |
| 2× | 63926 | 67622 | −0.056 | [−0.077, −0.035] | 1.00 | 70% |
| 4× | 63926 | 79622 (+25%) | +0.013 | [−0.006, +0.031] | 0.09 | 42% |

D's win **survives B at 2× the latent width** and only closes at 4× (≈+25%
total params) — so the advantage is the *dynamics*, not raw capacity: a
memoryless control needs several times the latent width merely to draw level
(the same shape as the E-A1 capacity sweep). *(Single seed; pool seeds for a
firm crossover.)*

## At scale: 5-seed × 5-cohort × 2-backbone sweep (Tier-1, 2×A100)

The single-seed numbers above are honest but noisy. The Tier-1 sweep settles
them: **5 cohorts** (2013-01 unclocked; **2017-04, 2019-07, 2021-04, 2023-04**
clocked — a 6-year era span) × **5 seeds** × **{mlp, conv}** = **50 runs** on 2
A100s, 100–120 players/cohort, session split, pooled by averaging each player's
`D − B` across seeds then bootstrapping over players. Reproduce:
`scripts/tier1_sweep.sh` + `scripts/sweep_pool.py` (50 raw cells +
`results/tier1_pooled.txt` in `results/`).

**Timing — the headline, robust across a 6-year era span (clocked cohorts):**

| cohort | mlp timing D−B (P) | conv timing D−B (P) |
|--------|-------------------:|--------------------:|
| 2017-04 | −0.0262 (**1.00**) | −0.0329 (**1.00**) |
| 2019-07 | −0.0277 (**1.00**) | −0.0331 (**1.00**) |
| 2021-04 | −0.0239 (**1.00**) | −0.0319 (**1.00**) |
| 2023-04 | −0.0211 (**1.00**) | −0.0270 (**1.00**) |

The evolving latent beats the memoryless twin on think-time in **all 8 clocked
conditions** — 4 eras (2017→2023) × 2 backbones, 5 seeds each, **every P=1.00**,
effect size tightly clustered at ≈ −0.021 to −0.033. This is the robust pillar
at scale and across eras. *(2013 is unclocked — its "timing" is on degenerate
≈1 ms clocks and is excluded.)*

**Move — small and backbone-dependent:**

| cohort | mlp move D−B (P) | conv move D−B (P) |
|--------|-----------------:|------------------:|
| 2013-01 | −0.0027 (1.00) | −0.0149 (1.00) |
| 2017-04 | +0.0005 (0.18 null) | −0.0046 (1.00) |
| 2019-07 | −0.0009 (0.85 null) | −0.0083 (1.00) |
| 2021-04 | +0.0009 (0.13 null) | −0.0059 (1.00) |
| 2023-04 | −0.0048 (1.00) | −0.0175 (1.00) |

The **conv** trunk gives a small-but-significant move win on **every** cohort
(−0.005 to −0.018); the **mlp** trunk is **null** on three of four clocked
cohorts. So a move advantage exists but is small and needs the spatial backbone.

**Pooling corrected the single-seed story.** Last section's dramatic 2×2 swings
were *seed noise*: single-seed 2017 showed mlp +0.032 (D worse) and conv −0.030
(D wins); pooled over 5 seeds they are **+0.0005 (null)** and **−0.0046
(small)**. So we do **not** claim the conv trunk "rescues" the move channel —
the honest, at-scale picture is: **timing robustly wins everywhere; the move
effect is small (conv) to null (mlp).** Lead with timing.

## Reading

- **Dynamics beat a fixed per-player style** (E-C1, largest gap): identity
  alone is not enough; the *evolution* carries signal.
- **Dynamics beat memoryless history-conditioning at equal capacity** (E-C2):
  the dangerous objection, answered on real data.
- **The win survives the future-sessions split** (E-C3): it is real evolving
  dynamics, not habit memorized within one sitting.

## Caveats (state honestly)

- One month; from-scratch board backbone is weak (~3.0 NLL, not Maia-strong) —
  the D-vs-B *comparison* is valid for any shared backbone, but a stronger /
  engine-informed move model should raise the ceiling and (per the synthetic
  positive-control finding) widen the gap.
- `2013-01` has no `[%clk]` → move-NLL only; timing (E-C6) needs a 2017+ archive.
- Move-fraction (E-C2) ≈ session split (E-C3) here because the median session
  boundary lands near 70% of moves; the session split is the principled form.

## Verbal vs hidden injection channel (E-E1 / RQ6)

Nobody compares text-memory vs soft-vector as *interchangeable* injection
channels (design.md §9). We can, cheaply and without an LLM: the **same**
evolving recurrence (same seed) is delivered to the head either as the full
hidden vector (`hidden`) or as the few interpretable anchored `DIMENSIONS` the
*verbal* text encodes (`verbal`, via the injector's `readout`). Near-equal
capacity (63943p vs 63151p), held-out move-NLL:

| cohort | hidden (full vector) | verbal (anchored dims) | hidden−verbal | P(<0) |
|--------|---------------------:|-----------------------:|--------------:|------:|
| synthetic hidden-tilt | 1.262 | 1.331 | **−0.069** | 1.00 |
| real 2013-01 blitz (100) | 3.021 | 3.138 | **−0.117** | 1.00 |

**Hidden is significantly richer** on both (P=1.00): the full vector carries
un-anchored dynamics the few verbal dims drop. The verbal channel — the one
that is **portable to closed APIs** — is lossy by ~0.07 (synthetic) to ~0.12
(real) nats, and the gap is *larger on real data* (real behaviour is richer
than the 4 interpretable axes). So: open-weight soft-prompt injection buys
accuracy; verbal injection buys portability at a measurable cost. Reproduce:
`run_rq6(dataset)`.

## Generality — the same framework in a NON-game domain (E-D1 / RQ5)

The contribution is a **game-agnostic** dynamic-latent core. Test: port the
*exact* injector + trainer + per-player eval to **knowledge tracing**, swapping
only the encoder/oracle head (`KTBackbone`: a logistic correct/incorrect head
over item-difficulty + latent). A synthetic student mirrors the chess
hidden-tilt player — a hidden **frustration** `h` (leaky integral of recent
*errors*) lowers `P(correct)` and slows response time, not reconstructable from
the windowed `history_features`.

The chess pattern **reproduces exactly** (24 students × 120 items, pooled over
4 seeds, ~61% accuracy):

| channel | D−B (pooled, 96 students) | 95% CI | P(D−B<0) | D wins |
|---------|--------------------------:|--------|---------:|-------:|
| **response time** | **−0.050** | [−0.055, −0.045] | **1.000** | **100%** |
| response (correct/incorrect) | −0.001 | [−0.002, +0.001] | 0.77 | 57% |

Just like chess: **timing is where the evolving latent robustly helps**
(P=1.00, every student), while the discrete outcome (move / correct-incorrect)
is weak. The *same channel signature* — timing-robust, discrete-weak — holds in a
domain that is not a game, so the evolving-latent contribution is not a chess
artifact. (On *mechanism* we are careful: the shuffle controls (see the REAL-data
subsection below) show the KT timing edge is order-invariant
**individualization**, not tilt-*tracking* — the
memoryless twin already receives the windowed tilt features, so D's marginal win
is per-trajectory calibration. The claim here is generality of the *signature*,
not of the state-dynamics mechanism, which chess supports via concentration.)
Reproduce: `run_kt(build_kt_dataset(...))` (`gps.experiments.kt`).

**On REAL data (ASSISTments 2009).** We also ran the *unchanged* pipeline on a
real KT dataset (3114 students, 149 skills, 278k responses; item feature = each
skill's empirical difficulty; no response-time column, so the **correctness**
channel only). The loader is a first-class, tested module —
`gps.data.kt_csv.load_kt_csv` (parses the standard 5-column KT export straight
into a `TrajectoryDataset`; `tests/test_kt_csv.py`) — so these real-data results
reproduce from committed code, not a one-off script. On **500 students** (≥50 responses each), the evolving latent
beats the memoryless twin at predicting **real** student responses, and the win
is **seed-stable across 3 seeds**: D−B = −0.0095 / −0.0116 / −0.0090 (mean
≈ −0.010), **P(D−B<0)=1.00 in every seed**, every 95% CI excludes 0, D wins
64–73%. So the per-individual evolving latent extracts a real signal a
memoryless reader misses — a non-synthetic RQ5 result. (Notably the *response*
channel, weak on synthetic data, carries signal on real students.)

The effect is **robust to cohort definition**: a sweep over `n_students ∈
{150, 300, 500} × min_resp ∈ {30, 50}` gives a negative D−B (D beats memoryless)
in *all five* configurations, with P(D−B<0) ≥ 0.96 everywhere; the CI excludes 0
for n≥300 and significance strengthens **monotonically with cohort size** — the
signature of a real effect with tightening bootstrap CIs, not a small-sample
artifact (150-student was marginal at P=0.96–0.97). Full table + sweep script
(`scratchpad/real_kt_sensitivity.py`) in `results/real_kt.txt`.

**Cross-dataset *and* cross-platform replication.** The result is **not specific
to ASSISTments 2009**. Re-running the *identical* pipeline (500 students, 3 seeds)
reproduces it on two further real KT datasets:
- **ASSISTments 2017** (934k responses, same platform, different year): RQ5 D−B =
  −0.0145 / −0.0128 / −0.0143, **P=1.00 every seed**, D wins 65–69%; F Wasserstein
  **3.7× < average-person** (corr 0.92).
- **KDD Cup 2010 Bridge-to-Algebra** (1.8M responses, a *different platform* —
  Carnegie Learning Cognitive Tutor): RQ5 D−B = −0.0045 / −0.0041 / −0.0037,
  **P=1.00 every seed**, D wins 56–59%; F corr(pred, obs) = **0.98** (beats
  average-person, though it under-disperses). The effect is smaller here — a
  high-accuracy dataset leaves less headroom — but the *direction and
  significance are identical*.
- **Spanish vocabulary** (579k responses, a *different subject domain* —
  language, not math): RQ5 D−B = −0.0313 / −0.0329 / −0.0327, **P=1.00 every
  seed**, D wins 77–81% — the *largest* effect of all (language has rich
  per-student variation); F Wasserstein **3.3× < average-person** (corr 0.96).

So the real-KT finding holds across **8 datasets, multiple platforms, and 3
subject domains (math + language + engineering)** — ASSISTments 2009/2012/2015/
2017, KDD-Cup Algebra and Bridge-to-Algebra, Spanish, and Statics — significant
in every seed of every dataset; effect size varies with the domain (largest in
language, smallest on the high-accuracy set) but the sign and significance never
do. Builder `scratchpad/assist17_rq5.py`; raw in `results/real_kt.txt`; all runs
in W&B `gps-kt-scaling`.

**What predicts the effect size? Population heterogeneity — a synthesis across
the four datasets.** The naive guess (harder datasets → bigger effect) is
*wrong*: Spanish has a *high* accuracy yet the largest effect. What actually
tracks the latent's advantage is the **per-student accuracy spread** (the same
heterogeneity Milestone-F measures): across **eight** real datasets, |D−B|
correlates **Pearson 0.89** with observed spread. The relationship is a **strong
linear trend anchored by the extremes** — the least-heterogeneous population
(Bridge-to-Algebra, spread 0.10) has the smallest edge (−0.004) and the most
heterogeneous (Spanish, spread 0.26) the largest (−0.032). We are honest about
its limits: the **middle band** (six datasets at spread 0.12–0.19) is a noisy
plateau (all ≈ −0.008 to −0.014), so the *rank* monotonicity is looser
(**Spearman 0.74** at n=8, down from 0.90 at n=5 — filling in the middle revealed
the plateau). So it is a real *trend*, not a tight monotone law: the latent's
edge is clearly larger in high-heterogeneity populations and clearly smaller in
low ones, with a noisy interior. This is exactly the direction the
**individualization** account predicts — the more a population varies
student-to-student, the more an online per-individual latent can exploit — and it
**links RQ5 to Milestone-F** (the latent's heterogeneity-recovery is *why* it
predicts better). The evolving latent earns its keep where individuals differ
most. (All eight datasets logged to W&B `gps-kt-scaling`.)

**One principle unifies the KT and chess results.** "The latent's edge scales
with behavioral heterogeneity" is not only a KT-across-datasets fact — it is the
*same* thing the chess analyses show at two other granularities: the timing edge
**concentrates ≈3× for the weakest (most variable) players** (rating
stratification) and **2–8× under time pressure** (concentration), i.e. it grows
exactly in the sub-populations and contexts where human behaviour is most
variable. Across *populations* (KT datasets), across *players* (chess rating),
and across *contexts* (chess time pressure), the evolving latent buys the most
where behaviour is least predictable from the average — one mechanism, three
granularities.

**Temporal-shuffle control — what drives the edge (an honest refinement).** To
test whether D's advantage is an artifact of temporal autocorrelation in the
natural response order, we permuted *each student's* response sequence and re-ran
(n=500, seed 0). The edge does **not** collapse — it is undiminished: D−B =
−0.0151 shuffled vs −0.0095 real (both P=1.00, both CIs exclude 0). Both arms
lose absolute accuracy (the held-out tail becomes an order-randomized subset),
but D's edge *over the memoryless twin persists*. So on the correctness channel
the evolving latent's value is **order-invariant**: it comes from accumulating
per-student evidence (**online ability individualization** — B resets its state
each step and cannot) rather than from exploiting the *order* of responses
(learning-curve dynamics). Control: `scratchpad/real_kt_shuffle.py`.

The same shuffle on the **synthetic** cohort (where think-time is generated from
an order-dependent hidden tilt) tells the same story on the **timing** channel:
D−B = −0.0654 real vs −0.0635 shuffled (both P=1.00) — again undiminished. This
looks surprising until one remembers the **equal-inputs** design: the memoryless
twin B already receives the hand-crafted *instantaneous* history features
(`post_loss`, `momentum`, `time_pressure`, `fatigue`; `history_features`), so it
can already read the tilt at each step. The evolving latent's *marginal*
advantage is therefore **not** tilt-*tracking* (B has that) but the learned
per-student *accumulation/calibration* on top — i.e. individualization — which is
inherently order-invariant. This is mechanistically the **same capability** that
drives the Milestone-F heterogeneity recovery below. **Consequence for the
thesis:** the evidence that the *chess* think-time edge is genuinely
*state-dependent* (not just individualization) rests on the **concentration**
analysis (§Concentration: the edge concentrates 2–8× under time pressure and ≈3×
for weaker players — a uniform individualization edge would not), plus the
synthetic **state-recovery probe + causal clamp** (E-C4/RQ2: on a player with a
*known* hidden tilt, the evolving latent recovers the ordered integral at
R²=0.93 vs 0.65 and clamping it moves predictions) — *not* on shuffling. A
faithful chess shuffle is deliberately deferred: the timing edge is dominated by
**within-game** time-pressure (the clock), so a *game-level* shuffle would leave
it intact, while a *within-game* shuffle would violate clock monotonicity — so a
chess shuffle cannot cleanly isolate cross-game dynamics and is not a decisive
control here. Controls: `scratchpad/{real_kt,synth_kt}_shuffle.py`.

## Go — attempted; no robust effect under controls (E-D2, honest negative)

The thesis promises chess **and** Go, so we built a real-Go pipeline and ran the
timing D-vs-B — but it does **not** yield a robust effect, and we report that
honestly. Data: **OGS** (online-go.com) — the game *JSON* carries per-move
think-times (`moves = [[x, y, time_ms], …]`; the SGF export does not, and the JSON
needs a browser User-Agent), resolving the plan's flagged "confirm per-move timing
availability" risk. 800 games, 586 players, 82k moves. Oracle-free (the timing
head reads only the latent; no KataGo / board encoder needed). Trajectory = one
player's moves within a game; think-time = the move's OGS time; a byo-yomi
time-pressure proxy + move progression feed `history_features`; reuse `run_kt`'s
lognormal timing head; future/temporal split.

**A naive run looked positive — but a homogeneity control overturns it.** On the
mixed cohort (928 trajectories, all board sizes) the evolving latent *appeared*
to win: D−B = −0.0037 / −0.0022 / −0.0011, P=1.00 in 2/3. But restricting to
**19×19-only** games (592 trajectories, well-powered) the effect **vanishes**:

| cohort | D−B (3 seeds) | verdict |
|--------|--------------|---------|
| mixed sizes (9/13/19) | −0.0037 / −0.0022 / −0.0011 | looks positive (P=1.00 in 2/3) |
| **9×9** (n=209, fastest) | +0.0003 / −0.0036 / −0.0041 | *looked* weak (2/3 seeds P=1.00) |
| **13×13** (n=127) | +0.0005 / +0.0003 / −0.0002 | null |
| **19×19** (n=592, well-powered) | +0.0029 / +0.0002 / −0.0002 | null (one seed *worse*) |
| **9×9, 2.5× larger** (N=519) | −0.0015 / +0.0011 / +0.0001 | **null — collapses** (1/3 seeds sig, two wrong-sign) |

So the mixed-cohort effect was mostly a **board-size/speed confound** (the
evolving latent detecting the game *regime* — fast 9×9 ~1–2 s vs slow 19×19 —
i.e. per-trajectory think-time *level*). The one residual finding — a **weak,
seed-unstable 9×9 signal** (n=209, 2/3 seeds) — we chased to a **2.5× larger
9×9-only cohort** (a fresh OGS scan → 1554 games → N=519 trajectories) to settle
real-vs-noise. It **collapses to null**: D−B −0.0015 / +0.0011 / +0.0001 (mean
≈ −0.0001, only 1/3 seeds significant, two *wrong-signed*). The weak 9×9 effect
does **not** firm up with more power — it was **small-cohort noise**. A
**cross-game** framing (game W/L → post-loss/momentum) was also null (n=42, ns).
**We therefore make no Go claim at any board size:** under a homogeneity control
*and* a power check, the evolving latent does not beat the memoryless twin on
real-Go think-time. Go is **future work** — a cleaner test needs the true
byo-yomi clock (not a proxy) and/or the move channel (a Go board backbone).
Generality is claimed via **chess + knowledge tracing** (both real). Builders
`scratchpad/go/*.py` (size filter `argv[4]`, larger cohort `go_big.json`); raw
`results/go_timing.txt`. This is exactly the control a reviewer would run —
better we ran it, and then powered it up.

## Population heterogeneity — beating the "positive average person" (E-F2 / F)

The field's named-but-unsolved problem: behaviour models collapse to a
population-average person. Test (`run_population`, KT domain): give 40 students
**distinct skills** (`skill_spread`), train the per-individual latent model, and
compare how well it vs an average-person baseline reproduce the *distribution*
of per-student held-out accuracy.

| skill spread | observed spread | model spread | W1(model→obs) | W1(avg-person→obs) | corr(pred,obs) |
|-------------:|----------------:|-------------:|--------------:|-------------------:|---------------:|
| 0.0 | 0.108 | 0.031 | 0.064 | 0.088 | 0.87 |
| 1.0 | 0.163 | 0.127 | 0.033 | 0.135 | 0.93 |
| **1.5** | **0.206** | **0.184** | **0.032** | **0.174** | **0.96** |

As the real heterogeneity grows, the per-individual latent **reproduces the
population's accuracy spread** (0.184 vs observed 0.206) and matches the
observed distribution with **4–5× lower Wasserstein** than the average-person
baseline (a point mass, spread 0), while correctly ranking who is skilled
(corr 0.96).

The **generative precision/recall** (`gps/eval/distributional.py`,
Kynkäänniemi 2019) makes the failure of the average-person crisp (spread 1.5):

| model | JS↓ | precision (plausible) | recall (covers diversity) |
|-------|----:|----------------------:|--------------------------:|
| per-individual latent | **0.12** | 0.97 | **0.95** |
| average-person | 0.68 | 1.00 | **0.00** |

The average-person is perfectly plausible (precision 1.00) but covers **none**
of the population's diversity (recall 0.00) — the textbook "positive average
person". The per-individual latent is both plausible *and* diverse (0.97/0.95).
So the latent **recovers real heterogeneity** — the Milestone-F demonstration,
with the proper distributional eval. Reproduce:
`run_population(build_kt_dataset(skill_spread=1.5))`.

**On REAL students (ASSISTments 2009).** The same `run_population` on the **same
500-student cohort** used for RQ5 (correctness) recovers the *real* accuracy
distribution: observed spread 0.190, model 0.095; **Wasserstein-1D 0.074 (model)
vs 0.147 (average-person)** — 2.0× closer; JS 0.17 vs 0.61; **precision/recall
0.86/0.75 vs average-person 1.00/0.00**; corr(pred, observed) = 0.96. So on the
same real students, the evolving latent *both* beats the memoryless twin at
predicting responses (RQ5) *and* recovers the population accuracy distribution
(F) — the average-person is plausible but covers *none* of it. Non-synthetic
Milestone-F. Raw: `results/real_kt.txt`.

**Generating novel players (E-F1).** Beyond scoring the real cohort, we *sample*
200 never-seen players from a Gaussian prior fit to the real style latents
(`run_generation`) and score the generated population against the real one
(distributional, no pointwise ground truth). With a **full-covariance** prior
(a diagonal one under-disperses — a real methodological finding):

| population | accuracy spread | W1↓ | JS↓ | precision | recall |
|------------|----------------:|----:|----:|----------:|-------:|
| real | 0.165 | — | — | — | — |
| **generated (sampled latents)** | **0.155** | 0.024 | 0.124 | 0.93 | **1.00** |
| average-person | 0.000 | 0.17 | 0.68 | 1.00 | **0.00** |

Sampling latents generates a **realistic, fully-diverse** population — matching
the real accuracy spread, plausible (precision 0.93) **and** covering the entire
real diversity (recall 1.00) — versus the average-person, which covers none. So
the latent is not just a per-individual *predictor* but a per-individual
*generator*: a usable model of the population, beating the "positive average
person" on both pillars (recovery and generation).

**Scope (honest negative — chess-F does *not* cleanly extend).** We attempted to
extend F to real-chess think-time *levels* (per-player mean log-think-time;
`scratchpad/run_chess_f.py`). It does **not** cleanly beat the average-person:
the **evolving** state-injector fails to recover per-player level (corr(pred,
obs)=0.05) because it encodes behavioral *state*, not player *identity* — it
never sees a player's own past think-times, so cannot accumulate their level (KT
works because accumulated *accuracy* reveals skill). A **static per-player
embedding** *does* correlate (0.61) but the log-think-time head is
mis-calibrated in absolute scale (both over-disperse and offset → Wasserstein
worse than the point-mass baseline). So we claim **F on KT**, and note chess-F as
**future work** needing a calibrated per-individual timing head. Usefully, this
delineates the evolving latent as a **state model, not an identity model** —
consistent with the shuffle-control finding above.

## LLM policy: frozen verbal injection is a negative control (B5)

The LLM backbone (`SGLangBackbone`, sglang + Qwen3) is implemented and runs on
real chess. The first test (`gps.experiments.llm_inject`, 88 held-out 2013-blitz
moves) asks the persona-prompt question: does splicing a verbal state note
(tilt/fatigue/momentum) into a **frozen** LLM's prompt help it predict the
player's next move? A third **irrelevant-text control** isolates content from
token-count:

| Qwen3-8B condition | move-NLL | Δ vs none |
|--------------------|---------:|----------:|
| no injection | 3.957 | — |
| **state note** | 4.038 | **+0.081** |
| irrelevant control | 4.026 | +0.069 |

The state note is **no better than irrelevant filler** — a frozen LLM does not
*use* the state content (it has no learned mapping from "tilted" to *this
player's* moves). So the naive verbal / persona-prompt baseline (B5) **does not
work**, a clean negative control that motivates the *trained* dynamic latent
(the board-native E-C wins). (GPU; this is the one experiment not in the CPU
suite.)

### Trained LLM via RL (slime): RL learns, but state still doesn't help *moves*

We then **trained** an LLM with RL (slime GRPO, Qwen3-1.7B, 2×A100, real 2017
blitz): the model generates a move for a position; reward = match to the
player's *actual* move (dense: 1.0 exact / +0.3 same destination / +0.2 same
origin). Held-out move-match over 30 rollouts, with vs without the player's
verbal state in the prompt (**3 seeds each**):

| condition | frozen (eval 0) | RL final mean (sd) | RL gain |
|-----------|----------------:|-------------------:|--------:|
| with-state | ~0.076 | 0.0996 (0.0003) | **+0.024** |
| no-state | ~0.076 | 0.1009 (0.0102) | **+0.030** |

Two findings. (1) **RL works**: both conditions improve move-match ≈+0.025–0.030
over the frozen model — a *trained* injector genuinely learns, unlike the frozen
negative control above. The full slime RL pipeline for player-mimicry is
operational. (2) **State does not help move prediction**: with−no = −0.0013, a
clean **null** across 3 seeds (|Δ| ≪ the no-state seed sd of 0.010). This is
exactly what the board-native results predict — **move choice is stateless** (the
move-concentration is flat everywhere; the evolving state is legible in *timing*,
not moves). So the central channel asymmetry holds **even in an RL-trained LLM**.

We also RL-trained the LLM to predict the player's **think-time** (output a
number of seconds; reward = log-bucketed closeness). Here too **RL works
strongly** — the model reaches ~60% bucket-accuracy (from ≪ that frozen) — but
the **verbal state does not help**: no-state 0.627 vs with-state 0.594 (the
no-state model is *ahead throughout*; single-seed shown, multi-seed firm-up in
`results/slime_rl_llm.txt`). This is coherent, not contradictory: **position
complexity dominates think-time** (the B4+complexity result, Spearman ≈ 0.39),
which the LLM reads straight from the FEN, while the small *per-individual*
timing residual that the board-native **hidden** latent captures is not
recoverable from a **verbal** prompt (RQ6: hidden ≫ verbal). So across *both*
channels the RL-trained LLM confirms the same message: **a verbal state note
does not help; the state's value lives in the trained hidden latent, not the
prompt.** The genuine positive-LLM test is therefore a `HIDDEN` soft-prompt
(prefix-embedding) injection — future work, since slime/sglang RL is
text-native.

### SFT beats RL as a probe — and reproduces the timing/move asymmetry

The RL match-reward is *sparse* (exact-move ~3%), which is why the state effects
came out as nulls. So we also did **behavior-cloning SFT** (TRL, Qwen3-1.7B,
LoRA) — fine-tune on the player's actual move/time via cross-entropy, and read
the **held-out completion NLL** (a dense, direct signal). SFT works dramatically
(held-out NLL drops from ≈4–9 frozen to ≈0.25–0.31). And with this cleaner probe
the board-native asymmetry **appears in the LLM** (3 seeds each):

| channel | with-state NLL | no-state NLL | Δ (with−no) |
|---------|---------------:|-------------:|------------:|
| **timing** | 0.2497 | 0.2606 | **−0.0109** |
| move | 0.2967 | 0.2997 | −0.0030 |

The **timing** benefit is ~3.6× the move benefit and is **non-overlapping**
across seeds (every with-state run beats every no-state run: 0.249–0.250 vs
0.260–0.261). It is a genuine *learned* effect: the with-state timing prompt
starts at a much **worse** frozen NLL (9.1 vs 6.5 — the frozen model naively
echoes the clock number) yet ends up **better** after SFT. So **the player's
state helps an SFT-trained LLM predict think-time, far more than moves** — the
same channel asymmetry as board-native (state → timing ≫ moves), now with a
positive LLM result. **Full-param SFT confirms it** (timing Δ = −0.0113 ≈ the
LoRA −0.0109; move Δ = −0.0033) — the asymmetry holds across **LoRA and
full-param**. Raw: `results/slime_rl_llm.txt`.

**It is not a small-model artifact — a backbone-scaling trend (with an honest
capacity caveat).** We re-ran the SFT probe (LoRA, 3 seeds) across **four model
sizes**:

| backbone (LoRA) | timing Δ | move Δ |
|----------|---------:|-------:|
| Qwen3-0.6B | −0.0107 | −0.0036 |
| Qwen3-1.7B | −0.0116 | −0.0042 |
| Qwen3-4B | −0.0114 | **−0.0004** |
| Qwen3-8B | −0.0136 | **−0.0008** |

The **think-time** help is robust at every scale (−0.011 to −0.014). Under **LoRA**
the already-small **move** effect drops to ≈0 at ≥4B, so the asymmetry ratio grows
(≈3× → >10×, denominator-noisy). **But we then tested whether that move-collapse is
real or a capacity artifact by running full-param SFT (all weights, 3 seeds each) at
the two scales where LoRA collapses the move channel — 4B and 8B** — made feasible on
this old-kernel node by a **single-GPU paged-8-bit optimizer** (~30 GB at 4B / ~57 GB
at 8B, no FSDP/NCCL, sidestepping the multi-GPU hang):

| scale | LoRA move Δ | full-param move Δ (3-seed) | full-param timing Δ (3-seed) |
|-------|---------:|-------:|-------:|
| Qwen3-4B | **−0.0004** (null) | **−0.0072** (sd .0005, all<0) | −0.0110 (sd .0006) |
| Qwen3-8B | **−0.0008** (null) | **−0.0083** (sd .0015, all<0) | −0.0128 (sd .0003) |

The **timing benefit is invariant to adaptation method and scale** (full-param
−0.0110/−0.0128 ≈ LoRA −0.0114/−0.0136 at 4B/8B) — the strong, robust claim. But the
**move channel does *not* collapse under full fine-tuning**: at *both* scales where
LoRA drops move to ≈0, full FT recovers a stable **~−0.007/−0.008 move benefit** (all
3 seeds < 0 at each scale). So the LoRA "clean null at ≥4B" is a **LoRA-capacity
effect** (a fixed r=16 adapter is a shrinking fraction of a bigger model), **not a
property of scale**. The **timing ≫ move asymmetry still holds** under full
fine-tuning (timing ~1.5× move, every seed at both scales) but as a **graded** effect,
not a clean null. So the honest *when-not-what* story is: the timing benefit is robust
and does not wash out at scale (answering the "no-SOTA backbone" concern); the clean
move-**null** is specific to the low-capacity LoRA probe and the board-native policy,
while a full-capacity LLM shows a small but consistent move benefit too. All runs in
W&B `gps-llm-sft-scale` / `gps-llm-sft-8b-full`; raw in `results/slime_rl_llm.txt`.

**LLM arm, summarised.** Three methods, one consistent story: (i) *frozen*
verbal/persona-prompt = negative control (a state note ≈ irrelevant filler);
(ii) *RL* (GRPO) genuinely learns the task but its sparse match-reward can't
resolve the state effect (nulls); (iii) *SFT* (dense NLL probe) surfaces the
**timing ≫ move** asymmetry robustly. So a verbal prompt *can* carry the state's
timing signal once trained with a dense objective — but the effect is small, and
the board-native **hidden** latent remains the stronger channel (RQ6).

## Reproduce (CPU, deterministic)

```bash
# ingest a cohort (≈18MB archive; pass-2 parse ≈130s across 8 workers)
gps ingest 2013-01.pgn.zst --out out --speed blitz \
  --min-games 30 --min-sessions 3 --max-players 100 \
  --max-games-per-player 20 --workers 8

# E-C2 / E-C3 / E-C1 (set WANDB_API_KEY; CUDA_VISIBLE_DEVICES="" for determinism)
gps train-ec out/dataset.jsonl.gz --epochs 15 --batch-size 16            # E-C2
gps train-ec out/dataset.jsonl.gz --epochs 15 --batch-size 16 --split session       # E-C3
gps train-ec out/dataset.jsonl.gz --epochs 15 --batch-size 16 --split session --control static  # E-C1
```

Pool per-player `diff_per_player` across ≥3 seeds and bootstrap (single seeds
wobble; the pooled-over-players number is the trustworthy one, as in E-A1).
