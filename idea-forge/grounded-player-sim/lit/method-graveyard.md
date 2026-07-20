# Method graveyard (from this project's own execution history)

Methodological death causes already paid for in this project; the critic
should check the surviving claims do not still rest on any of these.

- n=8 rank correlation underpowered / died: the "0.89 heterogeneity scaling
  law" framing (retired 2026-07-13 after the fixed-loader replication:
  Spearman 0.476, dataset-bootstrap CI crosses zero, Pearson 0.138 without
  the Spanish anchor) / recurrence signal: <15 samples + a rank-correlation
  claim.
- Mixed-cohort regime confound / died: the Go "positive" (mixed board sizes;
  the latent detected fast-9×9-vs-slow-19×19 regime, not player state; null
  within every stratum, residual 9×9 signal collapsed on a 2.5× cohort) /
  recurrence signal: pooled heterogeneous cohorts showing an effect absent
  within strata.
- Single-seed swings on small nat gaps / died: single-seed 2017 move
  estimates (+0.032 mlp / −0.030 conv → pooled +0.0005 / −0.0046) /
  recurrence signal: any conclusion from 1 seed on a |gap| < 0.05-nat
  channel.
- Preprocessing fit on evaluation rows / died: KT skill-difficulty fit over
  the whole file including held-out rows (leakage; fixed loader + regression
  test) / recurrence signal: any feature fit before the temporal split.
- Decision-level bootstrap on within-player-correlated decisions / died:
  original concentration analysis (re-run player-bootstrapped) / recurrence
  signal: resampling units that are not the independent unit.
- High-variance bucket inflating raw ratios / died: raw 4.0–6.1× time-
  pressure concentration (survives variance normalization at 2.7–3.6×, so
  reported normalized) / recurrence signal: subgroup effect ratios without
  per-bucket variance control.
