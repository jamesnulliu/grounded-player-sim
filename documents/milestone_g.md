# Milestone G — LLM-agent deployment + released-SOTA benchmark (RESOLVED)

**Outcome.** The 2026-07 pivot *aspired* to make the LLM the headline (via a
hidden≫verbal win) and to answer the weak-backbone objection with a strong
backbone. The honest resolution:

* **G3 (done) — the discovery: hidden-vs-verbal ordering is backbone-dependent.**
  Injected state helps a real LLM's think-time (verbal−none −0.005, 3 seeds), but
  **hidden does NOT beat verbal in the LLM** (hidden−verbal +0.0034) — the LLM
  reads the note *semantically*, so text is the efficient channel. Board-native
  (no language prior) is the reverse: the richer hidden vector wins (RQ6,
  −0.069/−0.117). So *which channel is richer depends on the backbone's language
  prior* — a measured finding, not "hidden wins in the LLM." The LLM is a
  **deployment**/secondary result, not the headline. `results/g3_llm.txt`,
  `scripts/g3_hidden.py`.

* **G4 (LANDED 2026-07-06, extended to a 3rd cohort 2026-07-13) — the
  weak-backbone objection, answered with released weights.** Instead of
  retraining a strong backbone (a Maia move-backbone was deprioritized — see
  below), we tested whether the evolving latent adds think-time value *over a
  released model's own prediction*:
  - **Maia-2** (released human-move SOTA) move-entropy as a difficulty signal:
    strongest baseline Spearman **0.414 / 0.445 / 0.447 ≥ ChessMimic's 0.41**,
    latent still adds **(B4+z)−B4 = −0.025 / −0.029 / −0.039, P=1.00** (3
    cohorts × 5 seeds).
  - **Allie** (ICLR'25) — the airtight test with an *actual released think-time
    head* (Spearman **0.62 / 0.64 / 0.65** ≫ 0.41): latent still adds in the
    direct Allie-vs-Allie+z test on all three cohorts (**−0.023 P=1.00 /
    −0.018 P=0.998 / −0.033 P=1.00**); honestly *smaller* vs Allie, and
    against the fullest Elo+clock+Allie co-fit, significant on 2017 and 2021
    (−0.013 both) but **ns on 2019**. The 3rd cohort (2021-06, added
    2026-07-13 specifically to test whether the 2019 null made the co-fit
    result a coin-flip) resolves it 2-significant/1-null — "usually
    significant, cohort-dependent," not "marginal." Move-deviation-from-Maia
    R²≈0.009 (null).
  `results/g4_timing.txt`, `scripts/g4_{cache,run}_{maia,allie}.py`.

* **Why we did NOT retrain a Maia move-backbone (G1).** A strong backbone
  *absorbs the move signal* (the conv trunk already dropped move D−B −0.027→−0.003)
  and the timing head reads **only the latent**, so a stronger trunk cannot change
  the timing result — it is backbone-independent by construction. The
  weak-backbone objection is therefore answered *architecturally* (timing) and by
  the G4 released-model benchmark (add-on value), not by re-implementing Maia.

**Net.** Board-native timing (RQ6 + E-C6 + G4) is the headline; the move channel
and the LLM are honest secondary results. The three landed novelties
(when-not-what, the equal-capacity future-split control, the backbone-dependent
channel ordering) all stand; see `documents/related_work.md` for positioning and
`documents/paper_draft.md` for the synthesis.

## Landed LLM code (CPU-tested)

The hidden soft-prompt channel is built and unit-tested
(`tests/test_hidden_prefix.py`, `tests/test_llm_hidden.py`):
`gps.policy.hidden_prefix.HiddenPrefixProjector` (trainable `latent →
[n_prefix, hidden]` bridge), `prepend_prefix()`, the `SGLangBackbone` HIDDEN
wiring, and `gps.experiments.llm_hidden` (the SFT entry point). The SFT probe
results (state → timing ≫ moves, robust 0.6B→8B, LoRA→full FT) are in
`results/slime_rl_llm.txt`.
