# Scope (validation run, 2026-07-19)

**One-sentence problem:** Validate whether the nearly-completed
grounded-player-sim project — an evolving per-individual latent state that
predicts a person's future think-time but not their choices, isolated by an
equal-capacity memoryless twin on strict future splits — is a strong idea
**for ICLR specifically** (target: next ICLR cycle, primary area:
applications to neuroscience & cognitive science).

**Mode:** single validation round on an executed project (expand-mode
registration → dedup collection → adversarial critic with venue-fit
emphasis → disposition). Not a generative round; no direction menu.

**Hard constraints:**
- The empirical package is frozen (`documents/paper_readiness_plan.md`):
  no new experiments before the manuscript stabilizes; the critic's output
  is a go/no-go + framing guidance, not a redesign.
- Prior-art baseline: the project's own 2026-07-13 verified sweep
  (`documents/related_work.md`); the checker's job is what is NEW since
  then + anything that sweep missed, not re-deriving it.
- Venue fixed: ICLR (user-specified). Venue-fit adjudication is mandatory
  in the critic report.

**Search angles for the checker:**
1. Per-individual / dynamic state modeling in chess or games (post-2026-06).
2. Response-time / think-time prediction with evolving latent states
   (ML venues + psychometrics crossover).
3. Evolving-vs-static user state controls in simulators / recsys / KT.
4. ICLR 2026 accepted papers occupying the human-simulation +
   individualization space.

**User hard filters (criteria.md D):** assessed by critic as usual (dense /
general / uncrowded), but note the project is executed — filters inform
framing, not a kill.

**Model routing:** defaults per agent frontmatter.
