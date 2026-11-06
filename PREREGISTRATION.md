# Pre-registration: DRAFT, NOT FROZEN

**No scored run has happened, and this file does not yet bind one.** Before the first scored run it is
frozen, committed and tagged. Numbers marked *pilot* are fixed from a pilot that is never scored. Changing
the frozen copy later means a new pre-registration with the old one kept, and everything is reported either
way: a prediction that misses is written up as a miss. Predictions go on record when this freezes.

**Fixed budget means fixed budget.** Each comparison holds the mean context held in a session equal across
arms, so arms differ in what they keep, not how much.

| # | Hypothesis | Primary measure, comparison | Falsified if |
|---|---|---|---|
| H1 | At equal budget, contracting recalls more than compaction | Probe recall, T2 vs C1 | 95% CI on the detail difference includes zero or lies below *pilot* |
| H2 | Contracting produces fewer hallucinated details than compaction or masking | Hallucination rate on evicted facts, T2 vs C1 and C3 | CI includes no reduction |
| H3 | An event's presence alone tells the model there's more to look up | Correct lookups when needed, T2 vs T3 | No-marker is within 0.05 of marker: the stronger claim holds |
| H4 | Importance beats oldest-first | Probe recall, T2 vs T1 | CI includes zero |
| H5 | A region exists where performance improves and cost falls | Per configuration: cost ratio and recall difference, vs C0 and vs C1 | No configuration has a cost ratio CI below 1.0 with recall non-inferior within 0.05 |
| H6 | Batched schedules dominate literal continuous ones | Cost per successful task, T2 across schedules | An every-call schedule is non-dominated |
| H7 | The ladder beats one step to a stub | Lookups per correct recall, T2 vs T4 | CI includes zero |
| H8 | A meaningful share of a handoff brief is unimportant | Fraction of bytes ablatable with continuation score within 0.05 | Under 10% of bytes qualify |
| H9 | It transfers | Sign of T2 vs C1 and C3 on other models and providers | Opposite sign with CI excluding zero on 2 of 3 |

**Primary family (Holm-corrected at 0.05):** H1, H2 against C1, H4, and the H5 recall comparison against C0.
The rest are secondary, reported with intervals and no correction claim.

**Analysis.** 95% intervals by a seeded bootstrap of seeds (10,000 resamples), Holm on the primary family,
deterministic scorers pinned by test. No interim looks at outcomes after the freeze, except validity checks
that don't read arm-level scores. A failed run is retried only on a driver error, and every retry is listed.
