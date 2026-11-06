# Eunoe: the design

The experiment and its kit. Numbers from any private deployment are deliberately absent: everything here
can be reproduced from this repository. Marked **[D]** where read from a provider's documentation, and
**[C]** where claimed by a cited work and not re-run here.

## 1. The questions

Does contracting context beat compaction, on what measures, and at what cost? Is there a region where
performance improves and cost falls? What is acceptable to forget?

The idea: instead of summarising a whole history when it reaches a threshold, evict **details about
events** continuously, least important first, leaving the event in context until the very end. Oldest-first
is a control here and never the design. The event still being there is itself the signal that more can be
looked up; whether that alone is enough, or a marker is needed, is a hypothesis.

## 2. Why the cache is the hard part

A model's cached state for a token was computed attending to everything before it, and the cache key is a
hash of the whole prefix [D], so a change at position N invalidates everything after N. Editing an old
event means paying to rewrite whatever follows it. So literal continuous eviction should lose on cost, and
"continuous" has to mean continuous in what the model sees across a session, delivered in batches. There
are three ways to make that cheaper without any provider change: a batch at a threshold, an edit only when
the cache has already expired, and a generational layout that leaves a frozen region ahead of the first
edit, so reads survive up to it.

There is a newer second constraint. On recent Claude models, editing an earlier turn on the client also
fails a check on every later thinking block, and accounts created on or after 2026-08-31 get a 400 for it
[D]. So `thinking_policy` is a declared factor in every manifest, and the conformance suite tests it per
account:

| Policy | Mechanics | Cost |
|---|---|---|
| `allow` | Send failing blocks anyway; billed as ordinary input | Only accounts where the check is not enforced |
| `strip_on_edit` | Remove thinking from history at each edit, keep it from there | Valid everywhere; the model loses past reasoning |
| `drop_block` | `thinking-binding-controls-2026-08-01`, drops failing blocks | The docs warn it can raise output |
| `server_clear` | `clear_thinking_20251015`, keep N turns | Valid; invalidates at the clear; API only |
| `off` | A model or mode without thinking | Changes the model's ability, so never compared across this line |

The public default is `strip_on_edit`, the one every account can run.

## 3. Prior work

| Work | What it does | What it leaves open | Use here |
|---|---|---|---|
| [VISTA](https://arxiv.org/html/2606.30005) (Xu, Li, Zhang, 2026), closest published | Typed blocks and a dashboard; `archive` swaps a block for a handle and summary; agent chooses | One step, not a ladder; agent's judgment, not measured importance | Cite; run if its code is released |
| [LETHE](https://github.com/JesusGarcia9009/lethe-engineering), closest shape | Scores relevance, folds cold runs into notes, pages the rest out, leaves a stub | Needle test only; heuristic relevance; no cache-cost treatment | Arm `lethe` |
| [LCM](https://arxiv.org/html/2605.04050v1) (Ehrlich, Blackman, 2026) | Summary DAG over an immutable log, expand tool | Levels are spans of messages, not one event's detail | Arm `lcm`, via [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm) |
| [ACM](https://arxiv.org/abs/2607.23809) (Li et al., 2026) | Agent-called context editing with lossless offload | Agent decides | Cite |
| [The Complexity Trap](https://arxiv.org/abs/2508.21433v3) (Lindenbauer et al., 2025) | Masking old tool output vs summary on SWE-bench Verified; masking matched summary at about half the cost [C] | Heuristic triggers, one benchmark family | Control C3: contracting has to beat this too |
| [ACON](https://arxiv.org/abs/2510.00615v2) (Kang et al., 2025) | Optimises compression guidelines from paired failures | Improves a prompt, not a per-event measure | Cite the paired method |
| [Don't Break the Cache](https://arxiv.org/abs/2601.06007) (Lumer et al., 2026) | Cache placement across providers | No eviction at all | Cite |
| [Factory's probe evaluation](https://factory.com/news/evaluating-compression) (2025) | Recall, artifact, continuation and decision probes | Private data, LLM judge | Adopt the four kinds, score deterministically |
| [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) (2025) | Restorable compression: keep the path, drop the content | A practitioner's post | Cite; the reconstructible level |
| Anthropic context editing [D] | `clear_tool_uses_20250919`, oldest first, whole result | Oldest-first only | Control C4 |
| Anthropic compaction [D] | `compact_20260112` at a threshold | Whole-history summary | Control C1 |
| [LOCA-bench](https://arxiv.org/abs/2602.07962) (Zeng et al., 2026); [OOLONG](https://arxiv.org/abs/2511.02817) | Public long-context benchmarks | OOLONG has a contamination problem, per LCM's authors [C] | External task set, a later phase |
| [LongMemEval](https://arxiv.org/abs/2410.10813), [MemoryAgentBench](https://arxiv.org/abs/2507.05257), [HaluMem](https://arxiv.org/abs/2511.03506) | Memory benchmarks | They test memory stores, not eviction from a live prompt | Borrow abstention and hallucination scoring |
| [FadeMem](https://arxiv.org/abs/2601.18642) | Importance-weighted decay in stores | External memory, not the prompt | Cite for the importance question |
| llama.cpp `seq_rm`, [Lethe KV pruning](https://arxiv.org/abs/2511.06029) (Zeng et al., 2025) | Remove a range from a local KV cache without re-evaluating | Whether output matches full re-evaluation | The real test of reworking a cache, a later phase |

**Not novel:** evicting tool output, leaving a stub, lossless lookups, relevance scoring, hierarchical
summaries, batching near cache expiry, pricing the cache. **Novel as far as can be found:** a ladder of levels
for one event as the unit of eviction; importance measured by counterfactual ablation; a whole surface of
performance against cost across schedule, TTL and write premium, with a fair simple baseline; the
preserved-thinking interaction. About twenty searches and the papers named: "not found" is not "doesn't
exist", and the search is re-run before the pre-registration freezes.

## 4. Hypotheses

The full table, with falsifiers, is in [`../PREREGISTRATION.md`](../PREREGISTRATION.md), which is a draft until
it is frozen and tagged before the first scored run.

## 5. Levels of detail

An event moves down **one level at a time**, least important first, and leaves context only after every
other option is used.

| Event | L1 | L2 | L3 | Notes |
|---|---|---|---|---|
| Tool result | Head and tail, counts | Extracted facts | One line | Largest category |
| File read | Path, hash, line count | Same | Same | Reconstructible, because the file is the archive |
| Tool call input | Long values trimmed | Key arguments | Merged with its result | A call and its result move together |
| User message | Stays verbatim longest | Key sentences | First sentence | A default a signal can override |
| System note, reminder | Bare fact | Bare fact | Bare fact | |
| Assistant text | First and last paragraphs | Claims and decisions | First sentence | |
| Thinking | No level | No level | No level | Opaque: kept or dropped |

**Telling the model there's more** is a factor: `M0` nothing, `M1` a trailing `…`, `M2` `[more: recall e123]`.
The lookup tool is in every contracting arm, and instructions are equal within a comparison.

## 6. Arms and controls

| Arm | What it is | Why it's there |
|---|---|---|
| C0 | No management, up to the model's limit | The baseline |
| C1 | Provider compaction, or the CLI's, at budget | The thing to replace |
| C2 | A handoff-style restart: a fresh session and a code-built block | Contracting's non-continuous version |
| C3 | Observation masking, oldest first | The honest simple baseline |
| C4 | `clear_tool_uses_20250919`, tuned | The provider's own eviction |
| T1 | The ladder in oldest-first order | Isolates importance. A control only |
| T2 | The ladder in importance order, no marker | The design |
| T3 | T2 with `M2` | The marker question |
| T4 | T2's order, one step to a stub | Isolates the ladder |
| lethe, lcm | Try-first arms | See [`try-first.md`](try-first.md) |

**Fairness.** Every control gets its best documented configuration, written down before the run; budgets and
instructions match within a comparison; a control that would have won under a better config is reported rerun
under it. No human baseline is claimed.

**Schedule factors**, on T2 only: every 100 calls to a cap; a threshold batch; a larger batch; at cache expiry
keeping 50%; expiry plus threshold; and generational, past a frozen region. `schedules.grid_for` builds them at
any budget. Every-call stays available as a negative control.

**A layout rule.** An edit keeps the cache up to its first changed byte, so every contracting arm leaves the
floor (system prompt and tools) unedited.

## 7. Tasks and fixtures

A seeded procedural generator: stdlib, deterministic, no private data. It emits long scripted sessions in
invented domains (`oncall`, `smallbiz`, `research`), each with a tool surface against a simulated environment,
plus planted facts with gold, some later superseded; probes in four kinds; unanswerable probes; and tasks scored
by checking the environment's state. Gold comes by construction, new seeds avoid memorisation, and a stranger can
draw fresh sessions. A profile sets a session's shape, and `shares` can be set to match a measured workload.

External sets (LOCA-bench, LongMemEval's abstention items) come in a later phase, pinned by commit and reported
apart. A private task set implements the same `TaskSet` and stays with its owner.

## 8. Importance signals

| Signal | What it uses | Where it can run |
|---|---|---|
| S0 oldest-first; S1 random | Position; a seeded draw | Controls |
| S2 largest-first | Size | Public |
| S3 kind prior | A taxonomy of event kinds | Public |
| S4 hindsight | Whether later turns used it | Replay only: an oracle ceiling, never deployable |
| S5 access count | Lookups so far | Public |
| S6 reference density | Later events naming its identifiers so far | Public |
| S7 small model at write time | A rating from a cheap model, with its own cost line | Public |
| S8 ablation estimate | A table from a counterfactual study | Public |
| S9 frequency times weight | A rare but heavy event survives | Public stand-in |

**Counterfactual ablation:** for a checkpoint and a group of events, fork, step the group down or remove it, ask
the following turns' probes, compare with the unedited fork, and score the loss. Groups are tested hierarchically
with sequential halving. That is why `fork` is a conformance step.

## 9. Measures

| Measure | Definition |
|---|---|
| Probe recall (primary) | Share of answerable probes correct against gold, by kind and by level at probe time |
| Hallucination of evicted detail | Share of probes on an evicted fact answered with a specific wrong value |
| Correct abstention | Share of unanswerable probes admitted as unknown |
| Task success | Environment-checked completion of embedded tasks |
| Lookups | Count, tokens, precision, and missed lookups |
| Staleness | Probes answered from a superseded fact |
| Cost | Per session and per successful task: usage times a dated price file times TTL |
| Cache behaviour | Reads, writes by TTL, rewrites |
| Latency | Per turn, and the worst turn, because a rewrite is one slow turn |
| Context held | Mean and peak, which has to match across arms within tolerance |

## 10. Sample sizes

Everything is within-seed paired: every arm sees the same generated session. These are estimates until a
pilot's variance replaces them, and the frozen pre-registration fixes N from the pilot.

| Test | Effect | Assumed spread | α, power 0.80 | Needed |
|---|---:|---|---:|---:|
| Probe recall, session mean, paired | +0.10 | SD of differences 0.15 | 0.0125 after Holm | 25 seeds |
| Non-inferiority, margin 0.05 | | 0.15 | 0.025 one-sided | 71 seeds |
| Task success, McNemar | +0.15 | 30% discordant | 0.0125 | 146 pairs |

95% intervals by a seeded bootstrap of seeds, Holm on the primary family, deterministic scorers pinned by test.

## 11. Reworking the cache

| Alternative | How it's tested |
|---|---|
| Batch at expiry | A schedule factor |
| Batch at a threshold | A schedule factor |
| Generational layout | A schedule factor. As far as can be found, untested for this |
| Keep-alive pings | An API arm in a later phase |
| Server-side edits | Controls C1 and C4 |
| A reworked cache, simulated | `eunoe replay`'s "free edits" row: what a provider feature would be worth |
| A reworked cache, real | A later phase: remove a range from a local model's KV cache and compare with full re-evaluation |

## 12. How a context reaches a model

Per harness in [`harness-conformance.md`](harness-conformance.md): editing a session file on the two CLIs,
editing a conversation store on Antigravity, sending a changed list on the APIs.

## 13. Threats to validity

| Threat | What the design does |
|---|---|
| Replay undercounts what a lean context needs | Demand regret is a lower bound; forks and live runs test it directly |
| Instructions drive lookups | Equal within a comparison, published, swapped in a robustness run |
| Synthetic isn't real | External sets and private runs reported separately |
| Planted facts are easy | Decision and continuation probes, tasks, human trials |
| Importance overfit | Separate training and test seeds; policies freeze before the grid |
| Budget not really equal | Mean held context measured, matched and reported |
| A weak control flatters the treatment | Best documented config, written first; rerun if beaten |
| Prices and betas move | Dated price files; cost recomputed from usage; beta headers in the manifest |
| Provider cache isn't deterministic | Raw usage kept; repeats |
| Account effects | Conformance per login; `thinking_policy` declared; `strip_on_edit` as public default |
| Model updates mid-campaign | Pinned ids; order randomised by seed; no silent fallback |
| Contamination | Fresh seeds for the headline; external sets reported apart |
| Author bias | Pre-registration, mechanical scoring, a public kit, every miss reported |
| Undocumented session formats | Conformance per version; versions pinned in manifests |

## 14. What private runs may report

A private task set can report aggregates and nothing they came from: never message text, tool arguments, file
names, people, verbatim reasons or session ids, and only after its owner reviews the exact aggregates. A hash
manifest of the inputs shows the numbers came from a fixed set. The reproducible claim is always the public set.
