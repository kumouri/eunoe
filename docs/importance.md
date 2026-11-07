# Importance: what fork ablation found on generated sessions

**Status: a pilot on generated sessions, not a scored result.** Nothing here is frozen, and `PREREGISTRATION.md` still binds
nothing. The seeds used here (1001, and 2001 to 2003) are spent: a scored grid has to use others.

To run it:

```bash
python -m eunoe ablate train --door scripted --out eunoe-results/forks   # offline, proves the machinery
python -m eunoe ablate fit   --out eunoe-results/forks
python -m eunoe ablate test  --door scripted --out eunoe-results/forks
python -m eunoe ablate train --model claude-sonnet-5-5 --out eunoe-results/forks   # real, on your own login
```

## How a fork works, and what it costs

Each condition is one edited context, written once as a session file, then forked for its questions. So a condition costs one
rewrite and a few reads. Measured on `claude-sonnet-5-5` through the CLI: a generated session held about 132k tokens unedited,
and a condition wrote that once at $0.53 at list, with each later call on it reading rather than writing.

Every condition answers the same questions, so differences are paired by session. Questions go in one numbered batch, because
a rewrite is the expensive part. Items in a batch can't show each other's gold. `eunoe.forks.transcript` does the same on a real
Claude Code session file, using the same renderer, so you can run this on your own sessions; rows hold counts and hashes only.

## Does a code-built level carry a fact in a form a model uses?

**When the fact is still literally there, yes. Whether it is still there is the whole question.** Generated sessions, three
sessions on one training seed, about 26 questions each:

| Condition | Held | Fact still literally held | Recall | Recall when held | Recall when gone | Specific wrong answers |
|---|---:|---:|---:|---:|---:|---:|
| Nothing changed | 132k | 100% | 97% | 97% | | 0% |
| Same again, the noise floor | 132k | 100% | 96% | 96% | | 0% |
| Everything at L1, first version | 50k | 5% | 5% | 100% | 0% | 6% |
| Everything at L2 | 24k | 55% | 55% | 100% | 0% | 0% |
| Everything at L3 | 18k | 0% | 0% | | 0% | 9% |

Read across, a level's recall is almost exactly the share of facts it still literally holds. A model reads a fact from a digest
as readily as from the whole output, and when the fact is gone it mostly says so.

**The first L1 held less than L2, which a ladder can't allow.** Trimming kept an output's head and tail and an assistant turn's first
and last paragraphs, and most facts sit in the middle, so a fact gone at L1 came back at L2. L1 now keeps the details and sentences L2
would. On the one session where the re-run got that far, facts held at L1 went from 5% to 46%, recall from about 5% to 42%, at 56k held
rather than 50k. The rest of the re-run stopped at the flag limit below, so that figure is one session.

**L2 keeps too little of a tool's output.** Its share is 6% of an output, and details on noisy logs fill that before they reach the
fact, so facts in tool output mostly don't survive: 65% lost, against 0% for facts in assistant prose. That is what to improve first.

## What each kind costs, on this fixture

One kind taken to L3 with the rest verbatim, against the unedited session, with a 95% interval across the three sessions:

| Kind at L3 | Recall difference | Interval |
|---|---:|---|
| Tool output | -0.44 | -0.58 to -0.31 |
| Assistant text | -0.40 | -0.42 to -0.39 |
| File reads | -0.13 | -0.23 to -0.04 |
| Loaded files | 0.00 | -0.04 to 0.04 |
| User messages | 0.00 | -0.04 to 0.04 |
| System notes | +0.03 | 0.00 to 0.04 |

**This table reads the generator, not the world.** The generator plants facts only in tool output, assistant text and file reads, so
the rest show no loss by construction. It proves the method runs and finds what is there, and it can't say what matters in a
real session.

## Which signal?

Seven orderings at one budget, half of what the session holds, on nine sessions the training never saw. About 69k held in every
arm, so they differ in what they keep. The unedited session scored 99%.

| Signal | Recall, first answer | Against S0 | Against S8 |
|---|---:|---|---|
| S8 ablation table | **65%** | +0.36 (0.30 to 0.40) | |
| S7 small model's rating | 61% | +0.30 (0.23 to 0.37) | -0.04 (-0.09 to 0.01) |
| S3 kind prior | 59% | +0.25 (0.19 to 0.31) | -0.05 (-0.15 to 0.04) |
| S1 random, a control | 42% | +0.08 (0.02 to 0.14) | -0.24 (-0.34 to -0.13) |
| S6 reference density | 41% | +0.09 (0.01 to 0.21) | -0.25 (-0.33 to -0.14) |
| S0 oldest first, a control | 31% | | -0.36 (-0.40 to -0.30) |
| S9 frequency times weight, the public stand-in | 29% | -0.04 (-0.14 to 0.12) | -0.36 (-0.46 to -0.23) |

S8, S7 and S3 can't be told apart here. Each beats oldest-first with its interval clear of zero, and S8 beats random too. **Take that with the same
caution as the table above:** S8 and S3 read the kind of an event, and the generator decides by kind where facts go. So this is
evidence that a trained table works when importance really does follow kind, and it doesn't show that importance does. S7 can't
have learned the generator's layout from training, so it is the more interesting of the three, and it costs a small model's call per
batch of events.

## The ladder against one step to a stub (H7)

At the same budget, on nine sessions, with the same signal:

| | Ladder | One step to a stub | Difference, 95% interval |
|---|---:|---:|---|
| Recall, first answer | 29% | 26% | +0.01 (-0.13 to 0.14) |
| Recall, after lookups | 97% | 56% | **+0.40 (0.13 to 0.68)** |
| Lookups per correct answer | 0.70 | 0.54 | |
| Sessions with no lookups at all | 0 of 8 | 5 of 9 | |

**They hold about the same, and the difference is whether the model looks.** Facing a stub, it often answered `UNKNOWN` for
everything without trying, and in those sessions recall stayed under 20% in three of the five. Facing the ladder's detail, it looked things
up in every session and nearly always found them. H7's own measure, lookups per correct recall, goes the opposite way to its prediction,
because the ladder got far more correct answers by looking more. No marker was used in either arm, so this also bears on H3. All of it is
one budget, one instruction text and one fixture, so it sets up H7 and H3 for the grid and settles neither.

## What goes when thinking goes

A transcript stores thinking as an empty string and a signature, so a replay can't see what it carried. This check gives a model a
small knapsack to solve, with only the total in its reply, then asks in forks which items made that total. Answers were right every
time, in every arm, on both models: this measures what it costs to get there.

| Model, trials | Arm | Context held | Follow-up output tokens | Follow-ups right |
|---|---|---:|---:|---:|
| `claude-sonnet-5-5`, 12 | As the CLI left it | 8,503 | 54 | 24 of 24 |
| | Rewritten unchanged by the kit | 8,503 | 54 | 24 of 24 |
| | Thinking stripped | 7,391 | 483 | 24 of 24 |
| | Thinking kept, an earlier turn edited | 7,390 | 490 | 24 of 24 |
| `claude-opus-5-5`, 8 | As the CLI left it | 8,656 | 32 | 16 of 16 |
| | Rewritten unchanged by the kit | 8,656 | 32 | 16 of 16 |
| | Thinking stripped | 7,460 | 765 | 16 of 16 |
| | Thinking kept, an earlier turn edited | 7,459 | 706 | 16 of 16 |

Three things follow, on two logins that have never got a 400 for this:

- **Stripping costs work, not right answers, on this task.** A follow-up took 9 to 24 times the output, because the model solved the
  puzzle again. The docs warn of this. A puzzle it can re-solve from what it can see can't show a loss of a conclusion it couldn't
  reach again, so "no wrong answers" is an upper bound on how harmless this is.
- **On this door, an edit takes thinking away whether or not you strip it.** The edited arm holds what the stripped arm holds, to within
  a token, and it pays the same output to get there. The docs say failing blocks reach the model on such a login, which this contradicts,
  and the response's `input_transformations` was empty, so why isn't known. For a contracting arm on a CLI, `allow` and `strip_on_edit` look like
  the same thing after the first edit. A replay that charges `allow` for holding thinking overstates what it holds.
- **Rewriting a session through the kit loses nothing,** so the forks above aren't measuring their own plumbing.

**The first version of this check asked outright for what the reasoning held, and the safeguards refused it.** That is what they're
for, so it stopped there rather than being reworded. Don't build a probe that way.

## Codex: why an answer sometimes doesn't hold after an edit

Conformance saw the answer hold 7 of 11 times edited under reasoning items, against 6 of 6 unedited. This repeats it in pairs: ten
sessions on `gpt-6-luna`, the same question in two forks of each arm.

| Arm | Held | Declined | Words sent |
|---|---:|---:|---|
| Nothing changed | 17 of 20 | 3 | Padding, then the question |
| Padding swapped for other padding, an edit that changes no meaning | 18 of 20 | 2 | Same |
| Reasoning items removed, no edit | 17 of 20 | 3 | Same |
| Padding removed, reasoning items removed | 17 of 20 | 3 | Short message |
| **Padding removed, reasoning items kept** | **12 of 20** | **8** | Short message |
| **Never had padding, never edited** | **12 of 20** | **8** | Short message |

**The edit isn't the cause, because the same words with no edit at all do exactly as badly (12 against 12, p = 1.0).** Nothing is
broken and no answer is wrong: a failure is always the model declining, in words about not revealing a codeword or hidden content. It
declines 15% of the time unedited too, so conformance's 6 of 6 was a small sample. Several refusals name the words "ignore it", which sit
right beside the codeword once the padding is gone, so that is the likely reason, though it is a reading of the refusals and not a measurement.
Conformance's question was the problem, and a probe shouldn't look like asking for a secret.

Not settled: whether reasoning items add to it. The short message declined 8 of 20 with them and 3 of 20 without, p = 0.16 at this sample, and
forks of one session aren't independent. Nothing here says Codex's edits are unsafe, and nothing says dropping reasoning helps.

## Safeguards flagged some of this

Across generated sessions, 12 of 134 conditions had their batch refused with the message that safeguards flagged it, on invented
data, with no pattern a table can show. Four of those came in the last nine, so the rate may not be stable. A refused batch is recorded and
never retried, and a run stops at a limit you can set with `--max-errors`. A numbered batch of questions about what is held in context seems
to be what draws it, so a run should expect some missing conditions, and pairing by session survives them. Asking in smaller pieces or from
ordinary follow-ups is untested here.

## What this doesn't show

- **Nothing about a real session.** Everything above is generated, and the generator decides where facts live.
- **One budget.** Half of what is held. Signals may separate differently at other shares.
- **One instruction text.** Lookup rates depend on how the offer is worded.
- **Sonnet 5.5 on the forks.** Other models may read digests or stubs differently.
- **Small samples.** Three sessions for levels and kinds, nine for signals, and missing conditions where a batch was refused.

Everything here was billed to a subscription and priced at list from recorded usage, about $48 on `claude-sonnet-5-5`
for the forks.
