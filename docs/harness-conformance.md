# Harness conformance

Whether a harness and an account can really run a contracting arm, measured by the suite you can run yourself:
`python -m eunoe conform --harness <name>`. Everything in it is synthetic, and it costs a few cents of a plan at
most. **Run it on your own login before trusting a result**, because some of this depends on the account.

## The steps

| Step | What it does | What a failure means |
|---|---|---|
| 1 establish | A session with a codeword and padding | The harness can't run here |
| 2 control | Resume with no edit, and read the cache split | Resume doesn't work |
| 3 edit | Change the codeword inside the session, and ask for it | An edit doesn't reach the model |
| 4 shrink | Remove the padding, and check the context shrank | Eviction wouldn't save anything |
| 5a fork | Resume into a new session, and check the original is untouched | No ablation door |
| 5b thinking | The same edit under real thinking blocks | `thinking_policy = allow` isn't available on this account |
| 6 apply | Write a whole rendered context, tool blocks included, and continue it | A contracting arm has to shorten events in place instead |

Steps 1 to 4 are required. A harness or account that fails a step gets the arms that need it marked unavailable,
never silently approximated. Exit 0 means it conforms, 1 that a required step failed, and 2 that it can't be
reached.

## Measured, 2026-11-06

Each on its smallest model, in an empty scratch directory, with no tools, no MCP servers and every metered key
scrubbed. **API dollars spent: 0.**

| Harness | Version | 1–4 | 5a fork | 5b thinking | 6 apply |
|---|---|---|---|---|---|
| `claude-cli` | 2.1.309 | PASS | PASS | PASS | PASS |
| `claude-cli`, a second login | 2.1.309 | PASS | PASS | PASS | PASS |
| `claude-agent-sdk` | 0.2.179 | PASS | PASS | PASS | PASS |
| `codex` | 0.159.2 | PASS | PASS, by copy | Accepted; see below | PASS |
| `agy` | 1.5.2 | PASS | PASS, by copy | PASS | Unsupported |
| `ollama`, `qwen3.5:2b` | local | PASS | PASS, client-side | No thinking blocks | PASS, client-side |
| `scripted` | offline | PASS | PASS | PASS | PASS |

**Not run:** `claude-api`, `claude-api-server`, `openai-api` and `gemini-api` need a funded key. Their request
shapes, usage parsing, spend gate and step logic are tested offline against recorded shapes in
`tests/test_harnesses.py`, which is not the same as having run them. They say so with exit 2 until they have one.

## What each one taught

**Claude CLI.** Edit the session JSONL, and an edit lands at the next spawn. The file holds several copies of a
message's text (5 of the codeword here, 6 through the SDK), and all of them have to change. Reads fell from 11,635
to 3,055 at the edit, the cache restarting there, and a fork reused the edited prefix: 8,082 read, 76 written.
**New since the design was written:** a whole session can be written from a rendered context, tool blocks included,
so a contracting arm doesn't need to patch an existing file. The format is undocumented, so this is checked per
version.

**The thinking check.** Neither of two logins got a 400 for editing under thinking blocks. An account created on or
after 2026-08-31 should [D], so this column needs your own run, and `strip_on_edit` is the public default.

**Claude Agent SDK.** It ran on the CLI's login with every metered key blanked from its environment, so it is a
subscription harness here, and no key was involved. Same session files, same edits.

**Codex CLI.** Edit the rollout JSONL. Usage is cumulative across a resume, so each turn is a difference. `codex fork`
is interactive, so a fork here is the rollout copied under a new thread id, and `exec resume` accepts it. A whole-context
write has to keep the real opening rows, which hold the harness's instructions, and each new row needs an `ordinal` or
resume refuses. Cached tokens on an edit stayed at the part before the edit in most runs, and read 0 in one.

**Codex under reasoning: the edit is accepted, and the answer doesn't always hold.** The same turn asking for the
codeword, on `gpt-6-luna`: **unedited, 6 of 6 answered it; edited under its reasoning items, 7 of 11**. The rest were
refusals or "I don't know", and none was an error. That is a small sample on a synthetic codeword, and it doesn't say
why. It is why a step reports `INCONCLUSIVE` rather than `PASS` when the edit is accepted and the answer doesn't hold,
and why dropping thinking on edit stays the default until a scored run says otherwise.

**Antigravity (`agy`).** A conversation is a SQLite file whose rows hold protobuf, and the same text sits in
`steps.step_payload` and `gen_metadata.data`. The kit doesn't parse protobuf, so an edit keeps the byte length: a
codeword swaps for one of the same length, and padding is overwritten with spaces, which still shrank the context by
about 2,300 tokens here. A fork is the store copied under a new id, with `trajectory_meta.cascade_id` changed to match.
There is no door for a whole rendered context. Usage is cumulative. No cache reads were reported on any turn, which
may be the CLI's reporting and is not evidence of no caching. A tier is part of the model's name, so `--effort` conflicts
with `-low`. `agy --version` can hang, so the kit reads its changelog.

**Ollama.** You own the list, so an edit is sending something different. `prompt_eval_count` counts only what it
evaluated, so it can't show a read. It honours `OLLAMA_HOST`. Pass `--model` for one you have.

## What this doesn't show

Conformance says a door works, not that a contracting arm is any good. It uses one small model per harness, so a
behaviour specific to a larger model won't show. And these formats are undocumented, so a CLI upgrade can break any of
it, which is why every manifest pins a version.
