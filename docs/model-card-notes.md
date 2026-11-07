# Model notes

What a run has to pin, and the defaults so far. Each scored run's manifest records its own.

| Setting | Default | Why |
|---|---|---|
| Yardstick for list price | `claude-sonnet-5-5` | Cost is usage times a dated price file, whichever door a call went through |
| Conformance, Claude | `claude-haiku-5-5` | The smallest model that shows every behaviour |
| Conformance, Codex | `gpt-6-luna` | Same reason |
| Conformance, Antigravity | `gemini-3.8-flash-low`, and `-high` for the thinking step | A tier is part of the model's name there |
| Thinking policy | `strip_on_edit` | The one every account can run |
| TTL | Set explicitly | A CLI on a subscription defaults to one hour, and a key to five minutes [D], so the same session bills differently by door |

**Price files** live in `src/eunoe/cost/prices/`, named by the date they were read, and copied from the provider's
own page. A file with no per-token rates yet (OpenAI's) reports tokens and no list price rather than a guess.

**A subscription gives tokens, not dollars.** List price is a yardstick for comparing arms, not a bill.
