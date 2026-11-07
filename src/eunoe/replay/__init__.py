"""Read-only replay over your own CLI session files. No model call, no network, nothing kept but counts.

Usage is deduped by `message.id`, because the CLI writes one row per content block, each with the same usage,
and summing rows doubles every token count. The cost surface holds behaviour fixed: same calls, same output,
same new input, so it bounds cost and says nothing about quality.

An importance-and-regret replay needs a ledger per session and is left for the phase that uses it.
"""
