import unittest

from eunoe.analysis import holm, paired_bootstrap
from eunoe.cachesim import LOOKBACK_POSITIONS, PrefixCache, ThinkingBindingError, positions
from eunoe.cost import ListPrice
from eunoe.interfaces import Block, Budget, CallUsage, EvictionPolicy, ImportanceSignal, RenderedContext, Renderer, Schedule, ScheduleState, ToolSpec
from eunoe.ledger import ASSISTANT, FILE, SYSTEM_NOTE, THINKING, USER, Ledger
from eunoe.levels import Level, digest, estimate_tokens, find_details, headline, reconstructible_stub, trim
from eunoe.policies import LadderPolicy, MaskPolicy, StubPolicy
from eunoe.renderer import DefaultRenderer, render_context
from eunoe.rng import Rng
from eunoe.schedules import EveryN, Expiry, ExpiryThreshold, Generational, Never, Threshold, grid_for, parse_schedule
from eunoe.signals import SIGNALS, get_signal
from eunoe.wire import canonical, to_anthropic, to_chat_messages, to_gemini, to_openai_input, to_text

PINNED_STREAM = [832, 162, 600, 362, 902]
LOG = "\n".join([f"line {i} ok" for i in range(400)] + ["ERROR failed at token RB-AAAA-11"] + [f"tail {i}" for i in range(400)])


def ledger_with_events() -> Ledger:
    led = Ledger()
    led.append(USER, "Please look at INC-1001. We decided to roll back.")
    led.append("tool:fetch_logs", LOG, input={"service": "svc-a"})
    led.append(THINKING, "", signature="sig", tokens_hint=100)
    led.append(ASSISTANT, "First paragraph about INC-1001.\n\nMiddle part.\n\nMore middle.\n\nLast paragraph.")
    led.append(FILE, "body\n" * 100, source="docs/a.md")
    led.append(SYSTEM_NOTE, "Reminder: the review is due. Extra detail here.")
    return led


class LevelsTest(unittest.TestCase):
    def test_trim_keeps_structure_and_error_lines(self):
        out = trim(LOG)
        self.assertIn("line 0 ok", out)
        self.assertIn("tail 399", out)
        self.assertIn("ERROR failed", out)
        self.assertIn("lines", out)
        self.assertLess(len(out), len(LOG))

    def test_digest_prefers_details_on_error_lines(self):
        self.assertIn("RB-AAAA-11", digest(LOG))
        self.assertLess(len(digest(LOG)), len(LOG) // 4)

    def test_headline_is_one_short_line(self):
        self.assertEqual(headline("first\nsecond"), "first")
        self.assertLessEqual(len(headline("x" * 999)), 160)

    def test_reconstructible_stub_has_path_hash_and_lines(self):
        stub = reconstructible_stub("docs/a.md", "a\nb\n")
        self.assertIn("docs/a.md", stub)
        self.assertIn("sha256:", stub)
        self.assertIn("2 lines", stub)

    def test_find_details_sees_identifiers_paths_and_numbers(self):
        kinds = {k for _, k, _ in find_details("INC-1001 at runbooks/a-1.md on 2031-01-02 cost 12345 and 40%")}
        self.assertTrue({"id", "path", "date", "number"} <= kinds)


class RendererTest(unittest.TestCase):
    def test_rendering_is_pure_and_byte_stable(self):
        led = ledger_with_events()
        first = [canonical(b) for b in render_context(led).blocks]
        again = [canonical(b) for b in render_context(led).blocks]
        self.assertEqual(first, again)

    def test_a_level_change_only_changes_bytes_from_that_event_on(self):
        led = ledger_with_events()
        before = [canonical(b) for b in render_context(led).blocks]
        led.set_level("e00003", Level.L2)
        after = [canonical(b) for b in render_context(led).blocks]
        common = sum(1 for a, b in zip(before, after) if a == b)
        self.assertEqual(before[:3], after[:3])
        self.assertLess(common, len(before))
        self.assertEqual(before[-1], after[-1])

    def test_each_level_is_no_longer_than_the_last(self):
        led = ledger_with_events()
        for event in led.events:
            if event.kind == THINKING:
                continue
            sizes = [led.tokens_at(event, level) for level in Level]
            self.assertEqual(sizes[4], 0)
            self.assertLessEqual(sizes[3], sizes[0] + 40, event.kind)

    def test_call_and_result_stay_a_valid_pair_through_l2_and_merge_at_l3(self):
        led = ledger_with_events()
        event = led.by_id["e00001"]
        renderer = DefaultRenderer()
        for level in (Level.L0, Level.L1, Level.L2):
            blocks = renderer.render(event, level)
            self.assertEqual([b.type for b in blocks], ["tool_use", "tool_result"])
            self.assertEqual(blocks[0].tool_id, blocks[1].tool_id)
        self.assertEqual([b.type for b in renderer.render(event, Level.L3)], ["text"])
        self.assertEqual(renderer.render(event, Level.L4), [])

    def test_thinking_has_no_levels_only_kept_or_gone(self):
        led = ledger_with_events()
        event = led.by_id["e00002"]
        renderer = DefaultRenderer()
        self.assertEqual(renderer.render(event, Level.L0), renderer.render(event, Level.L2))
        self.assertEqual(renderer.render(event, Level.L4), [])

    def test_markers(self):
        led = ledger_with_events()
        event = led.by_id["e00003"]
        self.assertNotIn("recall", DefaultRenderer("M0").render(event, Level.L2)[0].text)
        self.assertTrue(DefaultRenderer("M1").render(event, Level.L2)[0].text.endswith("…"))
        self.assertIn("[more: recall e00003]", DefaultRenderer("M2").render(event, Level.L2)[0].text)
        self.assertNotIn("more", DefaultRenderer("M2").render(event, Level.L0)[0].text)
        with self.assertRaises(ValueError):
            DefaultRenderer("M9")

    def test_implements_the_protocol(self):
        self.assertIsInstance(DefaultRenderer(), Renderer)


class LedgerTest(unittest.TestCase):
    def test_archive_is_untouched_by_a_level_change(self):
        led = ledger_with_events()
        led.set_level("e00001", Level.L4)
        self.assertEqual(led.archive_get("e00001").text, LOG)
        self.assertEqual(led.lookups["e00001"], 1)

    def test_search_needs_every_word_and_counts_as_a_lookup(self):
        led = ledger_with_events()
        hits = led.search("RB-AAAA-11 ERROR")
        self.assertEqual([h.id for h in hits], ["e00001"])
        self.assertEqual(led.search("RB-AAAA-11 nothing-like-this"), [])

    def test_references_count_later_mentions(self):
        led = ledger_with_events()
        first = led.by_id["e00000"]
        self.assertGreaterEqual(led.reference_count(first), 1)


class PolicyTest(unittest.TestCase):
    def _ledger(self, n=40) -> Ledger:
        led = Ledger()
        for i in range(n):
            led.call = i
            led.append("tool:fetch_logs", f"header {i}\n" + "x" * 2000 + f"\nend {i}", input={"service": f"s{i}"})
        led.call = n
        return led

    def test_ladder_reaches_the_budget_stepping_one_level(self):
        led = self._ledger()
        policy = LadderPolicy("S0")
        budget = Budget(led.tokens_held() // 2, keep_recent_calls=2)
        changes = policy.plan(led, budget, ScheduleState())
        for change in changes:
            self.assertLess(change.level, Level.L4)
            led.set_level(change.event_id, change.level)
        self.assertLessEqual(led.tokens_held(), budget.target_tokens)

    def test_ladder_takes_detail_from_many_before_one_goes(self):
        led = self._ledger()
        budget = Budget(int(led.tokens_held() * 0.6), keep_recent_calls=2)
        changes = LadderPolicy("S0").plan(led, budget, ScheduleState())
        self.assertGreater(len(changes), 5)
        self.assertTrue(all(c.level <= Level.L2 for c in changes))

    def test_an_event_leaves_only_as_the_last_resort(self):
        led = self._ledger(30)
        changes = LadderPolicy("S0").plan(led, Budget(1, keep_recent_calls=0), ScheduleState())
        self.assertTrue(all(c.level == Level.L4 for c in changes))
        led2 = self._ledger(30)
        headlines = sum(led2.tokens_at(e, Level.L3) for e in led2.events)
        changes = LadderPolicy("S0").plan(led2, Budget(headlines + 200, keep_recent_calls=0), ScheduleState())
        self.assertFalse([c for c in changes if c.level == Level.L4])

    def test_recent_calls_and_the_frozen_region_are_never_touched(self):
        led = self._ledger()
        changes = LadderPolicy("S0").plan(led, Budget(1, keep_recent_calls=5), ScheduleState(frozen_upto=3))
        touched = {c.event_id for c in changes}
        self.assertNotIn("e00000", touched)
        self.assertNotIn("e00003", touched)
        self.assertNotIn("e00039", touched)
        self.assertIn("e00004", touched)

    def test_oldest_first_really_goes_oldest_first(self):
        led = self._ledger()
        budget = Budget(led.tokens_held() - 100, keep_recent_calls=2)
        changes = LadderPolicy("S0").plan(led, budget, ScheduleState())
        self.assertEqual(changes[0].event_id, "e00000")

    def test_stub_goes_straight_to_a_headline(self):
        led = self._ledger()
        changes = StubPolicy("S0").plan(led, Budget(led.tokens_held() // 2, keep_recent_calls=2), ScheduleState())
        self.assertTrue(all(c.level == Level.L3 for c in changes))

    def test_mask_touches_tool_results_only(self):
        led = self._ledger(10)
        led.append(USER, "y" * 4000)
        led.call = 100
        changes = MaskPolicy().plan(led, Budget(1, keep_recent_calls=0), ScheduleState())
        self.assertNotIn("e00010", {c.event_id for c in changes})

    def test_reconstructible_event_skips_to_path_and_hash(self):
        led = ledger_with_events()
        led.call = 100
        policy = LadderPolicy("S0")
        self.assertEqual(policy.next_level(led.by_id["e00004"], Level.L0), Level.L3)
        self.assertEqual(policy.next_level(led.by_id["e00001"], Level.L0), Level.L1)

    def test_thinking_is_never_a_candidate(self):
        led = ledger_with_events()
        led.call = 100
        changes = LadderPolicy("S0").plan(led, Budget(1, keep_recent_calls=0), ScheduleState())
        self.assertNotIn("e00002", {c.event_id for c in changes})

    def test_implements_the_protocol(self):
        self.assertIsInstance(LadderPolicy(), EvictionPolicy)
        self.assertIsInstance(MaskPolicy(), EvictionPolicy)


class SignalTest(unittest.TestCase):
    def test_every_signal_scores_and_implements_the_protocol(self):
        led = ledger_with_events()
        for name in SIGNALS:
            kwargs = {"future_uses": {}, "oracle": True} if name == "S4" else {}
            signal = get_signal(name, **kwargs)
            self.assertIsInstance(signal, ImportanceSignal, name)
            self.assertIsInstance(signal.score(led.events[0], led, 0.0), float, name)

    def test_the_oracle_refuses_without_saying_so(self):
        with self.assertRaises(ValueError):
            get_signal("S4", future_uses={})

    def test_random_is_seeded_and_stable(self):
        led = ledger_with_events()
        a = get_signal("S1", seed=1).score(led.events[0], led, 0.0)
        self.assertEqual(a, get_signal("S1", seed=1).score(led.events[0], led, 0.0))
        self.assertNotEqual(a, get_signal("S1", seed=2).score(led.events[0], led, 0.0))

    def test_user_outranks_a_note_on_the_kind_prior(self):
        led = ledger_with_events()
        signal = get_signal("S3")
        self.assertGreater(signal.score(led.by_id["e00000"], led, 0.0), signal.score(led.by_id["e00005"], led, 0.0))

    def test_unknown_signal(self):
        with self.assertRaises(ValueError):
            get_signal("S99")


class ScheduleTest(unittest.TestCase):
    def _led(self, tokens: int) -> Ledger:
        led = Ledger()
        led.append("tool:x", "z" * (tokens * 4), input={"a": "b"})
        return led

    def test_threshold_fires_past_the_trigger_only(self):
        sched = Threshold(1000, 500)
        self.assertFalse(sched.due(self._led(500), CallUsage(), 0))
        self.assertTrue(sched.due(self._led(2000), CallUsage(), 0))
        self.assertEqual(sched.target(self._led(2000), Budget(1)), 500)

    def test_a_target_it_could_not_reach_waits_for_a_batch_of_growth(self):
        sched = Threshold(1000, 500)
        sched.note_edit(reached=1800, wanted=500, pre=2000)
        self.assertFalse(sched.due(self._led(2000), CallUsage(), 0))
        self.assertTrue(sched.due(self._led(3500), CallUsage(), 0))

    def test_every_n(self):
        sched = EveryN(3, 100)
        sched.state.calls_since_edit = 2
        self.assertFalse(sched.due(self._led(500), CallUsage(), 0))
        sched.state.calls_since_edit = 3
        self.assertTrue(sched.due(self._led(500), CallUsage(), 0))

    def test_expiry_fires_only_past_the_ttl(self):
        sched = Expiry(0.5, "1h")
        self.assertFalse(sched.due(self._led(500), CallUsage(), 3000))
        self.assertTrue(sched.due(self._led(500), CallUsage(), 4000))
        self.assertEqual(Expiry(0.5, "5m").due(self._led(500), CallUsage(), 400), True)

    def test_expiry_plus_threshold_picks_its_target_by_why_it_fired(self):
        sched = ExpiryThreshold(0.5, 1000, 400)
        led = self._led(2000)
        self.assertTrue(sched.due(led, CallUsage(), 0))
        self.assertEqual(sched.target(led, Budget(1)), 400)
        self.assertTrue(sched.due(led, CallUsage(), 4000))
        self.assertGreater(sched.target(led, Budget(1)), 400)

    def test_never(self):
        self.assertFalse(Never().due(self._led(99999), CallUsage(), 99999))

    def test_parse_all_forms(self):
        for spec in ("never", "every:100:200000", "every_call:200000", "threshold:300000:200000", "expiry:0.5",
                     "expiry:0.5:5m", "expiry_threshold:0.5:300000:200000", "generational:300000:200000:0.5"):
            self.assertIsInstance(parse_schedule(spec), Schedule, spec)
        self.assertEqual(parse_schedule("generational:3:2").frozen_share, 0.5)
        for bad in ("nope", "threshold:x:y", "every:1"):
            with self.assertRaises(ValueError):
                parse_schedule(bad)

    def test_the_grid_is_the_six_factors(self):
        grid = grid_for(200_000)
        self.assertEqual(len(grid), 6)
        for spec in grid.values():
            parse_schedule(spec)
        self.assertIsInstance(parse_schedule(grid["generational"]), Generational)


class WireTest(unittest.TestCase):
    def _ctx(self) -> RenderedContext:
        led = ledger_with_events()
        led.set_level("e00003", Level.L3)
        ctx = render_context(led, "sys", [ToolSpec("t", "d", {"type": "object"})])
        ctx.blocks.append(Block("user", "text", text="Question?"))
        return ctx

    def test_anthropic_shape_is_valid(self):
        wire = to_anthropic(self._ctx())
        self.assertEqual(wire["system"], "sys")
        self.assertEqual(wire["messages"][0]["role"], "user")
        roles = [m["role"] for m in wire["messages"]]
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])))
        pending = set()
        for message in wire["messages"]:
            for block in message["content"]:
                if block["type"] == "tool_use":
                    pending.add(block["id"])
                if block["type"] == "tool_result":
                    pending.discard(block["tool_use_id"])
            self.assertFalse(pending and message["role"] == "user", "a tool_use went unanswered")
        self.assertEqual(message["role"], "user")

    def test_a_tool_result_opens_its_turn(self):
        ctx = RenderedContext("", [Block("assistant", "tool_use", name="t", input={}, tool_id="a"),
                                   Block("user", "text", text="hello"),
                                   Block("user", "tool_result", text="r", tool_id="a")])
        user = to_anthropic(ctx)["messages"][-1]
        self.assertEqual(user["content"][0]["type"], "tool_result")

    def test_thinking_replay_can_be_turned_off(self):
        types = {b["type"] for m in to_anthropic(self._ctx(), False)["messages"] for b in m["content"]}
        self.assertNotIn("thinking", types)

    def test_a_scripted_opening_turn_gets_a_user_first(self):
        ctx = RenderedContext("", [Block("assistant", "text", text="hi"), Block("user", "text", text="q")])
        self.assertEqual(to_anthropic(ctx)["messages"][0]["role"], "user")

    def test_other_providers(self):
        ctx = self._ctx()
        self.assertEqual(to_openai_input(ctx)["instructions"], "sys")
        self.assertTrue(any(i["type"] == "function_call_output" for i in to_openai_input(ctx)["input"]))
        contents = to_gemini(ctx)["contents"]
        self.assertTrue(any("functionResponse" in p for c in contents for p in c["parts"]))
        self.assertEqual(to_chat_messages(ctx)[0]["role"], "system")

    def test_to_text_leaves_thinking_out(self):
        self.assertIn("Question?", to_text(self._ctx()))


class CacheSimTest(unittest.TestCase):
    def _ctx(self, *texts) -> RenderedContext:
        return RenderedContext("", [Block("user", "text", text=t) for t in texts])

    def test_the_rule_is_the_documented_twenty(self):
        self.assertEqual(LOOKBACK_POSITIONS, 20)

    def test_a_prefix_reads_and_the_rest_writes(self):
        cache = PrefixCache("5m")
        first = cache.call(self._ctx("a" * 400, "b" * 400), 0.0)
        self.assertEqual(first.cache_read, 0)
        second = cache.call(self._ctx("a" * 400, "b" * 400, "c" * 400), 1.0)
        self.assertEqual(second.cache_read, first.cache_write)
        self.assertGreater(second.cache_write_5m, 0)

    def test_an_edit_reads_back_only_as_far_as_an_earlier_entry(self):
        cache = PrefixCache("5m")
        cache.call(self._ctx("a" * 400), 0.0)
        cache.call(self._ctx("a" * 400, "b" * 400), 1.0)
        cache.call(self._ctx("a" * 400, "b" * 400, "c" * 400), 2.0)
        edited = cache.call(self._ctx("a" * 400, "b" * 400, "C" * 400), 3.0)
        self.assertEqual(edited.cache_read, 2 * (estimate_tokens("a" * 400) + 3))

    def test_a_request_past_the_lookback_window_misses_with_every_byte_unchanged(self):
        short = [f"m{i}" for i in range(19)]
        cache = PrefixCache("5m")
        cache.call(self._ctx(*short[:1]), 0.0)
        self.assertGreater(cache.call(self._ctx(*short), 1.0).cache_read, 0)
        far = [f"m{i}" for i in range(25)]
        cache = PrefixCache("5m")
        cache.call(self._ctx(*far[:1]), 0.0)
        self.assertEqual(cache.call(self._ctx(*far), 1.0).cache_read, 0)

    def test_a_second_breakpoint_reaches_back_past_the_window(self):
        far = [f"m{i}" for i in range(25)]
        cache = PrefixCache("5m")
        cache.call(self._ctx(*far[:1]), 0.0)
        ctx = self._ctx(*far)
        ctx.meta["breakpoints"] = [0]
        self.assertGreater(cache.call(ctx, 1.0).cache_read, 0)

    def test_tool_runs_collapse_to_one_position(self):
        blocks = [Block("assistant", "tool_use", name="t", tool_id=str(i)) for i in range(5)]
        blocks += [Block("user", "tool_result", text="r", tool_id=str(i)) for i in range(5)]
        blocks += [Block("user", "text", text="q")]
        self.assertEqual(positions(blocks)[-1], 2)

    def test_a_frozen_region_keeps_its_reads_through_an_edit(self):
        stem = [f"m{i}" * 50 for i in range(10)]
        cache = PrefixCache("5m")
        first = self._ctx(*stem)
        first.meta["breakpoints"] = [4]
        cache.call(first, 0.0)
        edited = self._ctx(*stem[:5], "changed", *stem[6:])
        edited.meta["breakpoints"] = [4]
        kept = cache.call(edited, 1.0).cache_read
        self.assertEqual(kept, sum(estimate_tokens(s) + 3 for s in stem[:5]))
        plain = PrefixCache("5m")
        plain.call(self._ctx(*stem), 0.0)
        self.assertEqual(plain.call(self._ctx(*stem[:5], "changed", *stem[6:]), 1.0).cache_read, 0)

    def test_expiry_kills_the_cache(self):
        cache = PrefixCache("5m")
        cache.call(self._ctx("a" * 400), 0.0)
        self.assertEqual(cache.call(self._ctx("a" * 400), 400.0).cache_read, 0)
        cache = PrefixCache("1h")
        cache.call(self._ctx("a" * 400), 0.0)
        self.assertGreater(cache.call(self._ctx("a" * 400), 400.0).cache_read, 0)

    def test_binding_refuses_an_edit_before_kept_thinking(self):
        cache = PrefixCache("5m", binding=True)
        blocks = [Block("user", "text", text="a"), Block("assistant", "thinking", signature="s", tokens_hint=5)]
        cache.call(RenderedContext("", blocks), 0.0)
        with self.assertRaises(ThinkingBindingError):
            cache.call(RenderedContext("", [Block("user", "text", text="A"), blocks[1]]), 1.0)

    def test_binding_allows_stripping_the_thinking(self):
        cache = PrefixCache("5m", binding=True)
        blocks = [Block("user", "text", text="a"), Block("assistant", "thinking", signature="s", tokens_hint=5)]
        cache.call(RenderedContext("", blocks), 0.0)
        cache.call(RenderedContext("", [Block("user", "text", text="A")]), 1.0)


class CostTest(unittest.TestCase):
    def test_price_is_usage_times_the_dated_file(self):
        usage = CallUsage(input=1_000_000, cache_read=1_000_000, cache_write_5m=1_000_000, cache_write_1h=1_000_000,
                          output=1_000_000)
        self.assertAlmostEqual(ListPrice("claude-opus-5-5").price(usage), 4 + 0.2 + 5 + 8 + 20)
        self.assertAlmostEqual(ListPrice("claude-sonnet-5-5").price(usage), 2 + 0.1 + 2.5 + 4 + 10)

    def test_haiku_long_prompt_tier(self):
        short = CallUsage(input=1000, output=1000)
        long = CallUsage(input=200_000, output=1000)
        cost = ListPrice("claude-haiku-5-5")
        self.assertAlmostEqual(cost.price(short), (1000 * 0.1 + 1000 * 0.5) / 1e6)
        self.assertAlmostEqual(cost.price(long), (200_000 * 0.5 + 1000 * 2.5) / 1e6)

    def test_an_unknown_model_refuses(self):
        with self.assertRaises(ValueError):
            ListPrice("no-such-model")


class StatsTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_brackets_the_mean(self):
        a, b = [0.9, 0.8, 0.85, 0.9], [0.7, 0.75, 0.7, 0.8]
        one = paired_bootstrap(a, b, resamples=500)
        self.assertEqual(one, paired_bootstrap(a, b, resamples=500))
        self.assertLessEqual(one["low"], one["mean"])
        self.assertLessEqual(one["mean"], one["high"])
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_holm_stops_at_the_first_miss(self):
        out = holm({"a": 0.001, "b": 0.06, "c": 0.01}, 0.05)
        self.assertTrue(out["a"])
        self.assertTrue(out["c"])
        self.assertFalse(out["b"])


class RngTest(unittest.TestCase):
    def test_the_stream_is_pinned(self):
        rng = Rng("pin", 1)
        self.assertEqual([rng.randint(0, 999) for _ in range(5)], PINNED_STREAM)

    def test_shuffle_and_child_are_seeded(self):
        self.assertEqual(Rng("a").shuffle([1, 2, 3, 4]), Rng("a").shuffle([1, 2, 3, 4]))
        self.assertNotEqual(Rng("a").child("x").random(), Rng("a").child("y").random())


if __name__ == "__main__":
    unittest.main()
