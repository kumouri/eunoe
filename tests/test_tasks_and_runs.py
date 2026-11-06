import json
import unittest

from eunoe.agents import DefaultAgent
from eunoe.arms import ARMS, CONTROLS, RECALL_TOOL, TREATMENTS, build_arm
from eunoe.cost import ListPrice
from eunoe.demo import format_table, run_arms
from eunoe.drivers.scripted import ScriptedDriver
from eunoe.interfaces import Agent, BlindItem, Block, Driver, Judge, Probe, RenderedContext, TaskSet
from eunoe.judge import GoldJudge, normalize, parse_answer, score_probe
from eunoe.levels import Level
from eunoe.runner import SessionRun, aggregate
from eunoe.tasks.generator import DOMAINS, Generator, generate_session, get_profile, invented_name
from eunoe.rng import Rng
from eunoe.tasks.session import EventSpec

#: A fixture's bytes are part of what a manifest pins. A change here is a new fixture version, on purpose.
GOLDEN = {
    ("oncall", 1, "tiny"): "94eb6afdc234af326234fb8bd385e7aa2a4a71a92e73ef2c69bf48a12f4c74f9",
    ("smallbiz", 1, "tiny"): "bb5d762fdfc29edf830cbcd854129060b387e0508787f718d636035c5e22ee94",
    ("research", 1, "tiny"): "0d961991a31298b3c62ec470e4fef27d76486a3df355f46e7a41b25a95d5765a",
}


class GeneratorTest(unittest.TestCase):
    def test_a_seed_always_gives_the_same_bytes(self):
        for (domain, seed, profile), expected in GOLDEN.items():
            print_me = generate_session(domain, seed, get_profile(profile)).fingerprint()
            self.assertEqual(print_me, generate_session(domain, seed, get_profile(profile)).fingerprint())
            self.assertEqual(print_me, expected, f"{domain} fixture changed")

    def test_a_new_seed_gives_a_new_session(self):
        one = generate_session("oncall", 1, get_profile("tiny")).fingerprint()
        self.assertNotEqual(one, generate_session("oncall", 2, get_profile("tiny")).fingerprint())

    def test_probes_come_after_their_facts_and_have_gold(self):
        for domain in DOMAINS:
            session = generate_session(domain, 3, get_profile("demo"))
            facts = {f.key: f for f in session.facts}
            self.assertGreaterEqual(len(session.probes), 10)
            kinds = {p.kind for p in session.probes}
            self.assertIn("unanswerable", kinds)
            for probe in session.probes:
                if probe.kind == "unanswerable":
                    self.assertEqual(probe.gold, "")
                    self.assertNotIn(probe.key, facts)
                    continue
                self.assertGreater(probe.at_step, facts[probe.key].at_step)
                self.assertEqual(probe.gold, facts[probe.key].value)

    def test_gold_is_really_in_the_session_before_its_probe(self):
        for domain in DOMAINS:
            session = generate_session(domain, 2, get_profile("demo"))
            seen = ""
            for step in session.steps:
                if step.kind == "event":
                    seen += step.event.text
                if step.kind == "probe" and step.probe.gold:
                    self.assertIn(step.probe.gold, seen, step.probe.id)
                if step.kind == "task":
                    self.assertIn(step.task.gold, seen, step.task.id)

    def test_a_superseded_fact_appears_old_then_new(self):
        found = 0
        for seed in range(1, 8):
            session = generate_session("oncall", seed, get_profile("demo"))
            text = "\n".join(s.event.text for s in session.steps if s.kind == "event")
            for fact in session.facts:
                if fact.stale:
                    found += 1
                    self.assertLess(text.index(fact.stale[0]), text.rindex(fact.value))
        self.assertGreater(found, 0)

    def test_it_has_all_four_probe_kinds_and_tasks(self):
        session = generate_session("research", 1, get_profile("demo"))
        self.assertTrue({"recall", "artifact", "continuation", "decision"} <= {p.kind for p in session.probes})
        self.assertGreaterEqual(len(session.tasks), 1)

    def test_names_are_built_from_syllables(self):
        name = invented_name(Rng("x"))
        self.assertEqual(len(name.split()), 2)

    def test_it_is_a_task_set(self):
        self.assertIsInstance(Generator("tiny"), TaskSet)
        sessions = list(Generator("tiny").sessions(1))
        self.assertEqual({s.domain for s in sessions}, set(DOMAINS))

    def test_unknown_names_refuse(self):
        with self.assertRaises(ValueError):
            get_profile("nope")
        with self.assertRaises(ValueError):
            generate_session("nope", 1, get_profile("tiny"))

    def test_the_environment_answers_and_scores_a_task(self):
        session = generate_session("oncall", 1, get_profile("demo"))
        env = session.environment()
        task = session.tasks[0]
        self.assertFalse(env.check(task))
        self.assertIn("applied", env.execute("apply", {"subject": task.subject, "value": task.gold}))
        self.assertTrue(env.check(task))
        self.assertIn("no records", env.execute("fetch_logs", {"service": "not-a-service"}))


class JudgeTest(unittest.TestCase):
    def _probe(self, gold="RB-1", stale=()):
        return Probe("p", "recall", "q", "k", "s", gold, stale)

    def test_reads_the_answer_line(self):
        self.assertEqual(parse_answer("thinking aloud\nANSWER: RB-1"), "RB-1")
        self.assertEqual(parse_answer("RB-1"), "RB-1")

    def test_correct_stale_abstain_and_hallucinated(self):
        self.assertTrue(score_probe("ANSWER: rb-1.", self._probe()).correct)
        score = score_probe("ANSWER: OLD-9", self._probe(stale=("OLD-9",)))
        self.assertTrue(score.stale)
        self.assertFalse(score.hallucinated)
        score = score_probe("ANSWER: UNKNOWN", self._probe())
        self.assertTrue(score.abstained)
        self.assertFalse(score.hallucinated)
        self.assertTrue(score_probe("ANSWER: RB-2", self._probe()).hallucinated)

    def test_unanswerable_rewards_only_an_admission(self):
        self.assertTrue(score_probe("ANSWER: UNKNOWN", self._probe(gold="")).correct)
        self.assertTrue(score_probe("ANSWER: RB-2", self._probe(gold="")).hallucinated)

    def test_normalize_ignores_quoting_and_case(self):
        self.assertEqual(normalize("`RB-1`."), "rb-1")

    def test_the_judge_sees_no_arm(self):
        judge = GoldJudge()
        self.assertIsInstance(judge, Judge)
        self.assertEqual(judge.grade(BlindItem("i", "q", "ANSWER: RB-1", "RB-1")).score, 1.0)
        self.assertEqual(judge.grade(BlindItem("i", "q", "ANSWER: nope", "RB-1")).score, 0.0)
        self.assertEqual(set(BlindItem.__dataclass_fields__), {"id", "ask", "answer", "gold"})


class ScriptedReaderTest(unittest.TestCase):
    """The plumbing can't cheat: the reader answers from what it was handed, and nothing else."""

    def _request(self, blocks, **meta):
        return RenderedContext("", blocks, [], {"now": 1.0, "id": "p", "key": "token for X-1", "subject": "X-1",
                                                "prefix": "RB-", "mode": "probe", **meta})

    def test_it_answers_from_a_statement(self):
        driver = ScriptedDriver(guess_rate=0.0)
        turn = driver.call(self._request([Block("user", "text", text="The token for X-1 is RB-77.")]))
        self.assertEqual(turn.text, "ANSWER: RB-77")

    def test_the_newest_statement_wins(self):
        driver = ScriptedDriver(guess_rate=0.0)
        turn = driver.call(self._request([Block("user", "text", text="The token for X-1 is RB-1."),
                                          Block("user", "text", text="The token for X-1 is RB-2.")]))
        self.assertEqual(turn.text, "ANSWER: RB-2")

    def test_with_the_fact_gone_it_cannot_know_it(self):
        driver = ScriptedDriver(guess_rate=0.0, blind_lookup_rate=0.0)
        turn = driver.call(self._request([Block("user", "text", text="nothing relevant")]))
        self.assertEqual(turn.text, "ANSWER: UNKNOWN")

    def test_a_digest_token_near_its_subject_can_be_read(self):
        driver = ScriptedDriver(guess_rate=0.0)
        turn = driver.call(self._request([Block("user", "tool_result", text="X-1; RB-55; other")]))
        self.assertEqual(turn.text, "ANSWER: RB-55")

    def test_two_tokens_of_the_shape_are_ambiguous(self):
        driver = ScriptedDriver(guess_rate=0.0, blind_lookup_rate=0.0, lookup_rate=0.0)
        turn = driver.call(self._request([Block("user", "tool_result", text="X-1; RB-55; RB-66")]))
        self.assertEqual(turn.text, "ANSWER: UNKNOWN")

    def test_a_marker_always_earns_a_lookup(self):
        driver = ScriptedDriver(guess_rate=0.0, lookup_rate=0.0, blind_lookup_rate=0.0)
        request = self._request([Block("user", "text", text="X-1 summary [more: recall e00001]")])
        request.tools = [RECALL_TOOL]
        self.assertEqual(driver.call(request).tool_calls[0].name, "recall")

    def test_it_implements_the_protocol_and_cannot_spend(self):
        driver = ScriptedDriver()
        self.assertIsInstance(driver, Driver)
        self.assertFalse(driver.capabilities.spends)


class ArmTest(unittest.TestCase):
    def _session(self):
        return generate_session("oncall", 1, get_profile("tiny"))

    def test_every_builtin_arm_builds(self):
        for arm in CONTROLS + TREATMENTS:
            strategy = build_arm(arm, budget=2000, tools=[])
            self.assertEqual(strategy.name, arm)

    def test_external_arms_refuse_loudly_when_absent(self):
        with self.assertRaises(RuntimeError):
            build_arm("lcm", budget=2000, tools=[], )
        with self.assertRaises(ValueError):
            build_arm("nope", budget=2000)

    def test_only_contracting_arms_get_the_recall_tool(self):
        self.assertFalse(build_arm("C0", budget=2000).has_recall)
        self.assertFalse(build_arm("C3", budget=2000).has_recall)
        self.assertTrue(build_arm("T2", budget=2000).has_recall)

    def test_t3_carries_the_marker_and_t2_does_not(self):
        self.assertEqual(build_arm("T3", budget=2000).ledger.renderer.marker, "M2")
        self.assertEqual(build_arm("T2", budget=2000).ledger.renderer.marker, "M0")

    def test_an_edit_strips_thinking_on_the_default_policy(self):
        strategy = build_arm("T2", budget=100, keep_recent=0)
        strategy.observe(EventSpec("thinking", signature="s", tokens_hint=50), 0.0)
        strategy.observe(EventSpec("tool:x", "y" * 4000, input={"a": "b"}), 0.0)
        strategy.prepare(1.0, None, 1.0)
        self.assertEqual(strategy.ledger.level("e00000"), Level.L4)

    def test_allow_keeps_thinking_through_an_edit(self):
        strategy = build_arm("T2", budget=100, keep_recent=0, thinking="allow")
        strategy.observe(EventSpec("thinking", signature="s", tokens_hint=50), 0.0)
        strategy.observe(EventSpec("tool:x", "y" * 4000, input={"a": "b"}), 0.0)
        strategy.prepare(1.0, None, 1.0)
        self.assertEqual(strategy.ledger.level("e00000"), Level.L0)

    def test_off_never_holds_thinking(self):
        strategy = build_arm("C0", budget=100, thinking="off")
        strategy.observe(EventSpec("thinking", signature="s", tokens_hint=50), 0.0)
        self.assertEqual(strategy.ledger.events, [])

    def test_a_bad_thinking_policy_refuses(self):
        with self.assertRaises(ValueError):
            build_arm("C0", budget=100, thinking="whatever")

    def test_a_lookup_reaches_the_verbatim_archive_after_eviction(self):
        strategy = build_arm("T2", budget=100, keep_recent=0)
        strategy.observe(EventSpec("tool:x", "The token for X-1 is RB-1.\n" + "y" * 4000, input={"a": "b"}), 0.0)
        strategy.prepare(1.0, None, 1.0)
        self.assertNotEqual(strategy.ledger.level("e00000"), Level.L0)
        self.assertIn("The token for X-1 is RB-1.", strategy.recall("X-1"))
        self.assertIn("The token for X-1 is RB-1.", strategy.recall("e00000"))
        self.assertEqual(strategy.recall("not-there-at-all"), "no match")

    def test_an_emulated_control_says_so_and_pays_for_its_summary(self):
        strategy = build_arm("C1", budget=100)
        self.assertTrue(strategy.manifest()["emulated"])
        for _ in range(10):
            strategy.observe(EventSpec("tool:x", "y" * 2000, input={"a": "b"}), 0.0)
        strategy.prepare(1.0, None, 1.0)
        self.assertEqual(strategy.edits, 1)
        self.assertTrue(strategy.overhead)
        self.assertEqual(strategy.prefix[0].event_id, "summary")

    def test_a_server_side_control_asks_the_provider_and_edits_nothing(self):
        for arm, kind in (("C1", "compact_20260112"), ("C4", "clear_tool_uses_20250919")):
            strategy = build_arm(arm, budget=100, server_side=True)
            strategy.observe(EventSpec("tool:x", "y" * 4000, input={"a": "b"}), 0.0)
            ctx = strategy.prepare(1.0, None, 1.0)
            self.assertEqual(ctx.meta["context_management"]["edits"][0]["type"], kind)
            self.assertEqual(strategy.edits, 0)
            self.assertFalse(strategy.manifest()["emulated"])

    def test_masking_keeps_the_call_and_placeholders_the_result(self):
        strategy = build_arm("C3", budget=50, keep_recent=0)
        strategy.observe(EventSpec("tool:x", "y" * 4000, input={"a": "b"}), 0.0)
        ctx = strategy.prepare(1.0, None, 1.0)
        self.assertEqual([b.type for b in ctx.blocks], ["tool_use", "tool_result"])
        self.assertIn("omitted", ctx.blocks[1].text)


class RunnerTest(unittest.TestCase):
    def _run(self, arm="T2", profile="tiny", budget=2000, **kw):
        session = generate_session("oncall", 1, get_profile(profile))
        agent = DefaultAgent(DOMAINS["oncall"].brief, session.tool_specs)
        self.assertIsInstance(agent, Agent)
        strategy = build_arm(arm, budget=budget, system=agent.instructions, tools=agent.tools(), **kw)
        return SessionRun(session, strategy, ScriptedDriver(seed="1:oncall"), ListPrice(), "5m").run(), session

    def test_no_management_recalls_everything_it_was_told(self):
        result, session = self._run("C0")
        answerable = [s for s in result.scores if s.kind != "unanswerable"]
        self.assertTrue(all(s.correct for s in answerable), [s.answer for s in answerable])
        self.assertEqual(result.edits, 0)
        self.assertEqual(result.tasks_ok, result.tasks_total)

    def test_probes_never_enter_the_ledger(self):
        session = generate_session("oncall", 1, get_profile("tiny"))
        strategy = build_arm("C0", budget=2000)
        SessionRun(session, strategy, ScriptedDriver(), ListPrice(), "5m").run()
        for event in strategy.ledger.events:
            for probe in session.probes:
                self.assertNotIn(probe.ask, event.text)

    def test_everything_gets_scored_and_priced(self):
        result, session = self._run("T2")
        self.assertEqual(len(result.scores), len(session.probes))
        self.assertGreater(result.cost_usd, 0)
        self.assertGreater(result.calls, 0)
        self.assertEqual(result.manifest["fixture"], session.fingerprint())
        self.assertEqual(result.manifest["signal"], "FrequencyWeight")

    def test_every_arm_sees_the_same_sessions(self):
        rows = run_arms(("C0", "T2"), "tiny", (1,), domains=("oncall",), budget=2000)
        self.assertEqual(rows["C0"][1][0].manifest["fixture"], rows["T2"][1][0].manifest["fixture"])

    def test_an_edit_shows_up_in_the_cache(self):
        rows = run_arms(("C0", "T2"), "demo", (1,), domains=("oncall",), budget=8000)
        self.assertGreater(rows["T2"][0]["edits"], 0)
        self.assertLess(rows["T2"][0]["mean_context"], rows["C0"][0]["mean_context"])
        self.assertGreater(rows["C0"][0]["probe_recall"], 0.95)

    def test_eviction_really_loses_facts_for_a_summary_arm(self):
        rows = run_arms(("C0", "C1"), "demo", (1,), budget=8000)
        self.assertLess(rows["C1"][0]["probe_recall"], rows["C0"][0]["probe_recall"])

    def test_bound_account_with_allow_records_the_failure(self):
        session = generate_session("oncall", 1, get_profile("demo"))
        strategy = build_arm("T2", budget=5000, thinking="allow")
        result = SessionRun(session, strategy, ScriptedDriver(binding=True), ListPrice(), "5m").run()
        self.assertTrue(result.errors)
        self.assertIn("400", result.errors[0])

    def test_bound_account_with_strip_on_edit_runs_clean(self):
        session = generate_session("oncall", 1, get_profile("demo"))
        strategy = build_arm("T2", budget=5000, thinking="strip_on_edit")
        result = SessionRun(session, strategy, ScriptedDriver(binding=True), ListPrice(), "5m").run()
        self.assertEqual(result.errors, [])

    def test_aggregate_and_table(self):
        rows = run_arms(("C0",), "tiny", (1,), domains=("oncall",), budget=2000)
        agg = rows["C0"][0]
        for key in ("probe_recall", "hallucination_on_evicted", "correct_abstention", "task_success", "lookups",
                    "cost_usd", "mean_context", "peak_context", "staleness", "cache_read", "cache_write"):
            self.assertIn(key, agg)
        self.assertIn("C0", format_table(rows))
        json.dumps(agg)
        self.assertEqual(aggregate([])["calls"], 0)


if __name__ == "__main__":
    unittest.main()
