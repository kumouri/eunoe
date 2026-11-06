"""Every harness offline: argument shapes, usage arithmetic, editing and the spend gate."""
import contextlib
import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from eunoe.cost import ListPrice
from eunoe.drivers.base import CliResult, OfflineError, SpendGate, SpendRefused, http_post, is_offline, run_cli, scrubbed_env
from eunoe.drivers.live import ClaudeApiDriver, ClaudeCliDriver, _from_text, make_driver
from eunoe.editors import HARNESSES, SUBSCRIPTION_OR_FREE, get_harness
from eunoe.editors.agy import AgyHarness
from eunoe.editors.base import Harness, Turn, Unsupported, edit_jsonl, replace_in_json
from eunoe.editors.claude_cli import ClaudeCliHarness, claude_usage, session_rows
from eunoe.editors.codex import CodexHarness, _rollout_rows
from eunoe.editors.conformance import (
    CODEWORD, NEW_CODEWORD, Suite, arms_available, demo_context, format_report, make_padding, report_to_json, scrub,
)
from eunoe.editors.scripted import ScriptedHarness
from eunoe.editors.server_side import ClaudeServerHarness
from eunoe.editors.stateless import ClaudeApiHarness, GeminiApiHarness, OllamaHarness, OpenAiResponsesHarness
from eunoe.interfaces import Block, CallUsage, ContextEditor, RenderedContext


class SuiteTest(unittest.TestCase):
    def test_the_offline_harness_conforms_and_implements_the_protocol(self):
        harness = ScriptedHarness()
        self.assertIsInstance(harness, ContextEditor)
        report = Suite(harness).run()
        self.assertTrue(report.conforms, format_report(report))
        self.assertEqual({s.status for s in report.steps}, {"PASS"})
        self.assertTrue(all(report.arms.values()), report.arms)
        self.assertEqual(report.spent_usd, 0.0)

    def test_a_bound_account_loses_only_the_arms_that_need_it(self):
        report = Suite(ScriptedHarness(binding=True)).run()
        by_step = {s.step: s for s in report.steps}
        self.assertEqual(by_step["5b thinking"].status, "FAIL")
        self.assertIn("400", by_step["5b thinking"].note)
        self.assertFalse(report.arms["thinking_policy_allow"])
        self.assertTrue(report.arms["thinking_policy_strip_on_edit"])
        self.assertTrue(report.conforms)

    def test_an_accepted_edit_with_a_wrong_answer_is_inconclusive_not_a_400(self):
        class Forgetful(ScriptedHarness):
            def _reply(self, prompt):
                if self.thinking and "only the codeword" in prompt:
                    return "I don't know."
                return super()._reply(prompt)

        report = Suite(Forgetful()).run()
        by_step = {s.step: s for s in report.steps}
        self.assertEqual(by_step["5b thinking"].status, "INCONCLUSIVE")
        self.assertIn("I don't know", by_step["5b thinking"].note)
        self.assertFalse(report.arms["thinking_policy_allow"])
        self.assertTrue(report.conforms)

    def test_an_edit_that_does_not_land_is_a_failure(self):
        class Deaf(ScriptedHarness):
            def replace(self, old, new):
                return 1  # claims it edited, and did not

        report = Suite(Deaf()).run()
        self.assertFalse(report.conforms)
        self.assertEqual(report.steps[-1].status, "FAIL")
        self.assertFalse(report.arms["contracting_arms"])

    def test_a_harness_that_cannot_answer_fails_at_step_one(self):
        class Broken(ScriptedHarness):
            def start(self, prompt, thinking=False):
                return Turn(error="no login")

        report = Suite(Broken()).run()
        self.assertEqual(len(report.steps), 1)
        self.assertFalse(report.conforms)

    def test_an_unavailable_harness_is_named_not_failed(self):
        class Absent(ScriptedHarness):
            def available(self):
                return False, "needs a key"

        report = Suite(Absent()).run()
        self.assertFalse(report.available)
        self.assertIn("needs a key", format_report(report))

    def test_a_missing_door_is_unsupported_not_approximated(self):
        class NoFork(ScriptedHarness):
            def fork(self, prompt):
                raise Unsupported("no fork")

            def apply(self, session, rendered):
                raise Unsupported("no writes")

        report = Suite(NoFork()).run()
        by_step = {s.step: s.status for s in report.steps}
        self.assertEqual(by_step["5a fork"], "UNSUPPORTED")
        self.assertEqual(by_step["6 apply"], "UNSUPPORTED")
        self.assertFalse(report.arms["ablation_forks"])
        self.assertTrue(report.conforms)

    def test_a_report_never_holds_a_path_or_a_session_id(self):
        noisy = ("error at C:\\Users\\alice\\x\\y.jsonl and /home/alice/z.jsonl for "
                 "123e4567-e89b-12d3-a456-426614174000")
        self.assertNotIn("alice", scrub(noisy))
        self.assertNotIn("123e4567", scrub(noisy))

        class Noisy(ScriptedHarness):
            def start(self, prompt, thinking=False):
                return Turn(error=noisy)

        text = report_to_json(Suite(Noisy()).run())
        self.assertNotIn("alice", text)
        self.assertNotIn("123e4567", text)

    def test_padding_is_deterministic_and_numbered(self):
        self.assertEqual(make_padding(50), make_padding(50))
        self.assertEqual(len(set(make_padding(50).split())), 50)

    def test_your_own_harness_loads_by_path(self):
        harness = get_harness("eunoe.editors.scripted:ScriptedHarness")
        self.assertIsInstance(harness, Harness)
        self.assertTrue(Suite(harness).run().conforms)
        with self.assertRaises(ValueError):
            get_harness("nope")

    def test_all_only_means_what_cannot_spend(self):
        for name in SUBSCRIPTION_OR_FREE:
            self.assertIn(name, HARNESSES)
        self.assertNotIn("claude-api", SUBSCRIPTION_OR_FREE)
        self.assertNotIn("openai-api", SUBSCRIPTION_OR_FREE)
        self.assertNotIn("gemini-api", SUBSCRIPTION_OR_FREE)


class SpendTest(unittest.TestCase):
    def test_a_paid_call_refuses_without_permission(self):
        with self.assertRaises(SpendRefused):
            SpendGate().check("x")

    def test_the_cap_stops_a_run(self):
        gate = SpendGate(allow=True, cap_usd=0.01)
        gate.check("x")
        gate.record("x", "m", CallUsage(), 0.02)
        with self.assertRaises(SpendRefused):
            gate.check("x")

    def test_every_call_is_written_down(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "spend.jsonl")
            gate = SpendGate(allow=True, ledger_path=path)
            gate.record("claude-api", "claude-haiku-5-5", CallUsage(input=5, output=1), 0.5)
            row = json.loads(open(path, encoding="utf-8").read())
        self.assertEqual(row["list_usd"], 0.5)
        self.assertEqual(gate.spent_usd, 0.5)

    def test_offline_refuses_a_subprocess_and_a_request(self):
        self.assertTrue(is_offline())
        with self.assertRaises(OfflineError):
            run_cli(["claude"])
        with self.assertRaises(OfflineError):
            http_post("https://example.invalid", {}, {})

    def test_every_metered_key_is_scrubbed_from_a_child(self):
        env = scrubbed_env({"ANTHROPIC_API_KEY": "a", "ANTHROPIC_API_KEY_OTHER": "b", "OPENAI_API_KEY": "c",
                            "GEMINI_API_KEY": "d", "GOOGLE_API_KEY": "e", "GOOGLE_GENAI_USE_VERTEXAI": "f",
                            "CLAUDECODE": "1", "PATH": "keep"})
        self.assertEqual(env, {"PATH": "keep"})

    def test_a_paid_harness_is_unavailable_until_it_has_key_and_permission(self):
        self.assertIn("offline", ClaudeApiHarness(environ={"EUNOE_OFFLINE": "1"}).available()[1])
        self.assertIn("EUNOE_ANTHROPIC_API_KEY", ClaudeApiHarness(environ={}).available()[1])
        self.assertIn("--allow-spend", ClaudeApiHarness(environ={"EUNOE_ANTHROPIC_API_KEY": "k"}).available()[1])
        allowed = ClaudeApiHarness(gate=SpendGate(allow=True), environ={"EUNOE_ANTHROPIC_API_KEY": "k"})
        self.assertTrue(allowed.available()[0])

    def test_it_reads_only_its_own_named_key(self):
        harness = ClaudeApiHarness(environ={"ANTHROPIC_API_KEY": "personal", "ANTHROPIC_API_KEY_OTHER": "other"})
        self.assertFalse(harness.available()[0])
        self.assertEqual(harness.headers()["x-api-key"], "")


class JsonEditTest(unittest.TestCase):
    def test_replace_walks_keys_and_values(self):
        out, n = replace_in_json({"a": ["x old y", {"old": 1}], "b": 2}, "old", "new")
        self.assertEqual(n, 2)
        self.assertEqual(out, {"a": ["x new y", {"new": 1}], "b": 2})

    def test_edit_jsonl_changes_only_rows_that_hold_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "s.jsonl")
            untouched = '{"a":   "spacing kept",  "n": 1}\n'
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(untouched)
                fh.write(json.dumps({"m": "the old text"}) + "\n")
            self.assertEqual(edit_jsonl(path, "old", "new"), 1)
            lines = open(path, encoding="utf-8", newline="").read().splitlines(True)
        self.assertEqual(lines[0], untouched)
        self.assertEqual(json.loads(lines[1]), {"m": "the new text"})


class FakeRunner:
    """A recorded shape: answers from `script`, remembers what it was asked."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def __call__(self, argv, cwd=None, env=None, input_text=None, timeout=0):
        self.calls.append(dict(argv=argv, env=env, input=input_text, cwd=cwd))
        return self.script.pop(0)


def claude_result(text="OK", session="11111111-1111-1111-1111-111111111111", **usage):
    return CliResult(0, json.dumps({"result": text, "session_id": session, "is_error": False, "total_cost_usd": 0.001,
                                    "usage": {"input_tokens": 2, "cache_read_input_tokens": 10,
                                              "cache_creation_input_tokens": 5, "output_tokens": 1,
                                              "cache_creation": {"ephemeral_1h_input_tokens": 5}, **usage}}), "")


class ClaudeCliTest(unittest.TestCase):
    def _harness(self, *script, config_dir="cfg"):
        return ClaudeCliHarness(runner=FakeRunner(*script), config_dir=config_dir,
                                environ={"PATH": "x", "ANTHROPIC_API_KEY": "secret"})

    def test_a_run_has_no_tools_no_mcp_and_no_user_settings(self):
        harness = self._harness(claude_result())
        turn = harness.start("hello")
        call = harness.runner.calls[0]
        self.assertEqual(turn.text, "OK")
        for flag in ("--strict-mcp-config", "--session-id", "--setting-sources"):
            self.assertIn(flag, call["argv"])
        self.assertEqual(call["argv"][call["argv"].index("--tools") + 1], "")
        self.assertNotIn("user", call["argv"][call["argv"].index("--setting-sources") + 1])
        self.assertEqual(call["input"], "hello")
        harness.cleanup()

    def test_a_metered_key_never_reaches_the_child(self):
        harness = self._harness(claude_result())
        harness.start("hello")
        self.assertNotIn("ANTHROPIC_API_KEY", harness.runner.calls[0]["env"])
        harness.cleanup()

    def test_resume_and_fork_flags(self):
        forked = claude_result(session="22222222-2222-2222-2222-222222222222")
        harness = self._harness(claude_result(), claude_result(), forked)
        harness.start("a")
        harness.resume("b")
        self.assertEqual(harness.runner.calls[1]["argv"][harness.runner.calls[1]["argv"].index("--resume") + 1], harness.current)
        turn = harness.fork("c")
        self.assertIn("--fork-session", harness.runner.calls[2]["argv"])
        self.assertEqual(turn.error, "")
        self.assertIn("22222222-2222-2222-2222-222222222222", harness.sessions)
        harness.cleanup()

    def test_a_fork_that_kept_the_id_is_an_error(self):
        same = "33333333-3333-3333-3333-333333333333"
        harness = self._harness(claude_result(session=same))
        harness.current = same
        self.assertIn("same session", harness.fork("c").error)
        harness.cleanup()

    def test_thinking_asks_for_high_effort(self):
        harness = self._harness(claude_result())
        harness.start("a", thinking=True)
        self.assertEqual(harness.runner.calls[0]["argv"][harness.runner.calls[0]["argv"].index("--effort") + 1], "high")
        harness.cleanup()

    def test_garbage_and_errors_become_a_turn_error(self):
        harness = self._harness(CliResult(1, "not json", "boom"))
        self.assertIn("boom", harness.start("a").error)
        harness = self._harness(CliResult(0, json.dumps({"is_error": True, "result": "400 bad"}), ""))
        self.assertIn("400", harness.start("a").error)

    def test_usage(self):
        usage = claude_usage({"input_tokens": 1, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3,
                              "output_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 1,
                                                                     "ephemeral_1h_input_tokens": 2}})
        self.assertEqual((usage.input, usage.cache_read, usage.cache_write_5m, usage.cache_write_1h, usage.output),
                         (1, 2, 1, 2, 4))
        self.assertEqual(claude_usage({"cache_creation_input_tokens": 9}).cache_write_5m, 9)

    def test_the_scratch_project_folder_goes_with_what_the_cli_put_in_it(self):
        with tempfile.TemporaryDirectory() as config:
            harness = self._harness(config_dir=config)
            scratch = harness._dir()
            folder = os.path.join(config, "projects", "".join(c if c.isalnum() else "-" for c in os.path.realpath(scratch)))
            os.makedirs(os.path.join(folder, "memory"))
            harness.cleanup()
            self.assertFalse(os.path.exists(folder))
            self.assertFalse(os.path.exists(scratch))

    def test_session_files_get_edited_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as config:
            sid = "44444444-4444-4444-4444-444444444444"
            folder = os.path.join(config, "projects", "C--somewhere-eunoe-conform-abc")
            os.makedirs(folder)
            path = os.path.join(folder, f"{sid}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "user", "message": {"content": CODEWORD}}) + "\n")
                fh.write(json.dumps({"type": "last-prompt", "lastPrompt": CODEWORD}) + "\n")
                fh.write(json.dumps({"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": ""}]}}) + "\n")
            harness = self._harness(config_dir=config)
            harness.current = sid
            harness.sessions.append(sid)
            self.assertEqual(harness.replace(CODEWORD, NEW_CODEWORD), 2)
            self.assertEqual(harness.thinking_block_count(), 1)
            first = harness.snapshot()
            self.assertEqual(first, harness.snapshot())
            harness.cleanup()
            self.assertFalse(os.path.exists(path))

    def test_cleanup_leaves_everything_else_alone(self):
        with tempfile.TemporaryDirectory() as config:
            folder = os.path.join(config, "projects", "someone-elses-project")
            os.makedirs(folder)
            other = os.path.join(folder, "55555555-5555-5555-5555-555555555555.jsonl")
            open(other, "w").close()
            harness = self._harness(config_dir=config)
            harness.sessions.append("66666666-6666-6666-6666-666666666666")
            harness.cleanup()
            self.assertTrue(os.path.exists(other))

    def test_a_whole_context_becomes_a_valid_chain(self):
        rows = session_rows(demo_context(), "77777777-7777-7777-7777-777777777777", "cwd", "1.0", "m")
        self.assertIsNone(rows[0]["parentUuid"])
        for before, after in zip(rows, rows[1:]):
            self.assertEqual(after["parentUuid"], before["uuid"])
        self.assertEqual([r["type"] for r in rows], ["user", "assistant", "user", "assistant"])
        self.assertEqual(rows[1]["message"]["content"][0]["type"], "tool_use")
        self.assertEqual(rows[2]["message"]["content"][0]["type"], "tool_result")

    def test_apply_writes_where_a_resume_will_find_it(self):
        with tempfile.TemporaryDirectory() as config:
            harness = ClaudeCliHarness(runner=FakeRunner(), config_dir=config, environ={"EUNOE_OFFLINE": "1"})
            ref = harness.apply(None, demo_context())
            self.assertEqual(ref.id, harness.current)
            self.assertTrue(harness.session_path())
            harness.cleanup()
            self.assertIsNone(harness.session_path())


def codex_output(text="OK", thread="t-1", usage=None):
    rows = [{"type": "thread.started", "thread_id": thread},
            {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
            {"type": "turn.completed", "usage": usage or {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 5}}]
    return CliResult(0, "\n".join(json.dumps(r) for r in rows), "")


class CodexTest(unittest.TestCase):
    def test_usage_is_a_difference_because_it_is_cumulative(self):
        harness = CodexHarness(runner=FakeRunner(
            codex_output(), codex_output(usage={"input_tokens": 250, "cached_input_tokens": 140, "output_tokens": 9})),
            environ={"PATH": "x"})
        first = harness.start("a")
        self.assertEqual((first.usage.input, first.usage.cache_read, first.usage.output), (60, 40, 5))
        second = harness.resume("b")
        self.assertEqual((second.usage.input, second.usage.cache_read, second.usage.output), (50, 100, 4))

    def test_it_reads_only_and_ignores_user_config(self):
        harness = CodexHarness(runner=FakeRunner(codex_output()), environ={"PATH": "x", "OPENAI_API_KEY": "s"})
        harness.start("a")
        call = harness.runner.calls[0]
        self.assertIn("--ignore-user-config", call["argv"])
        self.assertEqual(call["argv"][call["argv"].index("--sandbox") + 1], "read-only")
        self.assertEqual(call["argv"][-1], "-")
        self.assertNotIn("OPENAI_API_KEY", call["env"])

    def test_a_failed_turn_is_an_error(self):
        harness = CodexHarness(runner=FakeRunner(CliResult(1, json.dumps({"type": "turn.failed", "message": "no"}), "")),
                               environ={"PATH": "x"})
        self.assertIn("no", harness.start("a").error)

    def test_a_fork_copies_the_rollout_and_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as home:
            folder = os.path.join(home, "sessions", "2031", "01", "01")
            os.makedirs(folder)
            thread = "88888888-8888-8888-8888-888888888888"
            path = os.path.join(folder, f"rollout-x-{thread}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "session_meta", "payload": {"id": thread}}) + "\n")
                fh.write(json.dumps({"type": "response_item", "payload": {"type": "reasoning"}}) + "\n")
            harness = CodexHarness(runner=FakeRunner(codex_output("x", thread)), environ={"PATH": "x", "CODEX_HOME": home})
            harness.thread = thread
            before = harness.snapshot()
            harness.fork("q")
            self.assertEqual(harness.snapshot(), before)
            self.assertEqual(harness.thread, thread)
            self.assertEqual(harness.thinking_block_count(), 1)
            copied = [f for f in os.listdir(folder) if thread not in f]
            self.assertEqual(len(copied), 1)
            self.assertNotIn(thread, open(os.path.join(folder, copied[0]), encoding="utf-8").read())
            harness.threads.append(thread)
            harness.cleanup()
            self.assertEqual(os.listdir(folder), [])

    def test_a_whole_context_keeps_the_opening_rows_and_numbers_its_own(self):
        rows = _rollout_rows(demo_context(), 7)
        self.assertEqual([r["ordinal"] for r in rows], list(range(7, 7 + len(rows))))
        self.assertEqual({r["type"] for r in rows}, {"response_item"})


def agy_output(text="OK", conv="c-1", **usage):
    return CliResult(0, json.dumps({"conversation_id": conv, "status": "SUCCESS", "response": text,
                                    "usage": {"input_tokens": 100, "output_tokens": 2, "thinking_tokens": 0,
                                              "cache_read_tokens": 0, **usage}}), "")


class AgyTest(unittest.TestCase):
    def _store(self, folder, conv, payload: bytes):
        os.makedirs(os.path.join(folder, "conversations"), exist_ok=True)
        path = os.path.join(folder, "conversations", f"{conv}.db")
        with contextlib.closing(sqlite3.connect(path)) as db:
            db.execute("create table steps (idx integer primary key, step_payload blob)")
            db.execute("create table gen_metadata (idx integer primary key, data blob)")
            db.execute("create table trajectory_meta (trajectory_id text, cascade_id text)")
            db.execute("insert into steps values (0, ?)", (payload,))
            db.execute("insert into gen_metadata values (0, ?)", (payload,))
            db.execute("insert into trajectory_meta values ('t', ?)", (conv,))
            db.commit()
        return path

    def test_usage_is_a_difference(self):
        harness = AgyHarness(runner=FakeRunner(agy_output(), agy_output(input_tokens=250, output_tokens=5)), environ={"PATH": "x"})
        self.assertEqual(harness.start("a").usage.input, 100)
        self.assertEqual(harness.resume("b").usage.input, 150)

    def test_plan_mode_and_conversation_flag(self):
        harness = AgyHarness(runner=FakeRunner(agy_output(), agy_output()), environ={"PATH": "x", "GEMINI_API_KEY": "s"})
        harness.start("a")
        harness.resume("b")
        first, second = harness.runner.calls
        self.assertEqual(first["argv"][first["argv"].index("--mode") + 1], "plan")
        self.assertEqual(second["argv"][second["argv"].index("--conversation") + 1], "c-1")
        self.assertNotIn("GEMINI_API_KEY", first["env"])

    def test_thinking_uses_a_tier_not_effort(self):
        harness = AgyHarness(runner=FakeRunner(agy_output()), environ={"PATH": "x"})
        harness.start("a", thinking=True)
        argv = harness.runner.calls[0]["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "gemini-3.8-flash-high")
        self.assertNotIn("--effort", argv)

    def test_errors_and_denied_tools_are_errors(self):
        bad = CliResult(1, json.dumps({"conversation_id": "", "status": "ERROR", "error": "conflict"}), "")
        self.assertIn("conflict", AgyHarness(runner=FakeRunner(bad), environ={"PATH": "x"}).start("a").error)
        denied = CliResult(0, json.dumps({"conversation_id": "c", "status": "SUCCESS", "response": "",
                                          "denied_actions": [{"action": "command"}], "usage": {}}), "")
        self.assertIn("denied", AgyHarness(runner=FakeRunner(denied), environ={"PATH": "x"}).start("a").error)

    def test_an_over_long_prompt_refuses_before_spawning(self):
        harness = AgyHarness(runner=FakeRunner(), environ={"PATH": "x"})
        self.assertIn("limit", harness.start("x" * 40_000).error)

    def test_edits_hold_the_length_and_a_longer_one_refuses(self):
        with tempfile.TemporaryDirectory() as home:
            conv = "conv-a"
            payload = b"\x12\x20prefix " + CODEWORD.encode() + b" and padding here\x00"
            path = self._store(home, conv, payload)
            harness = AgyHarness(runner=FakeRunner(), environ={"PATH": "x"})
            harness.home, harness.conversation = home, conv
            self.assertEqual(harness.replace(CODEWORD, NEW_CODEWORD), 2)
            self.assertEqual(harness.replace("padding here", ""), 2)
            with contextlib.closing(sqlite3.connect(path)) as db:
                stored = bytes(db.execute("select step_payload from steps").fetchone()[0])
            self.assertEqual(len(stored), len(payload))
            self.assertIn(NEW_CODEWORD.encode(), stored)
            self.assertNotIn(b"padding", stored)
            self.assertTrue(stored.startswith(b"\x12\x20prefix "))
            with self.assertRaises(Unsupported):
                harness.replace(NEW_CODEWORD, NEW_CODEWORD + "-longer")

    def test_a_fork_copies_the_store_under_a_new_id(self):
        with tempfile.TemporaryDirectory() as home:
            conv = "conv-b"
            self._store(home, conv, b"data")
            harness = AgyHarness(runner=FakeRunner(agy_output("x", "conv-new")), environ={"PATH": "x"})
            harness.home, harness.conversation = home, conv
            before = harness.snapshot()
            harness.fork("q")
            self.assertEqual(harness.conversation, conv)
            self.assertEqual(harness.snapshot(), before)
            new = [f for f in os.listdir(os.path.join(home, "conversations")) if not f.startswith(conv)]
            self.assertEqual(len(new), 1)
            with contextlib.closing(sqlite3.connect(os.path.join(home, "conversations", new[0]))) as db:
                self.assertEqual(db.execute("select cascade_id from trajectory_meta").fetchone()[0], new[0][:-3])


class FakeTransport:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, body, timeout=0):
        self.calls.append(dict(url=url, headers=headers, body=json.loads(json.dumps(body))))
        return self.responses.pop(0)


def claude_response(text="OK", **usage):
    return 200, {"id": "m", "content": [{"type": "text", "text": text}],
                 "usage": {"input_tokens": 3, "output_tokens": 1, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0, **usage}}


class StatelessTest(unittest.TestCase):
    def _claude(self, *responses, **kw):
        return ClaudeApiHarness(gate=SpendGate(allow=True), transport=FakeTransport(*responses),
                                environ={"EUNOE_ANTHROPIC_API_KEY": "k"}, **kw)

    def test_the_request_asks_for_the_cache_and_counts_the_call(self):
        harness = self._claude(claude_response(), ttl="1h")
        harness.start("hello")
        call = harness.transport.calls[0]
        self.assertEqual(call["body"]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(call["headers"]["x-api-key"], "k")
        self.assertNotIn("thinking", call["body"])
        self.assertEqual(len(harness.gate.rows), 1)
        self.assertGreater(harness.gate.spent_usd, 0)

    def test_thinking_is_adaptive_and_blocks_go_back_unchanged(self):
        thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
        harness = self._claude((200, {"id": "m", "content": [thinking, {"type": "text", "text": "ok"}], "usage": {}}),
                               claude_response())
        harness.start("a", thinking=True)
        self.assertEqual(harness.transport.calls[0]["body"]["thinking"], {"type": "adaptive"})
        harness.resume("b")
        self.assertEqual(harness.transport.calls[1]["body"]["messages"][1]["content"][0], thinking)
        self.assertEqual(harness.thinking_block_count(), 1)

    def test_an_edit_changes_what_is_sent_next(self):
        harness = self._claude(claude_response(), claude_response())
        harness.start(f"codeword {CODEWORD}")
        self.assertEqual(harness.replace(CODEWORD, NEW_CODEWORD), 1)
        harness.resume("again")
        self.assertIn(NEW_CODEWORD, json.dumps(harness.transport.calls[1]["body"]["messages"]))

    def test_a_400_is_a_turn_error_and_leaves_the_history_clean(self):
        harness = self._claude(claude_response(), (400, {"error": {"message": "bound to a different conversation"}}))
        harness.start("a")
        turn = harness.resume("b")
        self.assertIn("bound", turn.error)
        self.assertEqual(len(harness.messages), 2)

    def test_the_gate_stops_it_before_the_request(self):
        harness = ClaudeApiHarness(gate=SpendGate(), transport=FakeTransport(), environ={"EUNOE_ANTHROPIC_API_KEY": "k"})
        with self.assertRaises(SpendRefused):
            harness.start("a")
        self.assertEqual(harness.transport.calls, [])

    def test_the_whole_suite_runs_against_a_recorded_api(self):
        class Model:
            def __init__(self):
                self.calls = []

            def __call__(self, url, headers, body, timeout=0):
                self.calls.append(body)
                text = json.dumps(body["messages"])
                last = body["messages"][-1]["content"][-1]["text"]
                if "only the word OK" in last:
                    return claude_response("OK", input_tokens=4000)
                if "READY" in last:
                    return claude_response("READY", input_tokens=10, cache_read_input_tokens=4000)
                if "rollback token" in last:
                    return claude_response("RB-77X GREEN-12", input_tokens=40)
                if "codeword" in last:
                    for word in (NEW_CODEWORD, CODEWORD, "PLOVER-3306"):
                        if word in text:
                            return claude_response(word, input_tokens=len(text) // 4)
                return claude_response("7", input_tokens=len(text) // 4)

        harness = ClaudeApiHarness(gate=SpendGate(allow=True), transport=Model(), environ={"EUNOE_ANTHROPIC_API_KEY": "k"})
        report = Suite(harness, cost=ListPrice("claude-haiku-5-5")).run()
        self.assertTrue(report.conforms, format_report(report))
        self.assertGreater(report.spent_usd, 0)

    def test_openai_is_sent_whole_with_nothing_stored(self):
        response = (200, {"id": "r", "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
                          "usage": {"input_tokens": 10, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 4}}})
        harness = OpenAiResponsesHarness(gate=SpendGate(allow=True), transport=FakeTransport(response, response),
                                         environ={"EUNOE_OPENAI_API_KEY": "k"})
        turn = harness.start("a")
        self.assertEqual((turn.usage.input, turn.usage.cache_read), (6, 4))
        harness.resume("b")
        body = harness.transport.calls[1]["body"]
        self.assertNotIn("previous_response_id", body)
        self.assertFalse(body["store"])
        self.assertEqual(len(body["input"]), 3)

    def test_gemini_is_stateless(self):
        response = (200, {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}],
                          "usageMetadata": {"promptTokenCount": 10, "cachedContentTokenCount": 4, "candidatesTokenCount": 1}})
        harness = GeminiApiHarness(gate=SpendGate(allow=True), transport=FakeTransport(response, response),
                                   environ={"EUNOE_GEMINI_API_KEY": "k"})
        turn = harness.start("a")
        self.assertEqual((turn.usage.input, turn.usage.cache_read), (6, 4))
        harness.resume("b")
        self.assertEqual(len(harness.transport.calls[1]["body"]["contents"]), 3)
        self.assertEqual(harness.transport.calls[1]["headers"]["x-goog-api-key"], "k")

    def test_ollama_spends_nothing_and_honours_its_host(self):
        response = (200, {"message": {"content": "OK"}, "prompt_eval_count": 9, "eval_count": 1})
        harness = OllamaHarness(transport=FakeTransport(response), environ={"OLLAMA_HOST": "127.0.0.1:1234"})
        self.assertEqual(harness.url, "http://127.0.0.1:1234/api/chat")
        self.assertFalse(harness.spends)
        self.assertEqual(harness.start("a").usage.input, 9)
        self.assertEqual(harness.gate.spent_usd, 0.0)
        self.assertEqual(OllamaHarness(model="m", environ={"EUNOE_OLLAMA_MODEL": "n"}).model, "m")
        self.assertEqual(OllamaHarness(environ={"EUNOE_OLLAMA_MODEL": "n"}).model, "n")


class ServerSideTest(unittest.TestCase):
    def test_the_edits_and_betas_are_what_the_provider_documents(self):
        padding = make_padding(30)
        transport = FakeTransport(
            claude_response("READY", input_tokens=500), claude_response("UNKNOWN", input_tokens=50),
            (200, {"content": [{"type": "compaction", "content": "x"}, {"type": "text", "text": "UNKNOWN"}],
                   "usage": {"input_tokens": 40, "output_tokens": 1}}))
        harness = ClaudeServerHarness(gate=SpendGate(allow=True), transport=transport,
                                      environ={"EUNOE_ANTHROPIC_API_KEY": "k"})
        report = harness.run_suite(padding)
        self.assertEqual({s.status for s in report.steps}, {"PASS"}, [(s.step, s.note) for s in report.steps])
        clear, compact = transport.calls[1], transport.calls[2]
        self.assertEqual(clear["headers"]["anthropic-beta"], "context-management-2025-06-27")
        self.assertEqual(clear["body"]["context_management"]["edits"][0]["type"], "clear_tool_uses_20250919")
        self.assertEqual(compact["headers"]["anthropic-beta"], "compact-2026-01-12")
        self.assertEqual(compact["body"]["context_management"]["edits"][0]["type"], "compact_20260112")
        self.assertGreater(report.spent_usd, 0)

    def test_it_needs_a_key(self):
        report = ClaudeServerHarness(environ={}).run_suite("x")
        self.assertFalse(report.available)


class LiveDriverTest(unittest.TestCase):
    def test_api_driver_shapes_the_request_and_records_spend(self):
        transport = FakeTransport(claude_response())
        gate = SpendGate(allow=True)
        driver = ClaudeApiDriver("claude-haiku-5-5", gate, transport, {"EUNOE_ANTHROPIC_API_KEY": "k"})
        ctx = RenderedContext("sys", [Block("user", "text", text="q")])
        turn = driver.call(ctx)
        self.assertEqual(turn.text, "OK")
        self.assertEqual(transport.calls[0]["body"]["system"], "sys")
        self.assertEqual(len(gate.rows), 1)

    def test_api_driver_passes_server_edits_and_thinking_controls(self):
        driver = ClaudeApiDriver("claude-haiku-5-5", SpendGate(allow=True), FakeTransport(),
                                 {"EUNOE_ANTHROPIC_API_KEY": "k"})
        ctx = RenderedContext("", [Block("user", "text", text="q")], [],
                              {"context_management": {"edits": [{"type": "compact_20260112"}]}, "thinking_policy": "drop_block"})
        headers, body = driver.request(ctx)
        self.assertIn("compact-2026-01-12", headers["anthropic-beta"])
        self.assertIn("thinking-binding-controls-2026-08-01", headers["anthropic-beta"])
        self.assertEqual(body["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block")

    def test_api_driver_places_a_second_breakpoint(self):
        driver = ClaudeApiDriver("claude-haiku-5-5", SpendGate(allow=True), FakeTransport(),
                                 {"EUNOE_ANTHROPIC_API_KEY": "k"}, ttl="1h")
        ctx = RenderedContext("", [Block("user", "text", text="a"), Block("assistant", "text", text="b"),
                                   Block("user", "text", text="c")], [], {"breakpoints": [0]})
        _, body = driver.request(ctx)
        self.assertEqual(body["messages"][0]["content"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertNotIn("cache_control", body["messages"][2]["content"][0])
        self.assertEqual(body["cache_control"], {"type": "ephemeral", "ttl": "1h"})

    def test_api_driver_refuses_without_permission(self):
        driver = ClaudeApiDriver("claude-haiku-5-5", SpendGate(), FakeTransport(), {"EUNOE_ANTHROPIC_API_KEY": "k"})
        with self.assertRaises(SpendRefused):
            driver.call(RenderedContext("", [Block("user", "text", text="q")]))

    def test_text_tools_parse_and_tolerate_fences(self):
        turn = _from_text('`TOOL recall {"query": "X-1"}`', CallUsage(), 0.0)
        self.assertEqual((turn.tool_calls[0].name, turn.tool_calls[0].args), ("recall", {"query": "X-1"}))
        self.assertEqual(_from_text("ANSWER: 5", CallUsage(), 0.0).tool_calls, [])
        self.assertEqual(_from_text("TOOL recall not-json", CallUsage(), 0.0).tool_calls, [])
        self.assertEqual(_from_text('TOOL apply ["not", "an", "object"]', CallUsage(), 0.0).tool_calls, [])

    def _cli_driver(self, config, *script):
        driver = ClaudeCliDriver("claude-haiku-5-5", environ={"PATH": "x"})
        driver.harness.config_dir = config
        driver.harness.runner = FakeRunner(*script)
        driver.harness._version = "2.1.309"
        return driver

    def test_cli_driver_appends_and_never_rewrites_until_an_edit(self):
        with tempfile.TemporaryDirectory() as config:
            driver = self._cli_driver(config, claude_result("OK"), claude_result("OK"), claude_result("OK"))
            one = Block("user", "text", text="first", event_id="e1")
            two = Block("assistant", "text", text="second", event_id="e2")
            prompt = Block("user", "text", text="go", event_id="prompt")
            driver.call(RenderedContext("", [one, prompt]))
            self.assertEqual(driver.rewrites, 1)
            driver.call(RenderedContext("", [one, Block("user", "text", text="go", event_id="e3"),
                                             Block("assistant", "text", text="OK", event_id="e4"), two, prompt]))
            self.assertEqual(driver.rewrites, 1, "unchanged history must append, not rewrite")
            path = driver.harness.session_path()
            text = open(path, encoding="utf-8").read()
            self.assertIn("second", text)
            driver.call(RenderedContext("", [Block("user", "text", text="EDITED", event_id="e1"), prompt]))
            self.assertEqual(driver.rewrites, 2)
            driver.close()

    def test_cli_driver_forks_a_probe_so_it_teaches_nothing(self):
        with tempfile.TemporaryDirectory() as config:
            forked = claude_result("ANSWER: x", session="99999999-9999-9999-9999-999999999999")
            driver = self._cli_driver(config, forked)
            prompt = Block("user", "text", text="What is it?", event_id="prompt")
            turn = driver.call(RenderedContext("", [Block("user", "text", text="first", event_id="e1"), prompt], [],
                                               {"mode": "probe"}))
            self.assertEqual(turn.text, "ANSWER: x")
            self.assertIn("--fork-session", driver.harness.runner.calls[0]["argv"])
            self.assertEqual(len(driver.written), 1)
            driver.close()

    def test_cli_driver_carries_a_lookup_in_the_prompt(self):
        with tempfile.TemporaryDirectory() as config:
            forked = claude_result("ANSWER: x", session="99999999-9999-9999-9999-999999999998")
            driver = self._cli_driver(config, forked)
            blocks = [Block("user", "text", text="first", event_id="e1"),
                      Block("assistant", "tool_use", name="recall", input={"query": "q"}, tool_id="l"),
                      Block("user", "tool_result", text="the found text", tool_id="l"),
                      Block("user", "text", text="What is it?", event_id="prompt")]
            driver.call(RenderedContext("", blocks, [], {"mode": "probe", "looked_up": True}))
            self.assertIn("the found text", driver.harness.runner.calls[0]["input"])
            self.assertEqual(len(driver.written), 1)
            driver.close()

    def test_only_named_drivers_exist(self):
        self.assertIsInstance(make_driver("claude-cli", model="m", gate=SpendGate(), ttl="1h"), ClaudeCliDriver)
        with self.assertRaises(ValueError):
            make_driver("nope", model="m", gate=SpendGate(), ttl="1h")


class AgentSdkTest(unittest.TestCase):
    def test_absent_sdk_is_unavailable_not_broken(self):
        from eunoe.editors.agent_sdk import AgentSdkHarness

        harness = AgentSdkHarness(environ={"PATH": "x"})
        with mock.patch.object(AgentSdkHarness, "_sdk", return_value=None):
            ok, reason = harness.available()
        self.assertFalse(ok)
        self.assertIn("agent-sdk", reason)


if __name__ == "__main__":
    unittest.main()
