"""What keeps this repo publishable, and the replay's promise to keep no text.

Fixtures are generated, results stay out of the tree, and nothing in it holds a local path or a personal
mailbox. These are shape checks, so they run with no list of anything private.
"""
import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import unittest

from eunoe.cli import main
from eunoe.replay.surface import rates, real_cost, replay_cost, schedule_grid
from eunoe.replay.usage import read_usage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]+Users[\\/]+|(?<![\w.])/(?:home|Users)/)(?!(?:alice|you|your|runner)\b)[A-Za-z0-9._-]+")
_MAILBOX = re.compile(r"[A-Za-z0-9._%+-]+@(?:gmail|googlemail|proton|protonmail|pm|outlook|hotmail|yahoo|icloud)\.[a-z.]+")
_RAW_RESULTS = (".jsonl", ".sqlite", ".db")


def tracked_files() -> list:
    try:
        done = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except OSError:
        return []
    return done.stdout.splitlines() if done.returncode == 0 else []


class TreeTest(unittest.TestCase):
    def test_no_local_path_or_personal_mailbox_is_tracked(self):
        hits = []
        for name in tracked_files():
            try:
                text = open(os.path.join(ROOT, name), encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in (_LOCAL_PATH, _MAILBOX):
                hits += [f"{name}: {m.group(0)[:12]}…" for m in pattern.finditer(text)]
        self.assertEqual(hits, [])

    def test_no_raw_result_file_is_tracked(self):
        self.assertEqual([n for n in tracked_files() if n.endswith(_RAW_RESULTS)], [])

    def test_the_kit_imports_nothing_outside_the_standard_library(self):
        import sys

        allowed = set(sys.stdlib_module_names) | {"eunoe"}
        optional = {os.path.join("arms", "lethe.py"), os.path.join("arms", "hermes_lcm_worker.py"),
                    os.path.join("editors", "agent_sdk.py")}
        source = os.path.join(ROOT, "src", "eunoe")
        for folder, _, files in os.walk(source):
            for name in files:
                rel = os.path.relpath(os.path.join(folder, name), source)
                if not name.endswith(".py") or rel in optional:
                    continue
                text = open(os.path.join(folder, name), encoding="utf-8").read()
                for module in re.findall(r"^(?:from|import) ([A-Za-z_0-9]+)", text, re.M):
                    self.assertIn(module, allowed, f"{rel} imports {module}")


class ReplayTest(unittest.TestCase):
    MARKER = "PLANTED-SECRET-TEXT-7731"

    def _session(self, folder: str) -> str:
        path = os.path.join(folder, "s.jsonl")
        rows = []
        for i in range(30):
            usage = {"input_tokens": 2, "cache_read_input_tokens": 1000 * i, "cache_creation_input_tokens": 1000,
                     "output_tokens": 50, "cache_creation": {"ephemeral_1h_input_tokens": 1000}}
            ts = f"2031-01-01T00:{i:02d}:00.000Z"
            rows.append({"type": "user", "timestamp": ts, "message": {"content": self.MARKER}})
            # One row per content block, each carrying the same usage: what the CLI really writes.
            for block in range(2):
                rows.append({"type": "assistant", "timestamp": ts, "isSidechain": False, "message": {
                    "id": f"m{i}", "model": "x", "usage": usage, "content": [{"type": "text", "text": self.MARKER}]}})
        rows.append({"type": "system", "subtype": "compact_boundary", "content": self.MARKER})
        rows.append({"type": "assistant", "isSidechain": True, "timestamp": "2031-01-01T01:00:00Z",
                     "message": {"id": "side", "usage": {"input_tokens": 999}}})
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        return path

    def test_usage_dedupes_by_message_id_and_skips_sidechains(self):
        with tempfile.TemporaryDirectory() as folder:
            usage = read_usage(self._session(folder))
        self.assertEqual(usage["rows"], 60)
        self.assertEqual(len(usage["calls"]), 30)
        self.assertEqual(usage["boundaries"], 1)
        self.assertNotIn("side", {c.id for c in usage["calls"]})

    def test_no_text_reaches_the_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._session(folder)
            for flags in ([], ["--json"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(main(["replay", path, "--budget", "5000", "--static-tokens", "100",
                                           "--post-compact-tokens", "1000", *flags]), 0)
                self.assertNotIn(self.MARKER, out.getvalue())
                self.assertNotIn(folder, out.getvalue())
                self.assertIn("batch 1.5x to 1x", out.getvalue())

    def test_the_real_shape_reproduces_the_real_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = read_usage(self._session(folder))["calls"]
        price = rates("claude-opus-5-5")
        real = real_cost(calls, price)
        none = replay_cost(calls, schedule_grid(10**9, 1000)["no management"], price, static=100)["cost"]
        self.assertAlmostEqual(none / real, 1.0, delta=0.15)

    def test_editing_every_call_costs_more_than_a_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = read_usage(self._session(folder))["calls"]
        price = rates("claude-opus-5-5")
        grid = schedule_grid(5000, 1000)
        every = replay_cost(calls, grid["every call"], price, static=100)["cost"]
        batch = replay_cost(calls, grid["batch 1.5x to 1x"], price, static=100)["cost"]
        self.assertGreater(every, batch)
        free = replay_cost(calls, grid["free edits (no rewrite cost)"], price, static=100)["cost"]
        self.assertLess(free, every)


class CliTest(unittest.TestCase):
    def _run(self, *argv) -> tuple:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_the_demo_runs_and_says_what_it_is(self):
        code, text = self._run("demo", "--quick")
        self.assertEqual(code, 0)
        self.assertIn("not a model", text)
        self.assertIn("T2", text)

    def test_conform_on_the_offline_harness_writes_a_clean_report(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text = self._run("conform", "--harness", "scripted", "--out", folder)
            self.assertEqual(code, 0)
            self.assertIn("conforms: yes", text)
            report = open(os.path.join(folder, "conform-scripted.json"), encoding="utf-8").read()
        self.assertTrue(json.loads(report)["conforms"])
        self.assertNotIn(folder, report)

    def test_a_harness_it_cannot_reach_exits_2_and_says_why(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text = self._run("conform", "--harness", "claude-cli", "--out", folder)
        self.assertEqual(code, 2)
        self.assertIn("UNAVAILABLE", text)

    def test_a_paid_harness_never_runs_unasked(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text = self._run("conform", "--harness", "claude-api", "--out", folder)
        self.assertEqual(code, 2)
        self.assertIn("UNAVAILABLE", text)

    def test_lists_and_generate(self):
        self.assertIn("claude-cli", self._run("conform", "--list")[1])
        self.assertIn("T2", self._run("arms")[1])
        code, text = self._run("gen", "--domain", "research", "--seed", "4", "--profile", "tiny")
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(text)["fingerprint"]), 64)

    def test_run_from_a_config_writes_a_manifest_outside_the_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            config = os.path.join(folder, "c.toml")
            with open(config, "w", encoding="utf-8") as fh:
                fh.write('arms = ["C0", "T2"]\nprofile = "tiny"\nbudget = 2000\nseeds = 2\ndomains = ["oncall"]\n')
            code, text = self._run("run", "--config", config, "--out", folder)
            self.assertEqual(code, 0)
            self.assertIn("recall, T2 - C0", text)
            manifests = [f for f in os.listdir(folder) if f.startswith("run-")]
            self.assertEqual(len(manifests), 1)
            data = json.load(open(os.path.join(folder, manifests[0]), encoding="utf-8"))
            calls = open(os.path.join(folder, "calls.jsonl"), encoding="utf-8").read().splitlines()
        self.assertEqual(data["seeds"], [1, 2])
        self.assertTrue(calls)
        self.assertEqual(set(json.loads(calls[0])), {"arm", "session", "input", "cache_read", "cache_write", "output", "list_usd"})
        self.assertEqual(data["api_usd_spent"], 0.0)
        self.assertEqual(len(data["fixtures"]), 2)

    def test_a_real_model_run_waits_for_yes(self):
        with tempfile.TemporaryDirectory() as folder:
            config = os.path.join(folder, "c.toml")
            with open(config, "w", encoding="utf-8") as fh:
                fh.write('profile = "tiny"\nbudget = 2000\ndriver = "claude-api"\ndomains = ["oncall"]\n')
            code, text = self._run("run", "--config", config, "--out", folder)
        self.assertEqual(code, 2)
        self.assertIn("--yes", text)

    def test_a_paid_run_refuses_without_allow_spend(self):
        with tempfile.TemporaryDirectory() as folder:
            config = os.path.join(folder, "c.toml")
            with open(config, "w", encoding="utf-8") as fh:
                fh.write('profile = "tiny"\nbudget = 2000\ndriver = "claude-api"\ndomains = ["oncall"]\n')
            code, text = self._run("run", "--config", config, "--out", folder, "--yes")
        self.assertEqual(code, 3)
        self.assertIn("--allow-spend", text)


if __name__ == "__main__":
    unittest.main()
