"""Fork ablation: invented sessions, the scripted door, no model and nothing leaving the machine."""
import argparse
import json
import os
import tempfile
import unittest

from eunoe.cost import ListPrice
from eunoe.editors.base import Turn
from eunoe.forks import analysis, codex_check, conditions as C, demand, experiment, thinking_check
from eunoe.forks.generated import TASK, build
from eunoe.forks.probes import Item, batch_prompt, clean_batch, cloze, gold_forms, parse_batch, score_answer, strong_details
from eunoe.forks.run import PACE, Replay, Sink, archive_search, cell_tokens, rate_events, run_condition
from eunoe.forks.scripted import ScriptedDoor
from eunoe.forks.transcript import blocks_of, held_text, load_checkpoint, read_rows, render_rows, turn_boundaries, write_session
from eunoe.ledger import FILE, SYSTEM_NOTE, THINKING, USER
from eunoe.levels import Level
from eunoe.signals import AblationEstimate


def usage(context):
    return {"input_tokens": 1, "cache_read_input_tokens": context, "cache_creation_input_tokens": 0, "output_tokens": 5}


def row(kind, uid, parent, content=None, **extra):
    out = {"type": kind, "uuid": uid, "parentUuid": parent, "isSidechain": False, "sessionId": "old", "cwd": "/old",
           "timestamp": "2030-01-01T00:00:00Z"}
    if kind in ("user", "assistant"):
        out["message"] = {"role": kind, "content": content, **({"id": extra.pop("mid"), "usage": usage(extra.pop("ctx", 1000))} if kind == "assistant" else {})}
    out.update(extra)
    return out


def invented_session() -> list:
    return [
        row("attachment", "a0", None, attachment={"type": "instructions", "content": "floor words"}, rendered=[{"type": "text", "text": "floor words PLANTED-FLOOR-77"}]),
        row("user", "u1", "a0", [{"type": "text", "text": "Please look at the fleet. PLANTED-USER-11"}]),
        row("assistant", "t1", "u1", [{"type": "thinking", "thinking": "", "signature": "sig1"}], mid="m1", ctx=1000),
        row("assistant", "c1", "t1", [{"type": "tool_use", "id": "tool1", "name": "Bash", "input": {"command": "cat runbooks/amber-101.md"}}], mid="m1", ctx=1000),
        row("user", "r1", "c1", [{"type": "tool_result", "tool_use_id": "tool1", "content": "line one\n" * 400 + "the slug is deep-buried-value-slug here\n" + "line two\n" * 400}], toolUseResult={"copy": "COPY-OF-RESULT"}),
        row("assistant", "x1", "r1", [{"type": "text", "text": "Done. It went fine. The file is runbooks/amber-101.md."}], mid="m2", ctx=2000),
        row("attachment", "a1", "x1", attachment={"type": "nested_memory", "path": "docs/memory.md", "content": "memory body"}, rendered=[{"type": "text", "text": "memory body NESTED-22"}]),
        row("api-request", "q1", None),
        row("user", "u2", "a1", [{"type": "text", "text": "Next please."}]),
        row("assistant", "t2", "u2", [{"type": "thinking", "thinking": "", "signature": "sig2"}], mid="m3", ctx=3000),
        row("assistant", "x2", "t2", [{"type": "text", "text": "I will edit the deep-buried-value-slug entry today in the amber queue."}], mid="m3", ctx=3000),
    ]


class Transcript(unittest.TestCase):
    def setUp(self):
        self.rows = invented_session()
        self.cp = load_checkpoint(self.rows, 8, "cp")
        self.led = self.cp.ledger

    def by_kind(self, kind):
        return [e for e in self.led.events if e.kind == kind]

    def test_events_and_calls(self):
        self.assertEqual(len(self.by_kind(THINKING)), 1)
        tool = self.by_kind("tool:Bash")[0]
        self.assertIn("deep-buried-value-slug", tool.text)
        self.assertEqual(tool.input["command"], "cat runbooks/amber-101.md")
        self.assertEqual(self.led.call, 2)
        self.assertEqual(self.cp.label(self.by_kind(FILE)[1]), "attachment:nested_memory")

    def test_floor_is_what_came_before_the_first_call(self):
        self.assertEqual(self.cp.frozen_upto, 1)

    def test_classifier_splits_a_user_channel(self):
        cp = load_checkpoint(self.rows, 8, "cp", classify=lambda text, row: USER if "PLANTED-USER" in text else SYSTEM_NOTE)
        self.assertEqual({e.kind for e in cp.ledger.events if e.text.startswith("Please")}, {USER})

    def test_unedited_keeps_every_message_and_chains(self):
        rows = render_rows(self.cp, {}, False, "new", "/here")
        self.assertEqual({r["type"] for r in rows}, {"user", "assistant", "attachment"})
        self.assertEqual({r["sessionId"] for r in rows}, {"new"})
        self.assertEqual({r["cwd"] for r in rows}, {"/here"})
        self.assertTrue(all("toolUseResult" not in r for r in rows))
        self.assertNotIn("copy-of-result", held_text(rows))

    def test_chain_holds_after_a_removal(self):
        rows = render_rows(self.cp, {}, True, "new", "/here")
        ids = {r["uuid"] for r in rows}
        self.assertFalse(any(b.get("type") == "thinking" for r in rows for b in blocks_of(r)))
        for r in rows:
            self.assertTrue(r["parentUuid"] is None or r["parentUuid"] in ids, r["uuid"])

    def test_thinking_alone_on_a_message_does_not_stand(self):
        tool = self.by_kind("tool:Bash")[0]
        rows = render_rows(self.cp, {tool.id: Level.L4}, False, "new", "/here")
        self.assertFalse(any((r.get("message") or {}).get("id") == "m1" for r in rows))

    def test_levels_reach_every_copy(self):
        tool = self.by_kind("tool:Bash")[0]
        for level, kept in ((Level.L1, False), (Level.L3, False)):
            rows = render_rows(self.cp, {tool.id: level}, True, "new", "/here")
            self.assertEqual("line two\nline two\n" * 5 in held_text(rows) and level == Level.L0, kept, level)

    def test_a_tool_pair_stays_valid_at_every_level(self):
        tool = self.by_kind("tool:Bash")[0]
        for level in (Level.L1, Level.L2, Level.L3, Level.L4):
            rows = render_rows(self.cp, {tool.id: level}, True, "new", "/here")
            uses = {b["id"] for r in rows for b in blocks_of(r) if b.get("type") == "tool_use"}
            results = {b["tool_use_id"] for r in rows for b in blocks_of(r) if b.get("type") == "tool_result"}
            self.assertEqual(uses, results, level)

    def test_an_attachment_edit_replaces_its_copies(self):
        event = self.by_kind(FILE)[1]
        rows = render_rows(self.cp, {event.id: Level.L3}, True, "new", "/here")
        kept = next(r for r in rows if r.get("uuid") == "a1")
        self.assertNotIn("memory body", json.dumps(kept))
        self.assertIn("docs/memory.md", json.dumps(kept))
        rows = render_rows(self.cp, {event.id: Level.L4}, True, "new", "/here")
        self.assertFalse(any(r.get("uuid") == "a1" for r in rows))

    def test_nothing_past_the_cut_is_written(self):
        rows = render_rows(self.cp, {}, False, "new", "/here")
        self.assertNotIn("Next please", json.dumps(rows))

    def test_a_turn_boundary_never_splits_a_tool_call(self):
        rows = invented_session()
        marks = [i for i, _ in turn_boundaries(rows)]
        self.assertIn(8, marks)
        self.assertNotIn(4, marks)

    def test_no_boundary_past_a_compaction(self):
        rows = invented_session()
        rows.insert(6, {"type": "system", "subtype": "compact_boundary", "uuid": "b"})
        self.assertTrue(all(i < 6 for i, _ in turn_boundaries(rows)))

    def test_written_session_reads_back(self):
        with tempfile.TemporaryDirectory() as folder:
            sid, path = write_session(render_rows(self.cp, {}, False, "x", "/here"), folder, "abc")
            self.assertEqual(sid, "abc")
            self.assertEqual({r["sessionId"] for r in read_rows(path)}, {"abc"})


class LadderHolds(unittest.TestCase):
    def test_l1_holds_every_detail_l2_holds(self):
        from eunoe import levels
        from eunoe.tasks.generator import OnCall
        from eunoe.rng import Rng

        domain = OnCall(Rng("levels-test"))
        for size in (40, 200, 800):
            lines = [domain.filler_line() for _ in range(size)]
            lines.insert(size // 2, "The rollback token for INC-1234 is RB-ABCDE-12.")
            text = "\n".join(lines)
            trimmed = levels.trim(text)
            details = levels.digest(text).split("; ")
            # One can straddle the edge of the head or the tail, and be cut in two.
            self.assertLessEqual(sum(1 for d in details if d not in trimmed), 2, size)

    def test_prose_at_l1_keeps_the_sentences_l2_keeps(self):
        from eunoe import levels

        text = "Opening line.\n\nThe next step for INC-1234 is STEP-AMBER-12.\n\nNothing here.\n\nClosing line."
        self.assertIn("STEP-AMBER-12", levels.key_sentences(text))
        self.assertIn("STEP-AMBER-12", levels.first_last_paragraphs(text))
        self.assertNotIn("Nothing here", levels.first_last_paragraphs(text))

    def test_a_fact_in_the_middle_survives_l1_when_it_fits(self):
        from eunoe import levels

        text = "\n".join(["plain words with nothing in them at all"] * 300 + ["The token is RB-ABCDE-12."] + ["more plain words here"] * 300)
        self.assertIn("RB-ABCDE-12", levels.trim(text))


class Probes(unittest.TestCase):
    def test_cloze_blanks_every_copy(self):
        line = "please read runbooks/amber-101.md and then read runbooks/amber-101.md again today"
        ask = cloze(line, "runbooks/amber-101.md")
        self.assertNotIn("amber", ask)
        self.assertEqual(ask.count("_____"), 2)

    def test_cloze_needs_words_to_ask_with(self):
        self.assertEqual(cloze("cat a.md", "a.md"), "")

    def test_strong_details(self):
        found = {d for d, _, _ in strong_details("see deep-buried-value-slug and runbooks/amber-101.md and src/tests and 2030-01-01")}
        self.assertIn("deep-buried-value-slug", found)
        self.assertIn("runbooks/amber-101.md", found)
        self.assertNotIn("src/tests", found)
        self.assertNotIn("2030-01-01", found)

    def test_a_path_can_come_back_shorter(self):
        self.assertIn("amber-101.md", gold_forms("runbooks/amber-101.md"))
        item = Item("c", "x", "runbooks/amber-101.md")
        self.assertTrue(score_answer("amber-101.md", item)["correct"])

    def test_outcomes_are_exclusive(self):
        item = Item("c", "x", "GOOD-1", ("OLD-1",))
        for said, field in (("GOOD-1", "correct"), ("UNKNOWN", "abstained"), ("OLD-1", "stale"), ("OTHER-9", "hallucinated")):
            out = score_answer(said, item)
            self.assertEqual([k for k, v in out.items() if v], [field], said)

    def test_unanswerable_is_right_only_when_admitted(self):
        item = Item("u", "x")
        self.assertTrue(score_answer("UNKNOWN", item)["correct"])
        self.assertTrue(score_answer("RB-99", item)["hallucinated"])

    def test_parse_batch(self):
        out = parse_batch("1. ANSWER: a\n2. LOOKUP: b c\nnoise\n3) ANSWER: d\n1. ANSWER: e")
        self.assertEqual(out, {1: ("ANSWER", "e"), 2: ("LOOKUP", "b c"), 3: ("ANSWER", "d")})

    def test_a_batch_never_shows_another_items_gold(self):
        items = [Item("a", "what follows GOLD-TWO here", "GOLD-ONE"), Item("b", "something else", "GOLD-TWO"), Item("c", "plain", "GOLD-3")]
        self.assertEqual([i.id for i in clean_batch(items)], ["a", "c"])

    def test_batch_numbers_items(self):
        text = batch_prompt([Item("a", "first"), Item("b", "second")], "Answer.")
        self.assertIn("1. first", text)
        self.assertIn("2. second", text)


class Demand(unittest.TestCase):
    def test_a_later_first_use_is_a_probe(self):
        rows = invented_session()
        cp = load_checkpoint(rows, 8, "cp")
        items, census = demand.items_from(cp, limit=5)
        self.assertEqual([i.gold for i in items], ["deep-buried-value-slug"])
        self.assertEqual(items[0].tags["label"], "tool:Bash")
        self.assertEqual(sum(c["demand"] for c in census), 1)
        self.assertNotIn("deep-buried-value-slug", items[0].ask)

    def test_what_arrived_after_the_cut_is_no_reach_back(self):
        rows = invented_session()
        rows[8]["message"]["content"][0]["text"] = "Next please, deep-buried-value-slug again."
        self.assertEqual(demand.items_from(load_checkpoint(rows, 8, "cp"), limit=5)[0], [])

    def test_a_line_naming_a_credential_is_never_a_probe(self):
        rows = invented_session()
        rows[10]["message"]["content"][0]["text"] = "Check the api token for the deep-buried-value-slug entry in the amber queue."
        self.assertEqual(demand.items_from(load_checkpoint(rows, 8, "cp"), limit=5)[0], [])

    def test_a_command_is_never_a_probe(self):
        self.assertFalse(demand.is_prose("cat runbooks/amber-101.md | head -5 and then some other words to pad it out"))
        self.assertTrue(demand.is_prose("I will edit the deep-buried-value-slug entry today in the amber queue."))

    def test_next_message(self):
        prompt, details = demand.next_message(load_checkpoint(invented_session(), 8, "cp"))
        self.assertEqual(prompt, "Next please.")
        self.assertEqual(len(details), 1)
        self.assertNotIn("deep", json.dumps(details))


class Conditions(unittest.TestCase):
    def setUp(self):
        self.cp = build("oncall", 7, steps=60, mean_result_chars=400)
        self.led = self.cp.ledger

    def test_control_changes_nothing(self):
        self.assertEqual(C.ctrl().changed(), 0)
        self.assertFalse(C.ctrl().strip_thinking)
        self.assertTrue(C.strip_only().strip_thinking)

    def test_recent_tail_is_left_alone(self):
        cond = C.at_level(self.led, 2, keep_recent=5)
        tail = {e.id for e in self.led.events[-5:]}
        self.assertFalse(tail & set(cond.levels))

    def test_floor_is_left_alone(self):
        cond = C.at_level(self.led, 2, frozen_upto=3)
        self.assertFalse({e.id for e in self.led.events[:4]} & set(cond.levels))

    def test_kind_only_touches_its_kind(self):
        cond = C.kind_at(self.led, "assistant", 3)
        self.assertTrue(cond.levels)
        self.assertEqual({self.led.by_id[i].kind for i in cond.levels}, {"assistant"})

    def test_a_signal_reaches_its_budget(self):
        before = self.led.tokens_held()
        saved = dict(self.led.levels)
        cond = C.by_signal(self.led, "S9", "signal:S9", 0.5)
        self.assertEqual(dict(self.led.levels), saved, "planning must not change the ledger")
        self.led.levels.update(cond.levels)
        self.assertLessEqual(self.led.tokens_held(), before * 0.6)
        self.led.levels.clear()
        self.led.levels.update(saved)

    def test_stub_and_ladder_differ(self):
        ladder = C.by_signal(self.led, "S0", "a", 0.5)
        stub = C.by_signal(self.led, "S0", "b", 0.5, stub=True)
        self.assertIn(Level.L3, set(stub.levels.values()))
        self.assertNotEqual(ladder.levels, stub.levels)

    def test_signals_order_differently(self):
        self.assertNotEqual(C.by_signal(self.led, "S0", "a", 0.7).levels, C.by_signal(self.led, "S1", "b", 0.7).levels)


class Running(unittest.TestCase):
    def run_all(self, folder, cp, conds, items, **kwargs):
        door, cost = ScriptedDoor(), ListPrice("claude-sonnet-5-5")
        sink = Sink(os.path.join(folder, "rows.jsonl"))
        for cond in conds:
            run_condition(door, cp, cond, items, sink, cost, TASK, "test", **kwargs)
        return analysis.read(sink.path)

    def test_generated_control_reads_everything_and_L3_needs_lookups(self):
        cp = build("oncall", 7, steps=60, mean_result_chars=400)
        with tempfile.TemporaryDirectory() as folder:
            rows = self.run_all(folder, cp, [C.ctrl(), C.at_level(cp.ledger, 3, keep_recent=2)], cp.items)
        table = {r["condition"]: r for r in analysis.condition_table(rows)}
        self.assertEqual(table["ctrl"]["recall"], 1.0)
        self.assertLess(table["all:L3"]["recall"], 1.0)
        self.assertLess(table["all:L3"]["recall_diff"], 0)
        self.assertGreater(table["all:L3"]["lookups"], 0)
        after = {r["condition"]: r for r in analysis.condition_table(rows, stage="final")}
        self.assertGreater(after["all:L3"]["recall"], table["all:L3"]["recall"])

    def test_a_real_transcript_runs_through_the_same_door(self):
        cp = load_checkpoint(invented_session(), 8, "cp")
        items, _ = demand.items_from(cp, limit=5)
        with tempfile.TemporaryDirectory() as folder:
            rows = self.run_all(folder, cp, [C.ctrl(), C.at_level(cp.ledger, 3, keep_recent=1, frozen_upto=cp.frozen_upto)], items, lookup=False,
                                replay=Replay("Next please.", ()))
        fork = {r["condition"]: r for r in rows if r["row"] == "fork"}
        self.assertEqual(fork["ctrl"]["changed"], 0)
        self.assertGreater(fork["all:L3"]["changed"], 0)
        self.assertEqual(sum(1 for r in rows if r["row"] == "replay"), 2)

    def test_rows_hold_no_text(self):
        cp = load_checkpoint(invented_session(), 8, "cp")
        items, _ = demand.items_from(cp, limit=5)
        with tempfile.TemporaryDirectory() as folder:
            rows = self.run_all(folder, cp, [C.ctrl(), C.at_level(cp.ledger, 3, keep_recent=1)], items, replay=Replay("Next please.", ()))
        dumped = json.dumps(rows)
        for planted in ("deep-buried-value-slug", "PLANTED-USER", "amber", "Next please"):
            self.assertNotIn(planted, dumped)

    def test_a_finished_condition_is_skipped_on_a_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = Sink(os.path.join(folder, "rows.jsonl"))
            sink.write(row="done", checkpoint="cp", condition="ctrl")
            self.assertIn(("cp", "ctrl"), Sink(sink.path).done)

    def test_a_refused_call_is_recorded_as_flagged(self):
        class Refusing(ScriptedDoor):
            def fork(self, prompt):
                return Turn(error="API Error: safeguards flagged this message")

        cp = build("oncall", 7, steps=40, mean_result_chars=400)
        with tempfile.TemporaryDirectory() as folder:
            sink = Sink(os.path.join(folder, "rows.jsonl"))
            out = run_condition(Refusing(), cp, C.ctrl(), cp.items, sink, ListPrice("claude-sonnet-5-5"), TASK, "test")
        self.assertIn("flagged", out["errors"])

    def test_archive_search(self):
        cp = load_checkpoint(invented_session(), 8, "cp")
        self.assertIn("deep-buried-value-slug", archive_search(cp.ledger, "deep-buried-value-slug"))
        self.assertEqual(archive_search(cp.ledger, "zzzz qqqq"), "")

    def test_rating(self):
        cp = build("oncall", 7, steps=40, mean_result_chars=400)
        ratings, usages = rate_events(ScriptedDoor(), cp.ledger)
        self.assertTrue(ratings)
        self.assertTrue(usages)

    def test_cells(self):
        cells = cell_tokens(build("oncall", 7, steps=40, mean_result_chars=400))
        self.assertTrue(all(c["tokens"] > 0 for c in cells))


class Analysis(unittest.TestCase):
    def test_group_kind(self):
        self.assertEqual(analysis.group_kind("tool:Bash"), "shell")
        self.assertEqual(analysis.group_kind("tool:Read"), "read")
        self.assertEqual(analysis.group_kind("tool:Write"), "write")
        self.assertEqual(analysis.group_kind("tool:fetch_logs"), "tool")
        self.assertEqual(analysis.group_kind("attachment:instructions"), "file")
        self.assertEqual(analysis.group_kind("user"), "user")

    def rows(self):
        out = []
        for cp in ("a", "b", "c"):
            for cond, right in (("ctrl", (1, 1, 1, 1)), ("all:L3", (1, 0, 0, 1))):
                for i, ok in enumerate(right):
                    first = dict(correct=bool(ok), hallucinated=False, abstained=not ok, stale=False)
                    out.append(dict(row="item", checkpoint=cp, condition=cond, item=f"i{i}", answerable=True, answered=True, held=bool(ok),
                                    looked_up=False, first=first, final=first, tags=dict(label="tool:Bash" if i < 2 else "assistant", band="10-49", size="100-999")))
        return out

    def test_paired_difference(self):
        table = {r["condition"]: r for r in analysis.condition_table(self.rows())}
        self.assertEqual(table["all:L3"]["recall_diff"], -0.5)
        self.assertEqual(table["all:L3"]["lost_of_control_right"], 0.5)
        self.assertEqual(table["all:L3"]["recall_when_gone"], 0.0)

    def test_importance_by_axis(self):
        table = analysis.importance_table(self.rows(), "group", conditions=("all:L3",), resamples=200)
        by = {r["band"]: r for r in table}
        self.assertEqual(by["shell"]["all:L3_lost"], 0.5)
        self.assertEqual(by["shell"]["all:L3_n"], 6)

    def test_s8_table_scores_a_heavily_lost_dense_cell_higher(self):
        rows = self.rows()
        cells = [dict(label="tool:Bash", band="10-49", tokens=1000), dict(label="assistant", band="10-49", tokens=1000), dict(label="file", band="50-199", tokens=1000)]
        census = [dict(label="tool:Bash", band="10-49", demand=10), dict(label="assistant", band="10-49", demand=1)]
        table = analysis.train_s8(rows, cells, census)
        self.assertGreater(table["shell|10-49"], table["assistant|10-49"])
        self.assertGreater(table["assistant|10-49"], table["file|50-199"])

    def test_a_trained_table_orders_evictions(self):
        cp = build("oncall", 7, steps=60, mean_result_chars=400)
        low = experiment.GroupedEstimate({"tool|0-9": 5.0}, default=0.0)
        self.assertIsInstance(low, AblationEstimate)
        tool = next(e for e in cp.ledger.events if e.is_tool)
        self.assertGreaterEqual(low.score(tool, cp.ledger, 0.0), 0.0)

    def test_replay_overlap(self):
        rows = [dict(row="replay", checkpoint="a", condition="ctrl", details=["x", "y"], words=5, output=9, error=False),
                dict(row="replay", checkpoint="a", condition="c", details=["x"], words=3, output=4, error=False)]
        table = {r["condition"]: r for r in analysis.replay_overlap(rows)}
        self.assertEqual(table["c"]["overlap_with_control"], 0.5)


class Experiment(unittest.TestCase):
    def args(self, folder, stage):
        return argparse.Namespace(stage=stage, out=folder, model="claude-sonnet-5-5", rate_model="claude-haiku-5-5", seeds="5", domains="oncall",
                                  only="", marker="M0", max_errors=3, run_id="t")

    def test_train_fit_test_report_offline(self):
        small = dict(experiment.__dict__.get("FORK", {}))
        with tempfile.TemporaryDirectory() as folder:
            real_build = experiment.build
            experiment.build = lambda d, s, marker="M0": real_build(d, s, marker=marker, steps=60, mean_result_chars=400)
            try:
                self.assertEqual(experiment.run_stage("train", self.args(folder, "train"), door=ScriptedDoor()), 0)
                self.assertEqual(experiment.fit(self.args(folder, "fit")), 0)
                with open(os.path.join(folder, "s8.json"), encoding="utf-8") as fh:
                    self.assertTrue(json.load(fh))
                self.assertEqual(experiment.run_stage("test", self.args(folder, "test"), door=ScriptedDoor(), rate_door=ScriptedDoor()), 0)
                report = experiment.report(self.args(folder, "report"), "test")
            finally:
                experiment.build = real_build
            names = {r["condition"] for r in report["first"]}
            self.assertIn("signal:S8", names)
            self.assertIn("stub:S9", names)
            self.assertEqual(report["usage"]["errors"], 0)
        self.assertEqual(small, {})

    def test_training_and_test_seeds_never_meet(self):
        self.assertFalse(set(experiment.TRAIN_SEEDS) & set(experiment.TEST_SEEDS))


class ThinkingCheck(unittest.TestCase):
    def test_the_puzzle_is_deterministic_and_has_one_answer(self):
        one, two = thinking_check.puzzle(3), thinking_check.puzzle(3)
        self.assertEqual(one, two)
        self.assertNotEqual(one["ids"], thinking_check.puzzle(4)["ids"])
        self.assertIn(str(one["limit"]), thinking_check.first_prompt(one))
        self.assertNotIn(one["ids"][0] + ",", thinking_check.first_prompt(one))

    def test_score(self):
        p = dict(ids=["K01", "K03"])
        self.assertTrue(thinking_check.score("ANSWER: K03 K01", p))
        self.assertFalse(thinking_check.score("ANSWER: K03", p))
        self.assertFalse(thinking_check.score("K01 K03", p))

    def test_summary_reads_only_trials_whose_first_answer_was_right(self):
        t = [dict(first_right=True, thinking_blocks=1, arms={"native": dict(context=9, results=[dict(right=True, output=5, thinking=2, context=9)])}),
             dict(first_right=False, thinking_blocks=1, arms={"native": dict(context=9, results=[dict(right=False, output=5, thinking=2, context=9)])})]
        out = thinking_check.summarize(t)
        self.assertEqual(out["native"]["forks"], 1)
        self.assertEqual(out["native"]["right"], 1)


class CodexCheck(unittest.TestCase):
    def test_reasoning_items_come_out(self):
        lines = [json.dumps({"type": "response_item", "payload": {"type": "reasoning"}}) + "\n",
                 json.dumps({"type": "response_item", "payload": {"type": "message"}}) + "\n", "not json\n"]
        self.assertEqual(len(codex_check.without_reasoning(lines)), 2)

    def test_classify(self):
        self.assertEqual(codex_check.classify("PLOVER-3306", ""), "held")
        self.assertEqual(codex_check.classify("I can’t provide that.", ""), "declined")
        self.assertEqual(codex_check.classify("HERON-9021", ""), "wrong")
        self.assertEqual(codex_check.classify("", "boom"), "error")

    def test_summary(self):
        out = codex_check.summarize([dict(arms={"kept": [dict(result="held", input=9), dict(result="declined", input=3)]})])
        self.assertEqual(out["kept"], dict(forks=2, held=1, declined=1, mean_input=6))

    def test_pace_is_unchanged_for_the_door(self):
        self.assertTrue(PACE)


if __name__ == "__main__":
    unittest.main()
