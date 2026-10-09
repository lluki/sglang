"""Actual phase nesting and result-to-output joins without token values or extra waits."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.observability import request_timeline as T


class TestTimelinePhases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.patch = patch.object(T, "_directory", self.tmp.name)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def records(self):
        return [
            json.loads(line)
            for p in Path(self.tmp.name).glob("*.jsonl")
            for line in p.read_text().splitlines()
        ]

    def test_nested_actual_phase_parent_and_exception_preserved(self):
        @T.phase("outer")
        def operation(req):
            with T.scope(req.rid, "child"):
                raise RuntimeError("serving error")

        with self.assertRaisesRegex(RuntimeError, "serving error"):
            operation(NS(rid="r"))
        rows = {r["stage"]: r for r in self.records()}
        self.assertEqual(rows["child"]["parent_id"], rows["outer"]["span_id"])
        self.assertLessEqual(rows["outer"]["start_ns"], rows["child"]["start_ns"])
        self.assertLessEqual(rows["child"]["end_ns"], rows["outer"]["end_ns"])
        self.assertFalse(rows["child"]["successful"])
        self.assertFalse(rows["outer"]["successful"])
        self.assertNotIn("serving error", json.dumps(rows))
        self.assertEqual(getattr(T._local, "stacks", {}), {})

    def test_disabled_phase_does_not_mutate_result_or_request(self):
        batch = NS(reqs=[NS(rid="r")], forward_iter=3)
        result = NS()

        @T.phase("forward", requests="batch", forward_result=True)
        def forward(batch):
            return result

        with patch.object(T, "_directory", None):
            self.assertIs(forward(batch), result)
        self.assertFalse(hasattr(result, "_timeline_forward_id"))
        self.assertEqual(self.records(), [])

    def test_result_identity_survives_later_forward_on_same_batch(self):
        req = NS(rid="r", output_ids=[], send_token_offset=0)
        batch = NS(reqs=[req], forward_iter=0, forward_mode="extend")

        @T.phase("forward", requests="batch", forward_result=True)
        def forward(batch):
            batch.forward_iter += 1
            return NS()

        @T.phase("result", requests="batch", result_processing=True)
        def process(batch, result):
            req.output_ids.append(12345)
            T.token_commit(req, 1)
            req.send_token_offset = 1
            T.stream_send([req], [[12345]], lambda: None)

        first, second = forward(batch), forward(batch)
        self.assertNotEqual(first._timeline_forward_id, second._timeline_forward_id)
        process(batch, first)
        rows = self.records()
        send = next(r for r in rows if r["stage"] == "scheduler_stream_send")
        commit = next(r for r in rows if r["stage"] == "scheduler_token_commit")
        processed = next(r for r in rows if r["stage"] == "result")
        self.assertEqual(send["forward_id"], first._timeline_forward_id)
        self.assertEqual(commit["forward_id"], first._timeline_forward_id)
        self.assertEqual(send["parent_id"], processed["span_id"])
        self.assertEqual((send["output_token_start"], send["output_token_end"]), (0, 1))
        self.assertNotIn("12345", json.dumps(rows))

    def test_pending_membership_is_candidate_not_admission(self):
        class Scheduler:
            waiting_queue = [NS(rid="wait"), NS(rid="wait")]
            chunked_req = None

            @T.phase("next", requests="pending")
            def select(self, running_batch, last_batch):
                return None

        Scheduler().select(NS(reqs=[NS(rid="run")]), None)
        self.assertEqual({r["rid"] for r in self.records()}, {"wait", "run"})
        self.assertTrue(all(r["membership"] == "candidate" for r in self.records()))

    def test_received_membership_and_keyword_request_binding(self):
        @T.phase("receive", requests="received")
        def receive():
            return [NS(rid="one"), NS(rid="two")]

        @T.phase("request")
        def operation(*, recv_req):
            return 3

        self.assertEqual(len(receive()), 2)
        self.assertEqual(operation(recv_req=NS(rid="one")), 3)
        self.assertEqual([r["rid"] for r in self.records()], ["one", "two", "one"])

    def test_existing_wait_recorded_once_with_correct_result_join(self):
        req = NS(rid="r")
        result = NS(_timeline_forward_id="actual-forward")
        calls = []
        with T.batch_scope(NS(reqs=[req]), result, "copy_wait"):
            calls.append("existing synchronize")
        row = self.records()[0]
        self.assertEqual(calls, ["existing synchronize"])
        self.assertEqual(row["forward_id"], "actual-forward")
        self.assertEqual(row["category"], "framework_queue")
        self.assertTrue(row["envelope"])

    def test_only_emitted_members_and_frozen_ordinals(self):
        req = NS(rid="emitted", send_token_offset=4, _timeline_result_forward_id="f")
        calls = []

        def send():
            calls.append(1)
            req.send_token_offset = 9
            return "sent"

        self.assertEqual(T.stream_send([req], [[991, 992]], send), "sent")
        self.assertEqual(calls, [1])
        row = self.records()[0]
        self.assertEqual((row["output_token_start"], row["output_token_end"]), (2, 4))
        self.assertNotIn("991", json.dumps(row))
        self.assertNotIn("992", json.dumps(row))

    def test_broken_sink_never_fails_wait_or_send(self):
        with tempfile.NamedTemporaryFile() as file, patch.object(
            T, "_directory", file.name
        ), patch.object(T, "_warned", True):
            with T.scope("r", "phase"):
                pass
            self.assertEqual(
                T.stream_send([NS(rid="r", send_token_offset=1)], [[4]], lambda: 5), 5
            )


if __name__ == "__main__":
    unittest.main()
