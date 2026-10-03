"""Request timeline joins, first-token semantics, and diagnostic failure safety."""

import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.storage.nixlshard.hicache_nixlshard import HiCacheNixlShard
from sglang.srt.observability import request_timeline


class TestRequestTimeline(unittest.TestCase):
    def records(self, directory):
        return [
            json.loads(line)
            for path in Path(directory).glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]

    def backend(self, agent):
        backend = object.__new__(HiCacheNixlShard)
        backend.agent = agent
        backend._trace_requests = True
        backend._condition = threading.Condition()
        backend._stats = Counter()
        backend.build_marker = "test-build"
        return backend

    def test_empty_output_does_not_define_first_token_and_repeated_frames_do_not_duplicate(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            request_timeline, "_directory", directory
        ):
            state = SimpleNamespace()
            self.assertFalse(
                request_timeline.first_output(
                    state, "rid-1", {"text": "", "output_ids": []}
                )
            )
            self.assertEqual(self.records(directory), [])
            self.assertTrue(
                request_timeline.first_output(
                    state, "rid-1", {"text": "", "output_ids": [123]}
                )
            )
            self.assertFalse(
                request_timeline.first_output(
                    state, "rid-1", {"text": "x", "output_ids": [123, 124]}
                )
            )
            records = self.records(directory)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["stage"], "api_first_nonempty_output")
            self.assertEqual(records[0]["rid"], "rid-1")
            self.assertNotIn("output_ids", records[0])

    def test_native_trace_is_joined_before_release_and_failure_still_drains(self):
        for trace_error in (False, True):
            with self.subTest(
                trace_error=trace_error
            ), tempfile.TemporaryDirectory() as directory, patch.object(
                request_timeline, "_directory", directory
            ):
                order = []

                def trace(handle):
                    order.append(("trace", handle))
                    if trace_error:
                        raise RuntimeError("diagnostic-only failure")
                    return [
                        {
                            "stage": "staging_copy",
                            "start_ns": 10,
                            "end_ns": 20,
                            "bytes": 64,
                        }
                    ]

                agent = SimpleNamespace(
                    poll=lambda handle: ["success"],
                    trace=trace,
                    release=lambda handle: order.append(("release", handle)),
                )
                self.assertEqual(self.backend(agent)._wait(42, 1, "request-42"), [True])
                self.assertEqual(order, [("trace", 42), ("release", 42)])
                record = self.records(directory)[0]
                self.assertEqual(record["rid"], "request-42")
                self.assertEqual(record["batch_handle"], 42)
                self.assertEqual(
                    record["stage"],
                    "native_trace_error" if trace_error else "native_batch",
                )
                if not trace_error:
                    self.assertTrue(record["terminal_observed"])

    def test_default_off_and_broken_sink_never_fail_io(self):
        with patch.object(request_timeline, "_directory", None):
            self.assertFalse(request_timeline.emit("id", "test"))
        with tempfile.NamedTemporaryFile() as file, patch.object(
            request_timeline, "_directory", file.name
        ), patch.object(request_timeline, "_warned", True):
            self.assertFalse(request_timeline.emit("id", "test"))
            agent = SimpleNamespace(
                poll=lambda handle: ["success"],
                trace=lambda handle: [],
                release=lambda handle: None,
            )
            self.assertEqual(self.backend(agent)._wait(0, 1, "id"), [True])

    def test_concurrent_threads_keep_complete_json_records(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            request_timeline, "_directory", directory
        ):
            threads = [
                threading.Thread(
                    target=lambda index=i: request_timeline.emit(
                        str(index), "query", 100, 200
                    )
                )
                for i in range(16)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            records = self.records(directory)
            self.assertEqual(len(records), 16)
            self.assertEqual(
                {record["rid"] for record in records}, {str(i) for i in range(16)}
            )


if __name__ == "__main__":
    unittest.main()
