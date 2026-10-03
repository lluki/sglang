# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Opt-in adapter tests against the built NIXLShard POSIX/UCX runtime.

Set SGLANG_RUN_NIXLSHARD_NATIVE=1 and add the rebuilt nixlshard package to
PYTHONPATH. These tests create disposable files, never block devices.
"""

import json
import os
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch
from test_hicache_nixlshard import Pool, config

from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo
from sglang.srt.observability import request_timeline


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_NIXLSHARD_NATIVE") == "1",
    "requires the rebuilt native NIXLShard runtime",
)
class TestNativeHiCacheNixlShard(unittest.TestCase):
    def backend(self, pool, directory, direct_io, **overrides):
        storage_config = config(direct_io=direct_io)
        storage_config.extra_config["agent"] = {
            "name": "hicache-test-" + uuid.uuid4().hex,
            "disks": [
                {
                    "path": os.path.join(directory, "cache.bin"),
                    "capacity_bytes": 16 * 1024 * 1024,
                    "unit_bytes": 4096,
                    "metadata_bytes": 1024 * 1024,
                    "create": True,
                }
            ],
            "listen_host": "127.0.0.1",
            "listen_port": 0,
            "max_inflight": 32,
            "timeout_ms": 5000,
            "staging_slot_bytes": 4096,
            "staging_slots": 4,
            "workers": 2,
            "direct_io": direct_io,
        }
        storage_config.extra_config["agent"].update(overrides)
        backend = StorageBackendFactory.create_backend(
            "nixlshard", storage_config, pool
        )
        self.addCleanup(backend.close)
        backend.register_mem_pool_host(pool)
        self.assertNotEqual(backend.build_marker, "unknown")
        return backend

    def test_trace_clock_brackets_local_and_remote(self):
        import nixlshard

        if not hasattr(nixlshard.Agent, "trace"):
            self.skipTest("requires the trace-capable native candidate")
        with tempfile.TemporaryDirectory(
            dir=os.environ.get("NIXLSHARD_TEST_DIR", "/raid/nixlshard-v2")
        ) as directory, patch.object(request_timeline, "_directory", directory):
            owner_pool = Pool("page_first_direct")
            owner = self.backend(
                owner_pool, directory, True, enable_trace=True, staging_slot_bytes=16384
            )
            expected = owner_pool.kv_buffer.clone()
            keys = ["trace-a", "trace-b"]
            self.assertEqual(owner.batch_set_v1(keys, torch.arange(4)), [True, True])
            reader_pool = Pool("page_first_direct")
            reader_pool.kv_buffer.zero_()
            owner_name = owner.storage_config.extra_config["agent"]["name"]
            reader = self.backend(
                reader_pool,
                directory,
                True,
                disks=[],
                enable_trace=True,
                peers={owner_name: owner.agent.endpoint()},
                remote_batch_limit=8,
                staging_slot_bytes=16384,
            )
            hint = HiCacheStorageExtraInfo(extra_info={"owner_hints": [owner_name] * 2})
            deadline = time.monotonic() + 5
            while reader.batch_exists(keys, hint) != 2:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            for backend, pool, rid, hints, expected_stage in (
                (owner, owner_pool, "local-clock", {}, "local_posix"),
                (
                    reader,
                    reader_pool,
                    "remote-clock",
                    {"owner_hints": [owner_name] * 2},
                    "remote_rpc",
                ),
            ):
                before = time.monotonic_ns()
                extra = HiCacheStorageExtraInfo(extra_info={"request_id": rid, **hints})
                self.assertEqual(
                    backend.batch_get_v1(keys, torch.arange(4), extra), [True, True]
                )
                after = time.monotonic_ns()
                records = [
                    json.loads(line)
                    for path in Path(directory).glob("request-timeline-*.jsonl")
                    for line in path.read_text().splitlines()
                ]
                batches = [
                    r
                    for r in records
                    if r["rid"] == rid and r["stage"] == "native_batch"
                ]
                self.assertEqual(len(batches), 1)
                self.assertTrue(batches[0]["terminal_observed"])
                events = batches[0]["events"]
                self.assertIn(expected_stage, [e["stage"] for e in events])
                self.assertIn("staging_copy", [e["stage"] for e in events])
                for event in events:
                    self.assertLessEqual(before, event["start_ns"])
                    self.assertLessEqual(event["start_ns"], event["end_ns"])
                    self.assertLessEqual(event["end_ns"], after)
                    if event["stage"] == "remote_rpc":
                        self.assertIn("owner_posix_ns", event, events)
                        self.assertGreater(event["owner_posix_ns"], 0)
                        self.assertGreater(event["owner_ucx_ns"], 0)
                        self.assertLessEqual(
                            event["owner_posix_ns"] + event["owner_ucx_ns"],
                            event["end_ns"] - event["start_ns"],
                        )
                torch.testing.assert_close(pool.kv_buffer[:, :4], expected[:, :4])

    def test_local_segmented_roundtrip_unaligned_pages(self):
        for layout in ("page_first", "page_first_direct", "page_head", "layer_first"):
            for direct_io in (False, True):
                with self.subTest(
                    layout=layout, direct_io=direct_io
                ), tempfile.TemporaryDirectory(
                    dir=os.environ.get("NIXLSHARD_TEST_DIR", "/raid/nixlshard-v2")
                ) as directory:
                    pool = Pool(layout)
                    backend = self.backend(pool, directory, direct_io)
                    expected = pool.kv_buffer.clone()
                    keys = ["page-a", "page-b", "page-c", "page-d"]
                    self.assertEqual(
                        backend.batch_set_v1(keys, torch.arange(8)), [True] * 4
                    )
                    self.assertEqual(backend.batch_exists(keys), 4)
                    pool.kv_buffer.zero_()
                    self.assertEqual(
                        backend.batch_get_v1(keys, torch.arange(8)), [True] * 4
                    )
                    torch.testing.assert_close(pool.kv_buffer, expected)
                    self.assertEqual(backend.get_stats()["get_hits"], 4)
                    backend.close()

    def test_concurrent_backup_prefetch_and_absent_page(self):
        with tempfile.TemporaryDirectory(
            dir=os.environ.get("NIXLSHARD_TEST_DIR", "/raid/nixlshard-v2")
        ) as directory:
            pool = Pool("layer_first")
            backend = self.backend(pool, directory, direct_io=True)
            expected = pool.kv_buffer[:, :, :2].clone()
            self.assertEqual(backend.batch_set_v1(["old"], torch.arange(2)), [True])
            with ThreadPoolExecutor(2) as workers:
                read = workers.submit(backend.batch_get_v1, ["old"], torch.arange(2, 4))
                write = workers.submit(
                    backend.batch_set_v1, ["new"], torch.arange(4, 6)
                )
                self.assertEqual(read.result(timeout=10), [True])
                self.assertEqual(write.result(timeout=10), [True])
            torch.testing.assert_close(pool.kv_buffer[:, :, 2:4], expected)
            self.assertEqual(backend.batch_exists(["old", "absent", "new"]), 1)
            self.assertEqual(
                backend.batch_get_v1(["absent"], torch.arange(6, 8)), [False]
            )
            backend.close()


if __name__ == "__main__":
    unittest.main()
