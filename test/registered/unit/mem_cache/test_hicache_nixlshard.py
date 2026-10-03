# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""CPU adapter contract tests; no model, native binding, or GPU required."""

import ctypes
import gc
import sys
import threading
import types
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory


class FakeAgent:
    def __init__(self, config):
        self.regions = {}
        self.values = {}
        self.submissions = []
        self.released = []
        self.handles = {}
        self.fail_keys = set()
        self.ready = None
        self.started = threading.Event()
        self.poll_barrier = None
        self.closed = False
        self.lock = threading.Lock()

    def register_memory(self, address, length):
        token = len(self.regions) + 1
        self.regions[token] = (address, length)
        return token

    def _submit(self, items, direction):
        with self.lock:
            handle = len(self.handles)
            self.handles[handle] = (items, direction)
            self.submissions.append((direction, items))
            self.started.set()
            return handle

    def batch_store(self, items):
        return self._submit(items, "set")

    def batch_load(self, items):
        return self._submit(items, "get")

    def poll(self, handle):
        if self.ready is not None and not self.ready.is_set():
            return None
        if self.poll_barrier is not None:
            self.poll_barrier.wait(timeout=3)
        items, direction = self.handles[handle]
        statuses = []
        for item in items:
            key = item["key"]
            if key in self.fail_keys:
                statuses.append("io_failure")
                continue
            segments = []
            for token, offset, length in item["segments"]:
                base, capacity = self.regions[token]
                assert 0 <= offset <= capacity - length
                segments.append((base + offset, length))
            if direction == "set":
                self.values[key] = b"".join(ctypes.string_at(p, n) for p, n in segments)
                statuses.append("success")
            elif key not in self.values:
                statuses.append("not_found")
            else:
                payload = self.values[key]
                offset = 0
                for pointer, length in segments:
                    ctypes.memmove(pointer, payload[offset : offset + length], length)
                    offset += length
                statuses.append("success")
        return statuses

    def release(self, handle):
        self.released.append(handle)

    def batch_exists(self, keys, hints=None):
        self.last_hints = hints
        return [key in self.values for key in keys]

    def close(self):
        self.closed = True


class Pool:
    def __init__(self, layout="page_first", dtype=torch.float32):
        self.layout = layout
        self.page_size = 2
        self.size = 8
        self.layer_num = 2
        self.head_num = 1
        self.head_dim = 4
        self.dtype = dtype
        shape = (2, 2, 8, 1, 4) if layout == "layer_first" else (2, 8, 2, 1, 4)
        self.kv_buffer = torch.arange(128).reshape(shape).to(dtype)

    def get_page_buffer_meta(self, indices):
        pointers, lengths = [], []
        for start in indices.tolist()[:: self.page_size]:
            if self.layout == "layer_first":
                segments = [
                    self.kv_buffer[kv, layer, start : start + self.page_size]
                    for layer in range(self.layer_num)
                    for kv in range(2)
                ]
            else:
                segments = [
                    self.kv_buffer[kv, start : start + self.page_size]
                    for kv in range(2)
                ]
            for segment in segments:
                pointers.append(segment.data_ptr())
                lengths.append(segment.numel() * segment.element_size())
        return pointers, lengths


def config(
    revision="abc123",
    direct_io=False,
    tp_rank=0,
    export_native_metrics=False,
    enable_storage_metrics=True,
):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=enable_storage_metrics,
        is_page_first_layout=True,
        model_name="Qwen/Qwen3-32B-FP8",
        extra_config={
            "model_revision": revision,
            "agent": {"direct_io": direct_io},
            "export_native_metrics": export_native_metrics,
        },
    )


class TestHiCacheNixlShard(unittest.TestCase):
    def setUp(self):
        # Restore only the mocked module; clearing all modules would reload
        # native MLIR/CUTLASS extension types imported by framework metrics.
        self.original_native_module = sys.modules.get("nixlshard")
        sys.modules["nixlshard"] = types.SimpleNamespace(Agent=FakeAgent)
        self.backends = []

    def tearDown(self):
        for backend in self.backends:
            backend.close()
        if self.original_native_module is None:
            sys.modules.pop("nixlshard", None)
        else:
            sys.modules["nixlshard"] = self.original_native_module

    def backend(self, pool=None, **kwargs):
        pool = pool or Pool()
        backend = StorageBackendFactory.create_backend(
            "nixlshard", config(**kwargs), pool
        )
        backend.register_mem_pool_host(pool)
        self.backends.append(backend)
        return backend

    def test_segmented_roundtrip_all_layouts(self):
        for layout in ("page_first", "page_first_direct", "page_head", "layer_first"):
            with self.subTest(layout=layout):
                pool = Pool(layout)
                backend = self.backend(pool)
                expected = pool.kv_buffer.clone()
                indices = torch.arange(8)
                self.assertEqual(
                    backend.batch_set_v1(["a", "b", "c", "d"], indices), [True] * 4
                )
                items = backend.agent.submissions[0][1]
                self.assertEqual(len(items), 4)
                self.assertEqual(
                    len(items[0]["segments"]), 4 if layout == "layer_first" else 2
                )
                pool.kv_buffer.zero_()
                self.assertEqual(
                    backend.batch_get_v1(["a", "b", "c", "d"], indices), [True] * 4
                )
                torch.testing.assert_close(pool.kv_buffer, expected)
                self.assertEqual(len(backend.agent.regions), 1)
                self.assertEqual(len(backend.agent.released), 2)

    def test_prefix_exists_and_miss_after_exists(self):
        backend = self.backend()
        self.assertEqual(
            backend.batch_set_v1(["a", "c"], torch.tensor([0, 1, 4, 5])), [True, True]
        )
        hints = HiCacheStorageExtraInfo(extra_info={"owner_hints": ["owner"] * 3})
        self.assertEqual(backend.batch_exists(["a", "b", "c"], hints), 1)
        self.assertEqual(backend.agent.last_hints, ["owner"] * 3)
        self.assertTrue(backend.exists("a"))
        del backend.agent.values[backend._keys(["a"])[0]]
        self.assertEqual(
            backend.batch_get_v1(["a", "b", "c"], torch.arange(6)), [False, False, True]
        )

    def test_page_validation_before_io(self):
        backend = self.backend()
        for indices in (
            torch.tensor([0]),
            torch.tensor([1, 2]),
            torch.tensor([0, 2]),
            torch.tensor([8, 9]),
            torch.tensor([0.0, 1.0]),
        ):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    backend.batch_set_v1(["a"], indices)
        self.assertEqual(backend.agent.submissions, [])
        self.assertEqual(
            backend.batch_get_v1([], torch.empty(0, dtype=torch.int64)), []
        )
        self.assertEqual(backend.batch_exists([]), 0)
        with self.assertRaises(ValueError):
            backend.batch_get_v1([], torch.tensor([0]))

    def test_failure_is_per_page_and_unaligned_layer_segments_keep_destination(self):
        pool = Pool("layer_first")
        backend = self.backend(pool, direct_io=True)
        expected = pool.kv_buffer.clone()
        self.assertEqual(
            backend.batch_set_v1(["a", "b"], torch.arange(4)), [True, True]
        )
        self.assertTrue(
            all(len(v) == backend._page_bytes for v in backend.agent.values.values())
        )
        backend.agent.fail_keys.add(backend._keys(["b"])[0])
        pool.kv_buffer.zero_()
        self.assertEqual(
            backend.batch_get_v1(["a", "b"], torch.arange(4)), [True, False]
        )
        torch.testing.assert_close(pool.kv_buffer[:, :, :2], expected[:, :, :2])
        self.assertEqual(pool.kv_buffer[:, :, 2:4].count_nonzero().item(), 0)
        self.assertEqual(backend.get_stats()["get_hits"], 1)

    def test_namespaces_separate_revision_layout_dtype_rank(self):
        base = self.backend()
        others = [
            self.backend(revision="different"),
            self.backend(tp_rank=1),
            self.backend(Pool("layer_first")),
            self.backend(Pool(dtype=torch.bfloat16)),
        ]
        self.assertEqual(len({base.namespace, *(b.namespace for b in others)}), 5)
        self.assertEqual(self.backend().namespace, base.namespace)
        with self.assertRaises(ValueError):
            StorageBackendFactory.create_backend(
                "nixlshard", config(revision=""), Pool()
            )

    def test_auxiliary_pools_cannot_silently_hit(self):
        backend = self.backend()
        auxiliary = [PoolTransfer(PoolName.MAMBA, keys=["a"])]
        with self.assertRaises(NotImplementedError):
            backend.batch_exists_v2(["a"], auxiliary)
        with self.assertRaises(NotImplementedError):
            backend.batch_get_v2(auxiliary)
        with self.assertRaises(NotImplementedError):
            backend.register_mem_host_pool_v2(Pool(), PoolName.MAMBA)
        transfer = PoolTransfer(PoolName.KV, host_indices=torch.arange(2), keys=["a"])
        self.assertEqual(backend.batch_set_v2([transfer]), {PoolName.KV: [True]})
        self.assertEqual(backend.batch_exists_v2(["a"]).kv_hit_pages, 1)

    def test_exceptional_poll_drains_handle_before_releasing_active_operation(self):
        backend = self.backend()
        with patch.object(
            backend.agent, "poll", side_effect=RuntimeError("poll failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "poll failed"):
                backend.batch_get_v1(["a"], torch.arange(2))
        self.assertEqual(backend.agent.released, [0])
        self.assertEqual(backend._active, 0)

    def test_registered_tensor_owner_retained_until_native_close(self):
        pool = Pool()
        backend = self.backend(pool)
        tensor = weakref.ref(pool.kv_buffer)
        pool.kv_buffer = None
        gc.collect()
        self.assertIsNotNone(tensor())
        backend.close()
        gc.collect()
        self.assertIsNone(tensor())

    def test_native_counters_preserve_adapter_page_counts(self):
        backend = self.backend()
        backend.batch_set_v1(["a"], torch.arange(2))
        with patch.object(
            backend.agent,
            "stats",
            create=True,
            return_value={"staged_bytes": 256, "set_hits": 99},
        ):
            stats = backend.get_stats()
        self.assertEqual(stats["set_hits"], 1)
        self.assertEqual(stats["native_set_hits"], 99)
        self.assertEqual(stats["native_staged_bytes"], 256)

    def test_native_metrics_opt_in_exports_actual_registry_and_snapshots_once(self):
        import prometheus_client
        from prometheus_client import CollectorRegistry, generate_latest
        from prometheus_client.parser import text_string_to_metric_families

        registry = CollectorRegistry()
        backend = self.backend(export_native_metrics=True)
        native = {
            "posix_read_ns": 2500000000,
            "posix_read_bytes": 8192,
            "timeout": 2,
            "unknown_key_or_peer_field": 123,
            "staging_free_slots": 8,
        }
        with (
            patch.object(prometheus_client, "REGISTRY", registry),
            patch.object(backend.agent, "stats", create=True, return_value=native),
        ):
            snapshot = backend.get_stats()
            backend.get_stats()
            exported = generate_latest(registry).decode()
            samples = [
                sample
                for family in text_string_to_metric_families(exported)
                for sample in family.samples
            ]

            def value(name, label, component):
                return next(
                    sample.value
                    for sample in samples
                    if sample.name == name and sample.labels.get(label) == component
                )

            self.assertEqual(
                value(
                    "sglang:nixlshard_component_seconds_total",
                    "component",
                    "posix_read",
                ),
                2.5,
            )
            self.assertEqual(
                value(
                    "sglang:nixlshard_component_bytes_total", "component", "posix_read"
                ),
                8192,
            )
            self.assertEqual(
                value("sglang:nixlshard_events_total", "event", "timeout"), 2
            )
            self.assertNotIn("unknown_key_or_peer_field", exported)
            self.assertNotIn("staging_free_slots", exported)
            self.assertEqual(snapshot["native_posix_read_ns"], 2500000000)
            backend.close()

    def test_diagnostic_export_failure_does_not_prevent_quiescent_close(self):
        backend = self.backend()
        backend._native_exporter = Mock()
        backend._native_exporter.observe.side_effect = OSError(
            "diagnostic collector failed"
        )
        with (
            patch.object(backend.agent, "stats", create=True, return_value={}),
            self.assertLogs(
                "sglang.srt.mem_cache.storage.nixlshard.hicache_nixlshard",
                level="ERROR",
            ),
        ):
            backend.close()
        self.assertTrue(backend._closed)
        self.assertIsNone(backend._host_buffer)

    def test_native_metrics_require_both_configuration_gates(self):
        from sglang.srt.mem_cache.storage.nixlshard.native_metrics import (
            NativeMetricsExporter,
        )

        for enabled, opt_in in ((False, True), (True, False), (False, False)):
            with (
                self.subTest(enable_metrics=enabled, export_native_metrics=opt_in),
                patch.object(
                    NativeMetricsExporter,
                    "__init__",
                    side_effect=AssertionError("metrics exporter imported/created"),
                ),
            ):
                backend = self.backend(
                    enable_storage_metrics=enabled, export_native_metrics=opt_in
                )
                with patch.object(
                    backend.agent,
                    "stats",
                    create=True,
                    return_value={"posix_read_ns": 123},
                ):
                    self.assertEqual(backend.get_stats()["native_posix_read_ns"], 123)
                self.assertIsNone(backend._native_exporter)
                backend.close()

    def test_framework_metrics_accept_counter_snapshot_and_drain_samples_once(self):
        from sglang.srt.observability.metrics_collector import (
            StorageMetrics,
            StorageMetricsCollector,
        )

        backend = self.backend()
        backend.batch_set_v1(["a"], torch.arange(2))
        backend.batch_get_v1(["a"], torch.arange(2))
        snapshot = backend.get_stats()
        self.assertIsInstance(snapshot, StorageMetrics)
        self.assertEqual(snapshot.backup_pgs, [1])
        self.assertEqual(snapshot.prefetch_pgs, [1])
        collector = object.__new__(StorageMetricsCollector)
        collector._log_histogram = Mock()
        for name in (
            "histogram_prefetch_pgs",
            "histogram_backup_pgs",
            "histogram_prefetch_bandwidth",
            "histogram_backup_bandwidth",
        ):
            setattr(collector, name, Mock())
        collector.log_storage_metrics(snapshot)
        self.assertEqual(collector._log_histogram.call_count, 4)
        drained = backend.get_stats()
        self.assertEqual(drained.backup_pgs, [])
        self.assertEqual(drained.prefetch_pgs, [])
        self.assertEqual(drained["get_hits"], 1)

    def test_controller_selects_registered_page_interface_and_prefix_policy(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        controller = object.__new__(HiCacheController)
        pool = Pool()
        controller.enable_storage = False
        controller.storage_host_pool = pool
        controller.mem_pool_host = pool
        controller.page_size = pool.page_size
        controller.host_memory_mode = "cache"
        controller.storage_stop_event = threading.Event()
        controller._stop_storage_threads = Mock()
        controller._generate_storage_config = Mock(return_value=config())
        controller._create_sync_groups = Mock(return_value=[])
        controller._start_storage_threads = Mock()
        controller.attach_storage_backend("nixlshard")
        backend = controller.storage_backend
        self.backends.append(backend)
        self.assertIs(
            controller.page_get_func.__func__, HiCacheController._page_get_zero_copy
        )
        self.assertTrue(
            controller.page_set_func(["a", "c"], torch.tensor([0, 1, 4, 5]))
        )
        operation = types.SimpleNamespace(request_id="contract-test")
        self.assertEqual(
            controller.page_get_func(operation, ["a", "b", "c"], torch.arange(6)), 1
        )
        backend.agent.fail_keys.add(backend._keys(["b"])[0])
        self.assertFalse(controller.page_set_func(["a", "b"], torch.arange(4)))

    def test_concurrent_backup_prefetch_keep_independent_page_descriptors(self):
        backend = self.backend(direct_io=True)
        backend.batch_set_v1(["old"], torch.arange(2))
        backend.agent.poll_barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as workers:
            get = workers.submit(backend.batch_get_v1, ["old"], torch.arange(2, 4))
            put = workers.submit(backend.batch_set_v1, ["new"], torch.arange(4, 6))
            self.assertEqual(get.result(timeout=4), [True])
            self.assertEqual(put.result(timeout=4), [True])
        get_items = next(
            items
            for direction, items in backend.agent.submissions
            if direction == "get"
        )
        put_items = next(
            items
            for direction, items in reversed(backend.agent.submissions)
            if direction == "set"
        )
        get_offsets = {offset for _, offset, _ in get_items[0]["segments"]}
        put_offsets = {offset for _, offset, _ in put_items[0]["segments"]}
        self.assertTrue(get_offsets.isdisjoint(put_offsets))
        self.assertEqual(len(backend.agent.regions), 1)

    def test_close_waits_for_active_transfer_and_refuses_new_work(self):
        backend = self.backend()
        backend.agent.ready = threading.Event()
        with ThreadPoolExecutor(2) as workers:
            transfer = workers.submit(backend.batch_set_v1, ["a"], torch.arange(2))
            self.assertTrue(backend.agent.started.wait(2))
            closing = workers.submit(backend.close)
            with backend._condition:
                self.assertTrue(
                    backend._condition.wait_for(lambda: backend._closing, timeout=2)
                )
            self.assertFalse(closing.done())
            self.assertFalse(backend.agent.closed)
            with self.assertRaises(RuntimeError):
                backend.exists("a")
            backend.agent.ready.set()
            self.assertEqual(transfer.result(timeout=2), [True])
            closing.result(timeout=2)
        self.assertTrue(backend.agent.closed)


if __name__ == "__main__":
    unittest.main()
