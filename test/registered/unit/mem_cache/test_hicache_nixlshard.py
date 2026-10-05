# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""CPU adapter contract tests; no model, native binding, or GPU required."""

import ctypes
import gc
import hashlib
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
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory


class FakeAgent:
    authoritative_g3_supported = True

    def __init__(self, config):
        self.config = config
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
        self.quiescence = {}

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

    def is_quiescent(self, handle):
        event = self.quiescence.get(handle)
        return event is None or event.is_set()

    def release(self, handle):
        assert self.is_quiescent(handle)
        self.released.append(handle)

    def batch_exists(self, keys, hints=None, g3_instance=""):
        self.last_hints = hints
        self.last_g3_instance = g3_instance
        return [key in self.values for key in keys]

    def close(self):
        self.closed = True


class Pool:
    # Exercise the real framework allocation/lease methods on a tiny CPU pool.
    clear = HostKVCache.clear
    destroy = HostKVCache.destroy
    alloc = HostKVCache.alloc
    free = HostKVCache.free
    available_size = HostKVCache.available_size
    _merge_release_slots = HostKVCache._merge_release_slots
    acquire_io_lease = HostKVCache.acquire_io_lease
    release_io_lease = HostKVCache.release_io_lease
    logical_size = property(lambda self: self.size)
    logical_page_size = property(lambda self: self.page_size)

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
        self.device = "cpu"
        self.lock = threading.RLock()
        self.clear()
        self.alloc(self.size)

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


def page_keys(*labels):
    return [hashlib.sha256(label.encode()).hexdigest() for label in labels]


def config(
    revision="abc123",
    direct_io=False,
    direct_receive=False,
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
            "agent": {
                "direct_io": direct_io,
                "direct_receive": direct_receive,
                "numa_node": 0,
            },
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
                    backend.batch_set_v1(page_keys("a", "b", "c", "d"), indices),
                    [True] * 4,
                )
                items = backend.agent.submissions[0][1]
                self.assertEqual(len(items), 4)
                self.assertEqual(
                    len(items[0]["segments"]), 4 if layout == "layer_first" else 2
                )
                pool.kv_buffer.zero_()
                self.assertEqual(
                    backend.batch_get_v1(page_keys("a", "b", "c", "d"), indices),
                    [True] * 4,
                )
                torch.testing.assert_close(pool.kv_buffer, expected)
                self.assertEqual(len(backend.agent.regions), 1)
                self.assertEqual(len(backend.agent.released), 2)

    def test_direct_receive_requires_capable_native_and_boolean_config(self):
        for direct in (True, "true", 1):
            with self.subTest(direct=direct), self.assertRaises(ValueError):
                self.backend(direct_receive=direct)
        self.assertEqual(self.backends, [])

    def test_direct_receive_uses_final_kv_rows_and_same_storage_namespace(self):
        sys.modules["nixlshard"].direct_receive_supported = True
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = Pool(layout)
                staged = self.backend(pool)
                direct = self.backend(pool, direct_receive=True)
                self.assertTrue(direct.direct_receive)
                self.assertEqual(direct.namespace, staged.namespace)
                expected = pool.kv_buffer[:, 2:4].clone()
                indices = torch.tensor([2, 3])
                self.assertEqual(direct.batch_set_v1(page_keys("a"), indices), [True])
                pool.kv_buffer[:, 2:4].zero_()
                self.assertEqual(direct.batch_get_v1(page_keys("a"), indices), [True])
                torch.testing.assert_close(pool.kv_buffer[:, 2:4], expected)
                token, capacity = next(iter(direct.agent.regions.items()))
                self.assertEqual(
                    capacity, (pool.kv_buffer.data_ptr(), pool.kv_buffer.nbytes)
                )
                segments = direct.agent.submissions[-1][1][0]["segments"]
                self.assertEqual(len(segments), 2)
                for kv, (region, offset, length) in enumerate(segments):
                    destination = pool.kv_buffer[kv, 2:4]
                    self.assertEqual(region, token)
                    self.assertEqual(capacity[0] + offset, destination.data_ptr())
                    self.assertEqual(length, destination.nbytes)

    def test_direct_receive_rejects_unsupported_layout_without_registration(self):
        sys.modules["nixlshard"].direct_receive_supported = True
        for layout in ("page_head", "layer_first"):
            with self.subTest(layout=layout):
                backend = StorageBackendFactory.create_backend(
                    "nixlshard", config(direct_receive=True), Pool(layout)
                )
                self.backends.append(backend)
                with self.assertRaisesRegex(NotImplementedError, "direct_receive"):
                    backend.register_mem_pool_host(Pool(layout))
                self.assertIsNone(backend.agent)

    def test_failed_direct_receive_defers_free_until_late_dma_quiescence(self):
        sys.modules["nixlshard"].direct_receive_supported = True
        pool = Pool("page_first_direct")
        backend = self.backend(pool, direct_receive=True)
        unsafe = threading.Event()
        backend.agent.quiescence[0] = unsafe
        original_poll = backend.agent.poll
        with patch.object(
            backend.agent,
            "poll",
            side_effect=lambda h: page_keys("timeout") if h == 0 else original_poll(h),
        ):
            self.assertEqual(
                backend.batch_get_v1(page_keys("late"), torch.arange(2)), [False]
            )
            pool.free(torch.arange(2))  # Controller completed_req/abort tail release.
            self.assertEqual(pool.available_size(), 0)
            self.assertTrue(pool.slot_used[:2].all())
            with self.assertRaisesRegex(RuntimeError, "I/O leases"):
                pool.clear()
            with self.assertRaisesRegex(RuntimeError, "I/O lease"):
                backend.batch_get_v1(page_keys("overlap"), torch.arange(2))
            # Healthy unrelated destinations still complete while the failed
            # remote connection may write the quarantined rows at any time.
            self.assertEqual(
                backend.batch_get_v1(page_keys("healthy-miss"), torch.arange(2, 4)),
                [False],
            )
            self.assertEqual(backend.agent.released, [1])
            with self.assertRaisesRegex(RuntimeError, "quarantines"):
                backend.close()
            self.assertIs(backend._host_buffer, pool.kv_buffer)
            self.assertFalse(backend.agent.closed)
            # Simulate the previously posted NIC write arriving after timeout
            # and framework free. It cannot corrupt a newly allocated page.
            pool.kv_buffer[:, :2].fill_(999)
            self.assertIsNone(pool.alloc(2))
            unsafe.set()
            with backend._condition:
                self.assertTrue(
                    backend._condition.wait_for(
                        lambda: not backend._quarantined, timeout=2
                    )
                )
            self.assertEqual(pool.alloc(2).tolist(), [0, 1])
            self.assertEqual(backend.agent.released, [1, 0])
        backend.close()

    def test_quarantine_thread_start_failure_roots_pool_and_can_retry(self):
        from sglang.srt.mem_cache.storage.nixlshard.hicache_nixlshard import (
            _quarantine_owners,
        )

        sys.modules["nixlshard"].direct_receive_supported = True
        pool = Pool()
        backend = self.backend(pool, direct_receive=True)
        unsafe = threading.Event()
        backend.agent.quiescence[0] = unsafe
        with patch.object(
            backend.agent, "poll", return_value=page_keys("timeout")
        ), patch.object(
            threading.Thread, "start", side_effect=RuntimeError("no thread resources")
        ), self.assertLogs(
            "sglang.srt.mem_cache.storage.nixlshard.hicache_nixlshard", level="ERROR"
        ):
            self.assertEqual(
                backend.batch_get_v1(page_keys("late"), torch.arange(2)), [False]
            )
        self.assertIsNone(backend._quarantine_thread)
        pool.free(torch.arange(2))
        reference, tensor = weakref.ref(backend), weakref.ref(pool.kv_buffer)
        self.backends.remove(backend)
        del backend, pool
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertIsNotNone(tensor())
        backend = reference()
        self.assertIn(backend, _quarantine_owners)
        unsafe.set()
        with backend._condition:
            backend._start_quarantine_reaper()
            self.assertTrue(
                backend._condition.wait_for(
                    lambda: backend not in _quarantine_owners, timeout=2
                )
            )
        backend.close()
        del backend
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(tensor())

    def test_unsafe_native_success_is_not_published_and_quarantine_is_bounded(self):
        sys.modules["nixlshard"].direct_receive_supported = True
        storage_config = config(direct_receive=True)
        storage_config.extra_config["agent"]["max_inflight"] = 1
        pool = Pool()
        backend = StorageBackendFactory.create_backend(
            "nixlshard", storage_config, pool
        )
        backend.register_mem_pool_host(pool)
        self.backends.append(backend)
        unsafe = threading.Event()
        backend.agent.quiescence[0] = unsafe
        try:
            with patch.object(backend.agent, "poll", return_value=["success"]):
                self.assertEqual(
                    backend.batch_get_v1(page_keys("unsafe"), torch.arange(2)), [False]
                )
                self.assertEqual(
                    backend.batch_get_v1(page_keys("bounded"), torch.arange(2, 4)),
                    [False],
                )
            self.assertEqual(len(backend.agent.submissions), 1)
            self.assertEqual(backend._direct_live, 1)
            self.assertEqual(backend._stats["direct_quarantine_busy"], 1)
        finally:
            unsafe.set()
            with backend._condition:
                self.assertTrue(
                    backend._condition.wait_for(
                        lambda: not backend._quarantined, timeout=2
                    )
                )

    def test_host_io_lease_defers_only_leased_rows_and_rejects_aliases(self):
        pool = Pool()
        lease = pool.acquire_io_lease(torch.arange(2))
        with self.assertRaises(ValueError):
            pool.acquire_io_lease(torch.tensor([2, 2]))
        with self.assertRaisesRegex(RuntimeError, "already"):
            pool.acquire_io_lease(torch.arange(2))
        pool.free(torch.arange(4))
        self.assertEqual(pool.alloc(2).tolist(), [2, 3])
        with self.assertRaisesRegex(AssertionError, "Double-free"):
            pool.free(torch.arange(2))
        self.assertIsNone(pool.alloc(2))
        pool.release_io_lease(lease)
        self.assertEqual(pool.alloc(2).tolist(), [0, 1])
        with self.assertRaises(KeyError):
            pool.release_io_lease(lease)
        pool.clear()  # Reset allowed only after all external DMA is quiescent.
        with self.assertRaisesRegex(RuntimeError, "allocated"):
            pool.acquire_io_lease(torch.arange(2))

    def test_host_destroy_rejects_dma_lease_before_unpin_or_buffer_mutation(self):
        from sglang.srt.mem_cache.pool_host import base

        pool = Pool()
        pool.pin_memory = True
        original = pool.kv_buffer
        lease = pool.acquire_io_lease(torch.arange(2))
        with patch.object(base, "_is_cuda", True), patch.object(
            base, "_cuda_host_unregister"
        ) as unregister:
            with self.assertRaisesRegex(RuntimeError, "I/O leases"):
                pool.destroy()
            self.assertIs(pool.kv_buffer, original)
            self.assertFalse(getattr(pool, "_destroyed", False))
            unregister.assert_not_called()
            pool.release_io_lease(lease)
            pool.destroy()
            unregister.assert_called_once_with(original)
            self.assertIsNone(pool.kv_buffer)
            self.assertTrue(pool._destroyed)
            pool.destroy()
            unregister.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "destroyed"):
                pool.acquire_io_lease(torch.arange(2))
        partial = Pool.__new__(Pool)
        partial.destroy()
        self.assertTrue(partial._destroyed)
        self.assertIsNone(partial.kv_buffer)

    def test_detach_preserves_direct_backend_and_groups_until_quiescent_close(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        controller = HiCacheController.__new__(HiCacheController)
        controller._stop_storage_threads = Mock()
        controller._destroy_sync_groups = Mock()
        controller.storage_backend = Mock(direct_receive=True)
        controller.storage_backend.close.side_effect = RuntimeError(
            "quarantined receive"
        )
        controller.enable_storage = True
        controller.prefetch_hits_sync_groups = [object()]
        controller.prefetch_completion_sync_groups = [object()]
        retained = controller.storage_backend
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            controller.detach_storage_backend()
        self.assertIs(controller.storage_backend, retained)
        self.assertTrue(controller.enable_storage)
        controller._destroy_sync_groups.assert_not_called()
        self.assertEqual(len(controller.prefetch_hits_sync_groups), 1)

    def test_prefix_exists_and_miss_after_exists(self):
        backend = self.backend()
        self.assertEqual(
            backend.batch_set_v1(page_keys("a", "c"), torch.tensor([0, 1, 4, 5])),
            [True, True],
        )
        hints = HiCacheStorageExtraInfo(extra_info={"owner_hints": ["owner"] * 3})
        self.assertEqual(backend.batch_exists(page_keys("a", "b", "c"), hints), 1)
        self.assertEqual(backend.agent.last_hints, ["owner"] * 3)
        self.assertTrue(backend.exists(page_keys("a")[0]))
        del backend.agent.values[backend._keys(page_keys("a"))[0]]
        self.assertEqual(
            backend.batch_get_v1(page_keys("a", "b", "c"), torch.arange(6)),
            [False, False, True],
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
                    backend.batch_set_v1(page_keys("a"), indices)
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
            backend.batch_set_v1(page_keys("a", "b"), torch.arange(4)), [True, True]
        )
        self.assertTrue(
            all(len(v) == backend._page_bytes for v in backend.agent.values.values())
        )
        backend.agent.fail_keys.add(backend._keys(page_keys("b"))[0])
        pool.kv_buffer.zero_()
        self.assertEqual(
            backend.batch_get_v1(page_keys("a", "b"), torch.arange(4)), [True, False]
        )
        torch.testing.assert_close(pool.kv_buffer[:, :, :2], expected[:, :, :2])
        self.assertEqual(pool.kv_buffer[:, :, 2:4].count_nonzero().item(), 0)
        self.assertEqual(backend.get_stats()["get_hits"], 1)

    def test_schema_and_geometry_are_bound_before_ssd_open(self):
        storage_config = config()
        disk = {"path": "/disposable/mock", "create": True, "unit_bytes": 65536}
        storage_config.extra_config["agent"].update(
            disks=[disk], numa_node=3, g3_instance="model-kv"
        )
        original = dict(disk)
        backend = StorageBackendFactory.create_backend(
            "nixlshard", storage_config, Pool()
        )
        self.backends.append(backend)
        self.assertIsNone(backend.agent)
        pool = Pool()
        backend.register_mem_pool_host(pool)
        self.assertEqual(disk, original)
        native = backend.agent.config
        self.assertEqual(native["namespace_id"], backend.namespace_identity)
        self.assertEqual(native["registration_mode"], "EXPLICIT")
        self.assertEqual(native["g3_instance"], "model-kv")
        self.assertEqual(native["numa_node"], 3)
        self.assertEqual(native["disks"][0]["namespace_id"], backend.namespace_identity)
        self.assertEqual(native["disks"][0]["numa_node"], 3)
        self.assertEqual(native["disks"][0]["key_bytes"], 32)
        self.assertEqual(native["disks"][0]["min_object_bytes"], backend._page_bytes)
        self.assertEqual(native["disks"][0]["max_object_bytes"], backend._page_bytes)
        self.assertEqual(native["disks"][0]["unit_bytes"], 4096)
        self.assertEqual(native["disks"][0]["metadata_alignment"], 4096)
        backend.register_mem_pool_host(pool)
        self.assertEqual(len(backend.agent.regions), 1)

    def test_full_binary_digest_and_affinity_survive_both_transfer_modes(self):
        digest = bytes(range(32))
        for direct in (False, True):
            with self.subTest(direct=direct):
                native = types.SimpleNamespace(
                    Agent=FakeAgent, direct_receive_supported=True
                )
                with patch.dict(sys.modules, {"nixlshard": native}):
                    storage_config = config(direct_receive=direct)
                    storage_config.extra_config["agent"].update(
                        numa_node=5, g3_instance="assigned-nvme"
                    )
                    backend = StorageBackendFactory.create_backend(
                        "nixlshard", storage_config, Pool()
                    )
                self.backends.append(backend)
                backend.register_mem_pool_host(Pool())
                self.assertEqual(
                    backend.batch_set_v1([digest.hex()], torch.arange(2)), [True]
                )
                self.assertEqual(
                    backend.batch_get_v1([digest.hex()], torch.arange(2)), [True]
                )
                for _, items in backend.agent.submissions:
                    self.assertEqual(items[0]["key"], digest)
                    self.assertEqual(items[0]["g3_instance"], "assigned-nvme")
                    self.assertEqual(items[0]["numa"], 5)
                self.assertEqual(backend.batch_exists([digest.hex()]), 1)
                self.assertEqual(backend.agent.last_g3_instance, "assigned-nvme")

    def test_invalid_or_short_keys_fail_before_transfer(self):
        backend = self.backend()
        for key in ("short", "0" * 62, "0" * 66, "g" * 64, b"a" * 32, "\u00e9" * 64):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    backend.batch_set_v1([key], torch.arange(2))
                with self.assertRaises(ValueError):
                    backend.batch_exists([key])
        self.assertEqual(backend.agent.submissions, [])

    def test_readable_namespace_digest_is_not_exact_identity(self):
        with patch(
            "sglang.srt.mem_cache.storage.nixlshard.hicache_nixlshard.hashlib.sha256"
        ) as digest:
            digest.return_value.hexdigest.return_value = "same-readable-id"
            left = self.backend(revision="revision-left")
            right = self.backend(revision="revision-right")
        self.assertEqual(left.namespace, right.namespace)
        self.assertNotEqual(left.namespace_identity, right.namespace_identity)
        self.assertNotEqual(
            left.agent.config["namespace_id"], right.agent.config["namespace_id"]
        )

    def test_bad_instance_affinity_and_old_native_are_rejected(self):
        for field, value in (
            ("numa_node", None),
            ("numa_node", -1),
            ("numa_node", True),
            ("g3_instance", ""),
            ("registration_mode", "AUTOMATIC"),
        ):
            with self.subTest(field=field, value=value):
                storage_config = config()
                storage_config.extra_config["agent"][field] = value
                with self.assertRaises(ValueError):
                    StorageBackendFactory.create_backend(
                        "nixlshard", storage_config, Pool()
                    )
        old_agent = type("OldAgent", (), {"__init__": Mock()})
        with patch.dict(
            sys.modules, {"nixlshard": types.SimpleNamespace(Agent=old_agent)}
        ):
            with self.assertRaisesRegex(ValueError, "authoritative-G3"):
                StorageBackendFactory.create_backend("nixlshard", config(), Pool())
        old_agent.__init__.assert_not_called()

    def test_mismatched_disk_identity_fails_before_agent_open(self):
        for mismatch in (
            {"namespace_id": "different-full-schema"},
            {"numa_node": 1},
            {"key_bytes": 16},
        ):
            with self.subTest(mismatch=mismatch):
                storage_config = config()
                storage_config.extra_config["agent"]["disks"] = [mismatch]
                backend = StorageBackendFactory.create_backend(
                    "nixlshard", storage_config, Pool()
                )
                self.backends.append(backend)
                with self.assertRaises(ValueError):
                    backend.register_mem_pool_host(Pool())
                self.assertIsNone(backend.agent)

    def test_unregistered_backend_can_close_without_opening_disks(self):
        backend = StorageBackendFactory.create_backend("nixlshard", config(), Pool())
        self.backends.append(backend)
        backend.close()
        self.assertIsNone(backend.agent)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.register_mem_pool_host(Pool())

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
        self.assertEqual(self.backend().namespace_identity, base.namespace_identity)
        with self.assertRaises(ValueError):
            StorageBackendFactory.create_backend(
                "nixlshard", config(revision=""), Pool()
            )

    def test_auxiliary_pools_cannot_silently_hit(self):
        backend = self.backend()
        auxiliary = [PoolTransfer(PoolName.MAMBA, keys=page_keys("a"))]
        with self.assertRaises(NotImplementedError):
            backend.batch_exists_v2(page_keys("a"), auxiliary)
        with self.assertRaises(NotImplementedError):
            backend.batch_get_v2(auxiliary)
        with self.assertRaises(NotImplementedError):
            backend.register_mem_host_pool_v2(Pool(), PoolName.MAMBA)
        transfer = PoolTransfer(
            PoolName.KV, host_indices=torch.arange(2), keys=page_keys("a")
        )
        self.assertEqual(backend.batch_set_v2([transfer]), {PoolName.KV: [True]})
        self.assertEqual(backend.batch_exists_v2(page_keys("a")).kv_hit_pages, 1)

    def test_exceptional_poll_drains_handle_before_releasing_active_operation(self):
        backend = self.backend()
        with patch.object(
            backend.agent, "poll", side_effect=RuntimeError("poll failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "poll failed"):
                backend.batch_get_v1(page_keys("a"), torch.arange(2))
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
        backend.batch_set_v1(page_keys("a"), torch.arange(2))
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
        backend.batch_set_v1(page_keys("a"), torch.arange(2))
        backend.batch_get_v1(page_keys("a"), torch.arange(2))
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
            controller.page_set_func(page_keys("a", "c"), torch.tensor([0, 1, 4, 5]))
        )
        operation = types.SimpleNamespace(request_id="contract-test")
        self.assertEqual(
            controller.page_get_func(
                operation, page_keys("a", "b", "c"), torch.arange(6)
            ),
            1,
        )
        backend.agent.fail_keys.add(backend._keys(page_keys("b"))[0])
        self.assertFalse(controller.page_set_func(page_keys("a", "b"), torch.arange(4)))

    def test_concurrent_backup_prefetch_keep_independent_page_descriptors(self):
        backend = self.backend(direct_io=True)
        backend.batch_set_v1(page_keys("old"), torch.arange(2))
        backend.agent.poll_barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as workers:
            get = workers.submit(
                backend.batch_get_v1, page_keys("old"), torch.arange(2, 4)
            )
            put = workers.submit(
                backend.batch_set_v1, page_keys("new"), torch.arange(4, 6)
            )
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
            transfer = workers.submit(
                backend.batch_set_v1, page_keys("a"), torch.arange(2)
            )
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
