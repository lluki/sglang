# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the whole-page NIXLShard public API adapter."""
import ctypes
import hashlib
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig, HiCacheStorageExtraInfo, PoolHitPolicy, PoolName, PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory


class FakeAgent:
    def __init__(self, config):
        self.config = config
        self.regions = {}
        self.values = {}
        self.handles = {}
        self.submissions = []
        self.released = []
        self.canceled = []
        self.fail_keys = set()
        self.ready = None
        self.started = threading.Event()
        self.closed = False
        self.lock = threading.Lock()
        self.registrations = []
        self.poll_fail_once = False

    def register_memory(self, address, length, device_id=0, metadata="", *, numa_node=None):
        assert isinstance(address, ctypes.c_void_p) and device_id == 0
        self.regions[(address.value, length)] = metadata
        self.registrations.append((address.value, length, device_id, metadata, numa_node))
        return [(address.value, length, device_id, metadata)]

    def deregister_memory(self, address, length, device_id=0):
        assert isinstance(address, ctypes.c_void_p) and device_id == 0
        for state in self.handles.values():
            if not state.get("terminal"):
                for item in state["items"]:
                    pointer, size, _ = item["buffer"]
                    assert not (address.value <= pointer < address.value + length)
        del self.regions[(address.value, length)]

    def _submit(self, items, direction):
        with self.lock:
            handle = len(self.handles) + 1
            self.handles[handle] = {"items": items, "direction": direction}
            self.submissions.append((direction, items))
            self.started.set()
            return handle

    def batch_store(self, items):
        return self._submit(items, "set")

    def batch_load(self, items):
        return self._submit(items, "get")

    def poll(self, handle):
        if self.poll_fail_once:
            self.poll_fail_once = False
            raise RuntimeError("transient polling failure")
        state = self.handles[handle]
        if state.get("terminal"):
            return state["terminal"]
        if self.ready is not None and not self.ready.is_set():
            return "IN_PROG"
        errors = {}
        for item in state["items"]:
            identity = (item["namespace"], item["key"])
            pointer, length, device = item["buffer"]
            assert device == 0
            assert any(base <= pointer and pointer + length <= base + size for base, size in self.regions)
            if handle in self.canceled or item["key"] in self.fail_keys:
                errors[identity] = "ERROR"
            elif state["direction"] == "set":
                self.values[identity] = ctypes.string_at(pointer, length)
            elif identity not in self.values:
                errors[identity] = "NOT_FOUND"
            else:
                payload = self.values[identity]
                assert len(payload) == length
                ctypes.memmove(pointer, payload, length)
        state["errors"] = errors
        state["terminal"] = "ERROR" if "ERROR" in errors.values() else ("NOT_FOUND" if errors else "SUCCESS")
        return state["terminal"]

    def get_errors(self, handle):
        assert self.handles[handle].get("terminal")
        return dict(self.handles[handle]["errors"])

    def cancel(self, handle):
        self.canceled.append(handle)

    def release(self, handle):
        assert self.handles[handle].get("terminal")
        self.released.append(handle)

    def batch_exists(self, items):
        self.last_exists = items
        return [(item["namespace"], item["key"]) in self.values for item in items]

    def trace(self, handle):
        assert self.handles[handle].get("terminal")
        return []

    def stats(self):
        return {"completed": len(self.released)}

    def close(self):
        assert not self.regions, "deregister before close"
        assert len(self.released) == len(self.handles)
        self.closed = True


class Pool:
    def __init__(self, layout="page_first", dtype=torch.float32):
        self.layout = layout
        self.page_size = 2
        self.size = 8
        self.dtype = dtype
        self.layer_num = 1
        self.head_num = 1
        self.head_dim = 8
        self.kv_buffer = (
            [torch.arange(32, dtype=dtype).reshape(8, 4),
             torch.arange(32, 64, dtype=dtype).reshape(8, 4)]
            if layout == "layer_first"
            else torch.arange(64, dtype=dtype).reshape(8, 8)
        )
        self.leases = {}
        self.next_lease = 0

    def get_page_buffer_meta(self, indices):
        pointers, lengths = [], []
        buffers = self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]
        for first in indices.tolist()[::self.page_size]:
            for buffer in buffers:
                page = buffer[first:first + self.page_size]
                pointers.append(page.data_ptr())
                lengths.append(page.numel() * page.element_size())
        return pointers, lengths

    def acquire_io_lease(self, indices):
        rows = set(indices.tolist())
        assert all(not rows.intersection(existing) for existing in self.leases.values())
        self.next_lease += 1
        self.leases[self.next_lease] = rows
        return self.next_lease

    def release_io_lease(self, lease):
        del self.leases[lease]

    def bytes(self):
        buffers = self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]
        return [ctypes.string_at(buffer.data_ptr(), buffer.numel() * buffer.element_size()) for buffer in buffers]

    def zero(self):
        buffers = self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]
        for buffer in buffers:
            buffer.zero_()


def page_keys(*labels):
    return [hashlib.sha256(label.encode()).hexdigest() for label in labels]


def config(direct_receive=False, enable_storage_metrics=False, revision="immutable-model-revision", **unused):
    return HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=enable_storage_metrics,
        is_page_first_layout=True, model_name="model",
        extra_config={"model_revision": revision, "agent": {
            "name": "test", "numa_node": 3, "direct_receive": direct_receive,
        }},
    )


class TestHiCacheNixlShard(unittest.TestCase):
    def setUp(self):
        self.native_patch = patch.dict(sys.modules, {"nixlshard": types.SimpleNamespace(Agent=FakeAgent)})
        self.native_patch.start()
        self.backends = []

    def tearDown(self):
        for backend in self.backends:
            backend.close()
        self.native_patch.stop()

    def backend(self, pool=None, **kwargs):
        pool = pool or Pool()
        backend = StorageBackendFactory.create_backend("nixlshard", config(**kwargs), pool)
        backend.register_mem_pool_host(pool)
        self.backends.append(backend)
        return backend

    def test_complete_page_roundtrip_contiguous_and_split(self):
        for layout in ("page_first", "layer_first"):
            with self.subTest(layout=layout):
                pool = Pool(layout)
                backend = self.backend(pool)
                expected = pool.bytes()
                keys = page_keys("a", "b", "c", "d")
                indices = torch.arange(8)
                self.assertEqual(backend.batch_set_v1(keys, indices), [True] * 4)
                items = backend.agent.submissions[0][1]
                self.assertEqual(items[0]["key"], keys[0].encode("ascii"))
                self.assertEqual(items[0]["namespace"], "kv")
                self.assertEqual(set(items[0]), {"key", "namespace", "buffer"})
                self.assertEqual(items[0]["buffer"][1], 64)
                pool.zero()
                self.assertEqual(backend.batch_get_v1(keys, indices), [True] * 4)
                self.assertEqual(pool.bytes(), expected)
                self.assertFalse(pool.leases)
                counters = backend.get_stats()
                self.assertEqual(counters.get("pack_bytes", 0), 256 if layout == "layer_first" else 0)
                self.assertEqual(counters.get("unpack_bytes", 0), 256 if layout == "layer_first" else 0)

    def test_v2_pool_namespaces_preserve_same_ascii_key(self):
        backend = self.backend(Pool("layer_first"))
        auxiliary = Pool()
        auxiliary.kv_buffer.add_(1000)
        backend.register_mem_host_pool_v2(auxiliary, PoolName.MAMBA)
        key = page_keys("same", "same2", "same3", "same4")
        transfers = [
            PoolTransfer(PoolName.KV, host_indices=torch.arange(8), keys=key),
            PoolTransfer(PoolName.MAMBA, host_indices=torch.arange(8), keys=key),
        ]
        expected = [backend.mem_pool_host.bytes(), auxiliary.bytes()]
        self.assertEqual(backend.batch_set_v2(transfers), {PoolName.KV: [True] * 4, PoolName.MAMBA: [True] * 4})
        self.assertNotEqual(backend.agent.values[("kv", key[0].encode())], backend.agent.values[("mamba", key[0].encode())])
        backend.mem_pool_host.zero()
        auxiliary.zero()
        self.assertEqual(backend.batch_get_v2(transfers), {PoolName.KV: [True] * 4, PoolName.MAMBA: [True] * 4})
        self.assertEqual([backend.mem_pool_host.bytes(), auxiliary.bytes()], expected)

    def test_partial_errors_prefix_exists_and_exact_keys(self):
        backend = self.backend()
        keys = ["a" * 64, "contains/slash\x00", "short ASCII"]
        backend.batch_set_v1(keys, torch.arange(6))
        del backend.agent.values[("kv", keys[1].encode())]
        self.assertEqual(backend.batch_exists(keys), 1)
        self.assertEqual(backend.batch_get_v1(keys, torch.arange(6)), [True, False, True])
        backend.agent.fail_keys.add(keys[2].encode())
        self.assertEqual(backend.batch_get_v1(keys, torch.arange(6)), [True, False, False])
        self.assertEqual(backend.agent.last_exists[1]["key"], b"contains/slash\x00")
        with self.assertRaises(ValueError):
            backend.batch_exists(["é"])
        with self.assertRaises(ValueError):
            backend.batch_exists(["x" * 257])

    def test_v2_trailing_pool_preserves_restorable_prefix_holes(self):
        backend = self.backend()
        secondary = Pool()
        backend.register_mem_host_pool_v2(secondary, PoolName.SWA)
        keys = ["a", "b", "c", "d"]
        backend.batch_set_v1(keys, torch.arange(8))
        backend.batch_set_v2([PoolTransfer(PoolName.SWA, host_indices=torch.arange(8), keys=keys)])
        del backend.agent.values[("swa", b"b")]
        result = backend.batch_exists_v2(keys, [
            PoolTransfer(PoolName.SWA, keys=["tail"], hit_policy=PoolHitPolicy.TRAILING_PAGES)
        ])
        self.assertEqual(result.kv_hit_pages, 4)
        self.assertEqual(result.restorable_prefix_pages, [1, 3, 4])
        result = backend.batch_exists_v2(keys, [PoolTransfer(PoolName.SWA)])
        self.assertEqual(result.kv_hit_pages, 1)

    def test_registration_numa_and_pointer_contract(self):
        backend = self.backend(Pool("layer_first"))
        self.assertTrue(all(region[2:] == (0, "", 3) for region in backend.agent.registrations))
        backend.batch_set_v1(["whole-page"], torch.arange(2))
        self.assertEqual(backend.agent.registrations[-1][4], 3)
        self.assertFalse(backend.mem_pool_host.leases)

    def test_deadline_cancels_then_drains_before_leases_or_buffers_return(self):
        backend = self.backend(Pool("layer_first"), direct_receive=True)
        backend.batch_set_v1(["page"], torch.arange(2))
        backend.agent.ready = threading.Event()
        info = HiCacheStorageExtraInfo(extra_info={"deadline_monotonic": time.monotonic() - 1})
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(backend.batch_get_v1, ["page"], torch.arange(2), info)
            end = time.monotonic() + 3
            while not backend.agent.canceled and time.monotonic() < end:
                time.sleep(.001)
            self.assertEqual(len(backend.agent.canceled), 1)
            self.assertFalse(future.done())
            self.assertTrue(backend.mem_pool_host.leases)
            self.assertGreater(len(backend.agent.regions), 2)
            backend.agent.ready.set()
            self.assertEqual(future.result(timeout=3), [False])
        self.assertFalse(backend.mem_pool_host.leases)
        self.assertEqual(len(backend.agent.regions), 2)

    def test_transient_poll_failure_is_canceled_and_drained(self):
        backend = self.backend(Pool("layer_first"))
        backend.agent.poll_fail_once = True
        with self.assertRaisesRegex(RuntimeError, "transient polling"):
            backend.batch_set_v1(["page"], torch.arange(2))
        self.assertEqual(len(backend.agent.canceled), 1)
        self.assertEqual(len(backend.agent.released), 1)
        self.assertFalse(backend.mem_pool_host.leases)

    def test_active_detach_waits_and_parallel_disjoint_prefetch_backup(self):
        backend = self.backend(Pool("layer_first"))
        backend.batch_set_v1(["load"], torch.arange(2))
        backend.agent.ready = threading.Event()
        backend.agent.started.clear()
        with ThreadPoolExecutor(3) as executor:
            load = executor.submit(backend.batch_get_v1, ["load"], torch.arange(2))
            self.assertTrue(backend.agent.started.wait(3))
            with self.assertRaises(ValueError):
                backend.batch_set_v1(["overlap"], torch.arange(2))
            store = executor.submit(backend.batch_set_v1, ["store"], torch.arange(2, 4))
            close = executor.submit(backend.close)
            time.sleep(.03)
            self.assertFalse(close.done())
            self.assertFalse(backend.agent.closed)
            backend.agent.ready.set()
            self.assertEqual(load.result(timeout=3), [True])
            self.assertEqual(store.result(timeout=3), [True])
            close.result(timeout=3)
        self.assertTrue(backend.agent.closed)
        self.assertFalse(backend.agent.regions)

    def test_duplicate_destinations_and_invalid_indices_fail_before_acceptance(self):
        backend = self.backend(Pool("layer_first"))
        for indices in (torch.tensor([0, 1, 0, 1]), torch.tensor([0, 2, 4, 5])):
            with self.assertRaises(ValueError):
                backend.batch_get_v1(["a", "b"], indices)
        self.assertFalse(backend.agent.submissions)
        with self.assertRaises(ValueError):
            backend.batch_set_v2([
                PoolTransfer(PoolName.KV, host_indices=torch.arange(2), keys=["a"]),
                PoolTransfer(PoolName.KV, host_indices=torch.arange(2), keys=["b"]),
            ])

    def test_explicit_factory_and_controller_interfaces_are_wired(self):
        backend = self.backend()
        self.assertEqual(backend.implementation_marker, "nixlshard-public-api-whole-pages-v1")
        from pathlib import Path
        root = Path(__file__).resolve().parents[4] / "python/sglang/srt"
        self.assertIn('"nixlshard"', (root / "managers/cache_controller.py").read_text())
        hybrid = (root / "mem_cache/hybrid_cache/hybrid_cache_controller.py").read_text()
        self.assertIn("register_mem_host_pool_v2", hybrid)
        self.assertIn("batch_get_v2", hybrid)
        self.assertIn("batch_set_v2", hybrid)


if __name__ == "__main__":
    unittest.main()
