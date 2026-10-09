# SPDX-License-Identifier: Apache-2.0
"""Real POSIX and UCX/TCP adapter tests in one sandbox container.

Enable SGLANG_RUN_NIXLSHARD_NATIVE=1 with the isolated native build on
PYTHONPATH. NIXLSHARD_TEST_DIR selects the disposable debug-file directory.
"""
import os
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
from test_hicache_nixlshard import Pool, config, page_keys
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo, PoolName, PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory


@unittest.skipUnless(
    os.environ.get("SGLANG_RUN_NIXLSHARD_NATIVE") == "1",
    "requires the rebuilt native NIXLShard runtime",
)
class TestNativeHiCacheNixlShard(unittest.TestCase):
    def backend(self, pool, directory, path="cache.bin", **overrides):
        storage = config()
        storage.extra_config["agent"] = {
            "name": "adapter-native-" + uuid.uuid4().hex,
            "listen_host": "127.0.0.1", "listen_port": 0,
            "disks": [{
                "path": os.path.join(directory, path),
                "capacity_bytes": 16 * 1024 * 1024,
                "unit_bytes": 4096, "create": True,
            }],
            "staging_slot_bytes": 4096, "staging_slots": 4,
            "workers": 2, "numa_node": 0,
        }
        storage.extra_config["agent"].update(overrides)
        backend = StorageBackendFactory.create_backend("nixlshard", storage, pool)
        self.addCleanup(backend.close)
        backend.register_mem_pool_host(pool)
        self.assertNotEqual(backend.build_marker, "unknown")
        return backend

    def directory(self):
        directory = tempfile.TemporaryDirectory(
            dir=os.environ.get("NIXLSHARD_TEST_DIR", "/workspace/build-tools"),
            prefix="adapter-native-",
        )
        self.addCleanup(directory.cleanup)
        return directory.name

    def test_local_complete_page_posix_roundtrip_with_pack_copy_accounting(self):
        for layout in ("page_first", "layer_first"):
            with self.subTest(layout=layout):
                pool = Pool(layout)
                backend = self.backend(pool, self.directory())
                expected = pool.bytes()
                keys = page_keys("a", "b", "c", "d")
                self.assertEqual(backend.batch_set_v1(keys, torch.arange(8)), [True] * 4)
                pool.zero()
                self.assertEqual(backend.batch_get_v1(keys, torch.arange(8)), [True] * 4)
                self.assertEqual(pool.bytes(), expected)
                stats = backend.agent.stats()
                self.assertGreater(stats.get("posix_write_bytes", 0), 0)
                self.assertGreater(stats.get("posix_read_bytes", 0), 0)
                self.assertEqual(backend.get_stats().get("pack_bytes", 0), 256 if layout == "layer_first" else 0)

    def test_two_agents_ucx_tcp_remote_staged_and_direct_whole_values(self):
        keys = page_keys("remote-a", "remote-b", "remote-c", "remote-d")
        owner_pool = Pool("layer_first")
        owner = self.backend(owner_pool, self.directory())
        expected = owner_pool.bytes()
        self.assertEqual(owner.batch_set_v1(keys, torch.arange(8)), [True] * 4)
        owner_name = owner.native_config["name"]
        for direct in (False, True):
            with self.subTest(direct=direct):
                reader_pool = Pool("layer_first")
                reader_pool.zero()
                reader = self.backend(reader_pool, self.directory(), disks=[], direct_receive=direct)
                reader.agent.add_peer(owner_name, owner.agent.endpoint())
                info = HiCacheStorageExtraInfo(extra_info={"location_hints": [[owner_name] for _ in range(4)]})
                deadline = time.monotonic() + 10
                while reader.batch_exists(keys, info) != 4:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                self.assertEqual(reader.batch_get_v1(keys, torch.arange(8), info), [True] * 4)
                self.assertEqual(reader_pool.bytes(), expected)
                self.assertGreater(reader.agent.stats().get("remote_read_bytes", 0), 0)
                self.assertGreater(owner.agent.stats().get("ucx_write_bytes", 0), 0)
                self.assertEqual(reader.get_stats().get("unpack_bytes", 0), 256)

    def test_clean_reopen_retains_ascii_keys_and_distinct_pool_namespaces(self):
        directory = self.directory()
        pool = Pool()
        backend = self.backend(pool, directory)
        secondary = Pool()
        secondary.kv_buffer.add_(100)
        backend.register_mem_host_pool_v2(secondary, PoolName.INDEXER)
        keys = page_keys("one", "two", "three", "four")
        transfers = [
            PoolTransfer(PoolName.KV, host_indices=torch.arange(8), keys=keys),
            PoolTransfer(PoolName.INDEXER, host_indices=torch.arange(8), keys=keys),
        ]
        expected = [pool.bytes(), secondary.bytes()]
        self.assertEqual(backend.batch_set_v2(transfers), {PoolName.KV: [True] * 4, PoolName.INDEXER: [True] * 4})
        backend.close()
        pool.zero()
        secondary.zero()
        reopened = self.backend(pool, directory)
        reopened.register_mem_host_pool_v2(secondary, PoolName.INDEXER)
        self.assertEqual(reopened.batch_get_v2(transfers), {PoolName.KV: [True] * 4, PoolName.INDEXER: [True] * 4})
        self.assertEqual([pool.bytes(), secondary.bytes()], expected)

    def test_concurrent_prefetch_backup_and_absence_after_exists(self):
        pool = Pool("layer_first")
        backend = self.backend(pool, self.directory())
        self.assertEqual(backend.batch_set_v1(["existing"], torch.arange(2)), [True])
        with ThreadPoolExecutor(2) as executor:
            load = executor.submit(backend.batch_get_v1, ["existing"], torch.arange(2))
            store = executor.submit(backend.batch_set_v1, ["new"], torch.arange(2, 4))
            self.assertEqual(load.result(timeout=10), [True])
            self.assertEqual(store.result(timeout=10), [True])
        self.assertEqual(backend.batch_get_v1(["missing"], torch.arange(4, 6)), [False])
        self.assertFalse(pool.leases)


if __name__ == "__main__":
    unittest.main()
