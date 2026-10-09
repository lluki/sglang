"""Diskless requesters must not retain persistent-backup locks or staging."""

import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingWriteThrough,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _controller(*, writable=True):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.enable_storage = True
    controller.storage_backend = SimpleNamespace(write_enabled=writable)
    controller.backup_queue = Queue()
    controller.mem_pool_host = SimpleNamespace(free=MagicMock(), entry_map={})
    return controller


def _cache(controller):
    cache = MagicMock()
    cache.enable_storage = True
    cache.cache_controller = controller
    cache.page_size = 2
    cache.hicache_storage_pass_prefix_keys = True
    cache.ongoing_backup = {}
    cache._build_sidecar_transfers.return_value = []
    cache._build_backup_sidecar.return_value = []
    cache.storage_existence_cache.covers_all.return_value = False
    return cache


class TestUnifiedStorageBackupAdmission(unittest.TestCase):
    def test_diskless_skips_spec_locks_and_controller_queue(self):
        controller = _controller(writable=False)
        cache = _cache(controller)

        UnifiedRadixCache.write_backup_storage(cache, 7)

        cache.tree_core.build_storage_backup_spec.assert_not_called()
        cache._build_sidecar_transfers.assert_not_called()
        cache.inc_host_lock_ref.assert_not_called()
        self.assertEqual(cache.ongoing_backup, {})
        self.assertTrue(controller.backup_queue.empty())

    def test_disabled_or_detached_storage_skips_spec(self):
        for detached in (False, True):
            with self.subTest(detached=detached):
                cache = _cache(None if detached else _controller())
                cache.enable_storage = detached
                UnifiedRadixCache.write_backup_storage(cache, 7)
                cache.tree_core.build_storage_backup_spec.assert_not_called()
                cache.inc_host_lock_ref.assert_not_called()
                self.assertEqual(cache.ongoing_backup, {})

    def test_controller_refusal_does_not_acquire_host_lock(self):
        controller = _controller()
        controller.write_storage = MagicMock(return_value=None)
        cache = _cache(controller)
        cache.tree_core.build_storage_backup_spec.return_value = SimpleNamespace(
            host_value=torch.arange(4),
            token_ids=[1, 2, 3, 4],
            hash_value=["a", "b"],
            prefix_keys=["prefix"],
            comp_xfers={},
        )

        UnifiedRadixCache.write_backup_storage(cache, 7)

        controller.write_storage.assert_called_once()
        cache.inc_host_lock_ref.assert_not_called()
        self.assertEqual(cache.ongoing_backup, {})
        self.assertTrue(controller.backup_queue.empty())

    def test_diskless_requester_keeps_normal_host_cache_backup(self):
        controller = _controller(writable=False)
        cache = _cache(controller)
        cache.buffer_pipeline = None
        device = torch.arange(4)
        host = torch.arange(10, 14)
        cache.tree_core.build_backup_spec.return_value = (device, {})
        cache._execute_kv_backup.return_value = host
        cache._backup_publish_node_ids.return_value = [7]

        written = UnifiedRadixCache._execute_and_commit_kv_backup(
            cache, SimpleNamespace(node_ids=[7])
        )

        self.assertEqual(written, 4)
        cache.tree_core.commit_backup.assert_called_once_with(7, host, {})
        cache._track_write_through_node.assert_called_once()
        lock_params = cache.inc_lock_ref.return_value.to_dec_params.return_value
        cache.ongoing_write_through = {7: _OngoingWriteThrough(7, lock_params, [7])}
        cache.write_backup_storage.side_effect = (
            lambda node_id: UnifiedRadixCache.write_backup_storage(cache, node_id)
        )

        UnifiedRadixCache._finish_write_through_ack(cache, 7)

        cache.tree_core.finish_write_through.assert_called_once_with([7], 7)
        cache.dec_lock_ref.assert_called_once_with(7, lock_params)
        cache.tree_core.build_storage_backup_spec.assert_not_called()
        cache.inc_host_lock_ref.assert_not_called()
        self.assertEqual(cache.ongoing_backup, {})
        self.assertTrue(controller.backup_queue.empty())

    def test_writable_backup_queues_and_holds_host_lock(self):
        controller = _controller()
        cache = _cache(controller)
        host = torch.arange(4)
        aux = PoolTransfer(name=PoolName.SWA, host_indices=torch.arange(2))
        spec = SimpleNamespace(
            host_value=host,
            token_ids=[1, 2, 3, 4],
            hash_value=["a", "b"],
            prefix_keys=["prefix"],
            comp_xfers={"swa": [aux]},
        )
        cache.tree_core.build_storage_backup_spec.return_value = spec
        lock_params = object()
        cache.inc_host_lock_ref.return_value.to_dec_params.return_value = lock_params

        UnifiedRadixCache.write_backup_storage(cache, 7)

        operation = controller.backup_queue.get_nowait()
        self.assertIs(operation.host_indices, host)
        self.assertEqual(operation.token_ids, spec.token_ids)
        self.assertEqual(operation.hash_value, spec.hash_value)
        self.assertEqual(operation.prefix_keys, spec.prefix_keys)
        self.assertEqual(operation.pool_transfers, [aux])
        self.assertEqual(cache.ongoing_backup, {operation.id: (7, lock_params)})
        cache.inc_host_lock_ref.assert_called_once_with(7)


class TestBufferStorageBackupAdmission(unittest.TestCase):
    def _pipeline(self, controller):
        cache = _cache(controller)
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline.reset()
        pipeline.write_backlog_cap = 64
        pipeline._backup_oversize = MagicMock(return_value=False)
        return pipeline, cache

    def _launch(self, pipeline, cache, controller, *, with_aux=False):
        host = torch.arange(10, 14)
        snapshot = BufferBackupSnapshot(
            node_id=7,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=["a", "b"],
            key=RadixKey(array("q", [1, 2, 3, 4])),
            prefix_keys=["prefix"],
        )
        cache.tree_core.snapshot_buffer_backup.return_value = snapshot
        controller.write = MagicMock(return_value=host)
        aux = []
        if with_aux:
            swa_host = torch.arange(20, 22)
            aux = [PoolTransfer(name=PoolName.SWA, host_indices=swa_host)]
            cache._build_backup_sidecar.return_value = [
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                    host_indices=swa_host,
                    indices_from_pool=PoolName.SWA,
                )
            ]
            controller.mem_pool_host.entry_map[PoolName.SWA] = SimpleNamespace(
                host_pool=SimpleNamespace(page_size=2, free=MagicMock())
            )
        pipeline.enqueue_backup_intent(7)
        intent = pipeline.pending_write_queue.popleft()
        self.assertTrue(
            pipeline._launch_backup_intent(intent, torch.arange(4), {"swa": aux})
        )
        self.assertEqual(pipeline.write_staged_tokens_, 4)
        self.assertEqual(pipeline.write_backlog_tokens_, 0)
        self.assertEqual(pipeline.inflight_backup_hashes, {"a": 1, "b": 1})
        controller.mem_pool_host.free.assert_not_called()
        cache.dec_lock_ref.assert_not_called()
        return host

    def test_diskless_admission_reserves_no_resources(self):
        controller = _controller(writable=False)
        pipeline, cache = self._pipeline(controller)

        pipeline.enqueue_backup_intent(7)
        pipeline.flush_pending_writes()

        cache.tree_core.snapshot_buffer_backup.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        cache._build_backup_sidecar.assert_not_called()
        self.assertEqual(pipeline.write_backlog_tokens_, 0)
        self.assertEqual(pipeline.write_staged_tokens_, 0)
        self.assertEqual(pipeline.inflight_backup_hashes, {})
        self.assertTrue(pipeline.is_idle())
        self.assertTrue(controller.backup_queue.empty())

    def test_disabled_or_detached_storage_does_not_snapshot(self):
        for detached in (False, True):
            with self.subTest(detached=detached):
                pipeline, cache = self._pipeline(None if detached else _controller())
                cache.enable_storage = detached
                pipeline.enqueue_backup_intent(7)
                cache.tree_core.snapshot_buffer_backup.assert_not_called()
                self.assertTrue(pipeline.is_idle())

    def test_refused_write_releases_staging_after_d2h_ack(self):
        controller = _controller()
        pipeline, cache = self._pipeline(controller)
        host = self._launch(pipeline, cache, controller, with_aux=True)
        controller.write_storage = MagicMock(return_value=None)

        pipeline.finish_backup_ack(7)

        cache.dec_lock_ref.assert_called_once()
        controller.write_storage.assert_called_once()
        controller.mem_pool_host.free.assert_called_once_with(host)
        controller.mem_pool_host.entry_map[
            PoolName.SWA
        ].host_pool.free.assert_called_once()
        cache.storage_existence_cache.add.assert_not_called()
        self.assertEqual(pipeline.write_staged_tokens_, 0)
        self.assertEqual(pipeline.inflight_backup_hashes, {})
        self.assertTrue(pipeline.is_idle())
        self.assertTrue(controller.backup_queue.empty())

    def test_readonly_at_ack_skips_specs_and_releases_staging(self):
        controller = _controller()
        pipeline, cache = self._pipeline(controller)
        host = self._launch(pipeline, cache, controller, with_aux=True)
        controller.storage_backend.write_enabled = False
        controller.write_storage = MagicMock()
        pipeline._aux_window_keys = MagicMock()

        pipeline.finish_backup_ack(7)

        pipeline._aux_window_keys.assert_not_called()
        controller.write_storage.assert_not_called()
        controller.mem_pool_host.free.assert_called_once_with(host)
        cache.storage_existence_cache.add.assert_not_called()
        self.assertEqual(pipeline.write_staged_tokens_, 0)
        self.assertTrue(pipeline.is_idle())

    def test_writable_staging_is_retained_until_storage_ack(self):
        controller = _controller()
        pipeline, cache = self._pipeline(controller)
        host = self._launch(pipeline, cache, controller)

        pipeline.finish_backup_ack(7)

        operation = controller.backup_queue.get_nowait()
        self.assertIs(operation.host_indices, host)
        self.assertEqual(operation.hash_value, ["a", "b"])
        cache.dec_lock_ref.assert_called_once()
        controller.mem_pool_host.free.assert_not_called()
        cache.storage_existence_cache.add.assert_not_called()
        self.assertIn(operation.id, pipeline.ongoing_backup)
        self.assertEqual(pipeline.write_staged_tokens_, 4)
        pipeline.finish_storage_write_ack(operation.id)
        controller.mem_pool_host.free.assert_called_once_with(host)
        cache.storage_existence_cache.add.assert_called_once_with(
            PoolName.KV, ["a", "b"]
        )
        self.assertEqual(pipeline.write_staged_tokens_, 0)
        self.assertEqual(pipeline.inflight_backup_hashes, {})
        self.assertTrue(pipeline.is_idle())


if __name__ == "__main__":
    unittest.main()
