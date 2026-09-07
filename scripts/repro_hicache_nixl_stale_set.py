#!/usr/bin/env python3
"""Reproduce Full-KV host-slot reuse during an in-flight NIXL POSIX set."""

import argparse
import os
import sys
import tempfile
import time
from array import array
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-core", choices=("python", "rust"), default="python")
    parser.add_argument("--set-delay", type=float, default=3.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tests = root / "test/registered/unit/mem_cache"
    sys.path.insert(0, str(tests))
    os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = args.tree_core
    os.environ["SGLANG_HICACHE_NIXL_BACKEND_PLUGIN"] = "POSIX"
    os.environ["SGLANG_HICACHE_NIXL_USE_DIRECT_IO"] = "0"
    os.environ["SGLANG_HICACHE_NIXL_SET_DELAY_S"] = str(args.set_delay)

    storage_dir = tempfile.TemporaryDirectory(prefix="sglang-nixl-stale-")
    os.environ["SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR"] = storage_dir.name

    import torch
    import test_unified_radix_cache_unittest as fixtures
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.runtime_context import get_parallel

    case = fixtures.Test_FULL_ps1("test_insert_and_match_basic")
    parallel = get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
    parallel.__enter__()
    try:
        cache, allocator, req_pool = fixtures.build_fixture(fixtures.CacheConfig())
        case._init_hicache(
            cache,
            write_policy="write_back",
            storage_backend="nixl",
            prefetch_threshold=1,
        )

        seq = [1, 2, 3, 4]
        case._insert(cache, allocator, req_pool, seq)
        leaf = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq)))
        ).last_device_node
        device_indices = fixtures._device_value(cache, leaf, ComponentType.FULL)
        case._fill_full_kv(allocator, device_indices, marker=7)

        # D->H completion automatically queues the write-back L2->L3 set.
        case._backup_node(cache, leaf)
        deadline = time.monotonic() + 5
        while not cache.ongoing_backup and time.monotonic() < deadline:
            time.sleep(0.001)
        assert cache.ongoing_backup, "storage backup was not queued"

        host_group = cache.cache_controller.mem_pool_host
        host_pool = host_group.anchor_entry.host_pool
        in_flight = fixtures._host_value(cache, leaf, ComponentType.FULL).clone()
        hashes = list(cache.tree_core.get_hash_values(leaf))
        expected_first = host_pool.get_data_page(in_flight[0]).clone()

        # The new request shares two pages and diverges, splitting [1,2,3,4].
        case._insert(cache, allocator, req_pool, [1, 2, 9, 10])
        split_parent = fixtures._node_parent(cache, leaf)
        exposed = fixtures._host_value(cache, split_parent, ComponentType.FULL).clone()
        evicted_while_in_flight = cache.evict_host(len(exposed))
        reused = None
        if evicted_while_in_flight:
            assert evicted_while_in_flight == len(exposed), (
                evicted_while_in_flight,
                exposed.tolist(),
            )
            print("reclaimed in-flight slots:", exposed.tolist())
            reused = host_group.alloc(host_group.available_size())
            assert set(exposed.tolist()).issubset(set(reused.tolist()))
            poison = host_pool.get_dummy_flat_data_page().fill_(99)
            for index in exposed:
                host_pool.set_from_flat_data_page(index, poison)
        else:
            print("split prefix remained pinned while SET was in flight")

        deadline = time.monotonic() + args.set_delay + 10
        while cache.ongoing_backup and time.monotonic() < deadline:
            cache.drain_storage_control_queues()
            time.sleep(0.01)
        assert not cache.ongoing_backup, "NIXL backup did not finish"

        if not evicted_while_in_flight:
            evicted_after_ack = cache.evict_host(len(exposed))
            assert evicted_after_ack == len(exposed), (
                evicted_after_ack,
                exposed.tolist(),
            )
            print("split prefix became evictable after SET ACK")
            # Reuse and overwrite only after NIXL has stopped reading the slots.
            reused = host_group.alloc(host_group.available_size())
            assert set(exposed.tolist()).issubset(set(reused.tolist()))
            poison = host_pool.get_dummy_flat_data_page().fill_(99)
            for index in exposed:
                host_pool.set_from_flat_data_page(index, poison)

        assert reused is not None
        host_group.free(reused)
        target = host_group.alloc(len(seq))
        results = cache.cache_controller.storage_backend.batch_get_v1(hashes, target)
        assert all(results), results
        stored_first = host_pool.get_data_page(target[0]).clone()

        print(
            "expected first page unique values:", torch.unique(expected_first).tolist()
        )
        print("stored first page unique values:  ", torch.unique(stored_first).tolist())
        if evicted_while_in_flight:
            assert not torch.equal(stored_first, expected_first)
            assert torch.all(stored_first == 99)
            print("REPRODUCED: NIXL stored data read from a reclaimed host slot")
        else:
            assert torch.equal(stored_first, expected_first)
            print("FIX VERIFIED: NIXL stored the original pinned data")
    finally:
        case.doCleanups()
        parallel.__exit__(None, None, None)
        storage_dir.cleanup()


if __name__ == "__main__":
    main()
