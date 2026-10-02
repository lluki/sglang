# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Correctness-checked synthetic CPU KV/file microbenchmark (not model TTFT)."""

import argparse
import json
import math
import os
import statistics
import tempfile
import time
import uuid

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory


class SyntheticMhaPool:
    """CPU pool matching SGLang's MHA page descriptor and serialization order."""

    def __init__(self, pages, page_size, layers, heads, head_dim, layout):
        self.page_size, self.layer_num = page_size, layers
        self.head_num, self.head_dim = heads, head_dim
        self.size = pages * page_size
        self.layout, self.dtype = layout, torch.bfloat16
        if layout == "layer_first":
            shape = (2, layers, self.size, heads, head_dim)
        elif layout == "page_first":
            shape = (2, self.size, layers, heads, head_dim)
        elif layout == "page_head":
            shape = (2, pages, heads, page_size, layers, head_dim)
        else:
            shape = (2, pages, layers, page_size, heads, head_dim)
        self.kv_buffer = torch.empty(shape, dtype=self.dtype)

    def get_page_buffer_meta(self, indices):
        pointers, lengths = [], []
        for start in indices.tolist()[:: self.page_size]:
            if self.layout == "layer_first":
                parts = [
                    self.kv_buffer[kv, layer, start : start + self.page_size]
                    for layer in range(self.layer_num)
                    for kv in range(2)
                ]
            elif self.layout == "page_first":
                parts = [
                    self.kv_buffer[kv, start : start + self.page_size]
                    for kv in range(2)
                ]
            else:
                parts = [self.kv_buffer[kv, start // self.page_size] for kv in range(2)]
            for part in parts:
                pointers.append(part.data_ptr())
                lengths.append(part.numel() * part.element_size())
        return pointers, lengths


def delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def measure(backend, call, page_count):
    before = backend.get_stats()
    start = time.perf_counter_ns()
    result = call()
    elapsed = time.perf_counter_ns() - start
    counters = delta(before, backend.get_stats())
    return result, {
        "wall_ns": elapsed,
        "wall_us_per_page": elapsed / page_count / 1000,
        "native_counters": {
            key.removeprefix("native_"): value
            for key, value in counters.items()
            if key.startswith("native_")
        },
    }


def run(args):
    dimensions = (
        args.pages,
        args.rounds,
        args.page_size,
        args.layers,
        args.kv_heads,
        args.head_dim,
    )
    if any(value <= 0 for value in dimensions) or args.pages > 128:
        raise ValueError("dimensions and rounds must be positive; pages must be <=128")
    pool = SyntheticMhaPool(
        args.pages,
        args.page_size,
        args.layers,
        args.kv_heads,
        args.head_dim,
        args.layout,
    )
    page_bytes = pool.kv_buffer.numel() * pool.kv_buffer.element_size() // args.pages
    unit = 65536
    padded_page_bytes = math.ceil(page_bytes / unit) * unit
    capacity = 16 * 1024 * 1024 + padded_page_bytes * args.pages * args.rounds + unit
    identity = "bench-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(
        prefix="nixlshard-bench-", dir=args.directory
    ) as directory:
        storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=args.layout != "layer_first",
            model_name="synthetic-mha",
            extra_config={
                "model_revision": "synthetic-mha-v1",
                "agent": {
                    "name": identity,
                    "disks": [
                        {
                            "path": os.path.join(directory, "cache.bin"),
                            "capacity_bytes": capacity,
                            "unit_bytes": unit,
                            "metadata_bytes": 16 * 1024 * 1024,
                            "create": True,
                        }
                    ],
                    "listen_host": "127.0.0.1",
                    "listen_port": 0,
                    "staging_slot_bytes": padded_page_bytes,
                    "staging_slots": 4,
                    "workers": 2,
                    "max_inflight": 64,
                    "timeout_ms": args.timeout_ms,
                    "direct_io": args.direct_io,
                },
            },
        )
        backend = StorageBackendFactory.create_backend(
            "nixlshard", storage_config, pool
        )
        try:
            backend.register_mem_pool_host(pool)
            if backend.build_marker == "unknown":
                raise RuntimeError("rebuilt native NIXLShard build marker is missing")
            indices = torch.arange(pool.size)
            samples = []
            operations = (
                ("store", "exists", "load", "checkpoint")
                if args.checkpoint
                else ("store", "exists", "load")
            )
            for iteration in range(args.rounds):
                # New immutable keys every round ensure stores perform payload I/O.
                keys = [f"{identity}/{iteration}/{page}" for page in range(args.pages)]
                pool.kv_buffer.copy_(
                    torch.arange(pool.kv_buffer.numel()).reshape(pool.kv_buffer.shape)
                    % 97
                    + iteration
                )
                expected = pool.kv_buffer.clone()
                stored, store_cost = measure(
                    backend, lambda: backend.batch_set_v1(keys, indices), args.pages
                )
                if stored != [True] * args.pages:
                    raise RuntimeError(
                        f"store failure: {stored}; stats={backend.get_stats()}"
                    )
                found, exists_cost = measure(
                    backend, lambda: backend.batch_exists(keys), args.pages
                )
                if found != args.pages:
                    raise RuntimeError(f"incomplete hit prefix {found}/{args.pages}")
                pool.kv_buffer.zero_()
                loaded, load_cost = measure(
                    backend, lambda: backend.batch_get_v1(keys, indices), args.pages
                )
                if loaded != [True] * args.pages or not torch.equal(
                    pool.kv_buffer, expected
                ):
                    raise RuntimeError(
                        f"roundtrip failed: {loaded}; stats={backend.get_stats()}"
                    )
                samples.append(
                    {
                        "iteration": iteration,
                        "store": store_cost,
                        "exists": exists_cost,
                        "load": load_cost,
                    }
                )
                if args.checkpoint:
                    status, checkpoint_cost = measure(
                        backend, backend.agent.checkpoint, args.pages
                    )
                    if status != "success":
                        raise RuntimeError(f"checkpoint failure: {status}")
                    samples[-1]["checkpoint"] = checkpoint_cost
            result = {
                "measurement": "synthetic_cpu_kv_local_file_adapter_microbenchmark",
                "build_marker": backend.build_marker,
                "namespace": backend.namespace,
                "configuration": vars(args),
                "page_bytes": page_bytes,
                "file_capacity_bytes": capacity,
                "all_pages_verified": args.pages * args.rounds,
                "summary_mean_wall_us_per_page": {
                    operation: statistics.mean(
                        s[operation]["wall_us_per_page"] for s in samples
                    )
                    for operation in operations
                },
                "summary_native_mean_ns_per_page": {
                    operation: {
                        key: sum(
                            s[operation]["native_counters"].get(key, 0) for s in samples
                        )
                        / (args.pages * args.rounds)
                        for key in {
                            key
                            for s in samples
                            for key in s[operation]["native_counters"]
                            if key.endswith("_ns")
                        }
                    }
                    for operation in operations
                },
                "counters": backend.get_stats(),
                "samples": samples,
                "limitations": "Synthetic CPU KV and a disposable regular file; no model execution, no TTFT, no RDMA measurement. Native cost counters may overlap and must not be added as a latency decomposition.",
            }
        finally:
            backend.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument(
        "--layout",
        choices=["page_first", "page_first_direct", "page_head", "layer_first"],
        default="page_first",
    )
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Measure an explicit metadata checkpoint after each round",
    )
    parser.add_argument(
        "--directory",
        default="/raid/nixlshard-v2",
        help="Existing directory for a disposable debug file (default: /raid/nixlshard-v2)",
    )
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
