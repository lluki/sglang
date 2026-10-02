# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Segmented KV pages in NIXLShard's local SSD and remote-owner cache.

Native poll completion and release must quiesce I/O before returning. Python
does not impose a second timeout that could release a late RDMA destination.
"""

import hashlib
import json
import logging
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from functools import lru_cache

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    PoolName,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _metrics_snapshot_class():
    # Metrics imports pull in model/quantization dependencies. Keep them optional
    # for standalone CPU/native benchmarks with framework metrics disabled.
    from sglang.srt.observability.metrics_collector import StorageMetrics

    class NixlShardStats(dict, StorageMetrics):
        """Framework histogram samples plus cumulative JSON counters."""

        def __init__(self, counters, samples):
            dict.__init__(self, counters)
            StorageMetrics.__init__(self, **samples)

    return NixlShardStats


class HiCacheNixlShard(HiCacheStorage):
    """MHA KV adapter. Auxiliary/hybrid pools are explicitly unsupported."""

    def __init__(self, storage_config):
        extra = storage_config.extra_config or {}
        revision = extra.get("model_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("nixlshard requires an immutable model_revision")
        config = extra.get("agent")
        if not isinstance(config, dict):
            raise ValueError("nixlshard requires an agent configuration dictionary")
        if storage_config.is_mla_model or storage_config.should_split_heads:
            raise NotImplementedError(
                "nixlshard initially supports unsplit MHA KV only"
            )
        import nixlshard

        self.storage_config = storage_config
        self.model_revision = revision
        self.agent = nixlshard.Agent(dict(config))
        self.build_marker = getattr(nixlshard, "__build_marker__", "unknown")
        self._condition = threading.Condition()
        self._active = 0
        self._closing = False
        self._closed = False
        self._close_error = None
        self._registrations = []
        self._host_buffer = None
        self._stats = Counter()
        self._metric_samples = {
            name: deque(maxlen=1024)
            for name in (
                "prefetch_pgs",
                "backup_pgs",
                "prefetch_bandwidth",
                "backup_bandwidth",
            )
        }
        self.mem_pool_host = None

    def register_mem_pool_host(self, pool):
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("nixlshard is closed")
            if self.mem_pool_host is not None:
                if self.mem_pool_host is pool:
                    return
                raise RuntimeError("detach nixlshard before replacing the host pool")
            if pool.layout not in (
                "page_first",
                "page_first_direct",
                "page_head",
                "layer_first",
            ):
                raise NotImplementedError(f"unsupported KV layout: {pool.layout}")
            kv = getattr(pool, "kv_buffer", None)
            if not isinstance(kv, torch.Tensor) or not kv.is_contiguous():
                raise NotImplementedError(
                    "nixlshard requires a contiguous MHA kv_buffer"
                )
            if kv.device.type != "cpu" or kv.shape[0] != 2:
                raise NotImplementedError(
                    "nixlshard requires CPU K and V in one buffer"
                )
            if pool.page_size <= 0 or pool.size < pool.page_size:
                raise ValueError("host pool must contain at least one complete page")
            sample = torch.arange(pool.page_size, dtype=torch.int64)
            ptrs, lengths = pool.get_page_buffer_meta(sample)
            lengths = [int(n) for n in lengths]
            expected_segments = (
                2 * pool.layer_num if pool.layout == "layer_first" else 2
            )
            if len(ptrs) != expected_segments or len(lengths) != expected_segments:
                raise ValueError("incomplete MHA page descriptors")
            self._base = kv.data_ptr()
            self._buffer_bytes = kv.numel() * kv.element_size()
            self._segment_lengths = lengths
            self._page_bytes = sum(lengths)
            if any(n <= 0 for n in lengths):
                raise ValueError("empty page segment")
            self._validate_ranges(ptrs, lengths)
            schema = {
                "version": "nixlshard-mha-segments-v1",
                "model": self.storage_config.model_name,
                "revision": self.model_revision,
                "dtype": str(pool.dtype),
                "page_size": pool.page_size,
                "layout": pool.layout,
                "layers": pool.layer_num,
                "heads": pool.head_num,
                "head_dim": pool.head_dim,
                "segments": lengths,
                "object_bytes": self._page_bytes,
                "ranks": {
                    name: getattr(self.storage_config, name)
                    for name in (
                        "tp_rank",
                        "tp_size",
                        "pp_rank",
                        "pp_size",
                        "attn_cp_rank",
                        "attn_cp_size",
                        "dp_rank",
                    )
                },
            }
            self.namespace = hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            self._token = self.agent.register_memory(self._base, self._buffer_bytes)
            self._registrations.append(self._token)
            self._host_buffer = kv
            # Native runtime packs ordered segments into bounded, preregistered
            # aligned slots. Alignment/noncontiguity never requires pool.flatten()
            # or an additional Python staging allocation per page/direction.
            self.mem_pool_host = pool
            logger.info(
                "HiCacheNixlShard native build=%s namespace=%s layout=%s bytes/page=%d",
                self.build_marker,
                self.namespace,
                pool.layout,
                self._page_bytes,
            )

    def _validate_ranges(self, ptrs, lengths):
        for ptr, length in zip(ptrs, lengths):
            if (
                ptr < self._base
                or length <= 0
                or ptr + length > self._base + self._buffer_bytes
            ):
                raise ValueError("page segment outside registered host memory")

    @contextmanager
    def _operation(self):
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("nixlshard is closed")
            if self.mem_pool_host is None:
                raise RuntimeError("register_mem_pool_host must be called first")
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _keys(self, keys):
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("cache keys must be nonempty strings")
        return [f"{self.namespace}:{key}" for key in keys]

    @staticmethod
    def _hints(extra_info, count):
        hints = (extra_info.extra_info or {}).get("owner_hints") if extra_info else None
        if hints is not None and (
            len(hints) != count or any(not isinstance(h, str) for h in hints)
        ):
            raise ValueError("owner_hints must contain one string per page")
        return hints

    def _page_segments(self, keys, indices):
        pool = self.mem_pool_host
        if (
            indices is None
            or indices.ndim != 1
            or indices.numel() != len(keys) * pool.page_size
        ):
            raise ValueError("host indices must contain one complete page per key")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("host indices must be integer token offsets")
        rows = indices.to(device="cpu", dtype=torch.int64).reshape(-1, pool.page_size)
        for row in rows:
            start = int(row[0])
            if (
                start < 0
                or start % pool.page_size
                or start + pool.page_size > pool.size
            ):
                raise ValueError("page start is unaligned or out of bounds")
            if not torch.equal(row, torch.arange(start, start + pool.page_size)):
                raise ValueError("tokens within each host page must be consecutive")
        ptrs, lengths = pool.get_page_buffer_meta(indices.cpu())
        stride = len(self._segment_lengths)
        if len(ptrs) != len(keys) * stride or list(
            lengths
        ) != self._segment_lengths * len(keys):
            raise ValueError("page segment descriptors changed after registration")
        self._validate_ranges(ptrs, lengths)
        return [
            list(zip(ptrs[i : i + stride], lengths[i : i + stride]))
            for i in range(0, len(ptrs), stride)
        ]

    def _wait(self, handle, count):
        try:
            while True:
                statuses = self.agent.poll(handle)
                if statuses is not None:
                    if len(statuses) != count:
                        raise RuntimeError(
                            "nixlshard returned an incomplete batch result"
                        )
                    with self._condition:
                        self._stats.update(statuses)
                    return [status == "success" for status in statuses]
                # Native progress does not depend on Python's GIL. Sleeping here
                # allows the other storage worker and controller to run.
                time.sleep(0.0005)
        finally:
            # Also drains on an exceptional poll; registered buffers remain owned.
            self.agent.release(handle)

    def _transfer(self, keys, host_indices, direction, extra_info):
        if not keys:
            if host_indices is not None and host_indices.numel():
                raise ValueError("empty key batch has nonempty host indices")
            return []
        with self._operation():
            start = time.perf_counter()
            namespaced = self._keys(keys)
            pages = self._page_segments(keys, host_indices)
            hints = self._hints(extra_info, len(keys))
            results = []
            for first in range(0, len(keys), STORAGE_BATCH_SIZE):
                batch = pages[first : first + STORAGE_BATCH_SIZE]
                items = []
                for i, segments in enumerate(batch):
                    descriptors = [
                        (self._token, ptr - self._base, length)
                        for ptr, length in segments
                    ]
                    item = {"key": namespaced[first + i], "segments": descriptors}
                    if hints is not None and direction == "get":
                        item["hint"] = hints[first + i]
                    items.append(item)
                submit = (
                    self.agent.batch_load
                    if direction == "get"
                    else self.agent.batch_store
                )
                completed = self._wait(submit(items), len(batch))
                results.extend(completed)
            with self._condition:
                self._stats[f"{direction}_pages"] += len(results)
                self._stats[f"{direction}_hits"] += sum(results)
                self._stats[f"{direction}_bytes"] += self._page_bytes * sum(results)
                if self.storage_config.enable_storage_metrics:
                    prefix = "prefetch" if direction == "get" else "backup"
                    self._metric_samples[f"{prefix}_pgs"].append(sum(results))
                    bandwidth = (
                        self._page_bytes
                        * sum(results)
                        / max(time.perf_counter() - start, 1e-9)
                        / (1024**3)
                    )
                    self._metric_samples[f"{prefix}_bandwidth"].append(bandwidth)
            return results

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self._transfer(keys, host_indices, "get", extra_info)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self._transfer(keys, host_indices, "set", extra_info)

    def exists(self, key):
        return self.batch_exists([key]) == 1

    def batch_exists(self, keys, extra_info=None):
        if not keys:
            return 0
        with self._operation():
            found = self.agent.batch_exists(
                self._keys(keys), self._hints(extra_info, len(keys))
            )
            if len(found) != len(keys):
                raise RuntimeError("nixlshard returned an incomplete exists result")
            return found.index(False) if False in found else len(found)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        if host_pool_name != PoolName.KV or host_pool is not self.mem_pool_host:
            raise NotImplementedError("nixlshard auxiliary pools are not supported")

    @staticmethod
    def _require_kv(transfers):
        if any(t.name != PoolName.KV for t in transfers):
            raise NotImplementedError("nixlshard auxiliary pools are not supported")
        if len(transfers) > 1:
            raise ValueError("duplicate KV pool transfers")

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        self._require_kv(pool_transfers or [])
        return PoolTransferResult(self.batch_exists(keys, extra_info), {})

    def batch_get_v2(self, transfers, extra_info=None):
        self._require_kv(transfers)
        return {
            t.name: self.batch_get_v1(t.keys or [], t.host_indices, extra_info)
            for t in transfers
        }

    def batch_set_v2(self, transfers, extra_info=None):
        self._require_kv(transfers)
        return {
            t.name: self.batch_set_v1(t.keys or [], t.host_indices, extra_info)
            for t in transfers
        }

    def get(self, *args, **kwargs):
        raise NotImplementedError("use batch_get_v1 with registered host page indices")

    def batch_get(self, *args, **kwargs):
        raise NotImplementedError("use batch_get_v1 with registered host page indices")

    def set(self, *args, **kwargs):
        raise NotImplementedError("use batch_set_v1 with registered host page indices")

    def batch_set(self, *args, **kwargs):
        raise NotImplementedError("use batch_set_v1 with registered host page indices")

    def clear(self):
        raise NotImplementedError("clearing shared SSD cache is not supported")

    def get_stats(self):
        with self._condition:
            stats = dict(self._stats)
            samples = {
                name: list(values) for name, values in self._metric_samples.items()
            }
            for values in self._metric_samples.values():
                values.clear()
        # Prefix native counters to preserve the adapter's page-level counts.
        # Native counters include private staging, POSIX, and peer-control costs.
        if hasattr(self.agent, "stats"):
            stats.update(
                {f"native_{key}": value for key, value in self.agent.stats().items()}
            )
        if not self.storage_config.enable_storage_metrics:
            return stats
        return _metrics_snapshot_class()(stats, samples)

    def close(self):
        with self._condition:
            if self._closed:
                return
            if self._closing:
                self._condition.wait_for(
                    lambda: self._closed or self._close_error is not None
                )
                if self._close_error is not None:
                    raise RuntimeError(
                        "nixlshard shutdown failed"
                    ) from self._close_error
                return
            self._closing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._active == 0)
        # Native close drains all I/O before deregistration. Retain tensor owners
        # through this call; active adapter operations have already released handles.
        try:
            self.agent.close()
        except BaseException as error:
            with self._condition:
                # Preserve registrations and tensor ownership if native shutdown
                # did not establish quiescence. Wake concurrent close callers.
                self._close_error = error
                self._condition.notify_all()
            raise
        with self._condition:
            self._registrations.clear()
            self._host_buffer = None
            self.mem_pool_host = None
            self._closed = True
            self._condition.notify_all()
