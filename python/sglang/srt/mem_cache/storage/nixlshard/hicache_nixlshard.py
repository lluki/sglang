# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Whole-page NIXLShard storage with exact framework keys and pool namespaces."""

import copy
import ctypes
import logging
import os
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    PoolHitPolicy,
    PoolName,
    PoolTransferResult,
)
from sglang.srt.observability import request_timeline

logger = logging.getLogger(__name__)


class HiCacheNixlShard(HiCacheStorage):
    """One Agent, the supplied SSD set, and one whole value per logical page.

    SGLang's normal controller uses the retained v1 page interface. Its hybrid
    controller registers additional pools and explicitly calls the v2 interface.
    The framework must salt page hashes for incompatible models/ranks/layouts;
    this adapter passes those complete ASCII keys unchanged.
    """

    implementation_marker = "nixlshard-public-api-whole-pages-v1"

    def __init__(self, storage_config):
        extra = storage_config.extra_config or {}
        revision = extra.get("model_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("nixlshard requires an immutable model_revision")
        config = extra.get("agent")
        if not isinstance(config, dict):
            raise ValueError("nixlshard requires an agent configuration dictionary")
        import nixlshard

        self.storage_config = storage_config
        self.model_revision = revision
        # Provider/callback objects may own locks and must retain their identity.
        self.native_config = {
            key: value if callable(value) else copy.deepcopy(value)
            for key, value in config.items()
        }
        self.write_enabled = bool(self.native_config.get("disks"))
        self.numa_node = config.get("numa_node", 0)
        self.direct_receive = config.get("direct_receive", False)
        self.build_marker = getattr(nixlshard, "__build_marker__", "unknown")
        self._condition = threading.Condition()
        self._active = 0
        self._busy = {}
        self._closing = False
        self._closed = False
        self._registrations = {}
        self.registered_pools = {}
        self.mem_pool_host = None
        self._stats = Counter()
        self._metric_samples = {
            name: deque(maxlen=1024)
            for name in ("prefetch_pgs", "backup_pgs", "prefetch_bandwidth", "backup_bandwidth")
        }
        self._native_exporter = None
        self._export_native_metrics = bool(extra.get("export_native_metrics", False))
        self._diagnostic_logging = bool(extra.get("diagnostic_logging", False))
        self._timeout = extra.get("transfer_timeout_seconds")
        if self._timeout is not None and (
            not isinstance(self._timeout, (int, float)) or self._timeout <= 0
        ):
            raise ValueError("transfer_timeout_seconds must be positive")
        self.agent = nixlshard.Agent(self.native_config)
        try:
            for owner, endpoint in extra.get("peers", {}).items():
                self.agent.add_peer(owner, endpoint)
        except BaseException:
            self.agent.close()
            raise

    @staticmethod
    def _keys(keys):
        result = []
        for key in keys:
            if not isinstance(key, (str, bytes)):
                raise TypeError("page keys must be ASCII strings or bytes")
            try:
                encoded = key.encode("ascii") if isinstance(key, str) else key
                encoded.decode("ascii")
            except UnicodeError as error:
                raise ValueError("page keys must be ASCII") from error
            if not 1 <= len(encoded) <= 256:
                raise ValueError("page keys must contain 1-256 ASCII bytes")
            result.append(encoded)
        return result

    @staticmethod
    def _hints(extra_info, count, namespace):
        values = (extra_info.extra_info or {}) if extra_info else {}
        if "owner_hints" in values:
            raise ValueError("use location_hints lists instead of owner_hints")
        hints = values.get("location_hints")
        if isinstance(hints, dict):
            hints = hints.get(namespace)
        if hints is None:
            return [[] for _ in range(count)]
        if (
            not isinstance(hints, list) or len(hints) != count
            or any(
                not isinstance(owners, list)
                or any(not isinstance(owner, str) or not owner for owner in owners)
                for owners in hints
            )
        ):
            raise ValueError("location_hints must contain one list of owner strings per page")
        return [list(owners) for owners in hints]

    @staticmethod
    def _tensor_buffers(pool):
        # Only direct host allocations are inspected; never recurse into GPU
        # device pools. Select the allocations actually covering sample pages.
        for value in vars(pool).values():
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (item for item in value if isinstance(item, torch.Tensor))

    def register_mem_pool_host(self, pool):
        self.register_mem_host_pool_v2(pool, PoolName.KV)
        self.mem_pool_host = pool

    def register_mem_host_pool_v2(self, pool, name):
        namespace = str(name)
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("nixlshard is closed")
            if namespace in self.registered_pools:
                if self.registered_pools[namespace] is pool:
                    return
                raise RuntimeError("detach nixlshard before replacing a host pool")
            if pool.page_size <= 0 or pool.size < pool.page_size:
                raise ValueError("host pool must contain at least one complete page")
            sample = torch.arange(pool.page_size, dtype=torch.int64)
            meta = pool.get_page_buffer_meta(sample)
            if meta is None:
                # The SGLang logical KV anchor has no stored value. Its required
                # physical sidecars establish usable prefix boundaries.
                if getattr(pool, "kv_buffer", None) is not None:
                    raise ValueError("host pool returned no page descriptors")
                self.registered_pools[namespace] = pool
                return
            pointers, lengths = meta
            ranges = [(int(p), int(n)) for p, n in zip(pointers, lengths)]
            if len(pointers) != len(lengths) or not ranges or any(n <= 0 for _, n in ranges):
                raise ValueError("incomplete host page descriptors")
            owned = []
            for tensor in self._tensor_buffers(pool):
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    continue
                base, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
                if size and any(base <= pointer and pointer + length <= base + size for pointer, length in ranges):
                    owned.append((base, size, tensor))
            for pointer, length in ranges:
                if not any(base <= pointer and pointer + length <= base + size for base, size, _ in owned):
                    raise ValueError("page buffer is not covered by a contiguous CPU allocation")
            newly_registered = []
            try:
                for base, size, tensor in owned:
                    if (base, size) in self._registrations:
                        continue
                    self.agent.register_memory(
                        ctypes.c_void_p(base), size, 0, "", numa_node=self.numa_node
                    )
                    self._registrations[(base, size)] = tensor
                    newly_registered.append((base, size))
            except BaseException:
                for base, size in reversed(newly_registered):
                    self.agent.deregister_memory(ctypes.c_void_p(base), size, 0)
                    del self._registrations[(base, size)]
                raise
            self.registered_pools[namespace] = pool
            logger.info(
                "HiCacheNixlShard implementation=%s native=%s pool=%s bytes/page=%d numa=%d "
                "page_components=%d contiguous_page=%s registered_allocations=%d alignment_mod4096=%d",
                self.implementation_marker, self.build_marker, namespace,
                sum(lengths), self.numa_node, len(ranges),
                all(ranges[i][0] + ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1)),
                len(owned), ranges[0][0] % 4096,
            )

    @contextmanager
    def _operation(self):
        operation = object()
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError("nixlshard is closed")
            if self.mem_pool_host is None:
                raise RuntimeError("register_mem_pool_host must be called first")
            self._active += 1
        try:
            yield operation
        finally:
            with self._condition:
                self._busy.pop(operation, None)
                self._active -= 1
                self._condition.notify_all()

    def _reserve_ranges(self, operation, pages, direction):
        ranges = [(pointer, pointer + length) for page in pages for pointer, length in page]
        if direction == "get":
            ordered = sorted(ranges)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("load page destinations overlap")
        with self._condition:
            for existing_direction, existing in self._busy.values():
                if direction == "get" or existing_direction == "get":
                    if any(a < d and c < b for a, b in ranges for c, d in existing):
                        raise ValueError("page memory overlaps an active prefetch or backup")
            self._busy[operation] = (direction, ranges)

    def _page_ranges(self, pool, keys, indices):
        if (
            indices is None or indices.ndim != 1
            or indices.numel() != len(keys) * pool.page_size
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("host indices must contain one complete integer-indexed page per key")
        rows = indices.to(device="cpu", dtype=torch.int64).reshape(-1, pool.page_size)
        pages = []
        for row in rows:
            first = int(row[0])
            if (
                first < 0 or first % pool.page_size or first + pool.page_size > pool.size
                or not torch.equal(row, torch.arange(first, first + pool.page_size))
            ):
                raise ValueError("host page indices must be aligned, consecutive and in bounds")
            meta = pool.get_page_buffer_meta(row)
            if meta is None:
                pages.append([])
                continue
            pointers, lengths = meta
            if len(pointers) != len(lengths) or not pointers:
                raise ValueError("incomplete host page descriptors")
            ranges = [(int(p), int(n)) for p, n in zip(pointers, lengths)]
            for pointer, length in ranges:
                if length <= 0 or not any(
                    base <= pointer and pointer + length <= base + size
                    for base, size in self._registrations
                ):
                    raise ValueError("host page is outside registered memory")
            pages.append(ranges)
        return pages

    def _wait(self, handle, items, extra_info, direction=None, batch_id=None):
        values = (extra_info.extra_info or {}) if extra_info else {}
        deadline = values.get("deadline_monotonic")
        if deadline is None and self._timeout is not None:
            deadline = time.monotonic() + self._timeout
        canceled = False
        polling_error = None
        while True:
            try:
                status = self.agent.poll(handle)
                if status not in ("IN_PROG", "SUCCESS", "NOT_FOUND", "ERROR"):
                    raise RuntimeError(f"unexpected NIXLShard status: {status}")
            except BaseException as error:
                # Preserve all allocations and leases while draining even if a
                # client interrupt or transient polling error occurs.
                polling_error = polling_error or error
                if not canceled:
                    self.agent.cancel(handle)
                    canceled = True
                time.sleep(0.0005)
                continue
            if status != "IN_PROG":
                break
            if deadline is not None and time.monotonic() >= deadline and not canceled:
                self.agent.cancel(handle)
                canceled = True
            time.sleep(0.0005)
        try:
            if polling_error is not None:
                raise polling_error
            errors = self.agent.get_errors(handle) if status != "SUCCESS" else {}
            results = [(item["namespace"], item["key"]) not in errors for item in items]
            request_id = values.get("request_id")
            if request_id and self.native_config.get("enable_trace"):
                try:
                    request_timeline.emit(
                        request_id, "native_batch", batch_handle=handle,
                        native_batch_id=batch_id, operation="GET" if direction == "get" else "PUT" if direction == "set" else None,
                        build_marker=self.build_marker, terminal_observed=True,
                        events=self.agent.trace(handle),
                    )
                except Exception as error:
                    logger.exception("NIXLShard trace collection failed")
                    request_timeline.emit(
                        request_id, "native_trace_error", batch_handle=handle,
                        error_type=type(error).__name__,
                    )
            return results
        finally:
            self.agent.release(handle)

    def _transfer(self, keys, indices, direction, extra_info, name=PoolName.KV):
        namespace = str(name)
        native_keys = self._keys(keys)
        if direction == "set" and native_keys and not self.write_enabled:
            raise RuntimeError("NIXLShard storage writes require an assigned local SSD")
        if not keys:
            if indices is not None and indices.numel():
                raise ValueError("empty keys have nonempty host indices")
            return []
        with self._operation() as operation:
            pool = self.registered_pools[namespace]
            pages = self._page_ranges(pool, keys, indices)
            self._reserve_ranges(operation, pages, direction)
            hints = self._hints(extra_info, len(keys), namespace)
            results = []
            started = time.perf_counter()
            tracing = self._diagnostic_logging or request_timeline.enabled()
            diagnostic_start_ns = time.monotonic_ns() if tracing else 0
            copies = Counter() if tracing else None
            request_id = ((extra_info.extra_info or {}) if extra_info else {}).get("request_id")
            transfer_id = f"{os.getpid()}:adapter:{diagnostic_start_ns}"
            for first in range(0, len(keys), min(STORAGE_BATCH_SIZE, 128)):
                batch = pages[first:first + min(STORAGE_BATCH_SIZE, 128)]
                begin = first * pool.page_size
                lease = (
                    pool.acquire_io_lease(indices[begin:begin + len(batch) * pool.page_size])
                    if callable(getattr(pool, "acquire_io_lease", None)) else None
                )
                packed = []
                items = []
                batch_id = f"{transfer_id}:batch:{first}"
                layouts = []
                try:
                    for index, ranges in enumerate(batch):
                        if not ranges:
                            continue
                        length = sum(n for _, n in ranges)
                        contiguous = all(
                            ranges[i][0] + ranges[i][1] == ranges[i + 1][0]
                            for i in range(len(ranges) - 1)
                        )
                        if contiguous:
                            pointer = ranges[0][0]
                        else:
                            allocation = ctypes.create_string_buffer(length)
                            pointer = ctypes.addressof(allocation)
                            self.agent.register_memory(
                                ctypes.c_void_p(pointer), length, 0, "",
                                numa_node=self.numa_node,
                            )
                            packed.append((index, pointer, length, allocation, ranges))
                            if direction == "set":
                                offset = 0
                                copied_at = time.monotonic_ns()
                                for source, size in ranges:
                                    ctypes.memmove(pointer + offset, source, size)
                                    offset += size
                                copied_end = time.monotonic_ns()
                                copy_ns = copied_end - copied_at
                                request_timeline.emit(request_id, "framework_pack", copied_at, copied_end,
                                                      category="framework_pack", operation="PUT", bytes=length,
                                                      native_batch_id=batch_id, object_index=len(items),
                                                      span_id=f"{batch_id}:pack:{len(items)}", parent_id=transfer_id)
                                with self._condition:
                                    self._stats["pack_bytes"] += length
                                    self._stats["pack_ns"] += copy_ns
                                if copies is not None:
                                    copies["pack_bytes"] += length
                                    copies["pack_ns"] += copy_ns
                        item = {
                            "key": native_keys[first + index], "namespace": namespace,
                            "buffer": (pointer, length, 0),
                        }
                        if direction == "get":
                            item["location_hints"] = hints[first + index]
                        items.append(item)
                        layouts.append({"object_index": len(items) - 1, "page_index": first + index,
                                        "bytes": length, "destination_kind": "framework_pool" if contiguous else "registered_packed_buffer",
                                        "framework_pool_direct_eligible": contiguous,
                                        "component_bytes": [size for _, size in ranges],
                                        "destination_alignment_mod4096": pointer % 4096,
                                        "component_alignment_mod4096": [p % 4096 for p, _ in ranges]})
                    if items:
                        submit = self.agent.batch_load if direction == "get" else self.agent.batch_store
                        handle = submit(items)
                        completed = self._wait(handle, items, extra_info, direction, batch_id)
                        if request_id and tracing:
                            request_timeline.emit(request_id, "receive_witness", native_batch_id=batch_id,
                                                  operation="GET" if direction == "get" else "PUT",
                                                  direct_receive=self.direct_receive,
                                                  objects=[dict(layout, successful=success) for layout, success in zip(layouts, completed)])
                        physical = iter(completed)
                        batch_results = [next(physical) if ranges else True for ranges in batch]
                    else:
                        batch_results = [True] * len(batch)
                    if direction == "get":
                        for index, pointer, length, allocation, ranges in packed:
                            if batch_results[index]:
                                offset = 0
                                copied_at = time.monotonic_ns()
                                for destination, size in ranges:
                                    ctypes.memmove(destination, pointer + offset, size)
                                    offset += size
                                copied_end = time.monotonic_ns()
                                copy_ns = copied_end - copied_at
                                object_index = next(layout["object_index"] for layout in layouts if layout["page_index"] == first + index)
                                request_timeline.emit(request_id, "framework_unpack", copied_at, copied_end,
                                                      category="framework_unpack", operation="GET", bytes=length,
                                                      native_batch_id=batch_id, object_index=object_index,
                                                      span_id=f"{batch_id}:unpack:{object_index}", parent_id=transfer_id)
                                with self._condition:
                                    self._stats["unpack_bytes"] += length
                                    self._stats["unpack_ns"] += copy_ns
                                if copies is not None:
                                    copies["unpack_bytes"] += length
                                    copies["unpack_ns"] += copy_ns
                    if copies is not None:
                        packed_indices = {index for index, *_ in packed}
                        for index, (ranges, hit) in enumerate(zip(batch, batch_results)):
                            if hit and ranges:
                                copies["physical_components"] += len(ranges)
                                copies["packed_buffer_bytes" if index in packed_indices else "framework_pool_bytes"] += sum(n for _, n in ranges)
                    results.extend(batch_results)
                finally:
                    # _wait observes terminal polling before release; all native
                    # references have drained before unregistering allocations.
                    for _, pointer, length, _, _ in reversed(packed):
                        self.agent.deregister_memory(ctypes.c_void_p(pointer), length, 0)
                    if lease is not None:
                        pool.release_io_lease(lease)
            elapsed = max(time.perf_counter() - started, 1e-9)
            with self._condition:
                self._stats[f"{direction}_pages"] += len(results)
                self._stats[f"{direction}_hits"] += sum(results)
                transferred = sum(sum(n for _, n in page) for page, hit in zip(pages, results) if hit)
                self._stats[f"{direction}_bytes"] += transferred
                prefix = "prefetch" if direction == "get" else "backup"
                self._metric_samples[f"{prefix}_pgs"].append(sum(results))
                self._metric_samples[f"{prefix}_bandwidth"].append(transferred / elapsed / 1024**3)
            if tracing:
                fields = {key: copies[key] for key in (
                    "physical_components", "framework_pool_bytes", "packed_buffer_bytes",
                    "pack_bytes", "pack_ns", "unpack_bytes", "unpack_ns",
                )}
                logger.info("HiCacheNixlShard transfer direction=%s pool=%s pages=%d hits=%d copies=%s",
                            direction, namespace, len(results), sum(results), fields)
                request_timeline.emit(request_id, "adapter_transfer", start_ns=diagnostic_start_ns,
                                      end_ns=time.monotonic_ns(), direction=direction, pool=namespace,
                                      span_id=transfer_id, envelope=True, operation="GET" if direction == "get" else "PUT",
                                      pages=len(results), hits=sum(results), **fields)
            return results

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self._transfer(keys, host_indices, "get", extra_info)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self._transfer(keys, host_indices, "set", extra_info)

    def _exists(self, keys, name, extra_info):
        namespace = str(name)
        native_keys = self._keys(keys)
        pool = self.registered_pools[namespace]
        if getattr(pool, "kv_buffer", False) is None:
            return [True] * len(keys)
        hints = self._hints(extra_info, len(keys), namespace)
        found = []
        for first in range(0, len(keys), 128):
            items = [
                {"key": key, "namespace": namespace, "location_hints": hints[first + i]}
                for i, key in enumerate(native_keys[first:first + 128])
            ]
            started = time.monotonic_ns()
            current = self.agent.batch_exists(items)
            request_id = ((extra_info.extra_info or {}) if extra_info else {}).get("request_id")
            request_timeline.emit(request_id, "key_location", started, time.monotonic_ns(),
                                  category="key_location", operation="EXISTS", objects=len(items),
                                  found=sum(current), pool=namespace,
                                  span_id=f"{os.getpid()}:exists:{started}")
            if len(current) != len(items):
                raise RuntimeError("incomplete NIXLShard existence result")
            found.extend(current)
        return found

    @staticmethod
    def _prefix(found):
        return found.index(False) if False in found else len(found)

    def batch_exists(self, keys, extra_info=None):
        with self._operation():
            return self._prefix(self._exists(keys, PoolName.KV, extra_info))

    def exists(self, key):
        return self.batch_exists([key]) == 1

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        with self._operation():
            kv_pages = self._prefix(self._exists(keys, PoolName.KV, extra_info))
            restorable = set(range(1, kv_pages + 1))
            hits = {PoolName.KV: kv_pages} if kv_pages else {}
            seen = set()
            for transfer in pool_transfers or []:
                namespace = str(transfer.name)
                if namespace in seen:
                    raise ValueError("duplicate pool transfers")
                seen.add(namespace)
                found = self._exists(keys, transfer.name, extra_info)[:kv_pages]
                if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                    valid = set(range(1, self._prefix(found) + 1))
                elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    trailing = max(1, len(transfer.keys or []))
                    valid = {
                        prefix for prefix in range(1, kv_pages + 1)
                        if all(found[max(0, prefix - trailing):prefix])
                    }
                else:
                    raise ValueError("unsupported pool hit policy")
                if valid:
                    hits[transfer.name] = max(valid)
                restorable &= valid
            points = sorted(restorable)
            return PoolTransferResult(points[-1] if points else 0, hits, points)

    def _transfer_v2(self, transfers, direction, extra_info):
        names = [str(transfer.name) for transfer in transfers]
        if len(set(names)) != len(names):
            raise ValueError("duplicate pool transfers")
        return {
            transfer.name: self._transfer(
                transfer.keys or [], transfer.host_indices, direction, extra_info, transfer.name
            )
            for transfer in transfers
        }

    def batch_get_v2(self, transfers, extra_info=None):
        return self._transfer_v2(transfers, "get", extra_info)

    def batch_set_v2(self, transfers, extra_info=None):
        return self._transfer_v2(transfers, "set", extra_info)

    def get(self, *args, **kwargs):
        raise NotImplementedError("use registered whole-page batch_get_v1/v2")

    def batch_get(self, *args, **kwargs):
        raise NotImplementedError("use registered whole-page batch_get_v1/v2")

    def set(self, *args, **kwargs):
        raise NotImplementedError("use registered whole-page batch_set_v1/v2")

    def batch_set(self, *args, **kwargs):
        raise NotImplementedError("use registered whole-page batch_set_v1/v2")

    def clear(self):
        raise NotImplementedError("clearing shared SSD caches is not supported")

    def get_stats(self):
        with self._condition:
            counters = dict(self._stats)
            samples = {key: list(values) for key, values in self._metric_samples.items()}
            for values in self._metric_samples.values():
                values.clear()
        native = self.agent.stats()
        counters.update({f"native_{key}": value for key, value in native.items()})
        if not self.storage_config.enable_storage_metrics:
            return counters
        from sglang.srt.observability.metrics_collector import StorageMetrics

        if self._export_native_metrics:
            from .native_metrics import NativeMetricsExporter
            with self._condition:
                if self._native_exporter is None:
                    self._native_exporter = NativeMetricsExporter(self.storage_config)
                self._native_exporter.observe(native)
        return StorageMetrics(**samples)

    def close(self):
        with self._condition:
            if self._closed:
                return
            if self._closing:
                self._condition.wait_for(lambda: not self._closing)
                if self._closed:
                    return
            self._closing = True
            self._condition.wait_for(lambda: self._active == 0)
        try:
            for base, size in list(self._registrations):
                self.agent.deregister_memory(ctypes.c_void_p(base), size, 0)
                del self._registrations[(base, size)]
            self.agent.close()
        except BaseException:
            with self._condition:
                self._closing = False
                self._condition.notify_all()
            raise
        with self._condition:
            self.registered_pools.clear()
            self.mem_pool_host = None
            self._closed = True
            self._closing = False
            self._condition.notify_all()
