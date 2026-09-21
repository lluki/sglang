"""SGLang HiCache adapter for NIXLShard.

The adapter deliberately uses SGLang's dynamic-backend entry point.  It does
not register a second built-in ``nixl`` backend.  ``ShardClient`` and this
adapter must live in the SGLang worker process: host-pool addresses are passed
directly to the client and remain valid only until the synchronous batch call
returns.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

_SUCCESS = {"OK", "SUCCESS", "SUCCEEDED", "DONE", "HIT", "FOUND", "TRUE"}
_ALREADY_PRESENT = {"ALREADY_PRESENT", "EXISTS", "EXISTING"}
_MISS = {"MISS", "NOT_FOUND", "NOTFOUND", "ABSENT", "FALSE"}


@dataclass(frozen=True)
class _PreparedTransfer:
    transfer: PoolTransfer
    keys: list[str]
    buffers: list[list[tuple[int, int]]]
    # Strong references make the raw addresses above valid for the duration of
    # the synchronous ShardClient call.
    host_pool: Any
    host_indices: torch.Tensor


def _b64(value: Any) -> str:
    encoded = base64.urlsafe_b64encode(str(value).encode("utf-8")).decode("ascii")
    return encoded.rstrip("=") or "_"


def _result_items(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    if isinstance(results, dict):
        for field in ("results", "items", "values"):
            value = results.get(field)
            if isinstance(value, (list, tuple)):
                return list(value)
    for field in ("results", "items", "values"):
        value = getattr(results, field, None)
        if isinstance(value, (list, tuple)):
            return list(value)
    return [results]


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _status_name(result: Any) -> str:
    if isinstance(result, bool):
        return "OK" if result else "MISS"
    raw = _field(result, "status", result)
    if isinstance(raw, Enum):
        raw = raw.name
    elif hasattr(raw, "name"):
        raw = raw.name
    token = str(raw).strip().upper()
    # Handles values rendered as ``Status.OK`` without importing the client enum.
    return token.rsplit(".", 1)[-1]


def _result_ok(result: Any, operation: str) -> bool:
    status = _status_name(result)
    if status in _SUCCESS:
        return True
    if status in _ALREADY_PRESENT:
        return operation in ("set", "exists")
    if status in _MISS:
        return False
    return False


class HiCacheNixlShard(HiCacheStorage):
    """Dynamic HiCache backend backed by a process-local ``ShardClient``.

    Production configuration is read from ``storage_config.extra_config``.
    Tests may inject a client through the second constructor argument.  The
    client contract is intentionally small::

        batch_exists(keys)
        batch_get(keys, destinations)
        batch_set(keys, sources)

    Sources and destinations contain one list of ``(address, size)`` pairs per
    logical page. Calls are batched and synchronous; each returned item has a
    ``status`` and may have a ``value`` for clients that return read data
    instead of writing directly.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.storage_config = storage_config
        self.registered_pools: dict[PoolName, Any] = {}

        config = dict(storage_config.extra_config or {})
        injected = dict(kwargs or {})
        if not config.get("interface_v1"):
            raise ValueError(
                "NIXLShard requires interface_v1=1 for the zero-copy HiCache API"
            )
        self._schema_version = str(config.get("key_schema_version", 1))
        self._model_revision = str(config.get("model_revision", ""))
        namespace = config.get("deployment_namespace") or self._model_revision
        if not namespace:
            raise ValueError(
                "NIXLShard requires deployment_namespace or model_revision "
                "to prevent cache aliases across deployments"
            )
        self._deployment_namespace = str(namespace)
        self._pool_formats: dict[PoolName, str] = {}
        self._metrics = {
            "exists_pages": 0,
            "exists_hits": 0,
            "get_pages": 0,
            "get_hits": 0,
            "set_pages": 0,
            "set_successes": 0,
        }
        self._client = injected.get("client")
        if self._client is None:
            self._client = self._create_client(config, injected)

    @staticmethod
    def _create_client(config: dict[str, Any], injected: dict[str, Any]) -> Any:
        client_config = dict(config.get("client_config") or {})
        client_config.update(injected.get("client_config") or {})
        client_factory = injected.get("client_factory") or config.get("client_factory")
        if client_factory is not None:
            if isinstance(client_factory, str):
                module_name, separator, function_name = client_factory.partition(":")
                if not separator:
                    raise ValueError("client_factory must use 'module:function' syntax")
                client_factory = getattr(
                    importlib.import_module(module_name), function_name
                )
            return client_factory(client_config)

        client_cls = injected.get("client_class")
        client_config_cls = injected.get("client_config_class")
        if client_cls is None:
            try:
                from nixl_shard import ClientConfig, ShardClient
            except ImportError as exc:
                try:
                    from nixl_shard.client import ClientConfig, ShardClient
                except ImportError:
                    raise ImportError(
                        "NIXLShard HiCache requires nixl_shard.ShardClient in the "
                        "SGLang worker environment"
                    ) from exc
            client_cls = ShardClient
            client_config_cls = ClientConfig
        if client_config_cls is not None:
            return client_cls(client_config_cls(**client_config))
        return client_cls(**client_config)

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        super().register_mem_pool_host(mem_pool_host)
        self.registered_pools[PoolName.KV] = mem_pool_host
        self._probe_component_count(PoolName.KV, mem_pool_host)

    def register_mem_host_pool_v2(
        self, host_pool: Any, host_pool_name: PoolName
    ) -> None:
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._probe_component_count(host_pool_name, host_pool)

    def _probe_component_count(self, pool_name: PoolName, host_pool: Any) -> int:
        get_meta = getattr(host_pool, "get_page_buffer_meta", None)
        if get_meta is None:
            raise ValueError(
                f"NIXLShard pool {pool_name} does not expose get_page_buffer_meta; "
                "a zero-copy-capable host-pool layout is required"
            )
        page_size = int(getattr(host_pool, "page_size", 1) or 1)
        try:
            ptrs, sizes = get_meta(torch.arange(page_size, dtype=torch.int64))
        except Exception as exc:
            layout = getattr(host_pool, "layout", "unknown")
            raise ValueError(
                f"NIXLShard cannot address pool {pool_name} with layout {layout!r}"
            ) from exc
        if not ptrs or len(ptrs) != len(sizes):
            raise ValueError(
                f"NIXLShard pool {pool_name} returned invalid buffer metadata"
            )

        format_fields = {
            "class": f"{type(host_pool).__module__}.{type(host_pool).__qualname__}",
            "component_sizes": tuple(int(size) for size in sizes),
        }
        for name in (
            "layout",
            "page_size",
            "dtype",
            "kv_cache_dtype",
            "layer_num",
            "head_num",
            "head_dim",
            "kv_cache_dim",
            "num_mamba_layers",
            "temporal_dtype",
            "temporal_state_elem_size",
            "conv_state_shapes",
            "split_factor",
            "tp_lcm",
        ):
            if hasattr(host_pool, name):
                format_fields[name] = str(getattr(host_pool, name))
        encoded = repr(sorted(format_fields.items())).encode("utf-8")
        self._pool_formats[pool_name] = hashlib.sha256(encoded).hexdigest()[:24]
        return len(ptrs)

    def _storage_key(
        self,
        page_key: str,
        pool_name: PoolName,
        layout: str,
    ) -> str:
        cfg = self.storage_config
        # All free-form fields use unpadded URL-safe base64. Rank fields are
        # labelled segments, so no input can alias another tuple through
        # delimiter injection.
        return "/".join(
            (
                "nixlshard",
                f"schema={_b64(self._schema_version)}",
                f"model={_b64(cfg.model_name or '')}",
                f"revision={_b64(self._model_revision)}",
                f"deployment={_b64(self._deployment_namespace)}",
                f"format={self._pool_formats[pool_name]}",
                f"tp={cfg.tp_rank}-of-{cfg.tp_size}",
                f"pp={cfg.pp_rank}-of-{cfg.pp_size}",
                f"acp={cfg.attn_cp_rank}-of-{cfg.attn_cp_size}",
                f"dp={cfg.dp_rank}",
                f"pool={_b64(pool_name)}",
                f"layout={_b64(layout)}",
                f"page={_b64(page_key)}",
            )
        )

    def _page_keys(self, keys: Sequence[str], pool_name: PoolName) -> list[str]:
        pool = self.registered_pools[pool_name]
        layout = str(getattr(pool, "layout", "unknown"))
        return [self._storage_key(key, pool_name, layout) for key in keys]

    def _prepare_transfer(self, transfer: PoolTransfer) -> Optional[_PreparedTransfer]:
        keys = list(transfer.keys or [])
        pool = self.registered_pools.get(transfer.name)
        indices = transfer.host_indices
        if pool is None:
            logger.error("NIXLShard host pool %s is not registered", transfer.name)
            return None
        page_size = int(getattr(pool, "page_size", 1) or 1)
        expected = len(keys) * page_size
        if indices is None or indices.numel() != expected:
            logger.error(
                "NIXLShard indices length mismatch for %s: expected %d, got %d",
                transfer.name,
                expected,
                indices.numel() if indices is not None else 0,
            )
            return None
        if not keys:
            return _PreparedTransfer(transfer, [], [], pool, indices)

        try:
            ptrs, sizes = pool.get_page_buffer_meta(indices)
        except Exception:
            logger.exception("NIXLShard failed to address host pool %s", transfer.name)
            return None
        if len(ptrs) != len(sizes) or len(ptrs) % len(keys):
            logger.error(
                "NIXLShard metadata mismatch for %s: pages=%d ptrs=%d sizes=%d",
                transfer.name,
                len(keys),
                len(ptrs),
                len(sizes),
            )
            return None
        component_count = len(ptrs) // len(keys)
        components = [(int(ptr), int(size)) for ptr, size in zip(ptrs, sizes)]
        if any(ptr <= 0 or size <= 0 for ptr, size in components):
            logger.error("NIXLShard received an invalid buffer for %s", transfer.name)
            return None
        buffers = [
            components[i : i + component_count]
            for i in range(0, len(components), component_count)
        ]
        return _PreparedTransfer(
            transfer,
            self._page_keys(keys, transfer.name),
            buffers,
            pool,
            indices,
        )

    @staticmethod
    def _copy_value(result: Any, destination: list[tuple[int, int]]) -> bool:
        value = _field(result, "value")
        if value is None:
            # The normal zero-copy path has already populated destination.
            return True
        expected_size = sum(size for _, size in destination)
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or not value.is_contiguous():
                return False
            size = value.numel() * value.element_size()
            if size != expected_size:
                return False
            source = value.data_ptr()
            offset = 0
            for address, region_size in destination:
                ctypes.memmove(address, source + offset, region_size)
                offset += region_size
            return True
        try:
            payload = bytes(value)
        except (TypeError, ValueError):
            return False
        if len(payload) != expected_size:
            return False
        offset = 0
        for address, region_size in destination:
            ctypes.memmove(address, payload[offset : offset + region_size], region_size)
            offset += region_size
        return True

    def _call_client(
        self,
        operation: str,
        keys: list[str],
        buffers: Optional[list[list[tuple[int, int]]]] = None,
    ) -> list[bool]:
        if not keys:
            return []
        try:
            if operation == "exists":
                raw = self._client.batch_exists(keys)
            elif operation == "get":
                raw = self._client.batch_get(keys, buffers)
            else:
                raw = self._client.batch_set(keys, buffers)
        except Exception:
            logger.exception("NIXLShard batch_%s failed", operation)
            return [False] * len(keys)

        items = _result_items(raw)
        if len(items) != len(keys):
            logger.error(
                "NIXLShard batch_%s result mismatch: requested=%d returned=%d",
                operation,
                len(keys),
                len(items),
            )
        answer: list[bool] = []
        for i in range(len(keys)):
            if i >= len(items):
                answer.append(False)
                continue
            ok = _result_ok(items[i], operation)
            if ok and operation in ("get", "set"):
                expected_size = sum(size for _, size in buffers[i])
                logical_length = _field(items[i], "logical_length")
                if logical_length is not None:
                    try:
                        ok = int(logical_length) == expected_size
                    except (TypeError, ValueError):
                        ok = False
                if not ok:
                    logger.error(
                        "NIXLShard batch_%s length mismatch for key %s: "
                        "expected=%d returned=%r",
                        operation,
                        keys[i],
                        expected_size,
                        logical_length,
                    )
            if ok and operation == "get":
                ok = self._copy_value(items[i], buffers[i])
            answer.append(ok)
        successes = sum(answer)
        self._metrics[f"{operation}_pages"] += len(keys)
        result_metric = "set_successes" if operation == "set" else f"{operation}_hits"
        self._metrics[result_metric] += successes
        logger.debug(
            "NIXLShard batch_%s pages=%d successes=%d cumulative=%s",
            operation,
            len(keys),
            successes,
            self._metrics,
        )
        return answer

    def _batch_io_v2(
        self, operation: str, transfers: List[PoolTransfer]
    ) -> dict[str, List[bool]]:
        output: dict[str, List[bool]] = {}
        prepared: list[_PreparedTransfer] = []
        all_keys: list[str] = []
        all_buffers: list[list[tuple[int, int]]] = []
        for transfer in transfers:
            item = self._prepare_transfer(transfer)
            if item is None:
                output[transfer.name] = [False] * len(transfer.keys or [])
                continue
            if not item.keys:
                output[transfer.name] = []
                continue
            prepared.append(item)
            all_keys.extend(item.keys)
            all_buffers.extend(item.buffers)

        page_results = self._call_client(operation, all_keys, all_buffers)
        cursor = 0
        for item in prepared:
            count = len(item.keys)
            output[item.transfer.name] = page_results[cursor : cursor + count]
            cursor += count
        return output

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        if PoolName.KV not in self.registered_pools:
            return PoolTransferResult.empty()

        requests: list[tuple[PoolName, PoolHitPolicy, int, int]] = []
        all_keys: list[str] = []

        def add(pool_name: PoolName, policy: PoolHitPolicy, trailing: int) -> bool:
            if pool_name not in self.registered_pools:
                return False
            start = len(all_keys)
            all_keys.extend(self._page_keys(keys, pool_name))
            requests.append((pool_name, policy, trailing, start))
            return True

        add(PoolName.KV, PoolHitPolicy.ALL_PAGES, len(keys))
        for transfer in pool_transfers or []:
            if not add(
                transfer.name,
                transfer.hit_policy,
                max(1, len(transfer.keys or [])),
            ):
                return PoolTransferResult.empty()

        physical_results = self._call_client("exists", all_keys)
        by_pool: dict[PoolName, list[bool]] = {}
        for pool_name, _, _, start in requests:
            by_pool[pool_name] = physical_results[start : start + len(keys)]

        kv_pages = next(
            (i for i, hit in enumerate(by_pool[PoolName.KV]) if not hit), len(keys)
        )
        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        for pool_name, policy, trailing, _ in requests[1:]:
            page_exists = by_pool[pool_name][:kv_pages]
            boundary = 0
            if policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i, hit in enumerate(page_exists) if not hit), kv_pages
                )
            else:
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        page_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[pool_name] = boundary
            final_pages = min(final_pages, boundary)
        return PoolTransferResult(final_pages, hit_count)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2("get", transfers)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2("set", transfers)

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        return self.batch_exists_v2(keys, extra_info=extra_info).kv_hit_pages

    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) == 1

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self.batch_get_v2(
            [PoolTransfer(PoolName.KV, host_indices=host_indices, keys=keys)],
            extra_info,
        ).get(PoolName.KV, [False] * len(keys))

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self.batch_set_v2(
            [PoolTransfer(PoolName.KV, host_indices=host_indices, keys=keys)],
            extra_info,
        ).get(PoolName.KV, [False] * len(keys))

    def get(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("use batch_get_v1 or batch_get_v2")

    def batch_get(self, *args: Any, **kwargs: Any) -> List[None]:
        raise NotImplementedError("use batch_get_v1 or batch_get_v2")

    def set(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError("use batch_set_v1 or batch_set_v2")

    def batch_set(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError("use batch_set_v1 or batch_set_v2")

    def clear(self) -> bool:
        clear = getattr(self._client, "clear", None)
        if clear is None:
            return False
        return bool(clear())

    def get_metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def close(self) -> None:
        logger.info("NIXLShard final adapter metrics: %s", self._metrics)
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
