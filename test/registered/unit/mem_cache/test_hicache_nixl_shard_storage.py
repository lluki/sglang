"""Focused tests for the NIXLShard dynamic HiCache adapter."""

import ctypes
import logging
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.nixl_shard.hicache_nixl_shard import (
    HiCacheNixlShard,
)


class Status(Enum):
    OK = "ok"
    MISS = "miss"
    ALREADY_PRESENT = "already_present"
    ERROR = "error"


@dataclass
class Result:
    status: Status
    value: bytes | None = None
    logical_length: int | None = None


class FakeShardClient:
    def __init__(self, value_reads: bool = False):
        self.data = {}
        self.value_reads = value_reads
        self.calls = []

    def batch_exists(self, keys):
        self.calls.append(("exists", list(keys)))
        return [Result(Status.OK if key in self.data else Status.MISS) for key in keys]

    def batch_set(self, keys, sources):
        self.calls.append(("set", list(keys)))
        results = []
        for key, regions in zip(keys, sources):
            if key in self.data:
                results.append(
                    Result(Status.ALREADY_PRESENT, logical_length=len(self.data[key]))
                )
            else:
                self.data[key] = b"".join(
                    ctypes.string_at(address, size) for address, size in regions
                )
                results.append(Result(Status.OK, logical_length=len(self.data[key])))
        return results

    def batch_get(self, keys, destinations):
        self.calls.append(("get", list(keys)))
        results = []
        for key, regions in zip(keys, destinations):
            value = self.data.get(key)
            if value is None:
                results.append(Result(Status.MISS))
            elif self.value_reads:
                results.append(Result(Status.OK, value, logical_length=len(value)))
            else:
                offset = 0
                for address, size in regions:
                    ctypes.memmove(address, value[offset : offset + size], size)
                    offset += size
                results.append(Result(Status.OK, logical_length=len(value)))
        return results


class FakeHostPool:
    def __init__(
        self,
        pages=4,
        components=2,
        component_bytes=8,
        layout="page_first",
        page_size=1,
    ):
        self.page_size = page_size
        self.layout = layout
        self.buffers = torch.zeros(
            (pages, components, component_bytes), dtype=torch.uint8
        )
        self.dtype = self.buffers.dtype

    def get_page_buffer_meta(self, indices):
        ptrs = []
        sizes = []
        component_bytes = self.buffers.shape[-1]
        for index in indices.tolist():
            for component in range(self.buffers.shape[1]):
                ptrs.append(self.buffers[index, component].data_ptr())
                sizes.append(component_bytes)
        return ptrs, sizes


def make_config(extra_config=None):
    options = {
        "deployment_namespace": "unit-test-deployment",
        "interface_v1": 1,
    }
    options.update(extra_config or {})
    return HiCacheStorageConfig(
        tp_rank=1,
        tp_size=4,
        pp_rank=2,
        pp_size=3,
        attn_cp_rank=0,
        attn_cp_size=2,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="org/model",
        dp_rank=7,
        extra_config=options,
    )


def make_backend(client=None, config=None):
    return HiCacheNixlShard(
        config or make_config(), {"client": client or FakeShardClient()}
    )


def transfer(pool_name, keys, indices=None, policy=PoolHitPolicy.ALL_PAGES):
    return PoolTransfer(
        name=pool_name,
        keys=keys,
        host_indices=torch.tensor(
            list(range(len(keys))) if indices is None else indices, dtype=torch.int64
        ),
        hit_policy=policy,
    )


def test_dynamic_factory_loads_without_builtin_registration():
    client = FakeShardClient()
    config = make_config(
        {
            "backend_name": "nixl_shard",
            "module_path": "sglang.srt.mem_cache.storage.nixl_shard.hicache_nixl_shard",
            "class_name": "HiCacheNixlShard",
        }
    )
    backend = StorageBackendFactory.create_backend(
        "dynamic", config, mem_pool_host=None, client=client
    )
    assert isinstance(backend, HiCacheNixlShard)
    assert backend._client is client


def test_client_factory_receives_config_and_returns_live_client():
    client = FakeShardClient()
    captured = []

    def factory(options):
        captured.append(options)
        return client

    backend = HiCacheNixlShard(
        make_config(
            {
                "client_factory": factory,
                "client_config": {"metadata_endpoint": "127.0.0.1:7420"},
            }
        )
    )
    assert backend._client is client
    assert captured == [{"metadata_endpoint": "127.0.0.1:7420"}]


def test_key_namespace_is_deterministic_and_delimiter_safe():
    backend = make_backend(config=make_config({"model_revision": "rev/a"}))
    pool = FakeHostPool(components=1)
    backend.register_mem_pool_host(pool)

    first = backend._storage_key("page/a", PoolName.KV, "page/first")
    second = backend._storage_key("page/a", PoolName.KV, "page/first")
    delimiter_variant = backend._storage_key("page", PoolName.KV, "a/page/first")

    assert first == second
    assert first != delimiter_variant
    assert "org/model" not in first
    assert "tp=1-of-4" in first
    assert "pp=2-of-3" in first
    assert "dp=7" in first


def test_batch_set_and_get_use_one_page_key_with_scatter_gather_buffers():
    client = FakeShardClient(value_reads=True)
    backend = make_backend(client)
    kv_pool = FakeHostPool(components=2)
    mamba_pool = FakeHostPool(components=3)
    backend.register_mem_pool_host(kv_pool)
    backend.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
    kv_pool.buffers[0].fill_(11)
    kv_pool.buffers[1].fill_(12)
    mamba_pool.buffers[0].fill_(21)

    transfers = [
        transfer(PoolName.KV, ["p0", "p1"]),
        transfer(PoolName.MAMBA, ["p0"]),
    ]
    set_result = backend.batch_set_v2(transfers)
    assert set_result == {PoolName.KV: [True, True], PoolName.MAMBA: [True]}
    assert [call[0] for call in client.calls].count("set") == 1
    assert len(client.calls[-1][1]) == 3
    assert [len(value) for value in client.data.values()] == [16, 16, 24]

    kv_pool.buffers.zero_()
    mamba_pool.buffers.zero_()
    get_result = backend.batch_get_v2(transfers)
    assert get_result == {PoolName.KV: [True, True], PoolName.MAMBA: [True]}
    assert [call[0] for call in client.calls].count("get") == 1
    assert torch.all(kv_pool.buffers[0] == 11)
    assert torch.all(kv_pool.buffers[1] == 12)
    assert torch.all(mamba_pool.buffers[0] == 21)
    assert backend.get_metrics() == {
        "exists_pages": 0,
        "exists_hits": 0,
        "get_pages": 3,
        "get_hits": 3,
        "set_pages": 3,
        "set_successes": 3,
    }


def test_miss_fails_only_its_logical_page():
    client = FakeShardClient()
    backend = make_backend(client)
    pool = FakeHostPool(components=2)
    backend.register_mem_pool_host(pool)
    prepared = backend._prepare_transfer(transfer(PoolName.KV, ["p0", "p1"]))
    client.data[prepared.keys[1]] = b"x" * 16

    result = backend.batch_get_v2([transfer(PoolName.KV, ["p0", "p1"])])
    assert result[PoolName.KV] == [False, True]


def test_failed_io_warning_is_bounded_and_redacts_keys(caplog):
    class FailingClient:
        def batch_set(self, keys, sources):
            return [
                SimpleNamespace(
                    status=Status.ERROR,
                    detail=f"request exceeds limit for {keys[0]} " + "x" * 300,
                )
            ]

        def batch_get(self, keys, destinations):
            return [SimpleNamespace(status=Status.MISS, detail="cache miss")]

    backend = make_backend(FailingClient())
    key = "private-full-page-key"
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert backend._call_client("set", [key], [[(1, 8)]]) == [False]
        assert backend._call_client("get", [key], [[(1, 8)]]) == [False]
    warnings = [
        record.message
        for record in caplog.records
        if "batch_set failures" in record.message
    ]
    assert len(warnings) == 3
    assert all("first_status=ERROR" in message for message in warnings)
    assert all("request exceeds limit" in message for message in warnings)
    assert all(key not in message and len(message) < 300 for message in warnings)
    assert "further failure warnings suppressed" in warnings[-1]
    assert not any("batch_get failures" in record.message for record in caplog.records)

    class FailingGetClient(FailingClient):
        def batch_get(self, keys, destinations):
            return [SimpleNamespace(status=Status.ERROR, detail="remote read failed")]

    caplog.clear()
    read_backend = make_backend(FailingGetClient())
    with caplog.at_level(logging.WARNING):
        assert read_backend._call_client("get", [key], [[(1, 8)]]) == [False]
    assert len(caplog.records) == 1
    assert "batch_get failures=1/1 first_status=ERROR" in caplog.records[0].message
    assert "remote read failed" in caplog.records[0].message


def test_exists_uses_one_batch_and_honors_pool_hit_policies():
    client = FakeShardClient()
    backend = make_backend(client)
    kv_pool = FakeHostPool(components=1)
    mamba_pool = FakeHostPool(components=1)
    backend.register_mem_pool_host(kv_pool)
    backend.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
    keys = ["p0", "p1", "p2"]

    kv_keys = backend._page_keys(keys, PoolName.KV)
    mamba_keys = backend._page_keys(keys, PoolName.MAMBA)
    client.data.update({key: b"x" for key in kv_keys})
    client.data.update({mamba_keys[1]: b"x", mamba_keys[2]: b"x"})

    result = backend.batch_exists_v2(
        keys,
        [
            PoolTransfer(
                PoolName.MAMBA,
                keys=["tail0", "tail1"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )
    assert result.kv_hit_pages == 3
    assert result.extra_pool_hit_pages[PoolName.MAMBA] == 3
    assert [call[0] for call in client.calls] == ["exists"]


def test_already_present_is_idempotent_set_success():
    client = FakeShardClient()
    backend = make_backend(client)
    pool = FakeHostPool(components=1)
    backend.register_mem_pool_host(pool)
    item = transfer(PoolName.KV, ["p0"])

    assert backend.batch_set_v2([item])[PoolName.KV] == [True]
    pool.buffers.fill_(99)
    assert backend.batch_set_v2([item])[PoolName.KV] == [True]
    assert next(iter(client.data.values())) == bytes(8)


def test_mismatched_logical_length_fails_get_and_existing_set():
    class WrongLengthClient(FakeShardClient):
        def batch_get(self, keys, destinations):
            return [Result(Status.OK, logical_length=1) for _ in keys]

        def batch_set(self, keys, sources):
            return [Result(Status.ALREADY_PRESENT, logical_length=1) for _ in keys]

    backend = make_backend(WrongLengthClient())
    pool = FakeHostPool(components=2, component_bytes=8)
    backend.register_mem_pool_host(pool)
    item = transfer(PoolName.KV, ["p0"])

    assert backend.batch_set_v2([item])[PoolName.KV] == [False]
    assert backend.batch_get_v2([item])[PoolName.KV] == [False]


def test_multi_token_page_preserves_page_major_component_ordering():
    client = FakeShardClient(value_reads=True)
    backend = make_backend(client)
    pool = FakeHostPool(pages=4, components=3, component_bytes=5, page_size=2)
    backend.register_mem_pool_host(pool)
    pool.buffers[0].fill_(10)
    pool.buffers[1].fill_(11)
    pool.buffers[2].fill_(20)
    pool.buffers[3].fill_(21)
    item = transfer(PoolName.KV, ["p0", "p1"], indices=[0, 1, 2, 3])

    assert backend.batch_set_v2([item])[PoolName.KV] == [True, True]
    expected = pool.buffers.clone()
    pool.buffers.zero_()
    assert backend.batch_get_v2([item])[PoolName.KV] == [True, True]
    assert torch.equal(pool.buffers, expected)


def test_pool_format_fingerprint_prevents_layout_aliases():
    backend = make_backend()
    first_pool = FakeHostPool(components=1, page_size=1)
    backend.register_mem_pool_host(first_pool)
    first = backend._page_keys(["p0"], PoolName.KV)[0]

    second_pool = FakeHostPool(components=1, page_size=2)
    backend.register_mem_pool_host(second_pool)
    second = backend._page_keys(["p0"], PoolName.KV)[0]
    assert first != second


def test_deployment_namespace_is_required():
    try:
        make_backend(config=make_config({"deployment_namespace": ""}))
    except ValueError as exc:
        assert "deployment_namespace" in str(exc)
    else:
        raise AssertionError("empty deployment namespace must be rejected")


def test_zero_copy_interface_v1_is_required():
    try:
        make_backend(config=make_config({"interface_v1": 0}))
    except ValueError as exc:
        assert "interface_v1=1" in str(exc)
    else:
        raise AssertionError("copying legacy interface must be rejected")


def test_v1_methods_share_the_batched_zero_copy_path():
    client = FakeShardClient(value_reads=True)
    backend = make_backend(client)
    pool = FakeHostPool(components=1)
    backend.register_mem_pool_host(pool)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    pool.buffers[0].fill_(31)
    pool.buffers[1].fill_(32)

    assert backend.batch_set_v1(["p0", "p1"], indices) == [True, True]
    assert backend.batch_exists(["p0", "p1"]) == 2
    pool.buffers.zero_()
    assert backend.batch_get_v1(["p0", "p1"], indices) == [True, True]
    assert torch.all(pool.buffers[0] == 31)
    assert torch.all(pool.buffers[1] == 32)


def test_registration_rejects_pool_without_zero_copy_metadata():
    backend = make_backend()

    class CopyOnlyPool:
        page_size = 1
        layout = "layer_first"

    try:
        backend.register_mem_pool_host(CopyOnlyPool())
    except ValueError as exc:
        assert "zero-copy-capable" in str(exc)
    else:
        raise AssertionError("copy-only pool must be rejected")


def test_live_nixl_shard_cold_set_and_l3_hit(tmp_path):
    from nixl_shard import (
        AgentConfig,
        ClientConfig,
        DeviceRegistration,
        ServiceConfig,
        ShardAgent,
        ShardClient,
        ShardNamingService,
    )

    service = ShardNamingService(ServiceConfig(auto_evict=False))
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "hicache-device.bin",
            16 * 4096,
            logical_device_id=1,
            create=True,
            direct_io=False,
        )
    )
    service.register_device(
        DeviceRegistration(1, "inproc://hicache", agent.size, 4096, 1)
    )
    client = ShardClient(ClientConfig(service=service, agent=agent))
    backend = make_backend(client)
    pool = FakeHostPool(pages=2, components=2, component_bytes=64)
    backend.register_mem_pool_host(pool)
    indices = torch.tensor([0], dtype=torch.int64)
    expected = torch.full_like(pool.buffers[0], 47)

    try:
        assert backend.batch_exists(["cold-page"]) == 0
        pool.buffers[0].copy_(expected)
        assert backend.batch_set_v1(["cold-page"], indices) == [True]
        assert service.get_metrics()["objects_ready"] == 1
        pool.buffers[0].zero_()
        assert backend.batch_exists(["cold-page"]) == 1
        assert backend.batch_get_v1(["cold-page"], indices) == [True]
        assert torch.equal(pool.buffers[0], expected)
        assert client.get_metrics()["hits"] == 1
    finally:
        backend.close()
        agent.close(timeout=5)
