# NIXLShard HiCache adapter

This adapter follows the public API in the NIXLShard design repository.
The SGLang source used for this migration is
`72726e814f09016d4ad077f2f3cd4ef7e7968f5e`.

Select `--hicache-storage-backend nixlshard`. The registered factory constructs
`HiCacheNixlShard(storage_config)`. The ordinary `CacheController` explicitly
uses the retained v1 page methods; `HybridCacheController` registers its physical
pools and calls the v2 transfer methods. The implementation marker is
`nixlshard-public-api-whole-pages-v1`; the adapter also logs the loaded native
build marker.

## Configuration and identity

Example storage extra configuration:

```json
{
  "model_revision": "immutable-model-weight-revision",
  "agent": {
    "name": "inference-rank-0",
    "numa_node": 0,
    "staging_slot_bytes": 33554432,
    "staging_slots": 8,
    "disks": [{
      "path": "/raid/dedicated-nixlshard-cache.bin",
      "capacity_bytes": 1073741824,
      "create": true
    }]
  }
}
```

The disk list is supplied unchanged to one Agent at construction. Its format
limits must accommodate every physical pool's complete logical page. Registration
records the configured NUMA hint; transfer descriptors contain no NUMA selector.

Standalone discovery uses an Agent `metadata_endpoint` of
`{"type": "etcd", "hosts": ["http://127.0.0.1:2379"]}`. Framework-managed
inventory can supply an extra-configuration `key_change_callback`, a `peers`
mapping of owner names to numeric IPv4/control-port endpoints, and
`owner_hints` through `HiCacheStorageExtraInfo.extra_info`. Hints may be a list
for one pool or a mapping from pool name to per-page lists.

Every v2 pool uses exactly `namespace=str(transfer.name)`: `kv`, `swa`,
`mamba`, `indexer`, and other registered pool names. The v1 methods use `kv`.
ASCII keys, including full SHA-256 hex strings, slash and control bytes, pass
unchanged to the native API. There is no hex decoding, truncation, pool suffix,
or constructor namespace.

Use SGLang's request `cache_salt` to distinguish incompatible model revisions,
ranks, dtypes and page layouts when constructing framework page hashes.
`model_revision` documents the selected immutable weights; it does not salt
keys automatically. All clients sharing a namespace/key must agree on the
complete value and its length.

## Pages, ownership and completion

One logical page is one whole-value object. The adapter obtains addresses and
lengths from each host pool's actual `get_page_buffer_meta` result. Adjacent
components use one direct contiguous buffer. Split components are packed into
owned, registered buffers for stores and unpacked after successful loads.
`pack_bytes`, `pack_ns`, `unpack_bytes` and `unpack_ns` count these copies.
Physical K/V or auxiliary components never become additional keys.

Contiguous CPU allocations are registered with `ctypes.c_void_p`, length,
device ID zero, metadata and the NUMA hint. Exact address/length/device ID
registrations are deregistered before native close. Host-pool I/O leases and
adapter memory-range reservations prevent page reuse and overlapping prefetch
or backup while an operation owns memory. Close waits for active operations.

Polling uses scalar `IN_PROG`, `SUCCESS`, `NOT_FOUND` and `ERROR`. On terminal
failure, `get_errors` maps namespace/key pairs to ordered framework results.
Tracing is collected before handle release. A client deadline supplied as
`deadline_monotonic`, or extra `transfer_timeout_seconds`, requests cancellation
and continues polling until the native implementation returns ownership.
Unregistration and allocator lease release happen after terminal polling.

V1 existence reports the consecutive successful prefix. V2 existence preserves
per-pool results, all-page requirements and trailing-page restorable-prefix
sets, including holes that must be intersected across framework ranks. A logical
KV anchor has no physical object; its required physical sidecars determine the
usable storage prefix.

## Validation

The CPU suite uses a fake implementing the current public native API:

```sh
PYTHONPATH=python python -m unittest discover \
  -s test/registered/unit/mem_cache -p 'test_hicache_nixlshard*.py' -v
```

Enable the real native tests with the isolated NIXLShard package on
`PYTHONPATH` and its library/plugin paths configured:

```sh
SGLANG_RUN_NIXLSHARD_NATIVE=1 UCX_TLS=tcp,self,cuda_copy \
  PYTHONPATH="$NIXLSHARD_SITE:$NIXL_SITE:python" \
  python -m unittest discover -s test/registered/unit/mem_cache \
  -p 'test_hicache_nixlshard*.py' -v
```

`NIXLSHARD_TEST_DIR` selects a writable directory for disposable debug files.
The tests validate contiguous and split-page payload equality, multiple pool
namespaces, prefix and partial failures, NUMA registration, deadlines and drain,
active detach, concurrent prefetch/backup, clean recovery, actual POSIX counters,
and two-Agent UCX TCP payloads with staged and direct receive. These tests do
not establish RDMA performance.

Current correctness and payload-path measurements use the native tests above
and the standalone NIXLShard repository's supported tools.
Reproduce measurements with the migrated native library and this adapter's
implementation marker.

## Real-model measurements

`tools/bench_hicache_model.py` drives a running server with one request outstanding,
records the first received nonempty output event, and checks `cached_tokens_details`
and native byte counters before accepting a cache-tier result. It saves exact input
tokens/cache salts, streamed events and full-generation/background counter windows;
those cumulative timers are not a causal TTFT breakdown. `--replay-from` reuses a
prior run's cold references for a diskless requester measuring remote storage only.

Set `SGLANG_REQUEST_TIMELINE_DIR` before server launch and enable the Agent's
`enable_trace` to join native diagnostics to the client's supplied request IDs.
Extra configuration `diagnostic_logging=true` records per-operation physical
components, direct framework-pool bytes, packed-buffer bytes and pack/unpack copy
bytes/nanoseconds. Host registration logs actual component count and adjacency.
The Qwen3 MHA host pool separates K/V even in `page_first_direct`; native direct
receive into its owned packed buffer still requires framework-pool unpack copies
and must not be reported as framework receiver zero-copy.
