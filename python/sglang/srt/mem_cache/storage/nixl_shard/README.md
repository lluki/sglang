# NIXLShard HiCache backend

This dynamic backend connects SGLang HiCache to `nixl_shard.ShardClient`.
Unlike the built-in `nixl` backend, NIXLShard owns key placement and extent
allocation; the adapter never turns a cache key into a file name.

The adapter and `ShardClient` run in the SGLang worker process. Batch get/set
calls receive addresses in the registered HiCache host pools and must complete
synchronously before returning. This is what keeps the host-to-shard-agent path
zero-copy and makes host-buffer lifetime explicit.

Single-process functional configuration using the bundled bootstrap:

```text
--hicache-storage-backend dynamic
--hicache-storage-backend-extra-config '{
  "backend_name":"nixl_shard",
  "module_path":"sglang.srt.mem_cache.storage.nixl_shard.hicache_nixl_shard",
  "class_name":"HiCacheNixlShard",
  "interface_v1":1,
  "key_schema_version":1,
  "deployment_namespace":"model-revision-and-cache-format-v1",
  "client_factory":"nixl_shard.bootstrap:create_client",
  "client_config":{
    "embedded_service":true,
    "embedded_service_name":"sglang-worker-0",
    "file_path":"/tmp/nixlshard-worker-0.bin",
    "size":1073741824,
    "device_id":0,
    "logical_device_generation":1,
    "direct_io":false
  }
}'
```

For multiple workers, replace `embedded_service` with a loopback
`metadata_endpoint`, configure `tcp_listen_host`, and list the fixed V1 peers
in `remote_devices`. The bundled bootstrap registers and heartbeats the owned
device, and returns a live in-process `ShardClient`/`ShardAgent`. TCP is the
functional fallback transport; UCX is not exercised by this implementation.
Embedders and tests can instead inject an existing client through the dynamic
factory's `client=` keyword.

`interface_v1` is required. Without it, SGLang selects the deprecated copying
`batch_get`/`batch_set` contract instead of the zero-copy host-pool API.

A replaced device may reuse its ID only with a newer
`logical_device_generation`; its cache starts empty. V1 peer routes are static,
so restart workers with updated peer generations/endpoints after replacement.

`ShardClient` implements `batch_exists(keys)`,
`batch_get(keys, destinations)`, and `batch_set(keys, sources)`. Sources and
destinations contain one iovec (a list of `(address, size)` pairs) per logical
page. One page is stored as one contiguous NIXLShard extent even when its
HiCache host representation has multiple discontiguous components. Result
items expose a `status`; a get result may additionally expose `value` for a
copying fallback. `OK`/`HIT` are successes, `MISS` is not, and
`ALREADY_PRESENT` is an idempotent set success.

Storage keys include the schema version, required deployment namespace, model
revision, all parallel ranks, pool, host layout, a fingerprint of the pool's
byte format, and HiCache page identity. Free-form fields are URL-safe encoded
to prevent delimiter collisions. There is exactly one storage key per logical
page, and returned logical lengths must exactly match that page's iovec.
