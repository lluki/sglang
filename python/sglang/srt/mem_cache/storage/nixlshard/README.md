**Agent written**

# NIXLShard HiCache backend

Branch: `nixlshard-v2`. SGLang base: `20518d8518375f49be0d14ead7ea474dbc2721d0`.
This is a distinct backend selected with `--hicache-storage-backend nixlshard`.
It supports CPU MHA KV pools in `page_first`, `page_first_direct`, `page_head`,
and `layer_first` layouts. MLA, split-head persistence, and auxiliary/hybrid
pools fail explicitly. The controller uses the registered-page v1 interface;
KV-only v2 calls are also supported.

One page is one immutable object. Ordered K/V segments come from the concrete
host pool, which is registered once. The native runtime packs segments into
bounded registered aligned slots, including for unaligned and layer-first
pages. It handles POSIX I/O, owner-initiated UCX writes, deadlines, and safe
completion. The adapter holds the original host tensor until native shutdown.
Loads return one bool per page; `batch_exists` returns the consecutive hit
prefix. Controller read-prefix and all-page store policies remain unchanged.

## Configuration

Supply the JSON below with `--hicache-storage-backend-extra-config @/path/config.json`.
The client assigns disk paths and owns NUMA locality. This example initializes
a new disposable regular file; use a distinct file/agent per rank. Existing
block devices must already have the native format and use `create: false`.

```json
{
  "model_revision": "immutable-model-revision-or-local-snapshot-id",
  "prefetch_threshold": 64,
  "agent": {
    "name": "qwen-rank-0",
    "disks": [{
      "path": "/raid/nixlshard-v2/qwen-rank-0-cache.bin",
      "capacity_bytes": 1073741824,
      "unit_bytes": 65536,
      "metadata_bytes": 16777216,
      "create": true
    }],
    "listen_host": "127.0.0.1",
    "listen_port": 19010,
    "staging_slot_bytes": 33554432,
    "staging_slots": 8,
    "workers": 2,
    "max_inflight": 64,
    "timeout_ms": 5000,
    "direct_io": false
  }
}
```

The model revision is required. Namespaces additionally include model name,
dtype, layout, tokens/page, layer/head geometry, ordered segment sizes, and
rank topology. Qwen3-32B's TP1 BF16 host KV uses 16 MiB per 64-token page;
the staging slot must fit the complete page rounded to allocation units.

Caller hints can be supplied through
`HiCacheStorageExtraInfo(extra_info={"owner_hints": ["owner-name", ...]})`.
Configure the corresponding native `peers` map with `{ "owner-name":
{"host": "host-address", "port": 19010} }`, or supply `metadata_endpoint`
with a native metadata server's `{ "host": "host-address", "port": 19020 }`.
The SGLang adapter passes optional owner hints through; advertisements never
replace owner verification.

## Development validation

On the development container, activate the isolated NIXLShard prefix and use
the pinned SGLang checkout, rather than the image's bundled SGLang tree:

```bash
source /workspace/install/nixlshard/env.sh
export PYTHONPATH=/workspace/src/sglang/python:$PYTHONPATH
python -c 'import nixlshard; print(nixlshard.__file__, nixlshard.__build_marker__)'
python -c 'from sglang.srt.managers.cache_controller import HiCacheController; print(HiCacheController.__name__)'
python -m unittest discover -s test/registered/unit/mem_cache -p 'test_hicache_nixlshard.py' -v
SGLANG_RUN_NIXLSHARD_NATIVE=1 python -m unittest discover -s test/registered/unit/mem_cache -p 'test_hicache_nixlshard_native.py' -v
```

The native marker is exposed as `backend.build_marker` and logged with the
namespace, layout, and bytes/page during registration. Adapter counters track
get/set pages, hits, and successful bytes. Native staging, POSIX, control,
checkpoint, and failure counters appear in `get_stats()` with a `native_`
prefix. With framework metrics enabled, `get_stats()` also supplies the
`StorageMetrics` histogram samples expected by the SGLang collector; cumulative
counters remain accessible as mapping entries and serialize to JSON. Histogram
samples drain once per collection.

The isolated native prefix uses shared Abseil libraries so repeated agent
creation/destruction is safe. The original image's Torch remains 2.9.1+cu129;
Transformers was updated to the pinned source's 5.12.1 with tokenizers 0.22.2.
Full model serving additionally requires matching the pinned SGLang kernel and
CUDA runtime; an isolated serving runtime can supply those dependencies.

Opt-in native tests use disposable files under `/raid/nixlshard-v2` by default.
Set `NIXLSHARD_TEST_DIR` to select another existing test directory.

## Local microbenchmark

The harness verifies every round-trip before emitting JSON. Each store round
uses new immutable keys, so duplicate-key shortcuts cannot hide payload I/O.
It creates and removes a disposable file under `/raid/nixlshard-v2` by default;
`--directory` selects another existing directory:

```bash
python -m sglang.srt.mem_cache.storage.nixlshard.bench_local --pages 4 --rounds 5
python -m sglang.srt.mem_cache.storage.nixlshard.bench_local --checkpoint --pages 4 --rounds 3
python -m sglang.srt.mem_cache.storage.nixlshard.bench_local --layout layer_first --direct-io --pages 4 --rounds 5
python -m sglang.srt.mem_cache.storage.nixlshard.bench_local --layers 64 --kv-heads 8 --head-dim 128 --page-size 64 --pages 2 --rounds 3
```

The report includes the loaded native marker, page geometry, hit counts,
wall time/page, and native nanosecond counters/page for store, exists, and
load. `--checkpoint` additionally measures an explicit metadata checkpoint after
each round. These are synthetic CPU KV/file measurements. They do not measure model
TTFT or remote RDMA. Native cost counters may overlap; do not add them into a
latency decomposition. Filesystem/page-cache behavior and direct-I/O mode must
be reported when sharing results.
