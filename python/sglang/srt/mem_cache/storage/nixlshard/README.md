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

## Optional native diagnostics on /metrics

To export native counters, add `"export_native_metrics": true` at the top level
of the storage extra configuration and launch with `--enable-metrics`. Both
gates are required; export defaults off. Imports and registry creation remain
lazy, and CPU/native benchmarks with framework metrics disabled are unaffected.
This is a separate instrumented profile when comparing overhead.

| Metric family | Unit | Fixed label values |
| --- | --- | --- |
| `sglang:nixlshard_component_seconds_total` | seconds | `staging_copy`, `posix_read`, `posix_write`, `remote_control`, `exists_control`, `ucx_write`, `metadata_checkpoint` |
| `sglang:nixlshard_component_bytes_total` | bytes | `staging_copy`, `posix_read`, `posix_write`, `remote_read`, `ucx_write` |
| `sglang:nixlshard_events_total` | events | native worker statuses and the fixed resource/failure fields in `native_metrics.py` |

Labels identify the configured model and TP/DP/PP/attention-CP ranks and sizes,
plus a fixed component or event. Cache keys, peer names, incarnations and
arbitrary native fields never become labels. Current slot-usage gauges remain
available only in the `get_stats()` mapping.

The exporter converts cumulative native nanoseconds to seconds and adds only
the difference since its previous observation. Repeated snapshots do not add
counts twice. Families are reused on detach/reattach; a replacement Agent has
its own baseline, and its observed work adds to the existing model/rank totals.
A lower out-of-order snapshot is ignored rather than treated as an Agent reset.
The normal Prometheus multiprocess collector exposes worker counters at the
existing HTTP `/metrics` endpoint. Values update when the backend is collected,
rather than by querying the Agent from the HTTP process.

These are aggregate wall intervals and counters across native workers and
background work. `remote_control` includes owner SSD I/O, owner UCX write and
control response wait; it overlaps owner-side timers. POSIX intervals include
submission/progress/draining, and successful POSIX bytes include allocation
padding. Staging bytes are logical gather/scatter bytes and may count both
directions. Checkpoint timing covers periodic/explicit checkpoints; eager
reclamation is not timed separately. Event statuses describe batch workers,
rather than every incoming RPC. Failed exists RPCs contribute to their timer.

For native builds supporting `remote_batch_limit`, three event labels describe
grouped remote loads. `remote_load_batch_requests` counts requester group RPCs
that return a parsed response, including per-key failures. On the owner,
`remote_served_batch_requests` counts groups after successful POSIX reads,
before UCX completion. `remote_group_fallbacks` counts known-quiescent
`no_space` responses that cause a group to fall back to per-object loads.
These count groups, not pages, and omit the legacy single-object path; they
are not counts of all attempted or successful RPCs. The native default limit
is 1; a controlled batching profile can set it to 8 in the `agent` config.
Record the native build marker: older builds omit these fields, while the
exporter still creates zero-valued series, so zero alone does not establish
that grouping was disabled or unsupported.

Do not sum these timers into TTFT or label their sum attributable NIXLShard
overhead. They exclude Python/controller/framework work and overlap concurrent
I/O; attribution still requires a separately designed measurement. Record
export enablement, source/build markers, per-request cache provenance and raw
counter snapshots alongside any serving latency results.

## Optional bounded remote page batching

For 16 MiB logical pages, this addition to the existing `agent` configuration
allows up to eight consecutive pages belonging to the same owner per remote
load RPC, within a 128 MiB private staging slot:

```json
{
  "agent": {
    "remote_batch_limit": 8,
    "staging_slot_bytes": 134217728,
    "staging_slots": 4,
    "workers": 2
  }
}
```

Merge these fields into each endpoint's complete agent configuration. Use a
native build supporting grouped remote loads on both endpoints, with matching
model revision, KV schema and page geometry. Each agent reserves 512 MiB of
private staging in this example. The default `remote_batch_limit` remains 1;
grouping also respects slot capacity and falls back to individual loads after
a known-quiescent owner `no_space` response. Caller buffers retain the existing
registration, cancellation and completion rules.

The matched GB200 experiment used native `ff5aa4879101` and serving SGLang
`cfe719424937`, Qwen3-32B-FP8 with BF16 KV, page size 64, a 16 GB host cache,
and one request outstanding. Five measured streaming requests per context,
after one warmup, reduced median remote TTFT by 5.7% to 17.3% for 512 through
8192 tokens. At 8192 tokens, median TTFT was 415.27 ms with limit 1 and
343.26 ms with limit 8; both transferred exactly 2032 MiB from owner
file-backed storage to requester host memory. Group and cleanup counters
confirmed 16 transactions
with limit 8 versus 127 individual transactions, with no group fallbacks.
Raw run IDs are `20261003-135511-batch1` (corrected attempt 2) and
`20261003-141001-batch8`. These measurements retain the serving commit above;
subsequent documentation commits do not change that runtime pin.

For `page_first_direct`, each page has two contiguous 8 MiB segments, K then
V, separated by half the host pool. Native copies remain serial. Requester
staging timers were approximately 153–156 ms at 8192 tokens in both profiles.
A next experiment should isolate copies with the same registered buffers and
measure thread/memory placement before testing bounded copy parallelism that
joins before publishing completion. These timer windows overlap other work
and do not establish an additive TTFT breakdown.

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
