# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Opt-in observed native counters; aggregate timers overlap and are not TTFT."""

import threading
from weakref import WeakKeyDictionary

# Native Agent::elapsed/count fields. Never use caller keys, peer names or
# unknown future fields as metric labels.
TIMERS = (
    "staging_copy_ns",
    "posix_read_ns",
    "posix_write_ns",
    "remote_control_ns",
    "exists_control_ns",
    "ucx_write_ns",
    "metadata_checkpoint_ns",
)
BYTES = (
    "staging_copy_bytes",
    "posix_read_bytes",
    "posix_write_bytes",
    "remote_read_bytes",
    "ucx_write_bytes",
    "direct_receive_bytes",
    "direct_local_read_bytes",
)
EVENTS = (
    "success",
    "not_found",
    "not_ready",
    "busy",
    "no_space",
    "invalid_input",
    "io_error",
    "timeout",
    "canceled",
    "stores",
    "evictions",
    "staging_busy",
    "remote_unknown_peer",
    "remote_channel_busy",
    "remote_channel_not_ready",
    "quarantined_slots",
    "quarantines_released",
    "peer_connections",
    "peer_connect_failures",
    "remote_cleanup_requests",
    "remote_load_batch_requests",
    "remote_served_batch_requests",
    "remote_group_fallbacks",
    "direct_receive_segments",
    "direct_quarantined_batches",
    "direct_quarantines_released",
    "local_direct_fallbacks",
    "checkpoint_errors",
    "metadata_announce_errors",
    "metadata_connection_failures",
)
_LABELS = (
    "model_name",
    "tp_rank",
    "tp_size",
    "dp_rank",
    "pp_rank",
    "pp_size",
    "attn_cp_rank",
    "attn_cp_size",
)
_families_by_registry = WeakKeyDictionary()
_registry_lock = threading.Lock()


def _families(registry=None):
    # The module and prometheus imports are both lazy at the adapter boundary.
    from prometheus_client import REGISTRY, Counter

    registry = REGISTRY if registry is None else registry
    with _registry_lock:
        families = _families_by_registry.get(registry)
        if families is None:
            created = []
            try:
                for name, help_text, dimension in (
                    (
                        "sglang:nixlshard_component_seconds_total",
                        "Observed cumulative native component wall seconds. Timers overlap across "
                        "workers and remote control includes owner I/O; do not sum into TTFT.",
                        "component",
                    ),
                    (
                        "sglang:nixlshard_component_bytes_total",
                        "Observed cumulative native staging/payload bytes; component counts overlap.",
                        "component",
                    ),
                    (
                        "sglang:nixlshard_events_total",
                        "Observed cumulative native status/resource events, including background work.",
                        "event",
                    ),
                ):
                    created.append(
                        Counter(
                            name,
                            help_text,
                            labelnames=(*_LABELS, dimension),
                            registry=registry,
                        )
                    )
            except BaseException:
                for family in created:
                    registry.unregister(family)
                raise
            families = tuple(created)
            _families_by_registry[registry] = families
        return families


class NativeMetricsExporter:
    """Delta cumulative counters per Agent, retaining process totals on reattach.

    A fresh exporter belongs to a fresh native Agent. Repeated snapshots add no
    duplicate counts. Metric families are shared per registry, so replacement
    Agents with the same rank/model labels accumulate without duplicate
    registration. No unbounded per-Agent identity label is introduced.
    """

    def __init__(self, storage_config, registry=None):
        self.labels = {
            label: (
                str(getattr(storage_config, label) or "")
                if label == "model_name"
                else str(getattr(storage_config, label))
            )
            for label in _LABELS
        }
        seconds, byte_counts, events = _families(registry)
        self._series = {
            **{
                key: (
                    seconds.labels(**self.labels, component=key.removesuffix("_ns")),
                    1e9,
                )
                for key in TIMERS
            },
            **{
                key: (
                    byte_counts.labels(
                        **self.labels, component=key.removesuffix("_bytes")
                    ),
                    1,
                )
                for key in BYTES
            },
            **{key: (events.labels(**self.labels, event=key), 1) for key in EVENTS},
        }
        self._last = {}
        self._lock = threading.Lock()

    def observe(self, native):
        with self._lock:
            for key, (counter, divisor) in self._series.items():
                value = native.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    continue
                previous = self._last.get(key, 0)
                if value < previous:
                    # Native counters are monotonic for one Agent. Ignore a stale
                    # snapshot rather than misclassifying it as a new Agent.
                    continue
                if value > previous:
                    counter.inc((value - previous) / divisor)
                self._last[key] = value
