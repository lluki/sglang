"""Native Prometheus export lifetimes and real multiprocess scrape contracts."""

import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.mem_cache.storage.nixlshard.native_metrics import NativeMetricsExporter


def config(rank=0):
    return SimpleNamespace(
        model_name="test-model",
        tp_rank=rank,
        tp_size=2,
        dp_rank=0,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )


def value(registry, name, dimension, component, rank="0"):
    samples = [
        sample
        for family in text_string_to_metric_families(generate_latest(registry).decode())
        for sample in family.samples
    ]
    return next(
        sample.value
        for sample in samples
        if sample.name == name
        and sample.labels.get(dimension) == component
        and sample.labels["tp_rank"] == rank
    )


class TestNativeMetrics(unittest.TestCase):
    def test_reattached_agents_share_families_and_keep_totals_without_double_counts(
        self,
    ):
        registry = CollectorRegistry()
        first = NativeMetricsExporter(config(), registry)
        first.observe({"posix_read_ns": 1000000000, "posix_read_bytes": 10})
        first.observe({"posix_read_ns": 1000000000, "posix_read_bytes": 10})
        second = NativeMetricsExporter(
            config(), registry
        )  # replacement/reattached Agent
        second.observe({"posix_read_ns": 500000000, "posix_read_bytes": 4})
        first.observe({"posix_read_ns": 1500000000, "posix_read_bytes": 12})
        self.assertEqual(
            value(
                registry,
                "sglang:nixlshard_component_seconds_total",
                "component",
                "posix_read",
            ),
            2,
        )
        self.assertEqual(
            value(
                registry,
                "sglang:nixlshard_component_bytes_total",
                "component",
                "posix_read",
            ),
            16,
        )
        self.assertEqual(len(list(registry.collect())), 3)

    def test_registry_and_rank_isolation_and_stale_snapshot_cannot_inflate_counters(
        self,
    ):
        left, right = CollectorRegistry(), CollectorRegistry()
        exporter = NativeMetricsExporter(config(), left)
        exporter.observe({"staging_copy_bytes": 100, "busy": 3})
        exporter.observe(
            {"staging_copy_bytes": 90, "busy": 2}
        )  # stale concurrent snapshot
        exporter.observe({"staging_copy_bytes": 130, "busy": 4})
        other_rank = NativeMetricsExporter(config(1), left)
        other_rank.observe({"staging_copy_bytes": 9})
        other_registry = NativeMetricsExporter(config(), right)
        other_registry.observe({"staging_copy_bytes": 2})
        self.assertEqual(
            value(
                left,
                "sglang:nixlshard_component_bytes_total",
                "component",
                "staging_copy",
            ),
            130,
        )
        self.assertEqual(
            value(
                left,
                "sglang:nixlshard_component_bytes_total",
                "component",
                "staging_copy",
                rank="1",
            ),
            9,
        )
        self.assertEqual(
            value(
                right,
                "sglang:nixlshard_component_bytes_total",
                "component",
                "staging_copy",
            ),
            2,
        )
        self.assertEqual(
            value(left, "sglang:nixlshard_events_total", "event", "busy"), 4
        )

    def test_worker_counters_are_visible_to_actual_multiprocess_scraper(self):
        # The HTTP server scrapes mmap-backed worker metrics, not their registries.
        from prometheus_client import multiprocess

        source = """
from types import SimpleNamespace
from sglang.srt.mem_cache.storage.nixlshard.native_metrics import NativeMetricsExporter
config = SimpleNamespace(model_name="test-model", tp_rank=0, tp_size=2, dp_rank=0,
                         pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1)
exporter = NativeMetricsExporter(config)
exporter.observe({"posix_write_ns": 2000000000, "posix_write_bytes": 8192, "timeout": 1})
exporter.observe({"posix_write_ns": 2000000000, "posix_write_bytes": 8192, "timeout": 1})
batch_events = {"remote_load_batch_requests": 3, "remote_served_batch_requests": 4,
                "remote_group_fallbacks": 1}
exporter.observe(batch_events)
exporter.observe(batch_events)  # repeated observations must not count RPCs twice
exporter.observe({})  # an older native snapshot with absent fields does not reset
exporter.observe({"remote_load_batch_requests": 5, "remote_served_batch_requests": 6,
                  "remote_group_fallbacks": 2})
"""
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PROMETHEUS_MULTIPROC_DIR=directory)
            # A fresh interpreter selects prometheus' multiprocess Value class.
            process = subprocess.run(
                [sys.executable, "-c", source],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry, path=directory)
            self.assertEqual(
                value(
                    registry,
                    "sglang:nixlshard_component_seconds_total",
                    "component",
                    "posix_write",
                ),
                2,
            )
            self.assertEqual(
                value(
                    registry,
                    "sglang:nixlshard_component_bytes_total",
                    "component",
                    "posix_write",
                ),
                8192,
            )
            self.assertEqual(
                value(registry, "sglang:nixlshard_events_total", "event", "timeout"), 1
            )
            for event, expected in (
                ("remote_load_batch_requests", 5),
                ("remote_served_batch_requests", 6),
                ("remote_group_fallbacks", 2),
            ):
                with self.subTest(event=event):
                    self.assertEqual(
                        value(registry, "sglang:nixlshard_events_total", "event", event),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
