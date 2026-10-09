# SPDX-License-Identifier: Apache-2.0
"""Strict campaign witnesses reject plausible-looking partial or mixed hits."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[4] / "tools" / "bench_hicache_model.py"
spec = importlib.util.spec_from_file_location("nixlshard_model_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

class TestModelBenchmarkWitnesses(unittest.TestCase):
    def remote(self):
        return {"text": "4\n", "context_tokens": 512,
                "meta_info": {"cached_tokens": 448,
                              "cached_tokens_details": {"device": 0, "host": 0, "storage": 448}},
                "native_bytes_delta_full_generation_background":
                    {"remote_read": 7 * 16777216, "direct_receive": 7 * 16777216},
                "owner_stats_delta_full_generation_background":
                    {"posix_read_bytes": 7 * 16777216, "ssd_metadata_read_bytes": 7 * 4096}}

    def test_direct_remote_requires_complete_native_receive_and_positive_owner_ssd(self):
        row = self.remote()
        self.assertTrue(benchmark.verify_tier(row, "warm_remote_ssd", "4\n", 64, "native_direct")["exclusive_tier_verified"])
        for component, value in (("direct_receive", 0), ("direct_receive", 16777216), ("posix_read", 4096)):
            bad = copy.deepcopy(row)
            bad["native_bytes_delta_full_generation_background"][component] = value
            with self.assertRaises(RuntimeError):
                benchmark.verify_tier(bad, "warm_remote_ssd", "4\n", 64, "native_direct")
        for component in ("posix_read_bytes", "ssd_metadata_read_bytes"):
            bad = copy.deepcopy(row)
            bad["owner_stats_delta_full_generation_background"][component] = 0
            with self.assertRaises(RuntimeError):
                benchmark.verify_tier(bad, "warm_remote_ssd", "4\n", 64, "native_direct")

    def test_partial_and_mixed_hits_and_output_mismatch_are_never_samples(self):
        for cache in ({"storage": 384}, {"storage": 448, "device": 1}, {"storage": 448, "host": 1}):
            row = self.remote()
            row["meta_info"]["cached_tokens_details"] = cache
            with self.assertRaises(RuntimeError):
                benchmark.verify_tier(row, "warm_remote_ssd", "4\n", 64, "native_direct")
        with self.assertRaises(RuntimeError):
            benchmark.verify_tier(self.remote(), "warm_remote_ssd", "5\n", 64, "native_direct")

    def test_staged_remote_has_no_direct_bytes_but_separate_store_copy_is_allowed(self):
        row = self.remote()
        row["native_bytes_delta_full_generation_background"].update(direct_receive=0, staging_copy=117440512)
        self.assertTrue(benchmark.verify_tier(row, "warm_remote_ssd", "4\n", 64, "native_staged")["exact_output_match"])
        with self.assertRaises(RuntimeError):
            benchmark.verify_tier(self.remote(), "warm_remote_ssd", "4\n", 64, "native_staged")

    def test_cold_control_rejects_any_cache_source(self):
        row = self.remote()
        with self.assertRaises(RuntimeError):
            benchmark.verify_tier(row, "cold", "4\n", 64)
        row["meta_info"] = {"cached_tokens": 0, "cached_tokens_details": {"device": 0, "host": 0, "storage": 0}}
        self.assertTrue(benchmark.verify_tier(row, "cold", "4\n", 64)["semantic_answer_verified"])
