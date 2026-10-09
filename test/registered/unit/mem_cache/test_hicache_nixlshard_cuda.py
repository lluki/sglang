# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for clock bounds and CUDA activity-to-request stage joins."""
import unittest
import time
from unittest.mock import patch
from sglang.srt.observability.cuda_request_profile import (
    anchor_offset_bounds, calibrate_probe_groups, clock_probe_bounds, select_events,
)


class Event:
    def __init__(self, **values):
        self.values = values
    def __getattr__(self, name):
        if name not in self.values:
            raise AttributeError(name)
        return lambda: self.values[name]


def event(name, start, end, device="CPU", corr=0, thread=1):
    return Event(name=name, start_ns=start, end_ns=end, device_type=device,
                 correlation_id=corr, start_thread_id=thread, device_resource_id=7)


class TestCUDARequestActivity(unittest.TestCase):
    def test_gpu_clock_offset_is_bounded_by_launch_and_completion(self):
        self.assertEqual(clock_probe_bounds(100, 200, 130, 170), (-30, 30))
        with self.assertRaises(ValueError):
            clock_probe_bounds(100, 105, 120, 140)
        self.assertEqual(anchor_offset_bounds({
            "monotonic_before_ns": 490, "wall_ns": 1000,
            "monotonic_after_ns": 510}), (-510, -490))

    def test_async_h2d_and_model_kernels_join_launch_annotation_not_gpu_wall_envelope(self):
        anchors = [{"monotonic_before_ns": 500, "wall_ns": 1000,
                    "monotonic_after_ns": 500}]
        events = [
            event("sglang:clock_probe:before", 1100, 1200),
            event("cudaLaunchKernel", 1120, 1140, corr=1),
            event("probe", 1140, 1150, "CUDA", corr=1),
            event("sglang:h2d", 1220, 1310, thread=2),
            event("cudaLaunchKernel", 1250, 1260, corr=2, thread=2),
            event("copy_kv", 1270, 1350, "CUDA", corr=2),
            event("sglang:model_prefill", 1320, 1440, thread=7),
            event("cudaLaunchKernel", 1350, 1360, corr=3, thread=7),
            event("attention", 1380, 1490, "CUDA", corr=3),
            event("sglang:clock_probe:after", 1500, 1600),
            event("cudaLaunchKernel", 1520, 1530, corr=4),
            event("probe", 1530, 1540, "CUDA", corr=4),
        ]
        selected, calibration = select_events(events, anchors, [
            {"label": "before", "host_start_ns": 620, "host_end_ns": 670},
            {"label": "after", "host_start_ns": 1010, "host_end_ns": 1060},
        ])
        cpu = [e for e in selected if "CPU" in e["device_type"]]
        self.assertEqual(len(cpu), 4)
        self.assertEqual(sum(e["envelope"] for e in cpu), 2)
        selected = [e for e in selected if "CUDA" in e["device_type"]]
        self.assertEqual([e["category"] for e in selected], ["h2d", "model_prefill"])
        self.assertEqual(selected[0]["launch_event_indices"], [4])
        self.assertEqual([e["start_ns"] for e in selected], [770, 880])
        self.assertEqual(selected[0]["end_ns"], 850)
        self.assertFalse(calibration["gpu_clock_mapping_verified"], "one probe per site cannot satisfy the five-probe contract")
        self.assertEqual(calibration["gpu_probes"][0]["offset_lower_ns"], -20)
        self.assertEqual(calibration["gpu_probes"][0]["offset_upper_ns"], 20)

    def test_custom_operator_namespace_preserves_outer_stage_and_actual_kernel(self):
        events = [
            event("sglang:model_prefill", 100, 200),
            event("sglang::custom_op", 120, 160),
            event("cudaLaunchKernel", 130, 140, corr=7),
            event("sglang::example_kernel", 150, 190, "CUDA", corr=7),
        ]
        selected, _ = select_events(events, [], [])
        kernel = next(e for e in selected if "CUDA" in e["device_type"])
        self.assertEqual(kernel["category"], "model_prefill")
        self.assertEqual(kernel["joined_stage"], "model_prefill")
        self.assertEqual(kernel["parent_annotation"], 0)
        self.assertEqual(kernel["launch_event_indices"], [2])
        self.assertFalse(kernel["envelope"])
        self.assertEqual((kernel["index"], kernel["name"], kernel["start_ns"],
                          kernel["end_ns"], kernel["correlation_id"]),
                         (3, "sglang::example_kernel", 150, 190, 7))

    def test_unjoined_gpu_activity_is_other_and_not_falsely_attributed_to_prefill(self):
        selected, calibration = select_events(
            [event("foreign_kernel", 10, 20, "CUDA")], [], [])
        self.assertEqual(selected[0]["category"], "other")
        self.assertIsNone(selected[0]["parent_annotation"])
        self.assertFalse(calibration["gpu_clock_mapping_verified"])

    def test_completion_marks_state_without_collecting_before_stream_delivery(self):
        from sglang.srt.observability import cuda_request_profile as profile
        state = {"rid": "test-request"}
        with patch.object(profile, "_current", state), patch.object(profile, "stop") as stop:
            profile.mark_completed("other-request")
            self.assertNotIn("completed", state)
            profile.mark_completed("test-request")
            stop.assert_not_called()
            profile.flush_completed()
            stop.assert_called_once_with("test-request")

    def groups(self):
        return [{"label": f"{site}:{i}", "site": site, "probe_index": i,
                 "host_start_ns": 1000 * group + 100 * i, "host_end_ns": 1000 * group + 100 * i + 80,
                 "offset_lower_ns": -10 + i, "offset_upper_ns": 20 - i, "actual_gpu_kernels": 1}
                for group, site in enumerate(("before", "after")) for i in range(5)]

    def test_complete_actual_probe_groups_intersect_only_at_each_site_and_use_endpoint_hull(self):
        probes = self.groups()
        for p in probes[5:]:
            p["offset_lower_ns"] += 5
            p["offset_upper_ns"] += 5
        value = calibrate_probe_groups(probes)
        self.assertTrue(value["gpu_clock_mapping_verified"])
        self.assertEqual(value["gpu_probe_groups"][0]["offset_lower_ns"], -6)
        self.assertEqual(value["gpu_probe_groups"][0]["offset_upper_ns"], 16)
        self.assertEqual(value["gpu_request_offset_hull"]["offset_lower_ns"], -6)
        self.assertEqual(value["gpu_request_offset_hull"]["offset_upper_ns"], 21)

    def test_missing_duplicate_or_contradictory_probe_groups_are_unverified(self):
        for kind in ("missing", "duplicate", "contradictory", "no-kernel"):
            probes = self.groups()
            if kind == "missing":probes.pop()
            elif kind == "duplicate":probes[-1]["probe_index"] = 0
            elif kind == "no-kernel":probes[-1]["actual_gpu_kernels"] = 0
            else:probes[-1]["offset_lower_ns"] = 100
            value = calibrate_probe_groups(probes)
            self.assertFalse(value["gpu_clock_mapping_verified"])
            self.assertIsNone(value["gpu_request_offset_hull"])

    def test_model_sized_correlations_and_cpu_hierarchy_have_bounded_collection_cost(self):
        events = []
        for i in range(10000):
            begin = 100 * i
            events.extend([event("sglang:model_prefill", begin, begin + 60),
                           event("cudaLaunchKernel", begin + 10, begin + 20, corr=i + 1),
                           event("kernel", begin + 25, begin + 30, "CUDA", corr=i + 1)])
        started = time.monotonic()
        selected, _ = select_events(events, [], [])
        self.assertEqual(sum("CUDA" in e["device_type"] for e in selected), 10000)
        self.assertLess(time.monotonic() - started, 3, "model-sized trace collection regressed to quadratic work")


if __name__ == "__main__":
    unittest.main()
