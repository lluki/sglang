# SPDX-License-Identifier: Apache-2.0
"""Opt-in request-scoped CUPTI activity, with measured CPU/GPU clock bounds.

SGLANG_REQUEST_CUDA_ACTIVITY=1 enables this diagnostic mode. One request may
be active per scheduler process. Raw profiler arguments, inputs, stacks and
tensor addresses are never written; only selected timing/correlation fields
reach the request timeline. GPU clock probes synchronize their own stream;
serving kernels and transfer streams retain their normal scheduling.
"""
import contextlib
import heapq
import os
import threading
import time
from collections import defaultdict

from . import request_timeline

_enabled = os.environ.get("SGLANG_REQUEST_CUDA_ACTIVITY") == "1"
_current = None


def wall_anchor():
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return {"monotonic_before_ns": before, "wall_ns": wall,
            "monotonic_after_ns": after}


def anchor_offset_bounds(anchor):
    return (anchor["monotonic_before_ns"] - anchor["wall_ns"],
            anchor["monotonic_after_ns"] - anchor["wall_ns"])


def clock_probe_bounds(host_start, host_end, gpu_start, gpu_end):
    """GPU minus CPU offset; a completed kernel lies inside its host bracket."""
    if host_start > host_end or gpu_start > gpu_end:
        raise ValueError("invalid CUDA clock probe interval")
    lower, upper = gpu_end - host_end, gpu_start - host_start
    if lower > upper:
        raise ValueError("CUDA probe cannot fit its measured host bracket")
    return lower, upper


def _clock_probe(state, label):
    import torch
    for index in range(5):
        unique = f"{label}:{index}"
        with torch.cuda.stream(state["probe_stream"]), torch.profiler.record_function("sglang:clock_probe:" + unique):
            before = time.monotonic_ns()
            state["probe_tensor"].zero_()
            state["probe_stream"].synchronize()
            after = time.monotonic_ns()
        state["probes"].append({"label": unique, "site": label, "probe_index": index,
                                "host_start_ns": before, "host_end_ns": after})


def calibrate_probe_groups(calibrated, expected_per_site=5):
    """Intersect compatible same-site probes; retain a conservative endpoint hull."""
    groups = []
    for site in ("before", "after"):
        probes = [p for p in calibrated if p.get("site", p["label"].split(":", 1)[0]) == site]
        indices = {p.get("probe_index") for p in probes}
        complete = (len(probes) == expected_per_site and indices == set(range(expected_per_site))
                    and all(p.get("actual_gpu_kernels", 0) > 0 for p in probes))
        lower = max((p["offset_lower_ns"] for p in probes), default=None)
        upper = min((p["offset_upper_ns"] for p in probes), default=None)
        compatible = bool(probes) and lower <= upper
        groups.append({"site": site, "probe_count": len(probes), "expected_probe_count": expected_per_site,
                       "complete": complete, "compatible": compatible,
                       "offset_lower_ns": lower, "offset_upper_ns": upper,
                       "host_start_ns": min((p["host_start_ns"] for p in probes), default=None),
                       "host_end_ns": max((p["host_end_ns"] for p in probes), default=None),
                       "labels": [p["label"] for p in probes]})
    verified = all(g["complete"] and g["compatible"] for g in groups)
    hull = ({"offset_lower_ns": min(g["offset_lower_ns"] for g in groups),
             "offset_upper_ns": max(g["offset_upper_ns"] for g in groups),
             "valid_start_ns": groups[0]["host_start_ns"], "valid_end_ns": groups[1]["host_end_ns"]}
            if verified else None)
    return {"gpu_probe_groups": groups, "gpu_clock_mapping_verified": verified,
            "gpu_request_offset_hull": hull,
            "gpu_clock_conversion_model": "Kineto absolute CPU-wall timestamps mapped by measured wall/monotonic brackets; GPU residual offset measured by completed private-stream kernels",
            "gpu_clock_validity_model": "constant residual offset within each sequential five-probe site; bounded endpoint-hull drift over this enclosed request window; no extrapolation and no request-wide constant offset assumption"}


def start(rid):
    global _current
    if not _enabled or not request_timeline.enabled() or not rid:
        return
    if _current is not None:
        if _current["rid"] != rid:
            request_timeline.emit(rid, "cuda_activity_error",
                                  reason="another request profiler is active")
        return
    started = time.monotonic_ns()
    import torch
    if not torch.cuda.is_available():
        request_timeline.emit(rid, "cuda_activity_error", reason="CUDA unavailable")
        return
    try:
        # This tensor contains no model/cache data and is allocated before capture.
        probe_tensor = torch.empty(1, device="cuda")
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=torch._C._profiler._ExperimentalConfig(
                profile_all_threads=True, enable_cuda_sync_events=True,
                expose_kineto_event_metadata=True),
        )
        state = {"rid": str(rid), "profile": profile, "probe_tensor": probe_tensor,
                 "probe_stream": torch.cuda.Stream(),
                 "thread": threading.get_ident(), "anchors": [wall_anchor()],
                 "probes": []}
        profile.start()
        _current = state
        _clock_probe(state, "before")
    except Exception as error:
        if _current is not None:
            try:
                _current["profile"].stop()
            except Exception:
                pass
        _current = None
        request_timeline.emit(rid, "cuda_activity_error",
                              reason=type(error).__name__)
    finally:
        request_timeline.emit(rid, "cuda_profile_setup", started, time.monotonic_ns(),
                              category="framework_cpu", operation="DIAGNOSTIC",
                              source="profiler startup and private-stream clock calibration")


def scope(stage):
    if _current is None:
        return contextlib.nullcontext()
    import torch
    return torch.profiler.record_function("sglang:" + stage)


def _method(event, name, default=None):
    try:
        return getattr(event, name)()
    except (AttributeError, RuntimeError, TypeError):
        return default


def _stage_annotation(name):
    # C++ custom operators use the sglang:: namespace; only our explicit
    # record_function scopes carry request-stage attribution.
    return name in ("sglang:h2d", "sglang:model_prefill", "sglang:model_decode",
                    "sglang:d2h_backup") or name.startswith("sglang:clock_probe:")


def select_events(events, anchors, probes):
    """Extract absolute correlated activity; preserve missing joins explicitly."""
    bounds = [anchor_offset_bounds(a) for a in anchors]
    offset = (min(a for a, _ in bounds) + max(b for _, b in bounds)) // 2 if bounds else 0
    wall_error = max((abs(endpoint - offset) for pair in bounds for endpoint in pair), default=0)
    entries = []
    for index, event in enumerate(events):
        start, end = _method(event, "start_ns"), _method(event, "end_ns")
        if start is None or end is None or end < start:
            continue
        entries.append({
            "index": index, "name": str(_method(event, "name", ""))[:512],
            "start_ns": int(start) + offset, "end_ns": int(end) + offset,
            "device_type": str(_method(event, "device_type", "")),
            "device_index": _method(event, "device_index", -1),
            "stream_id": _method(event, "device_resource_id", -1),
            "thread_id": _method(event, "start_thread_id", -1),
            "correlation_id": _method(event, "correlation_id", 0),
            "linked_correlation_id": _method(event, "linked_correlation_id", 0),
            "external_id": _method(event, "external_id", 0),
            "flow_id": _method(event, "flow_id", 0),
            "flow_start": _method(event, "flow_start", False),
            "flow_type": str(_method(event, "flow_type", "")),
            "bytes": _method(event, "nbytes", 0),
        })
    annotations = [e for e in entries if "CPU" in e["device_type"] and _stage_annotation(e["name"])]
    cpu = [e for e in entries if "CPU" in e["device_type"]]
    by_correlation, by_external = defaultdict(list), defaultdict(list)
    for entry in cpu:
        if entry["correlation_id"]:
            by_correlation[entry["correlation_id"]].append(entry)
        if entry["external_id"]:
            by_external[entry["external_id"]].append(entry)
    annotations_by_thread = defaultdict(list)
    for annotation in annotations:
        annotations_by_thread[annotation["thread_id"]].append(annotation)
    cpu_by_thread = defaultdict(list)
    for entry in cpu:
        cpu_by_thread[entry["thread_id"]].append(entry)
    annotation_for_cpu = {}
    for thread, thread_cpu in cpu_by_thread.items():
        thread_annotations = sorted(annotations_by_thread[thread], key=lambda e: e["start_ns"])
        active, position = [], 0
        for entry in sorted(thread_cpu, key=lambda e: e["start_ns"]):
            while position < len(thread_annotations) and thread_annotations[position]["start_ns"] <= entry["start_ns"]:
                annotation = thread_annotations[position]
                heapq.heappush(active, (annotation["end_ns"] - annotation["start_ns"], annotation["index"], annotation))
                position += 1
            while active and active[0][2]["end_ns"] <= entry["start_ns"]:
                heapq.heappop(active)
            if active:
                annotation_for_cpu[entry["index"]] = active[0][2]
    selected = []
    retained_cpu = {}
    probe_events = {}
    for event in entries:
        if "CUDA" not in event["device_type"]:
            continue
        links = {event["correlation_id"], event["linked_correlation_id"]}
        links.discard(0)
        matches = {entry["index"]: entry for link in links for entry in by_correlation.get(link, ())}
        if event["external_id"]:
            matches.update((entry["index"], entry) for entry in by_external.get(event["external_id"], ()))
        launch = sorted(matches.values(), key=lambda entry: entry["index"])
        containers = [annotation_for_cpu[e["index"]] for e in launch if e["index"] in annotation_for_cpu]
        parent = min(containers, key=lambda a: a["end_ns"]-a["start_ns"]) if containers else None
        stage = parent["name"][7:] if parent else None
        gpu_envelope = _stage_annotation(event["name"]) or event["name"] in (
            "Context Sync", "Stream Sync", "Event Sync", "Stream Wait Event")
        if stage and stage.startswith("clock_probe:"):
            # CUPTI synchronization intervals can begin at earlier queued work.
            # Only actual probe kernels/copies establish the host-bracket bound.
            if not gpu_envelope:
                probe_events.setdefault(stage.split(":", 1)[1], []).append(event)
            continue
        if stage == "h2d":
            category = "h2d"
        elif stage in ("model_prefill", "model_decode"):
            category = "model_prefill"
        elif stage == "d2h_backup":
            category = "d2h_backup"
        else:
            category = "other"
        event.update(category=category, parent_annotation=parent["index"] if parent else None,
                     launch_event_indices=[e["index"] for e in launch],
                     joined_stage=stage, source="CUPTI activity via PyTorch Kineto",
                     envelope=gpu_envelope)
        selected.append(event)
        for entry in launch:
            retained_cpu[entry["index"]] = dict(entry, category="framework_cpu", envelope=False,
                                                joined_stage=stage, source="CUPTI correlated CPU launch")
        if parent:
            retained_cpu[parent["index"]] = dict(parent, category="framework_cpu", envelope=True,
                                                 joined_stage=stage, source="PyTorch named CPU annotation")
    calibrated = []
    # Per-thread profiler scopes are nested; a sorted stack retains that
    # hierarchy without comparing every CPU event with every other event.
    stacks = defaultdict(list)
    for event in sorted(retained_cpu.values(), key=lambda e: (e["thread_id"], e["start_ns"], -e["end_ns"], e["index"])):
        stack = stacks[event["thread_id"]]
        while stack and (stack[-1]["end_ns"] <= event["start_ns"] or stack[-1]["end_ns"] < event["end_ns"]):
            stack.pop()
        if stack:
            stack[-1]["envelope"] = True
            event["cpu_parent_index"] = stack[-1]["index"]
        event["envelope"] = event["envelope"] or "Synchronize" in event["name"] or "WaitEvent" in event["name"]
        stack.append(event)
    for probe in probes:
        gpu = probe_events.get(probe["label"], [])
        if not gpu:
            continue
        first = min(e["start_ns"] for e in gpu)
        last = max(e["end_ns"] for e in gpu)
        # Preserve even a rejected initial CUPTI probe's actual intervals;
        # contradictory bounds invalidate its site instead of discarding proof.
        lower, upper = last - probe["host_end_ns"], first - probe["host_start_ns"]
        calibrated.append(dict(probe, gpu_start_ns=first, gpu_end_ns=last, actual_gpu_kernels=len(gpu), gpu_kernels=gpu,
                               offset_lower_ns=lower, offset_upper_ns=upper,
                               probe_interval_compatible=lower <= upper))
    return sorted(selected + list(retained_cpu.values()), key=lambda e: e["index"]), {
        "wall_monotonic_anchors": anchors, "wall_anchor_uncertainty_ns": wall_error,
        "gpu_probe_host_brackets": probes, "gpu_probes": calibrated, **calibrate_probe_groups(calibrated),
        "gpu_clock_probe_scope": "CUPTI-converted GPU minus CPU monotonic offset; each kernel bracketed by CPU launch and synchronized completion",
        "correlation_join": "GPU correlation/external IDs to CPU launch contained by named stage annotation",
    }


def stop(rid):
    global _current
    state = _current
    if state is None or state["rid"] != str(rid):
        return
    if state["thread"] != threading.get_ident():
        request_timeline.emit(rid, "cuda_activity_error",
                              reason="profiler completion on another thread")
        return
    started = time.monotonic_ns()
    try:
        _clock_probe(state, "after")
        state["profile"].stop()
        state["anchors"].append(wall_anchor())
        events = state["profile"].profiler.kineto_results.events()
        selected, calibration = select_events(events, state["anchors"], state["probes"])
        request_timeline.emit(rid, "cuda_clock_calibration", **calibration)
        for event in selected:
            fields = dict(event)
            begin, end = fields.pop("start_ns"), fields.pop("end_ns")
            index = fields.pop("index")
            request_timeline.emit(
                rid, "cuda_activity", begin, end,
                span_id=f"{os.getpid()}:cupti:{rid}:{index}",
                clock_domain="requester:CLOCK_MONOTONIC",
                operation="PUT" if fields["joined_stage"] == "d2h_backup" else "GET" if fields["category"] == "h2d" else "MODEL",
                **fields)
    except Exception as error:
        request_timeline.emit(rid, "cuda_activity_error",
                              reason=type(error).__name__)
    finally:
        _current = None
        request_timeline.emit(rid, "cuda_profile_collection", started,
                              time.monotonic_ns(), category="framework_cpu",
                              source="diagnostic collection after generation")


def mark_completed(rid):
    if _current is not None and _current["rid"] == str(rid):
        _current["completed"] = True


def flush_completed():
    """Collect after scheduler output has been queued for detokenization."""
    if _current is not None and _current.get("completed"):
        request_timeline.emit(_current["rid"], "scheduler_profile_output_queued")
        stop(_current["rid"])
