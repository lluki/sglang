#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure real first-nonempty-token latency, failing on unverified cache tiers.

Run against an otherwise idle SGLang HiCache server with metrics enabled.
Preparation, pressure requests, flushes and complete-generation counter windows
are retained separately; cumulative native timers are not causal TTFT slices.
"""
import argparse
import json
import math
import pathlib
import re
import statistics
import time
import urllib.request
import uuid


def verify_tier(record, tier, expected, page_size, receive_mode=None):
    """Reject partial/mixed cache paths and mismatched deterministic outputs."""
    if record["text"] != expected:
        raise RuntimeError(f"{tier}: output differs from full recomputation")
    length = record["context_tokens"]
    cache = record["meta_info"].get("cached_tokens_details") or {}
    delta = record["native_bytes_delta_full_generation_background"]
    source = {"hot_gpu": "device", "warm_host": "host", "warm_local_ssd": "storage", "warm_remote_ssd": "storage"}.get(tier)
    if tier == "cold":
        if record["meta_info"].get("cached_tokens", 0) or any(cache.values()):
            raise RuntimeError("cold recomputation contains cache hits")
    else:
        if cache.get(source, 0) < length - page_size:
            raise RuntimeError(f"{tier}: insufficient positive tier witness: {cache}")
        if any(cache.get(other, 0) for other in {"device", "host", "storage"} - {source}):
            raise RuntimeError(f"{tier}: mixed cache source: {cache}")
    if tier == "warm_local_ssd" and (delta.get("posix_read", 0) <= 0 or delta.get("remote_read", 0)):
        raise RuntimeError(f"local SSD payload witness missing: {delta}")
    if tier == "warm_remote_ssd":
        remote = delta.get("remote_read", 0)
        direct = delta.get("direct_receive", 0)
        if remote <= 0 or delta.get("posix_read", 0):
            raise RuntimeError(f"remote payload witness missing or local payload present: {delta}")
        if receive_mode == "native_direct" and direct != remote:
            raise RuntimeError("successful remote GET bytes lack a complete native direct receive witness")
        if receive_mode == "native_staged" and direct:
            raise RuntimeError("staged receive unexpectedly contains direct GET bytes")
        owner = record.get("owner_stats_delta_full_generation_background")
        # Native metadata_read_bytes takes precedence over its historical alias.
        metadata = 0 if owner is None else owner.get("metadata_read_bytes", owner.get("ssd_metadata_read_bytes", 0))
        if owner is not None and (owner.get("posix_read_bytes", 0) != remote or metadata <= 0):
            raise RuntimeError("remote GET lacks matched positive owner payload/SSD metadata witnesses")
    return {"exact_output_match": True, "exclusive_tier_verified": True,
            "cache": cache, "semantic_answer_verified": expected.lstrip().startswith("4")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:19400")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--admin-key-file")
    parser.add_argument("--owner-url", help="Read-only counter HTTP endpoint for the isolated SSD owner")
    parser.add_argument("--receive-mode", choices=["native_direct", "native_staged"])
    parser.add_argument("--validation-attempts", type=int, default=1,
                        help="Bounded remote convergence attempts; failed setup rows are excluded")
    parser.add_argument("--timeline-dir", help="Wait for diagnostic collection before admitting the next request")
    parser.add_argument("--replay-from", help="Prior requests.jsonl: reuse exact cold prompts/salts for remote-only hits")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-only", action="store_true", help="Compile test shapes and flush RAM without taking tier samples")
    parser.add_argument("--pressure-tokens", type=int, default=8064)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--tiers", nargs="+", choices=["cold", "hot_gpu", "warm_host", "warm_local_ssd", "warm_remote_ssd"],
                        default=["cold", "hot_gpu", "warm_host", "warm_local_ssd"])
    args = parser.parse_args()
    if args.repetitions < 1 or any(n < 2 * args.page_size for n in args.contexts):
        parser.error("positive repetitions and at least two pages per context required")
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    run = uuid.uuid4().hex
    records = []
    request_number = 0
    measured = []
    timeline_offsets = {}
    collected = set()
    headers = {}
    if args.admin_key_file:
        headers["Authorization"] = "Bearer " + pathlib.Path(args.admin_key_file).read_text().strip()

    def text_get(path):
        with urllib.request.urlopen(urllib.request.Request(args.url + path, headers=headers), timeout=60) as response:
            return response.read().decode()

    def snapshot():
        raw = text_get("/metrics")
        counters = {}
        for line in raw.splitlines():
            if line.startswith("sglang:nixlshard_component_bytes_total{"):
                component = re.search(r'component="([^"]+)"', line)
                if component:
                    key = component.group(1)
                    counters[key] = counters.get(key, 0) + float(line.rsplit(" ", 1)[1])
        return raw, counters

    def owner_snapshot():
        if not args.owner_url:
            return None
        with urllib.request.urlopen(args.owner_url.rstrip("/") + "/stats", timeout=30) as response:
            return json.load(response)

    def wait_for_collection(identifier):
        if not args.timeline_dir:
            return
        deadline = time.monotonic() + 120
        while identifier not in collected:
            for path in pathlib.Path(args.timeline_dir).glob("request-timeline-*.jsonl"):
                with path.open() as stream:
                    stream.seek(timeline_offsets.get(path, 0))
                    while True:
                        position = stream.tell()
                        line = stream.readline()
                        if not line:break
                        if not line.endswith("\n"):
                            stream.seek(position);break
                        if '"stage":"cuda_profile_collection"' in line:
                            collected.add(json.loads(line)["rid"])
                    timeline_offsets[path] = stream.tell()
            if time.monotonic() >= deadline:
                raise RuntimeError("diagnostic collection did not finish before next request admission")
            if identifier not in collected:
                time.sleep(.05)

    def accept(record, tier, repetition, expected):
        witnesses = verify_tier(record, tier, expected, args.page_size, args.receive_mode)
        row = {key: value for key, value in record.items() if key not in {"cache_salt", "input_ids", "events"}}
        row.update(repetition=repetition, receive_mode=args.receive_mode,
                   clock_domain="requester:CLOCK_MONOTONIC", output_witnesses=witnesses,
                   accepted=True)
        with (output / "measured-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        measured.append(row)

    def generate(tokens, salt, label):
        nonlocal request_number
        request_number += 1
        identifier = f"{run}-{request_number}"
        before_raw, before = snapshot()
        owner_before = owner_snapshot()
        payload = {"input_ids": tokens, "cache_salt": salt, "stream": True, "rid": identifier,
                   "sampling_params": {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True}}
        request = urllib.request.Request(args.url + "/generate", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", **headers})
        started = time.monotonic_ns()
        first = None
        events = []
        final = None
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                received = time.monotonic_ns()
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                event = json.loads(body)
                events.append({"received_ns": received, "event": event})
                final = event
                if first is None and event.get("text"):
                    first = received
        ended = time.monotonic_ns()
        if first is None or final is None:
            raise RuntimeError(f"{label}: no nonempty first-token event")
        wait_for_collection(identifier)
        # This wait is outside TTFT and allows asynchronous backups to finish.
        time.sleep(.5)
        after_raw, after = snapshot()
        owner_after = owner_snapshot()
        info = final["meta_info"]
        record = {"request_id": identifier, "label": label, "context_tokens": len(tokens),
                  "cache_salt": salt, "input_ids": tokens, "ttft_ns": first-started, "generation_ns": ended-started,
                  "started_ns": started, "first_nonempty_event_ns": first, "ended_ns": ended,
                  "text": final["text"], "meta_info": info, "events": events,
                  "native_bytes_delta_full_generation_background": {
                      key: after.get(key, 0)-before.get(key, 0) for key in set(before) | set(after)}}
        if owner_before is not None:
            old, new = owner_before["stats"], owner_after["stats"]
            record["owner_build_marker"] = owner_after["build_marker"]
            record["owner_stats_delta_full_generation_background"] = {
                key: new.get(key, 0) - old.get(key, 0) for key in set(old) | set(new)
                if isinstance(new.get(key, 0), (int, float)) and isinstance(old.get(key, 0), (int, float))}
            (output / f"{identifier}-owner-before.json").write_text(json.dumps(owner_before))
            (output / f"{identifier}-owner-after.json").write_text(json.dumps(owner_after))
        (output / f"{identifier}-metrics-before.prom").write_text(before_raw)
        (output / f"{identifier}-metrics-after.prom").write_text(after_raw)
        records.append(record)
        with (output / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({key: record[key] for key in ("label", "context_tokens", "ttft_ns", "text")}
                         | {"cache": info.get("cached_tokens_details")}), flush=True)
        return record

    def flush():
        request = urllib.request.Request(args.url + "/flush_cache?timeout=30", data=b"", method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError("RAM cache flush failed")
        time.sleep(.5)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    empty = tokenizer.apply_chat_template([{"role": "user", "content":
                                           "What is two plus two? Answer with only the number."}],
                                          tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    # Insert inert contextual material before the final question; preserve the
    # chat template's assistant prefix and require the semantic answer to start 4.
    prefix = tokenizer.encode("Context notes: ", add_special_tokens=False)
    filler = tokenizer.encode(" The sky is blue and the grass is green.", add_special_tokens=False)

    def prompt(length):
        room = length - len(empty) - len(prefix)
        if room < 0:
            raise ValueError("context is too short for prompt")
        material = (filler * math.ceil(room / len(filler)))[:room]
        return empty[:3] + prefix + material + empty[3:]

    def redact(value):
        if isinstance(value, dict):
            # server_info also embeds credentials in its shell launch command.
            return {key: "<redacted>" if key in {"api_key", "admin_api_key", "launch_command"} else redact(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    (output / "server-info.json").write_text(json.dumps(redact(json.loads(text_get("/server_info"))), indent=2))
    (output / "configuration.json").write_text(json.dumps(vars(args) | {"run": run}, indent=2))
    if args.replay_from:
        if args.tiers != ["warm_remote_ssd"]:
            parser.error("--replay-from requires --tiers warm_remote_ssd")
        prior = [json.loads(line) for line in pathlib.Path(args.replay_from).read_text().splitlines()]
        for length in args.contexts:
            candidates = [row for row in prior if row["label"] == "cold" and row["context_tokens"] == length]
            if not candidates or "input_ids" not in candidates[-1]:
                raise RuntimeError(f"no exact cold prompt/salt reference for context {length}")
            reference = candidates[-1]
            for repetition in range(args.repetitions):
                for attempt in range(args.validation_attempts):
                    flush()
                    record = generate(reference["input_ids"], reference["cache_salt"], "warm_remote_ssd")
                    try:
                        accept(record, "warm_remote_ssd", repetition, reference["text"])
                    except RuntimeError as error:
                        with (output / "excluded-setup.jsonl").open("a") as stream:
                            stream.write(json.dumps({"request_id": record["request_id"], "context_tokens": length,
                                                     "repetition": repetition, "attempt": attempt,
                                                     "reason": str(error), "accepted": False}) + "\n")
                        if attempt + 1 == args.validation_attempts:
                            raise
                        time.sleep(1)
                    else:
                        break
        summarize(measured, args, output)
        return
    if "warm_remote_ssd" in args.tiers:
        parser.error("remote-only measurements require --replay-from")
    # Compile/warm actual shapes before measured requests, using isolated salts.
    for length in sorted(set(args.contexts + [args.pressure_tokens])):
        generate(prompt(length), f"{args.model_revision}:{run}:warmup:{length}", "setup_warmup")
    flush()
    if args.warmup_only:
        print("WARMUP_COMPLETE; no measured tier samples", flush=True)
        return

    for length in args.contexts:
        tokens = prompt(length)
        salt = f"{args.model_revision}:{run}:target:{length}"
        references = []
        for repetition in range(args.repetitions):
            cold_salt = salt if repetition == args.repetitions-1 else salt+f":cold:{repetition}"
            reference = generate(tokens, cold_salt, "cold" if "cold" in args.tiers else "setup_seed")
            if reference["meta_info"].get("cached_tokens", 0) != 0:
                raise RuntimeError("cold recomputation contains cache hits")
            if not reference["text"].lstrip().startswith("4"):
                raise RuntimeError("semantic output correctness failed")
            references.append(reference["text"])
            if "cold" in args.tiers:
                accept(reference, "cold", repetition, reference["text"])
        if len(set(references)) != 1:
            raise RuntimeError("deterministic cold outputs disagree")
        expected = references[-1]

        for tier in [tier for tier in args.tiers if tier != "cold"]:
            for repetition in range(args.repetitions):
                if tier == "warm_host":
                    generate(prompt(args.pressure_tokens), f"{args.model_revision}:{run}:pressure", "setup_pressure")
                elif tier == "warm_local_ssd":
                    flush()
                record = generate(tokens, salt, tier)
                accept(record, tier, repetition, expected)

    summarize(measured, args, output)


def summarize(records, args, output):
    summary = []
    for length in args.contexts:
        for tier in args.tiers:
            samples = sorted(r["ttft_ns"]/1e6 for r in records
                             if r["label"] == tier and r["context_tokens"] == length)
            if samples:
                summary.append({"context_tokens": length, "tier": tier, "count": len(samples),
                                "mean_ttft_ms": statistics.mean(samples), "median_ttft_ms": statistics.median(samples),
                                "p95_ttft_ms": samples[math.ceil(.95*len(samples))-1]})
    (output / "summary.json").write_text(json.dumps({"results": summary, "missing_tiers": [tier for tier in ["cold", "hot_gpu", "warm_host", "warm_local_ssd", "warm_remote_ssd"] if tier not in args.tiers],
        "causal_breakdown": "pending: full-generation counters are not request-joined critical-path attribution"}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
