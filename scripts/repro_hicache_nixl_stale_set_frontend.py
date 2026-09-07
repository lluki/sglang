#!/usr/bin/env python3
"""Reproduce a HiCache stale SET through the OpenAI completions API.

This script launches a real SGLang server with write-back HiCache and the NIXL
POSIX backend, creates a radix split while a delayed zero-copy SET is in flight,
then reloads the original prompt from L3 and compares its completion.

Example (run from an SGLang checkout):

    python3 scripts/repro_hicache_nixl_stale_set_frontend.py \
        --model Qwen/Qwen2.5-0.5B --gpu 0 --expect clean

Use ``--expect corruption`` on vulnerable code. The delay is enabled by the
test-only ``SGLANG_HICACHE_NIXL_SET_DELAY_S`` hook in this change.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def repeated_tokens(tokenizer, text: str, count: int) -> list[int]:
    unit = tokenizer.encode(text, add_special_tokens=False)
    if not unit:
        raise RuntimeError(f"tokenizer produced no tokens for {text!r}")
    return (unit * ((count + len(unit) - 1) // len(unit)))[:count]


def tokens_ending_in(tokenizer, filler: str, ending: str, count: int) -> list[int]:
    ending_ids = tokenizer.encode(ending, add_special_tokens=False)
    if len(ending_ids) >= count:
        raise ValueError(f"ending needs {len(ending_ids)} tokens, budget is {count}")
    return repeated_tokens(tokenizer, filler, count - len(ending_ids)) + ending_ids


def wait_for_server(base_url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"SGLang did not become healthy within {timeout_s}s")


def post_until_ok(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            return
        if response.status_code != 400:
            response.raise_for_status()
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {url}")


def flush_when_idle(base_url: str, timeout_s: float) -> None:
    response = requests.post(
        f"{base_url}/flush_cache",
        params={"timeout": timeout_s},
        timeout=timeout_s + 10,
    )
    response.raise_for_status()


def complete(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    label: str,
    max_tokens: int,
) -> dict:
    started = time.monotonic()
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt_ids,
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "return_token_ids": True,
            "return_cached_tokens_details": True,
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    result = {
        "label": label,
        "text": choice["text"],
        "token_ids": choice.get("token_ids"),
        "cached_tokens_details": (body.get("sglext") or {}).get(
            "cached_tokens_details"
        ),
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_requests(args, base_url: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    common = tokens_ending_in(
        tokenizer,
        " Background material about ordinary colors and shapes.",
        "\nImportant fact: the secret answer is ZEBRA4821.\n",
        args.common_tokens,
    )
    victim = common + tokens_ending_in(
        tokenizer,
        " Read the background carefully before answering.",
        "\nQuestion: What is the secret answer? Answer:",
        args.suffix_tokens,
    )
    diverger = common + tokens_ending_in(
        tokenizer,
        " This is a separate branch with an unrelated task.",
        "\nQuestion: Name a primary color. Answer:",
        args.splitter_suffix_tokens,
    )
    print(
        f"victim={len(victim)} tokens split_at={len(common)} "
        f"splitter={len(diverger)} device_pressure={args.device_pressure_tokens} "
        f"reclaimer={args.reclaimer_tokens}",
        flush=True,
    )

    post_until_ok(f"{base_url}/flush_cache")
    post_until_ok(f"{base_url}/hicache/storage-backend/clear")
    baseline = complete(base_url, args.model, victim, "victim-before", args.max_tokens)

    # Evict the victim from GPU. Its D->H completion starts the delayed SET.
    device_pressure = repeated_tokens(
        tokenizer,
        " unrelated device eviction stream orange violet silver",
        args.device_pressure_tokens,
    )
    complete(
        base_url,
        args.model,
        device_pressure,
        "evict-victim-to-host",
        args.max_tokens,
    )

    # Reload the common host prefix and diverge, splitting the in-flight node.
    complete(base_url, args.model, diverger, "splitter", args.max_tokens)

    # Force another D->H backup. Vulnerable code reclaims the split prefix.
    reclaimer = repeated_tokens(
        tokenizer,
        " unrelated host reclamation stream cyan magenta yellow",
        args.reclaimer_tokens,
    )
    complete(
        base_url,
        args.model,
        reclaimer,
        "force-host-reuse",
        args.max_tokens,
    )

    # The victim SET and at most one later SET are serialized by the backend.
    settle_s = args.set_delay * 2 + 5
    print(f"waiting {settle_s:.1f}s for delayed NIXL SETs", flush=True)
    time.sleep(settle_s)
    flush_when_idle(base_url, args.set_delay * 3 + 10)

    replay = complete(base_url, args.model, victim, "victim-from-l3", args.max_tokens)
    details = replay["cached_tokens_details"] or {}
    storage_tokens = details.get("storage", 0)
    if storage_tokens <= 0:
        raise RuntimeError(
            "replay did not load any tokens from L3; the race was not exercised"
        )

    changed = baseline["token_ids"] != replay["token_ids"]
    result = {
        "baseline": baseline,
        "replay": replay,
        "completion_changed": changed,
        "storage_hit_tokens": storage_tokens,
        "verdict": "CORRUPTION_REPRODUCED" if changed else "CLEAN",
    }
    print(result["verdict"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--sglang-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="checkout whose python/ package is used by the launched server",
    )
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value (inherited if omitted)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tree-core", choices=("python", "rust"), default="python")
    parser.add_argument("--set-delay", type=float, default=15.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--server-log", type=Path, default=Path("/tmp/sglang-stale-set.log"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/sglang-stale-set.json"))
    parser.add_argument("--common-tokens", type=int, default=704)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--splitter-suffix-tokens", type=int, default=704)
    parser.add_argument("--device-pressure-tokens", type=int, default=3500)
    parser.add_argument("--reclaimer-tokens", type=int, default=3000)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--expect", choices=("corruption", "clean", "either"), default="either"
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    with tempfile.TemporaryDirectory(prefix="sglang-nixl-stale-") as storage_dir:
        server_env = dict(os.environ)
        checkout_python = str(args.sglang_repo.resolve() / "python")
        inherited_pythonpath = server_env.get("PYTHONPATH")
        server_env["PYTHONPATH"] = (
            checkout_python
            if not inherited_pythonpath
            else checkout_python + os.pathsep + inherited_pythonpath
        )
        if args.gpu is not None:
            server_env["CUDA_VISIBLE_DEVICES"] = args.gpu
        server_env.update(
            {
                "SGLANG_HICACHE_NIXL_BACKEND_PLUGIN": "POSIX",
                "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR": storage_dir,
                "SGLANG_HICACHE_NIXL_USE_DIRECT_IO": "0",
                "SGLANG_HICACHE_NIXL_SET_DELAY_S": str(args.set_delay),
                "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": args.tree_core,
            }
        )
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--tp",
            "1",
            "--page-size",
            "64",
            "--hicache-mem-layout",
            "page_first",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "0.20",
            "--hicache-write-policy",
            "write_back",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "nixl",
            "--hicache-storage-backend-extra-config",
            '{"plugin":{"posix":{"use_aio":"true","active":true}}}',
            "--max-total-tokens",
            "4096",
            "--disable-cuda-graph",
        ]
        args.server_log.parent.mkdir(parents=True, exist_ok=True)
        print("starting:", " ".join(command), flush=True)
        print(f"server log: {args.server_log}", flush=True)
        with args.server_log.open("w") as server_log:
            process = subprocess.Popen(
                command,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_for_server(base_url, process, args.startup_timeout)
                result = run_requests(args, base_url)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            finally:
                stop_server(process)

    changed = result["completion_changed"]
    if args.expect == "corruption" and not changed:
        raise SystemExit("expected completion corruption, but replay matched")
    if args.expect == "clean" and changed:
        raise SystemExit("expected a clean replay, but completion changed")


if __name__ == "__main__":
    main()
