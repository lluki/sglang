"""Opt-in monotonic timing markers for L3 latency investigations.

Set SGLANG_L3_TIMING_PATH to a writable JSONL file before starting workers.
Each record is one atomic append, so worker processes can share the path.
"""

import json
import os
import time

_PATH = os.environ.get("SGLANG_L3_TIMING_PATH")


def trace_l3(event: str, rid: str, **fields) -> None:
    if not _PATH:
        return
    record = {
        "event": event,
        "ts_ns": time.monotonic_ns(),
        "rid": rid,
        "pid": os.getpid(),
        **fields,
    }
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    fd = os.open(_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
