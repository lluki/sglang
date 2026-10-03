# SPDX-License-Identifier: Apache-2.0
"""Opt-in request diagnostics on the requester's CLOCK_MONOTONIC timeline.

Set SGLANG_REQUEST_TIMELINE_DIR before launch. One JSONL file per process
avoids cross-process append races. Records contain request IDs and timing
metadata, never token IDs, cache keys, tensor addresses or model outputs.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

_directory = os.environ.get("SGLANG_REQUEST_TIMELINE_DIR")
_lock = threading.Lock()
_warned = False


def enabled():
    return bool(_directory)


def first_output(state, rid, output):
    """Ignore empty stream frames and record the first token-bearing output once."""
    if (
        enabled()
        and not getattr(state, "_timeline_first_output", False)
        and (output.get("output_ids") or output.get("text"))
    ):
        state._timeline_first_output = True
        return emit(rid, "api_first_nonempty_output")
    return False


def emit(rid, stage, start_ns=None, end_ns=None, **fields):
    """Flush one bounded record; diagnostics errors never fail serving I/O."""
    global _warned
    if not _directory or not rid:
        return False
    now = time.monotonic_ns()
    try:
        start_ns = now if start_ns is None else int(start_ns)
        end_ns = start_ns if end_ns is None else int(end_ns)
        record = dict(
            schema_version=1,
            clock="CLOCK_MONOTONIC",
            rid=str(rid),
            pid=os.getpid(),
            stage=stage,
            start_ns=start_ns,
            end_ns=end_ns,
            **fields,
        )
        if start_ns < 0 or end_ns < start_ns:
            raise ValueError("invalid request timeline interval")
        data = json.dumps(record, separators=(",", ":")) + "\n"
        if len(data) > 1024 * 1024:
            raise ValueError("request timeline record exceeds 1 MiB")
        with _lock:
            directory = Path(_directory)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"request-timeline-{os.getpid()}.jsonl").open(
                "a"
            ) as stream:
                stream.write(data)
                stream.flush()
        return True
    except (OSError, ValueError, TypeError):
        if not _warned:
            logging.getLogger(__name__).warning(
                "request timeline diagnostics could not be written"
            )
            _warned = True
        return False
