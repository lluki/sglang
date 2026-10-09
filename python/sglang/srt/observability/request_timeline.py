# SPDX-License-Identifier: Apache-2.0
"""Opt-in request diagnostics on the requester's CLOCK_MONOTONIC timeline.

Set SGLANG_REQUEST_TIMELINE_DIR before launch. One JSONL file per process
avoids cross-process append races. Records contain request IDs and timing
metadata, never token IDs, cache keys, tensor addresses or model outputs.
"""

import json
import inspect
import itertools
import logging
import os
import sys
import threading
import time
from pathlib import Path
from contextlib import ExitStack, contextmanager
from functools import wraps

_directory = os.environ.get("SGLANG_REQUEST_TIMELINE_DIR")
_lock = threading.Lock()
_warned = False
_sequence = itertools.count()
_local = threading.local()


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
        return emit(
            rid,
            "api_first_nonempty_output",
            completion_token_count=output.get("meta_info", {}).get("completion_tokens"),
        )
    return False


def _request_ids(objects):
    """Select request membership only; never serialize request data."""
    if objects is None:
        return []
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    result = []
    for obj in objects:
        rid = getattr(obj, "rid", None) or getattr(obj, "request_id", None)
        if isinstance(rid, str) and rid and rid not in result:
            result.append(rid)
    return result


@contextmanager
def scope(rid, stage, *, category="framework_cpu", details=None, **fields):
    """Measure actual synchronous execution; nesting is not causal selection.

    Phase IDs encode a diagnostic sequence, never object/tensor addresses.
    Parent IDs identify this thread's enclosing phase for the same request.
    Consumers must subtract nested children and resolve GPU/CPU overlap.
    """
    if not enabled() or not rid:
        yield None
        return
    stacks = getattr(_local, "stacks", None)
    if stacks is None:
        stacks = _local.stacks = {}
    stack = stacks.setdefault(rid, [])
    span_id = f"{os.getpid()}:phase:{next(_sequence)}"
    parent_id = stack[-1] if stack else None
    stack.append(span_id)
    start = time.monotonic_ns()
    successful = False
    try:
        yield span_id
        successful = True
    finally:
        end = time.monotonic_ns()
        stack.pop()
        if not stack:
            stacks.pop(rid, None)
        emit(
            rid,
            stage,
            start,
            end,
            span_id=span_id,
            parent_id=parent_id,
            thread_id=threading.get_ident(),
            category=category,
            envelope=True,
            successful=successful,
            **fields,
            **(details or {}),
        )


def phase(stage, *, requests="request", forward_result=False, result_processing=False):
    """Instrument a named function without changing serving or synchronization.

    Pending membership is candidate membership, not evidence of admission.
    Forward/result identity survives overlapped execution via the result object.
    """

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            if not enabled():
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            owner = bound.get("self")
            batch = bound.get("batch")
            if requests == "inputs":
                members = bound.get("recv_reqs", [])
            elif requests == "pending":
                members = list(getattr(owner, "waiting_queue", []))
                for name in ("running_batch", "last_batch"):
                    members += list(getattr(bound.get(name), "reqs", []))
                members += [getattr(owner, "chunked_req", None)]
            elif requests == "batch":
                members = getattr(batch, "reqs", [])
            else:
                members = bound.get("req", bound.get("recv_req"))
            ids = _request_ids(members)
            result = bound.get("result")
            forward_id = getattr(result, "_timeline_forward_id", None)
            if result_processing:
                for req in members:
                    req._timeline_result_forward_id = forward_id
            start = time.monotonic_ns()
            value = None
            successful = False
            details = dict(
                membership="candidate" if requests == "pending" else "member"
            )
            contexts = ExitStack()
            for rid in ids:
                contexts.enter_context(scope(rid, stage, details=details))
            try:
                value = function(*args, **kwargs)
                successful = True
                return value
            finally:
                end = time.monotonic_ns()
                if requests == "received":
                    ids = _request_ids(value)
                if forward_result and successful:
                    forward_id = f"{os.getpid()}:forward:{batch.forward_iter}"
                    if value is not None:
                        value._timeline_forward_id = forward_id
                details.update(
                    forward_id=forward_id,
                    forward_mode=str(getattr(batch, "forward_mode", "")),
                )
                # Close with the original exception state so failed operations
                # are recorded as failed and continue to raise normally.
                contexts.__exit__(*sys.exc_info())
                if requests == "received":
                    for rid in ids:
                        emit(
                            rid,
                            stage,
                            start,
                            end,
                            span_id=f"{os.getpid()}:phase:{next(_sequence)}",
                            thread_id=threading.get_ident(),
                            category="framework_cpu",
                            envelope=True,
                            successful=successful,
                            membership="member",
                        )

        return wrapped

    return decorate


@contextmanager
def batch_scope(batch, result, stage):
    """Record a synchronization already present in serving; add no GPU wait."""
    if not enabled():
        yield
        return
    with ExitStack() as contexts:
        for rid in _request_ids(getattr(batch, "reqs", [])):
            contexts.enter_context(
                scope(
                    rid,
                    stage,
                    category="framework_queue",
                    forward_id=getattr(result, "_timeline_forward_id", None),
                    forward_mode=str(getattr(batch, "forward_mode", "")),
                )
            )
        yield


def stream_send(reqs, output_ids, send):
    """Measure the actual emitted members and ordinal ranges, not token values."""
    if not enabled():
        return send()
    # Freeze ordinals before send; Req output may be extended by a later forward.
    witnesses = []
    for req, ids in zip(reqs, output_ids):
        end = req.send_token_offset
        witnesses.append(
            (
                req.rid,
                end - len(ids),
                end,
                getattr(req, "_timeline_result_forward_id", None),
            )
        )
    start = time.monotonic_ns()
    value = send()
    end = time.monotonic_ns()
    for rid, token_start, token_end, forward_id in witnesses:
        stack = getattr(_local, "stacks", {}).get(rid, [])
        emit(
            rid,
            "scheduler_stream_send",
            start,
            end,
            span_id=f"{os.getpid()}:phase:{next(_sequence)}",
            parent_id=stack[-1] if stack else None,
            category="stream_delivery",
            thread_id=threading.get_ident(),
            successful=True,
            forward_id=forward_id,
            output_token_start=token_start,
            output_token_end=token_end,
            output_token_count=token_end - token_start,
        )
    return value


def token_commit(req, count):
    """Bind committed output ordinals to their actual forward result."""
    if enabled():
        emit(
            req.rid,
            "scheduler_token_commit",
            forward_id=getattr(req, "_timeline_result_forward_id", None),
            output_token_start=len(req.output_ids) - count,
            output_token_end=len(req.output_ids),
            output_token_count=count,
        )


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
