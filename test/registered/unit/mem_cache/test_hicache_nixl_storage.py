"""Unit tests for the NIXL HiCache storage backend -- no server, no model loading."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from typing import List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
from sglang.test.test_utils import CustomTestCase

# Stress tests are opt-in: CI never sets this; set locally to exercise them.
STRESS_ENABLED = bool(os.environ.get("SGLANG_RUN_NIXL_STRESS"))


class MockMemPoolHost:
    """Minimal MHA-style HostKVCache stand-in supporting the v1 paths.

    zero_copy mode uses ``page_first`` so ``get_page_buffer_meta`` returns
    valid (k, v) pointers into ``kv_buffer``. Non-zero-copy uses
    ``layer_first`` so the slow path uses ``get_data_page`` /
    ``set_from_flat_data_page`` against the same buffer.
    """

    def __init__(
        self,
        is_zero_copy_mode: bool,
        page_size: int = 2,
        layer_num: int = 2,
        head_num: int = 2,
        head_dim: int = 4,
        num_pages: int = 4,
        dtype: torch.dtype = torch.float32,
    ):
        self.layout = "page_first" if is_zero_copy_mode else "layer_first"
        self.page_size = page_size
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.num_pages = num_pages
        self.size = page_size * num_pages
        self.pin_memory = False
        if is_zero_copy_mode:
            # page_first: (2, size, layer, head, head_dim)
            self.kv_buffer = torch.zeros(
                (2, self.size, layer_num, head_num, head_dim), dtype=dtype
            )
        else:
            # layer_first: (2, layer, size, head, head_dim)
            self.kv_buffer = torch.zeros(
                (2, layer_num, self.size, head_num, head_dim), dtype=dtype
            )

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        base = self.kv_buffer.data_ptr()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        idx_list = indices.tolist()
        for i in range(0, len(idx_list), self.page_size):
            k_ptr = base + idx_list[i] * (
                self.layer_num * self.head_num * self.head_dim * self.dtype.itemsize
            )
            ptr_list.append(k_ptr)
            ptr_list.append(k_ptr + v_offset)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        return ptr_list, [element_size] * len(ptr_list)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
        ).flatten()

    def get_data_page(self, index, flat=True):
        if hasattr(index, "item"):
            index = int(index.item())
        page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        return page.flatten() if flat else page

    def set_from_flat_data_page(self, index, data_page):
        if hasattr(index, "item"):
            index = int(index.item())
        self.kv_buffer[:, :, index : index + self.page_size, :, :] = data_page.reshape(
            2, self.layer_num, self.page_size, self.head_num, self.head_dim
        )

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        # Test tensors are too small to satisfy 4 KiB stride alignment; the
        # O_DIRECT path correctly falls back to copy mode in this case.
        return False


class MinioFixture:
    """Spin up a single-node MinIO server on localhost and create a bucket.

    Relies on MinIO's default ``minioadmin``/``minioadmin`` root credentials
    so no env vars need to be plumbed through.
    """

    user = "minioadmin"
    password = "minioadmin"

    def __init__(self, bucket: str = "hicache-test"):
        self.bucket = bucket
        self.api_port = self._find_free_port()
        self.data_dir = tempfile.mkdtemp(prefix="nixl_minio_")
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"127.0.0.1:{self.api_port}"

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @staticmethod
    def _minio_bin() -> str | None:
        path = shutil.which("minio") or "/usr/local/bin/minio"
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

    @classmethod
    def is_available(cls) -> bool:
        """True iff a minio binary and boto3 are both importable."""
        if cls._minio_bin() is None:
            return False
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return True

    def start(self) -> None:
        minio_bin = self._minio_bin()
        if minio_bin is None:
            raise FileNotFoundError("minio binary not available")

        self.proc = subprocess.Popen(
            [minio_bin, "server", "--address", self.endpoint, self.data_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"minio exited early with rc={self.proc.returncode}")
            try:
                with socket.create_connection(
                    ("127.0.0.1", self.api_port), timeout=0.5
                ):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            self.stop()
            raise RuntimeError("minio did not become ready within 15s")

        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{self.endpoint}",
            aws_access_key_id=self.user,
            aws_secret_access_key=self.password,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )
        s3.create_bucket(Bucket=self.bucket)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        shutil.rmtree(self.data_dir, ignore_errors=True)


def _make_storage_config(extra_config, *, tp_size=2):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        is_page_first_layout=False,
        model_name="test_model",
        enable_storage_metrics=False,
        extra_config=extra_config,
    )


def _build_hicache_or_skip(test_case, storage_config, file_path):
    try:
        return HiCacheNixl(storage_config=storage_config, file_path=file_path)
    except ImportError:
        test_case.skipTest("NIXL not available")


_POSIX_BUFFERED = {"plugin": {"posix": {"active": True}}, "use_direct_io": False}
_POSIX_DEFAULT = {"plugin": {"posix": {"active": True}}}

# Geometry shared by the FIFO + ENOSPC fixtures.
_TEST_PAGE_SIZE = 2
_TEST_LAYER_NUM = 2
_TEST_HEAD_NUM = 2
_TEST_HEAD_DIM = 4


def _probe_page_bytes() -> int:
    """On-disk page size from MockMemPoolHost (tracks mock's dtype)."""
    probe = MockMemPoolHost(
        is_zero_copy_mode=False,
        page_size=_TEST_PAGE_SIZE,
        layer_num=_TEST_LAYER_NUM,
        head_num=_TEST_HEAD_NUM,
        head_dim=_TEST_HEAD_DIM,
        num_pages=1,
    )
    sample = probe.get_dummy_flat_data_page()
    return sample.numel() * sample.element_size()


_TEST_PAGE_BYTES = _probe_page_bytes()
_TEST_TP_SIZE = 2
_TEST_KEY_SUFFIX = f"_test_model_0_{_TEST_TP_SIZE}"


def _total_limit(per_rank_bytes: int) -> int:
    """Tests think in 'per-rank pages'; storage_limit is total across ranks."""
    return per_rank_bytes * _TEST_TP_SIZE


def _make_posix_hicache_with_mock_pool(test_case, file_path, *, storage_limit=None):
    extra = dict(_POSIX_BUFFERED)
    if storage_limit is not None:
        extra["storage_limit"] = storage_limit
    hicache = _build_hicache_or_skip(test_case, _make_storage_config(extra), file_path)
    mock_host = MockMemPoolHost(
        is_zero_copy_mode=False,
        page_size=_TEST_PAGE_SIZE,
        layer_num=_TEST_LAYER_NUM,
        head_num=_TEST_HEAD_NUM,
        head_dim=_TEST_HEAD_DIM,
        num_pages=4,
    )
    hicache.register_mem_pool_host(mock_host)
    hicache.is_zero_copy = False
    return hicache, mock_host


class TestNixlUnified(CustomTestCase):
    """Unified test suite for all NIXL components."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = "/tmp/test_nixl_unified"
        os.makedirs(self.test_dir, exist_ok=True)
        self.storage_config = _make_storage_config(_POSIX_BUFFERED)
        self.hicache = _build_hicache_or_skip(self, self.storage_config, self.test_dir)

    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @staticmethod
    def _open_fds() -> int:
        return len(os.listdir("/proc/self/fd"))

    def test_storage_register_failure_closes_fds(self):
        """If NIXL register_memory raises after fds are opened, all fds are still closed."""
        files = [os.path.join(self.test_dir, f"fail_{i}.bin") for i in range(3)]
        buffers = [(0, 64) for _ in range(3)]

        fds_before = self._open_fds()

        orig = self.hicache.agent.register_memory

        def boom(*args, **kwargs):
            raise RuntimeError("simulated register_memory failure")

        self.hicache.agent.register_memory = boom
        try:
            with self.hicache.registry.storage(buffers, files, "WRITE") as descs:
                self.assertIsNone(
                    descs, "storage CM should yield None on register failure"
                )
        finally:
            self.hicache.agent.register_memory = orig

        self.assertEqual(
            self._open_fds(),
            fds_before,
            "fd leak after register_memory failure mid-storage",
        )

    def _assert_host_addrs_pre_registered(
        self, is_zero_copy_mode: bool, hicache: HiCacheNixl = None
    ):
        """Exercise the v1 path and assert every host xfer addr lies within a
        currently-registered host (DRAM/tensor) region.

        Spies are installed BEFORE ``register_mem_pool_host`` so the up-front
        pre-registration is captured too.
        """
        if hicache is None:
            hicache = self.hicache
        agent = hicache.agent

        # Map registration-handle id -> [(addr, size, mem_type), ...]
        active_regs: dict = {}
        # Capture items list per get_reg_descs call so we can attribute them
        # to the registration handle returned by the next register_memory call.
        pending: list = []

        orig_get_reg = agent.get_reg_descs

        def spy_get_reg(items, mem_type=None):
            # NIXL's register_memory calls get_reg_descs internally with an
            # already-built nixlRegDList; iterating that pybind11 type is
            # unsafe, so only record entries when the input is a plain list.
            if isinstance(items, list) and items:
                entries = []
                for it in items:
                    if isinstance(it, torch.Tensor):
                        entries.append(
                            (it.data_ptr(), it.numel() * it.element_size(), None)
                        )
                    elif isinstance(it, tuple):
                        entries.append((it[0], it[1], mem_type))
                pending.append(entries)
            return orig_get_reg(items, mem_type)

        orig_register = agent.register_memory

        def spy_register(reg_descs):
            reg = orig_register(reg_descs)
            entries = pending.pop(0) if pending else []
            active_regs[id(reg)] = entries
            return reg

        orig_dereg = agent.deregister_memory

        def spy_dereg(reg):
            active_regs.pop(id(reg), None)
            return orig_dereg(reg)

        last_host_xfer: list = []

        orig_get_xfer = agent.get_xfer_descs

        def spy_get_xfer(items, mem_type=None):
            if mem_type in (None, "DRAM"):
                ranges = []
                for it in items:
                    if isinstance(it, torch.Tensor):
                        ranges.append((it.data_ptr(), it.numel() * it.element_size()))
                    elif isinstance(it, tuple):
                        ranges.append((it[0], it[1]))
                last_host_xfer.clear()
                last_host_xfer.extend(ranges)
            return orig_get_xfer(items, mem_type)

        violations: list = []
        orig_init = agent.initialize_xfer

        def spy_init(direction, local, remote, agent_name):
            host_regs = [
                (a, s)
                for entries in active_regs.values()
                for (a, s, mt) in entries
                if mt in (None, "DRAM")
            ]
            for a, s in last_host_xfer:
                if not any(ra <= a and a + s <= ra + rs for (ra, rs) in host_regs):
                    violations.append((a, s, dict(host_regs=host_regs)))
            last_host_xfer.clear()
            return orig_init(direction, local, remote, agent_name)

        agent.get_reg_descs = spy_get_reg
        agent.register_memory = spy_register
        agent.deregister_memory = spy_dereg
        agent.get_xfer_descs = spy_get_xfer
        agent.initialize_xfer = spy_init
        try:
            mock_host = MockMemPoolHost(is_zero_copy_mode)
            hicache.register_mem_pool_host(mock_host)
            # Force the requested mode regardless of how register_mem_pool_host derives it.
            hicache.is_zero_copy = is_zero_copy_mode

            num_pages = 3
            keys = [
                f"compliance_{int(is_zero_copy_mode)}_{i}" for i in range(num_pages)
            ]
            host_indices = torch.arange(
                num_pages * mock_host.page_size, dtype=torch.int64
            )

            set_results = hicache.batch_set_v1(keys, host_indices)
            self.assertTrue(
                all(set_results),
                f"batch_set_v1 failed (zero_copy={is_zero_copy_mode}): {set_results}",
            )

            get_results = hicache.batch_get_v1(keys, host_indices)
            self.assertTrue(
                all(get_results),
                f"batch_get_v1 failed (zero_copy={is_zero_copy_mode}): {get_results}",
            )
        finally:
            agent.get_reg_descs = orig_get_reg
            agent.register_memory = orig_register
            agent.deregister_memory = orig_dereg
            agent.get_xfer_descs = orig_get_xfer
            agent.initialize_xfer = orig_init

        self.assertEqual(
            violations,
            [],
            f"Host xfer addrs not covered by registration (zero_copy={is_zero_copy_mode}): {violations}",
        )

    def test_nixl_api_contract_host_addrs_within_registered_region_zero_copy(self):
        """All host xfer addrs must lie within a registered region -- zero-copy."""
        self._assert_host_addrs_pre_registered(is_zero_copy_mode=True)

    def test_nixl_api_contract_host_addrs_within_registered_region_non_zero_copy(self):
        """All host xfer addrs must lie within a registered region -- non-zero-copy."""
        self._assert_host_addrs_pre_registered(is_zero_copy_mode=False)

    def _make_obj_hicache(self) -> HiCacheNixl:
        """Start a MinIO server (cleaned up via addCleanup) and return a
        HiCacheNixl wired to its OBJ backend. Skips the test if the backend
        cannot be constructed."""
        minio = MinioFixture()
        minio.start()
        self.addCleanup(minio.stop)

        obj_config = _make_storage_config(
            {
                "plugin": {
                    "obj": {
                        "active": True,
                        "endpoint_override": f"http://{minio.endpoint}",
                        "use_virtual_addressing": "false",
                        "access_key": minio.user,
                        "secret_key": minio.password,
                        "bucket": minio.bucket,
                    }
                }
            },
            tp_size=1,
        )
        try:
            return HiCacheNixl(storage_config=obj_config, file_path="")
        except Exception as e:
            self.skipTest(f"NIXL OBJ backend unavailable: {e}")

    @unittest.skipUnless(
        MinioFixture.is_available(), "minio binary or boto3 not available"
    )
    def test_nixl_api_contract_host_addrs_within_registered_region_obj(self):
        """Same property over the OBJ backend (MinIO fixture)."""
        self._assert_host_addrs_pre_registered(
            is_zero_copy_mode=False, hicache=self._make_obj_hicache()
        )

    def test_batch_set_v1_skips_on_nonzero_mla_rank(self):
        """batch_set_v1 is a no-op on nonzero MLA backup ranks.

        With backup_skip=True the early-return must fire before the host-regs
        check, so calling without register_mem_pool_host still returns all-True
        (the host-regs check would otherwise return all-False).
        """
        self.hicache.backup_skip = True
        results = self.hicache.batch_set_v1(
            ["key1", "key2"], torch.tensor([0, 1], dtype=torch.int64)
        )
        self.assertEqual(results, [True, True])

    def test_batch_exists_zero_copy_mla_uses_single_key_denominator(self):
        """Zero-copy MLA batch_exists counts one storage key per logical key."""
        self.hicache.is_zero_copy = True
        self.hicache.is_mla_model = True
        self.hicache.agent.query_memory = lambda *a, **kw: [object(), None]

        self.assertEqual(self.hicache.batch_exists(["key1", "key2"]), 1)

    def test_batch_exists_zero_copy_mha_uses_two_key_denominator(self):
        """Zero-copy non-MLA batch_exists counts k/v pairs per logical key."""
        self.hicache.is_zero_copy = True
        self.hicache.is_mla_model = False
        self.hicache.agent.query_memory = lambda *a, **kw: [
            object(),
            object(),
            None,
            None,
        ]

        self.assertEqual(self.hicache.batch_exists(["key1", "key2"]), 1)

    def _run_concurrent_stress(
        self, is_zero_copy_mode: bool, hicache: HiCacheNixl = None
    ):
        """One getter thread + one setter thread share the same HiCacheNixl
        for ``is_zero_copy_mode``. Defaults to ``self.hicache`` (FILE backend);
        pass ``hicache`` to exercise a different backend (e.g. OBJ).

        Phase 1 pre-seeds N preset pages and stores them under fixed keys.
        Phase 2 runs the getter (reads the presets back and verifies content)
        concurrently with the setter (writes a stream of fresh distinct keys
        from a disjoint source region). The kv_buffer regions touched by the
        two threads are disjoint so any data corruption observed is from the
        backend's shared state (bounce buffers, devId maps, fd pool).
        """
        if hicache is None:
            hicache = self.hicache

        # 8 preset pages, 8 getter dst pages, 8 setter src pages -> 24 in use.
        mock_host = MockMemPoolHost(is_zero_copy_mode=is_zero_copy_mode, num_pages=32)
        hicache.register_mem_pool_host(mock_host)
        hicache.is_zero_copy = is_zero_copy_mode

        page_size = mock_host.page_size
        dtype = mock_host.dtype
        num_pages = 8

        # Disjoint per-thread regions in kv_buffer (indexed by token index).
        preset_src = (0, num_pages)
        getter_dst = (num_pages, 2 * num_pages)
        setter_src = (2 * num_pages, 3 * num_pages)

        # zero_copy=page_first uses dim 1 for the token axis; non-zero-copy=
        # layer_first uses dim 2. All buffer accesses below go through this so
        # the rest of the harness stays layout-agnostic.
        def token_index(start_token: int, n_tokens: int):
            s = slice(start_token, start_token + n_tokens)
            if is_zero_copy_mode:
                return (slice(None), s, slice(None), slice(None), slice(None))
            return (slice(None), slice(None), s, slice(None), slice(None))

        def page_index(start_page: int, n_pages: int):
            return token_index(start_page * page_size, n_pages * page_size)

        def fill_pages(start_page: int, n_pages: int, value_fn):
            """value_fn(i) -> scalar value for page i."""
            for i in range(n_pages):
                idx = page_index(start_page + i, 1)
                shape = mock_host.kv_buffer[idx].shape
                mock_host.kv_buffer[idx] = torch.full(
                    shape, float(value_fn(i)), dtype=dtype
                )

        # Phase 1: distinct value per preset page so a wrong-page result is
        # detectable; setter source is constant (value irrelevant to the
        # test, just needs to be valid).
        fill_pages(preset_src[0], num_pages, lambda i: i + 1)
        fill_pages(setter_src[0], num_pages, lambda i: -1.0)

        preset_keys = [f"preset_{int(is_zero_copy_mode)}_{i}" for i in range(num_pages)]
        preset_indices = torch.arange(
            preset_src[0] * page_size,
            preset_src[1] * page_size,
            dtype=torch.int64,
        )
        self.assertTrue(
            all(hicache.batch_set_v1(preset_keys, preset_indices)),
            "phase 1: presetting keys failed",
        )

        # Expected per-page-i payload after a successful get into getter_dst.
        expected_pages = [
            mock_host.kv_buffer[page_index(preset_src[0] + i, 1)].clone()
            for i in range(num_pages)
        ]

        # Phase 2.
        stop = threading.Event()
        errors: List[str] = []
        errors_lock = threading.Lock()

        def record_error(msg: str):
            with errors_lock:
                errors.append(msg)

        def getter_loop():
            dst_indices = torch.arange(
                getter_dst[0] * page_size,
                getter_dst[1] * page_size,
                dtype=torch.int64,
            )
            loops = 0
            while not stop.is_set():
                # Zero the dst pages so a no-op get is observable.
                mock_host.kv_buffer[page_index(getter_dst[0], num_pages)] = 0.0
                ok = hicache.batch_get_v1(preset_keys, dst_indices)
                if not all(ok):
                    record_error(f"getter loop {loops}: batch_get_v1 returned {ok}")
                    return
                for i in range(num_pages):
                    got = mock_host.kv_buffer[page_index(getter_dst[0] + i, 1)]
                    if not torch.equal(got, expected_pages[i]):
                        record_error(f"getter loop {loops}: preset page {i} corrupted")
                        return
                loops += 1

        def setter_loop():
            src_indices = torch.arange(
                setter_src[0] * page_size,
                setter_src[1] * page_size,
                dtype=torch.int64,
            )
            loops = 0
            while not stop.is_set():
                keys = [
                    f"setter_{int(is_zero_copy_mode)}_{loops}_{i}"
                    for i in range(num_pages)
                ]
                ok = hicache.batch_set_v1(keys, src_indices)
                if not all(ok):
                    record_error(f"setter loop {loops}: batch_set_v1 returned {ok}")
                    return
                loops += 1

        t_get = threading.Thread(target=getter_loop, daemon=True)
        t_set = threading.Thread(target=setter_loop, daemon=True)
        t_get.start()
        t_set.start()

        # Bounded run: long enough to interleave many ops under NIXL I/O
        # GIL release, short enough for a unit test.
        time.sleep(3.0)
        stop.set()
        t_get.join(timeout=10)
        t_set.join(timeout=10)

        self.assertFalse(
            t_get.is_alive() or t_set.is_alive(),
            "stress threads failed to stop",
        )
        self.assertEqual(errors, [], f"concurrency errors: {errors}")

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    def test_concurrent_getter_setter_file_zero_copy(self):
        """Stress: concurrent getter+setter, FILE backend, zero-copy."""
        self._run_concurrent_stress(is_zero_copy_mode=True)

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    def test_concurrent_getter_setter_file_non_zero_copy(self):
        """Stress: concurrent getter+setter, FILE backend, non-zero-copy."""
        self._run_concurrent_stress(is_zero_copy_mode=False)

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    @unittest.skipUnless(
        MinioFixture.is_available(), "minio binary or boto3 not available"
    )
    def test_concurrent_getter_setter_obj_zero_copy(self):
        """Stress: concurrent getter+setter, OBJ backend (MinIO), zero-copy."""
        self._run_concurrent_stress(
            is_zero_copy_mode=True, hicache=self._make_obj_hicache()
        )

    @unittest.skipUnless(STRESS_ENABLED, "set SGLANG_RUN_NIXL_STRESS=1 to run")
    @unittest.skipUnless(
        MinioFixture.is_available(), "minio binary or boto3 not available"
    )
    def test_concurrent_getter_setter_obj_non_zero_copy(self):
        """Stress: concurrent getter+setter, OBJ backend (MinIO), non-zero-copy."""
        self._run_concurrent_stress(
            is_zero_copy_mode=False, hicache=self._make_obj_hicache()
        )


class TestNixlFifoEviction(CustomTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="nixl_fifo_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_hicache(self, storage_limit=None):
        return _make_posix_hicache_with_mock_pool(
            self, self.test_dir, storage_limit=storage_limit
        )

    def _set_one(self, hicache, key: str):
        indices = torch.arange(_TEST_PAGE_SIZE, dtype=torch.int64)
        return hicache.batch_set_v1([key], indices)

    def _file_for(self, key: str) -> str:
        return os.path.join(self.test_dir, key + _TEST_KEY_SUFFIX)

    def test_fifo_evicts_oldest_when_limit_exceeded(self):
        # Cap = n+1 pages so writing the n+1th key crosses 80% and evicts only the oldest.
        n = 4
        per_rank = (n + 1) * _TEST_PAGE_BYTES
        limit = _total_limit(per_rank)
        hicache, _ = self._make_hicache(storage_limit=limit)
        try:
            for i in range(n):
                self.assertTrue(
                    all(self._set_one(hicache, f"k{i}")),
                    f"baseline set k{i} failed",
                )
                time.sleep(0.01)
            for i in range(n):
                self.assertTrue(
                    os.path.exists(self._file_for(f"k{i}")),
                    f"baseline k{i} missing before eviction",
                )

            self.assertTrue(
                all(self._set_one(hicache, f"k{n}")),
                f"set k{n} should succeed after evicting k0",
            )

            self.assertFalse(
                os.path.exists(self._file_for("k0")),
                "oldest key k0 should have been evicted",
            )
            self.assertTrue(
                os.path.exists(self._file_for(f"k{n}")),
                f"new key k{n} should be present",
            )
            for i in range(1, n):
                self.assertTrue(
                    os.path.exists(self._file_for(f"k{i}")),
                    f"key k{i} should still be present (only oldest evicted)",
                )

            total = sum(
                os.path.getsize(os.path.join(self.test_dir, f))
                for f in os.listdir(self.test_dir)
            )
            self.assertLessEqual(
                total,
                per_rank,
                f"total on-disk {total} exceeds per-rank limit {per_rank}",
            )
        finally:
            hicache.close()

    def test_no_limit_means_no_eviction(self):
        n = 4
        hicache, _ = self._make_hicache(storage_limit=None)
        try:
            for i in range(n + 1):
                self.assertTrue(
                    all(self._set_one(hicache, f"k{i}")),
                    f"set k{i} failed",
                )
                time.sleep(0.005)
            for i in range(n + 1):
                self.assertTrue(
                    os.path.exists(self._file_for(f"k{i}")),
                    f"k{i} missing -- unlimited mode must not evict",
                )
        finally:
            hicache.close()

    def test_multi_evict_when_batch_larger_than_single_entry(self):
        n = 4
        per_rank = (n + 1) * _TEST_PAGE_BYTES
        limit = _total_limit(per_rank)
        hicache, _ = self._make_hicache(storage_limit=limit)
        try:
            for i in range(n):
                self.assertTrue(all(self._set_one(hicache, f"k{i}")), f"set k{i}")
                time.sleep(0.01)

            new_keys = ["m0", "m1", "m2"]
            indices = torch.arange(_TEST_PAGE_SIZE * len(new_keys), dtype=torch.int64)
            self.assertTrue(
                all(hicache.batch_set_v1(new_keys, indices)),
                "multi-key batch_set should succeed after multi-evict",
            )

            for evicted in ("k0", "k1", "k2"):
                self.assertFalse(
                    os.path.exists(self._file_for(evicted)),
                    f"{evicted} should have been evicted",
                )
            self.assertTrue(
                os.path.exists(self._file_for("k3")),
                "k3 (newest of the old set) should remain",
            )
            for new_key in new_keys:
                self.assertTrue(
                    os.path.exists(self._file_for(new_key)),
                    f"new key {new_key} should be present",
                )
            total = sum(
                os.path.getsize(os.path.join(self.test_dir, f))
                for f in os.listdir(self.test_dir)
            )
            self.assertLessEqual(total, per_rank)
        finally:
            hicache.close()

    def test_batch_larger_than_limit_rejected_cache_untouched(self):
        # Cap = 1.5 pages: 1-page seed fits, 2-page batch is rejected as oversized.
        per_rank = _TEST_PAGE_BYTES + _TEST_PAGE_BYTES // 2
        limit = _total_limit(per_rank)
        hicache, _ = self._make_hicache(storage_limit=limit)
        try:
            self.assertTrue(all(self._set_one(hicache, "keep")), "seed set")
            seed_path = self._file_for("keep")
            self.assertTrue(os.path.exists(seed_path))
            seed_mtime = os.path.getmtime(seed_path)

            indices = torch.arange(2 * _TEST_PAGE_SIZE, dtype=torch.int64)
            results = hicache.batch_set_v1(["big0", "big1"], indices)
            self.assertEqual(
                results,
                [False, False],
                "oversized batch must be rejected (all False)",
            )

            self.assertTrue(
                os.path.exists(seed_path),
                "existing file must not be evicted on rejected batch",
            )
            self.assertEqual(
                os.path.getmtime(seed_path),
                seed_mtime,
                "existing file should not have been rewritten",
            )
            self.assertFalse(os.path.exists(self._file_for("big0")))
            self.assertFalse(os.path.exists(self._file_for("big1")))
        finally:
            hicache.close()

    def test_eviction_order_is_ctime_not_name(self):
        n = 4
        per_rank = (n + 1) * _TEST_PAGE_BYTES
        limit = _total_limit(per_rank)
        hicache, _ = self._make_hicache(storage_limit=limit)
        try:
            order = ["z_first", "a_second", "m_third", "b_fourth"]
            for k in order:
                self.assertTrue(all(self._set_one(hicache, k)), f"set {k}")
                time.sleep(0.01)
            self.assertTrue(all(self._set_one(hicache, "newest")), "set newest")
            self.assertFalse(
                os.path.exists(self._file_for("z_first")),
                "oldest by ctime (z_first) should be evicted, not 'a_second' "
                "which is lexically first",
            )
            for survivor in ("a_second", "m_third", "b_fourth", "newest"):
                self.assertTrue(
                    os.path.exists(self._file_for(survivor)),
                    f"{survivor} should still be present",
                )
        finally:
            hicache.close()

    def test_counter_resyncs_on_external_deletion(self):
        # External deletion -> next reclaim syncs the counter, no spurious eviction.
        n = 4
        per_rank = (n + 1) * _TEST_PAGE_BYTES
        limit = _total_limit(per_rank)
        hicache, _ = self._make_hicache(storage_limit=limit)
        try:
            for i in range(n):
                self.assertTrue(all(self._set_one(hicache, f"k{i}")), f"set k{i}")
                time.sleep(0.01)
            # Externally remove k1 and k2.
            os.unlink(self._file_for("k1"))
            os.unlink(self._file_for("k2"))
            # Triggers reclaim (counter still at high-water), but scan resync should evict nothing.
            self.assertTrue(all(self._set_one(hicache, "kN")), "set kN")
            self.assertTrue(
                os.path.exists(self._file_for("k0")),
                "k0 must survive: counter resync should have shown room",
            )
            self.assertTrue(
                os.path.exists(self._file_for("k3")),
                "k3 must survive",
            )
            self.assertTrue(os.path.exists(self._file_for("kN")))
        finally:
            hicache.close()

    def test_other_rank_files_in_dir_are_ignored(self):
        # Seed a foreign-suffix file (simulates a peer tp_rank's data).
        n = 4
        per_rank = (n + 1) * _TEST_PAGE_BYTES
        limit = _total_limit(per_rank)
        foreign = os.path.join(self.test_dir, "foreign_test_model_1_2")
        with open(foreign, "wb") as f:
            f.write(b"\0" * _TEST_PAGE_BYTES)
        hicache, _ = self._make_hicache(storage_limit=limit)
        try:
            for i in range(n):
                self.assertTrue(all(self._set_one(hicache, f"k{i}")), f"set k{i}")
                time.sleep(0.01)
            # Force one eviction by writing one more key.
            self.assertTrue(all(self._set_one(hicache, "kN")), "set kN")
            self.assertTrue(
                os.path.exists(foreign),
                "another rank's file must not be evicted by this rank's FIFO",
            )
            self.assertFalse(
                os.path.exists(self._file_for("k0")),
                "this rank's oldest (k0) should have been evicted",
            )
        finally:
            hicache.close()


def _mount_cmd_prefix() -> Optional[list]:
    """Return the argv prefix to invoke mount/umount, or None if unavailable."""
    if not shutil.which("mount"):
        return None
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        return ["sudo", "-n"]
    return None


def _try_mount_tmpfs(path: str, size_bytes: int) -> Optional[str]:
    prefix = _mount_cmd_prefix()
    if prefix is None:
        return "mount/sudo not available"
    rc = subprocess.run(
        prefix
        + [
            "mount",
            "-t",
            "tmpfs",
            "-o",
            f"size={size_bytes},mode=0777",
            "tmpfs",
            path,
        ],
        capture_output=True,
    )
    if rc.returncode != 0:
        return rc.stderr.decode(errors="replace").strip() or "mount failed"
    return None


def _try_umount(path: str) -> None:
    prefix = _mount_cmd_prefix()
    if prefix is None:
        return
    subprocess.run(prefix + ["umount", path], capture_output=True)


class TestNixlEnospc(CustomTestCase):
    TMPFS_BYTES = 64 * 1024

    @classmethod
    def setUpClass(cls):
        cls.mount_point = tempfile.mkdtemp(prefix="nixl_enospc_")
        err = _try_mount_tmpfs(cls.mount_point, cls.TMPFS_BYTES)
        if err is not None:
            shutil.rmtree(cls.mount_point, ignore_errors=True)
            raise unittest.SkipTest(f"cannot mount tmpfs: {err}")
        cls._mounted = True

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_mounted", False):
            _try_umount(cls.mount_point)
        shutil.rmtree(cls.mount_point, ignore_errors=True)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="case_", dir=self.mount_point)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _set_one(self, hicache, key: str):
        return hicache.batch_set_v1(
            [key], torch.arange(_TEST_PAGE_SIZE, dtype=torch.int64)
        )

    def test_fifo_keeps_writes_within_small_tmpfs(self):
        # Per-rank cap comfortably below tmpfs capacity -> FIFO fires first.
        per_rank = 8 * _TEST_PAGE_BYTES
        limit = _total_limit(per_rank)
        hicache, _ = _make_posix_hicache_with_mock_pool(
            self, self.test_dir, storage_limit=limit
        )
        try:
            n_writes = (self.TMPFS_BYTES // _TEST_PAGE_BYTES) * 4
            for i in range(n_writes):
                self.assertTrue(
                    all(self._set_one(hicache, f"k{i}")),
                    f"set k{i} failed under FIFO with small tmpfs",
                )
            total = sum(
                os.path.getsize(os.path.join(self.test_dir, f))
                for f in os.listdir(self.test_dir)
            )
            self.assertLessEqual(total, per_rank)
        finally:
            hicache.close()

    def test_reactive_fifo_recovers_from_disk_full(self):
        # No storage_limit set; reactive FIFO step must still recover from a full FS.
        hicache, _ = _make_posix_hicache_with_mock_pool(
            self, self.test_dir, storage_limit=None
        )
        try:
            n_writes = (self.TMPFS_BYTES // _TEST_PAGE_BYTES) * 4
            for i in range(n_writes):
                self.assertTrue(
                    all(self._set_one(hicache, f"k{i}")),
                    f"set k{i} should be recovered by reactive FIFO",
                )
        finally:
            hicache.close()


class TestParseStorageLimit(CustomTestCase):
    def test_units_accepted(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import parse_storage_limit

        self.assertEqual(parse_storage_limit("1MB"), 1024**2)
        self.assertEqual(parse_storage_limit("2GB"), 2 * 1024**3)
        self.assertEqual(parse_storage_limit("1TB"), 1024**4)
        self.assertEqual(parse_storage_limit("10mb"), 10 * 1024**2)
        self.assertEqual(parse_storage_limit("  10 mb  "), 10 * 1024**2)
        self.assertEqual(parse_storage_limit(4096), 4096)
        self.assertEqual(parse_storage_limit("4096"), 4096)

    def test_invalid_rejected(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import parse_storage_limit

        bad_values = (
            "",
            "1KB",
            "1B",
            "1XB",
            "abc",
            "1.5MB",
            "1 PB",
            True,
            None,
            1.5,
            # Non-positive: would silently disable all writes downstream.
            0,
            -1,
            "0",
            "0mb",
            "-1mb",
        )
        for bad in bad_values:
            with self.assertRaises((ValueError, TypeError), msg=f"accepted {bad!r}"):
                parse_storage_limit(bad)

    def test_nested_under_plugin_honored(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlBackendConfig

        nested = NixlBackendConfig(
            {"plugin": {"posix": {"active": True, "storage_limit": "2MB"}}}
        )
        self.assertEqual(nested.get_storage_limit_bytes(), 2 * 1024**2)

        top = NixlBackendConfig({"storage_limit": "1MB", "plugin": {"posix": {}}})
        self.assertEqual(top.get_storage_limit_bytes(), 1024**2)

        absent = NixlBackendConfig({"plugin": {"posix": {"active": True}}})
        self.assertIsNone(absent.get_storage_limit_bytes())


@unittest.skipUnless(hasattr(os, "O_DIRECT"), "O_DIRECT not available on this platform")
class TestNixlDirectIO(CustomTestCase):
    """Tests for the O_DIRECT file I/O path in NixlFileManager and HiCacheNixl."""

    def setUp(self):
        self.test_dir = "/tmp/test_nixl_direct_io"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_open_file_sets_o_direct(self):
        """open_file sets O_DIRECT on the file descriptor when use_direct_io=True."""
        import fcntl

        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        fm = NixlFileManager(self.test_dir, use_direct_io=True)
        test_file = os.path.join(self.test_dir, "test_odirect.bin")
        fd = fm.open_file(test_file, create=True)
        try:
            self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)
        finally:
            os.close(fd)

    def test_open_file_no_o_direct(self):
        """open_file does not set O_DIRECT when use_direct_io=False."""
        import fcntl

        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager

        fm = NixlFileManager(self.test_dir, use_direct_io=False)
        test_file = os.path.join(self.test_dir, "test_buffered.bin")
        fd = fm.open_file(test_file, create=True)
        try:
            self.assertFalse(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)
        finally:
            os.close(fd)

    def _make_direct_io_hicache(self) -> HiCacheNixl:
        """Return a HiCacheNixl configured for O_DIRECT (default) with the POSIX backend."""
        # use_direct_io defaults to True (env var); omit the key.
        return _build_hicache_or_skip(
            self, _make_storage_config(_POSIX_DEFAULT, tp_size=1), self.test_dir
        )

    def test_needs_page_alignment_true_for_file_backend(self):
        """File-based backend + use_direct_io=True must set needs_page_alignment."""
        hicache = self._make_direct_io_hicache()
        self.assertTrue(hicache.needs_page_alignment)

    def test_odirect_unaligned_pool_falls_back_to_copy(self):
        """O_DIRECT with non-aligned pool strides falls back to copy mode."""
        hicache = self._make_direct_io_hicache()

        mock_host = MockMemPoolHost(is_zero_copy_mode=True)
        hicache.register_mem_pool_host(mock_host)

        # MockMemPoolHost.is_stride_page_aligned() returns False, so even though
        # the layout would otherwise enable zero-copy, the backend must fall back.
        self.assertFalse(hicache.is_zero_copy)
        self.assertIsNotNone(hicache._bounce_set)
        self.assertIsNotNone(hicache._bounce_get)

    def test_odirect_disabled_via_config(self):
        """Top-level use_direct_io=false in extra_config disables O_DIRECT."""
        hicache = _build_hicache_or_skip(
            self, _make_storage_config(_POSIX_BUFFERED, tp_size=1), self.test_dir
        )
        self.assertFalse(hicache.needs_page_alignment)
        self.assertFalse(hicache.file_manager.use_direct_io)


if __name__ == "__main__":
    unittest.main()
