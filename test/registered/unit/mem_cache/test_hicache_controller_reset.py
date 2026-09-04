import threading
import unittest
from queue import Queue

from sglang.srt.managers.cache_controller import HiCacheController


class TestHiCacheControllerReset(unittest.TestCase):
    def test_reset_waits_for_prefetch_io_thread(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.enable_storage = True
        controller.storage_stop_event = threading.Event()
        controller.write_queue = []
        controller.load_queue = []
        controller.ack_write_queue = []
        controller.ack_load_queue = []
        controller.prefetch_queue = Queue()
        controller.backup_queue = Queue()
        controller.prefetch_hit_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller.host_mem_release_queue = Queue()
        controller.prefetch_tokens_occupied = 1

        def finished_thread():
            thread = threading.Thread(target=lambda: None)
            thread.start()
            thread.join()
            return thread

        controller.prefetch_thread = finished_thread()
        controller.backup_thread = finished_thread()
        controller.prefetch_thread_func = lambda: None
        controller.backup_thread_func = lambda: None

        io_entered = threading.Event()
        allow_io_return = threading.Event()

        def delayed_io():
            io_entered.set()
            allow_io_return.wait()

        controller.prefetch_io_aux_thread = threading.Thread(target=delayed_io)
        controller.prefetch_io_aux_thread.start()
        self.assertTrue(io_entered.wait(1))

        reset_done = threading.Event()
        reset_errors = []

        def reset():
            try:
                controller.reset()
            except Exception as error:
                reset_errors.append(error)
            finally:
                reset_done.set()

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        try:
            self.assertFalse(reset_done.wait(0.1))
        finally:
            allow_io_return.set()
            reset_thread.join(2)

        self.assertTrue(reset_done.is_set())
        self.assertEqual(reset_errors, [])


if __name__ == "__main__":
    unittest.main()
