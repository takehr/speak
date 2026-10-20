import asyncio
import threading
import unittest
from unittest import mock

from speak import AudioLoop


class SpeakerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self):
        loop = object.__new__(AudioLoop)
        loop.running = True
        loop.state = "active"
        loop.audio_in_queue = asyncio.Queue()
        loop.audio_in_queue.put_nowait(b"\0\0" * 16)
        loop.assistant_audio_streaming = True
        loop._status = lambda *_args, **_kwargs: None
        loop._debug_every = lambda *_args: False
        return loop

    async def check_cancellation(self, blocked_operation):
        loop = self.make_loop()
        entered = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        closing = asyncio.Event()
        operations = []
        thread_ids = []

        def operation(name):
            operations.append(name)
            thread_ids.append(threading.get_ident())
            if name == blocked_operation:
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release audio operation")
                operations.append(name + " finished")

        class Stream:
            def write(self, _data):
                operation("write")

            def close(self):
                operation("close")
                closed.set()

        def open_stream(**_kwargs):
            operation("open")
            return Stream()

        def status(message):
            if message == "closing local speaker stream":
                closing.set()

        loop._status = status
        with mock.patch("speak.pya.open", side_effect=open_stream):
            task = asyncio.create_task(loop.play_audio())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.wait_for(closing.wait(), timeout=2)
                self.assertFalse(closed.is_set())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(closed.is_set())
        self.assertEqual(operations.count("close"), 1)
        self.assertLess(
            operations.index(blocked_operation + " finished"), operations.index("close")
        )
        self.assertEqual(len(set(thread_ids)), 1)

    async def test_cancelled_write_finishes_before_stream_closes(self):
        await self.check_cancellation("write")

    async def test_cancelled_open_still_closes_the_acquired_stream(self):
        await self.check_cancellation("open")

    async def test_failed_open_preserves_original_error(self):
        loop = self.make_loop()
        with mock.patch("speak.pya.open", side_effect=OSError("device unavailable")):
            with self.assertRaisesRegex(OSError, "device unavailable"):
                await loop.play_audio()


if __name__ == "__main__":
    unittest.main()
