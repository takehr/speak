import unittest
from unittest import mock

from speak import AudioLoop


class FakeDetector:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class MicrophoneRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self):
        loop = object.__new__(AudioLoop)
        loop.mic_index = 4
        loop.running = True
        loop.audio_stream = None
        loop.detector = FakeDetector()
        loop._status = lambda *_args, **_kwargs: None
        return loop

    async def test_retries_when_microphone_open_fails(self):
        loop = self.make_loop()

        class BrokenBackend:
            open_calls = 0

            def open(self, **_kwargs):
                self.open_calls += 1
                loop.running = False
                raise OSError(-9999, "Unanticipated host error")

        backend = BrokenBackend()
        with (
            mock.patch("speak.pya", backend),
            mock.patch("speak.MICROPHONE_RETRY_SECONDS", 0.0),
        ):
            await loop.listen_microphone()

        self.assertEqual(backend.open_calls, 1)
        self.assertIsNone(loop.audio_stream)

    async def test_closes_and_retries_when_microphone_read_fails(self):
        loop = self.make_loop()

        class BrokenStream:
            closed = False

            def read(self, *_args, **_kwargs):
                loop.running = False
                raise OSError(-9999, "Unanticipated host error")

            def close(self):
                self.closed = True

        stream = BrokenStream()

        class Backend:
            def open(self, **_kwargs):
                return stream

        with (
            mock.patch("speak.pya", Backend()),
            mock.patch("speak.MICROPHONE_RETRY_SECONDS", 0.0),
        ):
            await loop.listen_microphone()

        self.assertTrue(stream.closed)
        self.assertIsNone(loop.audio_stream)
        self.assertEqual(loop.detector.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
