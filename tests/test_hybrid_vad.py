import asyncio
import unittest

from speak import AudioLoop, HybridTurnDetector


class FakeVad:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def is_speech(self, _frame, _sample_rate):
        return next(self.decisions)


class HybridTurnDetectorTests(unittest.TestCase):
    def make_detector(self, decisions, max_turn_seconds=1.0):
        return HybridTurnDetector(
            frame_ms=20,
            start_seconds=0.06,
            end_seconds=0.08,
            max_turn_seconds=max_turn_seconds,
            start_speech_ratio=0.60,
            end_speech_ratio=0.20,
            vad=FakeVad(decisions),
        )

    def frames(self, detector, count):
        return bytes(detector.frame_bytes * count)

    def test_ignores_short_keyboard_like_impulses(self):
        decisions = [True, False, False, True, False, False]
        detector = self.make_detector(decisions)

        started, ended, forced = detector.feed(self.frames(detector, len(decisions)))

        self.assertFalse(started)
        self.assertFalse(ended)
        self.assertFalse(forced)

    def test_closes_turn_after_detected_speech_and_pause(self):
        decisions = [True, True, False, False, False, False, False]
        detector = self.make_detector(decisions)

        started, ended, forced = detector.feed(self.frames(detector, len(decisions)))

        self.assertTrue(started)
        self.assertTrue(ended)
        self.assertFalse(forced)
        self.assertFalse(detector.speech_active)

    def test_forces_end_for_never_ending_noise(self):
        decisions = [True] * 8
        detector = self.make_detector(decisions, max_turn_seconds=0.10)

        started, ended, forced = detector.feed(self.frames(detector, len(decisions)))

        self.assertTrue(started)
        self.assertTrue(ended)
        self.assertTrue(forced)


class AudioStreamEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_audio_stream_end_to_live_session(self):
        loop = object.__new__(AudioLoop)
        loop.running = True
        loop.state = "active"
        loop.out_queue = asyncio.Queue()
        loop._debug_every = lambda *_args: False
        loop._clear_out_queue_pressure = lambda: None

        class FakeSession:
            audio_stream_end = None

            async def send_realtime_input(self, **kwargs):
                self.audio_stream_end = kwargs.get("audio_stream_end")
                loop.running = False

        loop.session = FakeSession()
        await loop.out_queue.put({"kind": "audio_stream_end"})

        await loop.send_realtime()

        self.assertTrue(loop.session.audio_stream_end)


if __name__ == "__main__":
    unittest.main()
