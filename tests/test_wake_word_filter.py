import logging
import json
import unittest
from collections import deque
from unittest import mock

from speak import CHUNK_SIZE, LocalCommandDetector


def make_detector(*, active_seconds, min_confidence=0.80):
    detector = object.__new__(LocalCommandDetector)
    detector.logger = logging.getLogger("test-wake-word-filter")
    detector.wake_word = "hello"
    detector.mimic_wake_word = "copy mode"
    detector.translate_wake_word = "translate"
    detector.no_auto_start_wake_word = ""
    detector.exit_word = "see you"
    detector.cooldown_seconds = 1.5
    detector.wake_min_confidence = min_confidence
    detector.wake_min_active_seconds = 0.25
    detector.wake_min_rms = 45.0
    detector._wake_effective_rms = 45.0
    detector._wake_noise_samples = deque(maxlen=47)
    detector._wake_observed_seconds = 0.0
    detector._wake_active_run_seconds = 0.0
    detector._wake_max_active_seconds = active_seconds
    detector.last_trigger_at = {
        "wake": 0.0,
        "mimic_wake": 0.0,
        "translate_wake": 0.0,
        "no_auto_start_wake": 0.0,
        "exit": 0.0,
    }
    detector.last_phrase_trigger_at = {}
    return detector


class WakeWordFilterTests(unittest.IsolatedAsyncioTestCase):
    def observe_background(self, detector):
        # RMS values reported while the user was silent, all above the old 45 floor.
        for level in [171, 181, 282, 202, 177, 164] * 12:
            detector._observe_wake_activity(bytes(CHUNK_SIZE * 2), level)

    async def test_constant_background_rejects_confident_false_wake(self):
        detector = make_detector(active_seconds=0.0)
        self.observe_background(detector)
        self.assertGreater(detector._wake_effective_rms, 282)
        command = detector._match_command(
            "hello", state="idle", is_final=True,
            words=[{"word": "hello", "conf": 0.99}],
        )
        self.assertIsNone(command)

    async def test_sustained_speech_above_background_can_wake(self):
        detector = make_detector(active_seconds=0.0)
        self.observe_background(detector)
        for _ in range(5):
            detector._observe_wake_activity(bytes(CHUNK_SIZE * 2), 1000)
        command = detector._match_command(
            "hello", state="idle", is_final=True,
            words=[{"word": "hello", "conf": 0.95}],
        )
        self.assertEqual(command, "wake")

    async def test_noise_change_then_reset_preserves_calibration(self):
        detector = make_detector(active_seconds=0.0)
        detector.recognizer = mock.Mock()
        self.observe_background(detector)
        for _ in range(60):
            detector._observe_wake_activity(bytes(CHUNK_SIZE * 2), 500)
        detector.reset()
        detector._observe_wake_activity(bytes(CHUNK_SIZE * 2), 500)
        self.assertEqual(detector._wake_effective_rms, 1250)
        self.assertEqual(detector._wake_max_active_seconds, 0)

    async def test_initial_calibration_cannot_count_as_wake_activity(self):
        detector = make_detector(active_seconds=0.0)
        for level in [10, 10, 10, 1000, 1000, 1000, 1000]:
            detector._observe_wake_activity(bytes(CHUNK_SIZE * 2), level)
        self.assertEqual(detector._wake_max_active_seconds, 0)

    async def test_unrelated_sentence_containing_wake_word_is_ignored(self):
        detector = make_detector(active_seconds=0.5)
        command = detector._match_command(
            "hello how are you", state="idle", is_final=True,
            words=[{"word": "hello", "conf": 0.99}],
        )
        self.assertIsNone(command)

    async def test_alternate_phrase_requires_both_words_and_final_result(self):
        detector = make_detector(active_seconds=0.5)
        detector.no_auto_start_wake_word = "start talking"
        words = [{"word": word, "conf": 0.95} for word in ("start", "talking")]
        self.assertIsNone(detector._match_command(
            "start talking", state="idle", is_final=False, words=words,
        ))
        self.assertIsNone(detector._match_command(
            "start", state="idle", is_final=True, words=words[:1],
        ))
        self.assertEqual(detector._match_command(
            "start talking", state="idle", is_final=True, words=words,
        ), "no_auto_start_wake")

    async def test_feed_filters_false_final_result_after_background(self):
        detector = make_detector(active_seconds=0.0)
        detector.recognizer = mock.Mock()
        detector.recognizer.AcceptWaveform.return_value = False
        detector.recognizer.PartialResult.return_value = '{"partial": ""}'
        for level in [171, 181, 282, 202, 177, 164] * 12:
            self.assertIsNone(detector.feed(bytes(CHUNK_SIZE * 2), state="idle", rms=level))
        detector.recognizer.AcceptWaveform.return_value = True
        detector.recognizer.Result.return_value = json.dumps({
            "text": "hello", "result": [{"word": "hello", "conf": 0.99}],
        })
        self.assertIsNone(detector.feed(bytes(CHUNK_SIZE * 2), state="idle", rms=181))

    async def test_rejects_short_keyboard_like_audio(self):
        detector = make_detector(active_seconds=0.064)

        command = detector._match_command(
            "hello",
            state="idle",
            is_final=True,
            words=[{"word": "hello", "conf": 0.95}],
        )

        self.assertIsNone(command)

    async def test_accepts_confident_wake_word_with_sustained_audio(self):
        detector = make_detector(active_seconds=0.32)

        command = detector._match_command(
            "hello",
            state="idle",
            is_final=True,
            words=[{"word": "hello", "conf": 0.95}],
        )

        self.assertEqual(command, "wake")

    async def test_rejects_result_without_word_confidence(self):
        detector = make_detector(active_seconds=0.32)

        command = detector._match_command(
            "hello",
            state="idle",
            is_final=True,
            words=[],
        )

        self.assertIsNone(command)


if __name__ == "__main__":
    unittest.main()
