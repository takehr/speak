import logging
import unittest

from speak import LocalCommandDetector


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
