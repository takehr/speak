import unittest

from speak import build_live_config


class VadConfigTests(unittest.TestCase):
    def test_automatic_vad_is_enabled_and_noise_resistant(self):
        config = build_live_config(automatic_vad=True)
        realtime = config.realtime_input_config
        detection = realtime.automatic_activity_detection

        self.assertFalse(detection.disabled)
        self.assertEqual(
            detection.start_of_speech_sensitivity.value,
            "START_SENSITIVITY_LOW",
        )
        self.assertEqual(
            detection.end_of_speech_sensitivity.value,
            "END_SENSITIVITY_HIGH",
        )
        self.assertEqual(detection.prefix_padding_ms, 300)
        self.assertEqual(detection.silence_duration_ms, 800)
        self.assertEqual(
            realtime.turn_coverage.value,
            "TURN_INCLUDES_ONLY_ACTIVITY",
        )

    def test_manual_vad_disables_server_detection(self):
        config = build_live_config(automatic_vad=False)
        realtime = config.realtime_input_config

        self.assertTrue(realtime.automatic_activity_detection.disabled)
        self.assertEqual(realtime.turn_coverage.value, "TURN_INCLUDES_ALL_INPUT")


if __name__ == "__main__":
    unittest.main()
