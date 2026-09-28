import unittest

from speak import build_live_config


class VadConfigTests(unittest.TestCase):
    def test_server_vad_is_disabled_for_explicit_client_turns(self):
        config = build_live_config()
        realtime = config.realtime_input_config
        detection = realtime.automatic_activity_detection

        self.assertTrue(realtime.automatic_activity_detection.disabled)
        self.assertEqual(realtime.turn_coverage.value, "TURN_INCLUDES_ALL_INPUT")


if __name__ == "__main__":
    unittest.main()
