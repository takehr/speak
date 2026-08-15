import argparse
import asyncio
import datetime as dt
import unittest

from speak import AudioLoop, next_daily_occurrence, parse_clock_time


class DailyScheduleTests(unittest.TestCase):
    def test_parses_24_hour_clock_time(self):
        self.assertEqual(parse_clock_time("20:00"), dt.time(20, 0))

    def test_rejects_invalid_clock_time(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_clock_time("8pm")

    def test_next_occurrence_is_today_before_start_time(self):
        timezone = dt.timezone(dt.timedelta(hours=9))
        now = dt.datetime(2026, 8, 16, 19, 30, tzinfo=timezone)

        result = next_daily_occurrence(now, dt.time(20, 0))

        self.assertEqual(result, dt.datetime(2026, 8, 16, 20, 0, tzinfo=timezone))

    def test_next_occurrence_catches_up_within_one_minute(self):
        timezone = dt.timezone(dt.timedelta(hours=9))
        now = dt.datetime(2026, 8, 16, 20, 0, 30, tzinfo=timezone)

        result = next_daily_occurrence(now, dt.time(20, 0))

        self.assertEqual(result, dt.datetime(2026, 8, 16, 20, 0, tzinfo=timezone))

    def test_next_occurrence_moves_to_tomorrow_after_grace_period(self):
        timezone = dt.timezone(dt.timedelta(hours=9))
        now = dt.datetime(2026, 8, 16, 20, 1, 1, tzinfo=timezone)

        result = next_daily_occurrence(now, dt.time(20, 0))

        self.assertEqual(result, dt.datetime(2026, 8, 17, 20, 0, tzinfo=timezone))


class ScheduledResponseTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, timeout=0.01):
        loop = object.__new__(AudioLoop)
        loop.running = True
        loop.state = "active"
        loop.assistant_speaking = False
        loop.scheduled_response_timeout = timeout
        loop._scheduled_first_turn_complete_event = asyncio.Event()
        loop._scheduled_user_response_event = asyncio.Event()
        loop.session_stop_event = asyncio.Event()
        loop.control_queue = asyncio.Queue()
        loop._status = lambda *_args, **_kwargs: None
        return loop

    async def test_timeout_starts_only_after_first_gemini_turn(self):
        loop = self.make_loop()
        task = asyncio.create_task(loop.monitor_scheduled_response())

        await asyncio.sleep(0.02)
        self.assertTrue(loop.control_queue.empty())

        loop._scheduled_first_turn_complete_event.set()
        command = await asyncio.wait_for(loop.control_queue.get(), timeout=0.1)
        self.assertEqual(command, ("sleep", None, "scheduled-no-response"))

        loop.session_stop_event.set()
        await task

    async def test_user_response_keeps_scheduled_session_active(self):
        loop = self.make_loop()
        loop._scheduled_first_turn_complete_event.set()
        task = asyncio.create_task(loop.monitor_scheduled_response())
        loop._scheduled_user_response_event.set()

        await asyncio.sleep(0.02)
        self.assertTrue(loop.control_queue.empty())

        loop.session_stop_event.set()
        await task


if __name__ == "__main__":
    unittest.main()
