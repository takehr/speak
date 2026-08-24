import argparse
import asyncio
import datetime as dt
import tempfile
import unittest
from pathlib import Path

from speak import (
    DEFAULT_DAILY_START_TIMES,
    AudioLoop,
    load_daily_practice_date,
    next_daily_occurrence,
    next_scheduled_occurrence,
    parse_clock_time,
)


class DailyScheduleTests(unittest.TestCase):
    def test_default_schedule_runs_hourly_from_1_pm_through_8_pm(self):
        self.assertEqual(
            DEFAULT_DAILY_START_TIMES,
            tuple(dt.time(hour=hour) for hour in range(13, 21)),
        )

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

    def test_next_scheduled_occurrence_selects_next_hour(self):
        timezone = dt.timezone(dt.timedelta(hours=9))
        now = dt.datetime(2026, 8, 16, 14, 1, 1, tzinfo=timezone)

        result = next_scheduled_occurrence(now, DEFAULT_DAILY_START_TIMES)

        self.assertEqual(result, dt.datetime(2026, 8, 16, 15, 0, tzinfo=timezone))


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


class DailyPracticeCompletionTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, state_path):
        loop = object.__new__(AudioLoop)
        loop.running = True
        loop.state = "active"
        loop.session_stop_event = asyncio.Event()
        loop._session_mode = "prompt"
        loop._session_active_started_at = loop._loop_time() - 1.0
        loop._session_user_responded = True
        loop.daily_practice_min_seconds = 0.01
        loop.daily_practice_state_path = Path(state_path)
        loop._daily_practice_completed_date = None
        loop._status = lambda *_args, **_kwargs: None
        return loop

    async def test_one_minute_normal_session_with_response_completes_day(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "practice.txt"
            loop = self.make_loop(state_path)
            task = asyncio.create_task(loop.monitor_daily_practice_completion())

            await asyncio.sleep(0.01)

            self.assertEqual(loop._daily_practice_completed_date, dt.date.today())
            self.assertEqual(load_daily_practice_date(state_path), dt.date.today())
            loop.session_stop_event.set()
            await task

    async def test_session_without_user_response_does_not_complete_day(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = self.make_loop(Path(directory) / "practice.txt")
            loop._session_user_responded = False
            task = asyncio.create_task(loop.monitor_daily_practice_completion())

            await asyncio.sleep(0.01)

            self.assertIsNone(loop._daily_practice_completed_date)
            loop.state = "closing"
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

if __name__ == "__main__":
    unittest.main()
