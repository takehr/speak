import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from speak import AudioLoop, MODEL_AUDIO_ECHO_GRACE_SECONDS


class TurnTakingTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self):
        loop = object.__new__(AudioLoop)
        loop.running = True
        loop.state = "active"
        loop.strict_turns = True
        loop.assistant_speaking = True
        loop.assistant_audio_streaming = True
        loop._speaker_write_active = True
        loop._model_audio_suppressed_until = 0.0
        loop.local_output_active = False
        loop._voice_commands_suppressed_until = 0.0
        loop.audio_in_queue = asyncio.Queue()
        loop.out_queue = asyncio.Queue()
        loop._client_turn_audio = bytearray(b"stale audio")
        loop.hybrid_turn_detector = mock.Mock()
        loop._scheduled_first_turn_complete_event = None
        loop._scheduled_user_response_event = None
        loop._status = mock.Mock()
        loop._debug_every = lambda *_args: False
        return loop

    async def test_receive_completion_does_not_unmute_during_last_write(self):
        loop = self.make_loop()
        finished = asyncio.Event()

        class Session:
            calls = 0

            async def receive(self):
                self.calls += 1
                if self.calls > 1:
                    finished.set()
                    await asyncio.Event().wait()
                yield SimpleNamespace(server_content=None)

        loop.session = Session()
        task = asyncio.create_task(loop.receive_audio())
        try:
            await asyncio.wait_for(finished.wait(), timeout=1)
            self.assertFalse(loop.assistant_audio_streaming)
            self.assertTrue(loop.assistant_speaking)
            self.assertTrue(loop._voice_commands_suppressed())

            loop._loop_time = lambda: 100.0
            loop._speaker_write_active = False
            loop._finish_assistant_audio_if_idle()
            self.assertFalse(loop.assistant_speaking)
            self.assertTrue(loop._voice_commands_suppressed())
            loop._loop_time = lambda: 100.0 + MODEL_AUDIO_ECHO_GRACE_SECONDS + 0.01
            self.assertFalse(loop._voice_commands_suppressed())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_playback_finished_before_receive_still_keeps_mic_muted(self):
        loop = self.make_loop()
        loop._speaker_write_active = False
        loop._finish_assistant_audio_if_idle()
        self.assertTrue(loop._voice_commands_suppressed())
        loop.assistant_audio_streaming = False
        loop.audio_in_queue.put_nowait(b"queued audio")
        loop._finish_assistant_audio_if_idle()
        self.assertTrue(loop.assistant_speaking)

    async def run_microphone_sequence(self, user_replies):
        loop = self.make_loop()
        loop.mic_index = 0
        loop.automatic_vad = True
        loop.detector = mock.Mock()
        loop.detector.feed.return_value = None
        loop._pcm16_rms = lambda _data: 0.0
        loop._awaiting_model_response = False
        loop.user_activity_active = False
        loop._user_activity_started_at = 0.0
        loop.vad_start_rms = 45.0
        loop._mark_session_user_response = mock.Mock()
        clock = [100.0]
        loop._loop_time = lambda: clock[0]
        frames = [b"playback", b"last echo", b"grace", b"boundary echo", b"silence"]
        if user_replies:
            frames += [b"reply", b"pause"]
        frames = iter(frames)

        def read(*_args, **_kwargs):
            data = next(frames)
            if data == b"last echo":
                loop.assistant_audio_streaming = False
                loop._speaker_write_active = False
                loop._finish_assistant_audio_if_idle()
            if data == b"boundary echo":
                clock[0] += MODEL_AUDIO_ECHO_GRACE_SECONDS + 0.01
            if data == b"pause" or (data == b"silence" and not user_replies):
                loop.running = False
            return data

        loop.audio_stream = mock.Mock(read=read)
        loop.hybrid_turn_detector.feed.side_effect = [
            (False, False, False, b""),  # silence after the echo guard
            (True, False, False, b"reply"),
            (False, True, False, b"pause"),
        ]
        loop._enqueue_realtime_message = mock.AsyncMock(return_value=True)
        await loop.listen_microphone()

        expected = [mock.call(b"silence")]
        if user_replies:
            expected += [mock.call(b"reply"), mock.call(b"pause")]
        self.assertEqual(loop.hybrid_turn_detector.feed.call_args_list, expected)
        self.assertEqual(
            [call.args[0] for call in loop.detector.feed.call_args_list],
            [call.args[0] for call in expected],
        )
        if user_replies:
            loop._enqueue_realtime_message.assert_awaited_once_with(
                {"kind": "speech_turn", "data": b"replypause"}
            )
            loop._mark_session_user_response.assert_called_once_with("voice")
        else:
            loop._enqueue_realtime_message.assert_not_awaited()
            loop._mark_session_user_response.assert_not_called()

    async def test_echo_followed_by_silence_does_not_send_a_user_turn(self):
        await self.run_microphone_sequence(user_replies=False)

    async def test_actual_reply_after_echo_guard_is_sent_once(self):
        await self.run_microphone_sequence(user_replies=True)


if __name__ == "__main__":
    unittest.main()
