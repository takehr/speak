import contextlib
import socket
import unittest
from unittest import mock

import speak
from speak import AudioLoop


class FakeConnection:
    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def __aexit__(self, *_args):
        return False


class FakeLive:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.connect_calls = 0

    def connect(self, **_kwargs):
        outcome = self.outcomes[self.connect_calls]
        self.connect_calls += 1
        return FakeConnection(outcome)


class FakeClient:
    def __init__(self, outcomes):
        self.aio = type("FakeAio", (), {})()
        self.aio.live = FakeLive(outcomes)


class LiveConnectionRetryTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, attempts=3):
        loop = object.__new__(AudioLoop)
        loop.live_connect_attempts = attempts
        loop.live_connect_retry_seconds = 0.0
        loop.live_open_timeout = 30.0
        loop.use_websocket_proxy = True
        loop.live_force_ipv4 = False
        loop.model = "models/test"
        loop.live_config = {}
        loop._status = lambda *_args, **_kwargs: None
        return loop

    async def test_retries_opening_handshake_timeout(self):
        expected_session = object()
        fake_client = FakeClient(
            [
                TimeoutError("timed out during opening handshake"),
                TimeoutError("timed out during opening handshake"),
                expected_session,
            ]
        )
        loop = self.make_loop()

        with mock.patch("speak.get_client", return_value=fake_client):
            async with contextlib.AsyncExitStack() as stack:
                session = await loop._enter_live_session_with_retry(stack)

        self.assertIs(session, expected_session)
        self.assertEqual(fake_client.aio.live.connect_calls, 3)

    async def test_does_not_retry_non_network_configuration_error(self):
        fake_client = FakeClient([ValueError("invalid config")])
        loop = self.make_loop()

        with (
            mock.patch("speak.get_client", return_value=fake_client),
            self.assertRaisesRegex(ValueError, "invalid config"),
        ):
            async with contextlib.AsyncExitStack() as stack:
                await loop._enter_live_session_with_retry(stack)

        self.assertEqual(fake_client.aio.live.connect_calls, 1)


class LiveClientOptionsTests(unittest.TestCase):
    def test_configures_timeout_and_proxy_bypass_for_websocket(self):
        with (
            mock.patch.object(speak, "client", None),
            mock.patch.object(speak, "client_connection_options", None),
            mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            client = speak.get_client(
                live_open_timeout=45.0,
                use_proxy=False,
                force_ipv4=True,
            )

        websocket_options = client._api_client._websocket_ssl_ctx
        self.assertEqual(websocket_options["open_timeout"], 45.0)
        self.assertIsNone(websocket_options["proxy"])
        self.assertEqual(websocket_options["family"], socket.AF_INET)


if __name__ == "__main__":
    unittest.main()
