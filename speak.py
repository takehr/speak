"""
## Documentation
Quickstart:
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup
pip install google-genai opencv-python pyaudio pillow mss vosk pyttsx3
"""

import argparse
import asyncio
import base64
import contextlib
import datetime as dt
import io
import json
import logging
import os
import platform
import re
import shutil
import traceback
from pathlib import Path

import cv2
import pyaudio
import PIL.Image

from google import genai
from google.genai import errors
from google.genai import types

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    import vosk
except ImportError:
    vosk = None

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_MODE = "camera"
DEFAULT_WAKE_WORD = "gemini"
DEFAULT_EXIT_WORD = "see you"
DEFAULT_STT_MODEL_PATH = "./models/vosk-model-small-en-us-0.15"
LOG_DIR = Path("./logs")
PROMPTS_DIR = Path("./prompts")
IDLE_SOUND_PATH = Path("./VSQSE_0522_pirorin_01.mp3")

RECOVERABLE_ERROR_PATTERNS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "deadline",
    "internal",
    "rate limit",
    "resource exhausted",
    "server",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "unavailable",
)

client = None

# LiveConnectConfig（system_instruction は role='system' が安定しやすいです）
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    realtime_input_config=types.RealtimeInputConfig(
        turn_coverage="TURN_INCLUDES_ALL_INPUT"
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    system_instruction=types.Content(
        parts=[
            types.Part.from_text(
                text="Follow the user's roleplay setup. Keep responses natural and conversational."
            )
        ],
        role="system",
    ),
)

pya = pyaudio.PyAudio()


def normalize_phrase(text):
    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(normalized.split())


def phrase_in_text(phrase, text):
    if not phrase or not text:
        return False
    phrase_tokens = phrase.split()
    text_tokens = text.split()
    window = len(phrase_tokens)
    if window == 0 or window > len(text_tokens):
        return False
    return any(text_tokens[i : i + window] == phrase_tokens for i in range(len(text_tokens) - window + 1))


def list_input_devices():
    """PyAudioの入力デバイス一覧を表示"""
    for i in range(pya.get_device_count()):
        info = pya.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            name = info.get("name", "unknown")
            rate = int(info.get("defaultSampleRate", 0))
            chans = int(info.get("maxInputChannels", 0))
            print(f"{i}: {name} (inputs={chans}, defaultRate={rate})")


def configure_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{dt.date.today().isoformat()}.log"
    logger = logging.getLogger("speak")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_prompt_scenarios(prompts_dir=PROMPTS_DIR):
    scenarios = []
    for path in sorted(prompts_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title = path.stem.replace("_", " ").replace("-", " ").strip()
        first_line = content.splitlines()[0].strip()
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip() or title
        scenarios.append({"path": path, "title": title, "prompt": content})

    if not scenarios:
        raise RuntimeError(f"No prompt markdown files found in {prompts_dir}.")
    return scenarios


def select_daily_prompt(scenarios, today=None):
    if not scenarios:
        raise RuntimeError("No prompt scenarios are available.")
    current_date = today or dt.date.today()
    index = current_date.toordinal() % len(scenarios)
    return scenarios[index]


def get_client():
    global client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )
    return client


def iter_response_parts(response):
    if (
        not response.server_content
        or not response.server_content.model_turn
        or not response.server_content.model_turn.parts
    ):
        return []
    return response.server_content.model_turn.parts


class LocalSpeaker:
    def __init__(self, logger):
        self.logger = logger
        self.available = pyttsx3 is not None

    def _speak_blocking(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    async def speak(self, text):
        if not self.available:
            self.logger.warning("TTS unavailable: pyttsx3 is not installed.")
            return
        try:
            await asyncio.to_thread(self._speak_blocking, text)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.logger.warning("TTS failed: %s", exc)


class LocalSoundPlayer:
    def __init__(self, logger):
        self.logger = logger
        self.available = pygame is not None
        self._mixer_ready = False

    def _ensure_mixer(self):
        if not self.available:
            return False
        if self._mixer_ready:
            return True
        try:
            pygame.mixer.init()
            self._mixer_ready = True
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            self.logger.warning("pygame mixer init failed: %s", exc)
            return False

    def _play_blocking(self, path):
        if not self._ensure_mixer():
            return False
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(20)
            return True
        finally:
            with contextlib.suppress(Exception):
                pygame.mixer.music.stop()
            with contextlib.suppress(Exception):
                pygame.mixer.music.unload()
            with contextlib.suppress(Exception):
                pygame.mixer.quit()
            self._mixer_ready = False

    async def play(self, path):
        if not self.available:
            return False
        try:
            return await asyncio.to_thread(self._play_blocking, path)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.logger.warning("pygame sound playback failed: %s", exc)
            return False


class LocalCommandDetector:
    def __init__(
        self,
        model_path,
        wake_word,
        exit_word,
        logger,
        no_auto_start_wake_word=None,
        cooldown_seconds=1.5,
    ):
        if vosk is None:
            raise RuntimeError("vosk is required. Install it with `pip install vosk`.")

        resolved = Path(model_path)
        if not resolved.exists():
            raise RuntimeError(
                f"Vosk model not found at {resolved}. Set --stt-model-path or VOSK_MODEL_PATH."
            )

        self.logger = logger
        self.wake_word = normalize_phrase(wake_word)
        self.exit_word = normalize_phrase(exit_word)
        self.no_auto_start_wake_word = normalize_phrase(no_auto_start_wake_word or "")
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_at = {"wake": 0.0, "exit": 0.0, "no_auto_start_wake": 0.0}
        self.model = vosk.Model(str(resolved))
        self.recognizer = vosk.KaldiRecognizer(self.model, SEND_SAMPLE_RATE)
        self.recognizer.SetWords(False)

    def _should_emit(self, command, now):
        last = self.last_trigger_at.get(command, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        self.last_trigger_at[command] = now
        return True

    def _extract_text(self, payload):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""
        return normalize_phrase(data.get("text") or data.get("partial") or "")

    def _match_command(self, text):
        now = asyncio.get_running_loop().time()
        if phrase_in_text(self.exit_word, text) and self._should_emit("exit", now):
            return "exit"
        if (
            phrase_in_text(self.no_auto_start_wake_word, text)
            and self._should_emit("no_auto_start_wake", now)
        ):
            return "no_auto_start_wake"
        if phrase_in_text(self.wake_word, text) and self._should_emit("wake", now):
            return "wake"
        return None

    def feed(self, pcm_bytes):
        payloads = []
        if self.recognizer.AcceptWaveform(pcm_bytes):
            payloads.append(self.recognizer.Result())
        else:
            payloads.append(self.recognizer.PartialResult())

        for payload in payloads:
            text = self._extract_text(payload)
            if not text:
                continue
            command = self._match_command(text)
            if command is None:
                continue
            self.logger.info("Detected %s phrase from transcript=%r", command, text)
            if hasattr(self.recognizer, "Reset"):
                self.recognizer.Reset()
            return command
        return None


class AudioLoop:
    def __init__(
        self,
        video_mode=DEFAULT_MODE,
        auto_start=True,
        enable_text_input=True,
        mic_index=None,
        strict_turns=False,
        wake_word=DEFAULT_WAKE_WORD,
        exit_word=DEFAULT_EXIT_WORD,
        no_auto_start_wake_word=None,
        wake_word_enabled=True,
        stt_model_path=DEFAULT_STT_MODEL_PATH,
    ):
        self.video_mode = video_mode
        self.auto_start = auto_start
        self.enable_text_input = enable_text_input
        self.mic_index = mic_index
        self.strict_turns = strict_turns
        self.wake_word_enabled = wake_word_enabled
        self.no_auto_start_wake_word = normalize_phrase(no_auto_start_wake_word or "")
        self.prompt_scenarios = load_prompt_scenarios()
        self.daily_prompt = select_daily_prompt(self.prompt_scenarios)

        self.logger = configure_logging()
        self.speaker = LocalSpeaker(self.logger)
        self.sound_player = LocalSoundPlayer(self.logger)
        self.detector = LocalCommandDetector(
            model_path=stt_model_path,
            wake_word=wake_word,
            exit_word=exit_word,
            logger=self.logger,
            no_auto_start_wake_word=self.no_auto_start_wake_word,
        )

        self.audio_stream = None
        self.session = None
        self.audio_in_queue = None
        self.out_queue = None
        self.session_task = None
        self.session_stop_event = None
        self.input_task = None
        self.assistant_speaking = False
        self.running = True
        self.state = "idle"
        self.control_queue = asyncio.Queue()

    def _status(self, text, level="info"):
        getattr(self.logger, level)("%s", text)
        print(f"[status] {text}")

    def _set_state(self, new_state):
        previous_state = self.state
        if previous_state == new_state:
            return
        self.logger.info("state %s -> %s", previous_state, new_state)
        print(f"[state] {previous_state} -> {new_state}")
        self.state = new_state
        if previous_state == "active" and new_state == "idle":
            try:
                asyncio.get_running_loop().create_task(self._play_idle_sound())
            except RuntimeError:
                self.logger.warning("idle sound skipped: no running event loop")

    async def _announce(self, text, level="info"):
        getattr(self.logger, level)("%s", text)
        print(text)
        await self.speaker.speak(text)

    def _show_daily_prompt(self):
        path = self.daily_prompt["path"]
        title = self.daily_prompt["title"]
        message = f"[prompt] {title} ({path.as_posix()})"
        self.logger.info("daily prompt: %s", message)
        print(message)

    def _get_idle_sound_command(self):
        sound_path = str(IDLE_SOUND_PATH.resolve())
        system_name = platform.system()

        if system_name == "Darwin":
            if shutil.which("afplay") is not None:
                return ["afplay", sound_path]
            return None

        if system_name == "Windows":
            powershell = shutil.which("powershell") or shutil.which("powershell.exe") or shutil.which("pwsh")
            if powershell is None:
                return None
            script = (
                "$player = New-Object -ComObject WMPlayer.OCX;"
                f"$media = $player.newMedia('{sound_path}');"
                "$player.currentPlaylist.appendItem($media);"
                "$player.controls.play();"
                "while ($player.playState -ne 1) { Start-Sleep -Milliseconds 100 }"
            )
            return [powershell, "-NoProfile", "-Command", script]

        linux_candidates = (
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", sound_path],
            ["mpg123", "-q", sound_path],
            ["mpg321", "-q", sound_path],
            ["mpv", "--no-video", "--really-quiet", sound_path],
            ["cvlc", "--play-and-exit", "--quiet", sound_path],
        )
        for command in linux_candidates:
            if shutil.which(command[0]) is not None:
                return command
        return None

    async def _play_idle_sound(self):
        if not IDLE_SOUND_PATH.exists():
            self._status(f"idle sound not found: {IDLE_SOUND_PATH}", level="warning")
            return
        self._status(f"playing idle sound: {IDLE_SOUND_PATH.name}")
        if await self.sound_player.play(IDLE_SOUND_PATH):
            self._status("idle sound playback finished")
            return
        command = self._get_idle_sound_command()
        if command is None:
            self._status("idle sound skipped: no supported audio player is available", level="warning")
            return

        try:
            process = await asyncio.create_subprocess_exec(*command)
            await process.wait()
            self._status(f"idle sound playback finished with code {process.returncode}")
        except Exception as exc:  # pragma: no cover - environment dependent
            self._status(f"idle sound playback failed: {exc}", level="warning")

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        try:
            while self.running and self.state == "active":
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    break

                await asyncio.sleep(1.0)

                if self.out_queue is not None and not (
                    self.strict_turns and self.assistant_speaking
                ):
                    await self.out_queue.put(frame)
        finally:
            cap.release()

    def _get_screen(self):
        try:
            import mss  # pylint: disable=g-import-not-at-top
        except ImportError as exc:
            raise ImportError("Please install mss package using 'pip install mss'") from exc

        sct = mss.mss()
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)

        image_bytes = mss.tools.to_png(shot.rgb, shot.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while self.running and self.state == "active":
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            if self.out_queue is not None and not (
                self.strict_turns and self.assistant_speaking
            ):
                await self.out_queue.put(frame)

    async def send_realtime(self):
        while self.running and self.state == "active":
            if self.out_queue is None:
                await asyncio.sleep(0.01)
                continue
            msg = await self.out_queue.get()
            if self.session is not None:
                await self.session.send(input=msg)

    async def receive_audio(self):
        try:
            while self.running and self.state == "active":
                if self.session is None:
                    await asyncio.sleep(0.01)
                    continue

                turn = self.session.receive()
                async for response in turn:
                    for part in iter_response_parts(response):
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            self.assistant_speaking = True
                            if self.audio_in_queue is not None:
                                self.audio_in_queue.put_nowait(part.inline_data.data)
                            continue
                        if isinstance(part.text, str) and not getattr(part, "thought", False):
                            print(part.text, end="")

                self.assistant_speaking = False
                if not self.strict_turns and self.audio_in_queue is not None:
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
        except Exception as exc:
            if self._is_normal_session_close(exc):
                self.logger.info("receive_audio closed normally: %s", exc)
                return
            raise

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        try:
            while self.running and self.state == "active":
                if self.audio_in_queue is None:
                    await asyncio.sleep(0.01)
                    continue
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        finally:
            await asyncio.to_thread(stream.close)

    async def listen_microphone(self):
        if self.mic_index is None:
            mic_info = pya.get_default_input_device_info()
            mic_index = mic_info["index"]
        else:
            mic_index = self.mic_index

        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while self.running:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

            command = self.detector.feed(data)
            if command == "wake":
                if self.state == "idle":
                    start_mode = "prompt" if self.no_auto_start_wake_word and not self.auto_start else "default"
                    self._status(
                        f"wake word detected from voice; startup mode={start_mode}"
                    )
                    await self.control_queue.put(("wake", start_mode, "voice"))
            elif command == "no_auto_start_wake":
                if self.state == "idle" and not self.auto_start:
                    self._status("no-auto-start wake word detected from voice; startup mode=silent")
                    await self.control_queue.put(("wake", "silent", "voice"))
            elif command == "exit":
                self._status("exit word detected from voice")
                await self.control_queue.put(("sleep", None, "voice"))

            if self.out_queue is not None and self.state == "active" and not (
                self.strict_turns and self.assistant_speaking
            ):
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_text(self):
        while self.running:
            text = await asyncio.to_thread(input, "message > ")
            normalized = normalize_phrase(text)
            if normalized == "q":
                self._status("quit requested from text input")
                await self.control_queue.put(("quit", None, "text"))
                return
            if self.state == "idle" and normalized == self.detector.wake_word:
                start_mode = "prompt" if self.no_auto_start_wake_word and not self.auto_start else "default"
                self._status(
                    f"wake word detected from text; startup mode={start_mode}"
                )
                await self.control_queue.put(("wake", start_mode, "text"))
                continue
            if (
                self.state == "idle"
                and not self.auto_start
                and normalized == self.detector.no_auto_start_wake_word
            ):
                self._status("no-auto-start wake word detected from text; startup mode=silent")
                await self.control_queue.put(("wake", "silent", "text"))
                continue
            if self.state in {"connecting", "active"} and normalized == self.detector.exit_word:
                self._status("exit word detected from text")
                await self.control_queue.put(("sleep", None, "text"))
                continue
            if self.session is not None and self.state == "active":
                while self.strict_turns and self.assistant_speaking:
                    await asyncio.sleep(0.05)
                await self.session.send(input=text or ".", end_of_turn=True)

    def _is_recoverable_gemini_error(self, exc):
        message = normalize_phrase(str(exc))
        return any(pattern in message for pattern in RECOVERABLE_ERROR_PATTERNS)

    def _is_normal_session_close(self, exc):
        if isinstance(exc, asyncio.CancelledError):
            return True
        if isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 1000:
            return True
        message = normalize_phrase(str(exc))
        return message in {"1000 none", "1000 ok"} or "connection closed ok" in message

    async def _report_session_error(self, exc):
        message = str(exc).strip() or exc.__class__.__name__
        if self._is_recoverable_gemini_error(exc):
            await self._announce(
                f"Gemini server error. Returning to idle. You can say {self.detector.wake_word} to reconnect.",
                level="error",
            )
            self.logger.error("recoverable Gemini error: %s", message)
            return

        await self._announce(
            f"Session error. Returning to idle. Details: {message}",
            level="error",
        )
        self.logger.error("non-recoverable session error: %s", message)

    async def stop_session(self, reason):
        if self.state == "idle":
            self.logger.info("stop_session ignored in idle: reason=%s", reason)
            return

        self._status(f"stopping session; reason={reason}")
        self.logger.info("stop_session requested: reason=%s", reason)
        if self.state == "connecting" and self.session_task is not None:
            self.session_task.cancel()
        elif self.session_stop_event is not None:
            self.session_stop_event.set()

        if self.session_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.session_task

    async def _cleanup_session(self, session_tasks):
        self._status("cleaning up session tasks")
        for task in session_tasks:
            task.cancel()
        for task in session_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if self._is_normal_session_close(exc):
                    self.logger.info("session task closed normally: %s", exc)
                else:
                    self.logger.exception("session task cleanup failure: %s", exc)

        self.assistant_speaking = False
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.session_stop_event = None
        self.session_task = None
        self._set_state("idle")

    async def _run_session(self, send_opening_prompt):
        self._set_state("connecting")
        self._status(f"opening Gemini live session; send_opening_prompt={send_opening_prompt}")
        self.session_stop_event = asyncio.Event()
        session_tasks = []
        try:
            async with get_client().aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self._set_state("active")

                if send_opening_prompt:
                    self._status(f"sending opening prompt: {self.daily_prompt['title']}")
                    await self.session.send(input=self.daily_prompt["prompt"], end_of_turn=True)
                else:
                    self._status("session started without opening prompt")

                session_tasks.append(asyncio.create_task(self.send_realtime()))
                session_tasks.append(asyncio.create_task(self.receive_audio()))
                session_tasks.append(asyncio.create_task(self.play_audio()))

                if self.video_mode == "camera":
                    session_tasks.append(asyncio.create_task(self.get_frames()))
                elif self.video_mode == "screen":
                    session_tasks.append(asyncio.create_task(self.get_screen()))

                await self.session_stop_event.wait()
        except asyncio.CancelledError:
            self.logger.info("session cancelled while state=%s", self.state)
            self._status("session cancelled")
            raise
        except Exception as exc:
            self.logger.exception("session failure")
            self._status(f"session failure: {exc}", level="error")
            await self._report_session_error(exc)
        finally:
            await self._cleanup_session(session_tasks)

    async def start_session(self, reason, send_opening_prompt):
        if self.state != "idle":
            self.logger.info("start_session ignored in state=%s reason=%s", self.state, reason)
            return

        self.logger.info(
            "start_session requested: reason=%s send_opening_prompt=%s",
            reason,
            send_opening_prompt,
        )
        self._status(
            f"starting session; reason={reason}; send_opening_prompt={send_opening_prompt}"
        )
        self.session_task = asyncio.create_task(self._run_session(send_opening_prompt))

    async def run(self):
        background_tasks = []
        try:
            self.logger.info("application start")
            self._status("application start")
            self._show_daily_prompt()
            if self.no_auto_start_wake_word and not self.auto_start:
                startup_message = (
                    f"Idle. Say {self.detector.wake_word} to start with today's prompt,"
                    f" or say {self.detector.no_auto_start_wake_word} to start without it."
                    f" Say {self.detector.exit_word} to return to idle."
                )
            else:
                startup_message = (
                    f"Idle. Say {self.detector.wake_word} to start and {self.detector.exit_word}"
                    " to return to idle."
                )
            await self._announce(startup_message)

            background_tasks.append(asyncio.create_task(self.listen_microphone()))
            if self.enable_text_input:
                background_tasks.append(asyncio.create_task(self.send_text()))

            if not self.wake_word_enabled:
                self._status("wake word disabled at startup; starting session immediately")
                await self.control_queue.put(("wake", "default", "startup"))

            while self.running:
                command, start_mode, reason = await self.control_queue.get()
                self._status(
                    f"received control command={command} start_mode={start_mode} reason={reason}"
                )
                if command == "wake":
                    send_opening_prompt = self.auto_start or start_mode == "prompt"
                    await self.start_session(reason, send_opening_prompt=send_opening_prompt)
                elif command == "sleep":
                    await self.stop_session(reason)
                elif command == "quit":
                    self.running = False
                    await self.stop_session(reason)
                else:
                    self.logger.warning("unknown command=%s reason=%s", command, reason)
        except KeyboardInterrupt:
            self.logger.info("keyboard interrupt")
            self._status("keyboard interrupt received")
        except Exception as exc:
            self.logger.exception("fatal application error")
            self._status(f"fatal application error: {exc}", level="error")
            await self._announce(f"Fatal error: {exc}", level="error")
            traceback.print_exception(exc)
        finally:
            self.running = False
            self._status("application shutdown start")
            await self.stop_session("shutdown")
            for task in background_tasks:
                task.cancel()
            for task in background_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if self.audio_stream is not None:
                with contextlib.suppress(Exception):
                    self.audio_stream.close()
            self.logger.info("application stop")
            self._status("application stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Disable auto start (model won't speak first).",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable console text input (run hands-free).",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphone input devices and exit.",
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="PyAudio input device index (use --list-mics to find).",
    )
    parser.add_argument(
        "--strict-turns",
        action="store_true",
        help="Disable barge-in by not sending new input while the model is speaking.",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default=DEFAULT_WAKE_WORD,
        help="Phrase that starts a Gemini session.",
    )
    parser.add_argument(
        "--exit-word",
        type=str,
        default=DEFAULT_EXIT_WORD,
        help="Phrase that returns the app to idle.",
    )
    parser.add_argument(
        "--no-auto-start-wake-word",
        type=str,
        default=None,
        help="Alternate phrase that starts a Gemini session from idle when using --no-auto-start.",
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Start a session immediately, then fall back to wake-word mode after returning to idle.",
    )
    parser.add_argument(
        "--stt-model-path",
        type=str,
        default=os.environ.get("VOSK_MODEL_PATH", DEFAULT_STT_MODEL_PATH),
        help="Path to the local Vosk speech recognition model directory.",
    )

    args = parser.parse_args()

    if args.list_mics:
        list_input_devices()
        raise SystemExit(0)

    main = AudioLoop(
        video_mode=args.mode,
        auto_start=(not args.no_auto_start),
        enable_text_input=(not args.no_text),
        mic_index=args.mic_index,
        strict_turns=args.strict_turns,
        wake_word=args.wake_word,
        exit_word=args.exit_word,
        no_auto_start_wake_word=args.no_auto_start_wake_word,
        wake_word_enabled=(not args.no_wake_word),
        stt_model_path=args.stt_model_path,
    )
    asyncio.run(main.run())
