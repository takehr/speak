"""
## Documentation
Quickstart:
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup
pip install google-genai opencv-python pyaudio pillow mss vosk pyttsx3
"""

import argparse
import asyncio
import contextlib
import datetime as dt
import io
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
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
    import vosk
except ImportError:
    vosk = None

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
REALTIME_SEND_TIMEOUT_SECONDS = 10.0
OUT_QUEUE_STALL_TIMEOUT_SECONDS = 10.0
HEALTH_CHECK_INTERVAL_SECONDS = 1.0
USER_SPEECH_START_RMS = 45.0
USER_SPEECH_END_RMS = 25.0
USER_SPEECH_SILENCE_SECONDS = 0.8
VOICE_COMMAND_ECHO_GRACE_SECONDS = 2.0
WAKE_COMMAND_MIN_CONFIDENCE = 0.65

DEFAULT_MODEL = os.environ.get(
    "GEMINI_LIVE_MODEL",
    "models/gemini-2.5-flash-native-audio-preview-12-2025",
)
DEFAULT_MODE = "camera"
DEFAULT_WAKE_WORD = "gemini"
DEFAULT_MIMIC_WAKE_WORD = "copy mode"
DEFAULT_TRANSLATE_WAKE_WORD = "translate"
DEFAULT_EXIT_WORD = "see you"
DEFAULT_STT_MODEL_PATH = "./models/vosk-model-small-en-us-0.15"
LOG_DIR = Path("./logs")
PROMPTS_DIR = Path("./prompts")
IDLE_SOUND_PATH = Path("./VSQSE_0522_pirorin_01.mp3")
IDLE_SOUND_HELPER_PATH = Path("./play_idle_sound.py")

RECOVERABLE_ERROR_PATTERNS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "deadline",
    "billing",
    "internal",
    "quota",
    "rate limit",
    "resource exhausted",
    "server",
    "service unavailable",
    "spending cap",
    "temporarily unavailable",
    "timeout",
    "unavailable",
)
QUOTA_OR_BILLING_ERROR_PATTERNS = (
    "billing",
    "quota",
    "resource exhausted",
    "spending cap",
)
UNSUPPORTED_LIVE_OPERATION_PATTERNS = (
    "1008",
    "not implemented",
    "not supported",
    "not enabled",
    "policy violation",
)

client = None


def build_live_config(enable_search=False):
    tools = None
    if enable_search:
        tools = [{"google_search": {}}]

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            ),
            turn_coverage="TURN_INCLUDES_ALL_INPUT",
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
        system_instruction=(
            "Follow the user's roleplay setup. Keep responses natural and conversational."
        ),
        tools=tools,
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


def build_mimic_mode_prompt(daily_prompt):
    title = daily_prompt["title"]
    scenario = daily_prompt["prompt"].strip()
    return f"""You are an English shadowing coach for a Japanese learner.

Use today's scenario below as the source situation.

Scenario title: {title}

Scenario instructions:
{scenario}

Run the conversation in this order:
1. First, give one short model answer in natural spoken English that fits the scenario well. Make it sound polished, not textbook-like.
2. Tell the user to repeat your answer as closely as possible.
3. After the user speaks, compare their response with your model answer and give concise feedback in Japanese.
4. In the feedback, explicitly check:
   - whether the meaning and wording stayed close to the model answer
   - whether any words were dropped, changed, or added
   - whether pronunciation, stress, rhythm, or linking sounded unnatural
   - whether the user fully copied the answer or still sounded different from the model
5. Then give one short retry suggestion and ask the user to repeat again.

Important behavior rules:
- Keep each model answer short: about 1 to 3 sentences.
- Prioritize imitation quality over free conversation.
- If the user's answer is far from the model answer, point that out clearly but kindly.
- When judging pronunciation, use the user's audio and wording together.
- Keep feedback practical and specific, with a coach-like tone.
- Stay in this shadowing loop unless the user clearly wants to stop or change topics.
"""


def build_translate_mode_prompt(daily_prompt):
    title = daily_prompt["title"]
    scenario = daily_prompt["prompt"].strip()
    return f"""You are an English speaking coach for a Japanese learner doing instant sentence translation practice.

Use today's scenario below as the source situation.

Scenario title: {title}

Scenario instructions:
{scenario}

Run the conversation in this order:
1. First, come up with one short model answer in natural spoken English that fits the scenario well.
2. Translate only that model answer into natural Japanese and say the Japanese first with an item number, for example: "1つめの日本語は...".
3. Tell the user to express that Japanese in English.
4. After the user speaks, compare their English with your hidden model answer and give concise feedback in Japanese.
5. If the user's English is still unnatural, incomplete, or inaccurate, do not move to the next sentence yet. Explain what to fix, give a short hint or corrected example, and ask the user to try the same sentence again.
6. Only when the user has said a sufficiently natural and correct English version, briefly confirm that in Japanese and then give the next numbered Japanese prompt.
7. Continue the main practice until the user has successfully completed at least 10 different numbered Japanese-to-English practice items in the session.
8. After at least 10 numbered practice items are completed, run a quick review test using those practiced items before wrapping up.

Important behavior rules:
- Keep each individual target answer short: usually 1 English sentence, and at most 3 English sentences for a single item.
- The "1 to 3 sentences" limit applies only to one item. It is not the total session length.
- The total main practice must be at least 10 numbered items. Do not stop after 3 items.
- The Japanese should sound natural and should preserve the intended meaning of the English model answer.
- Do not reveal the English model answer before the user tries.
- Judge the user's answer mainly by meaning, naturalness, and whether it fits the scenario.
- If the user's answer is awkward but understandable, explain how to make it more natural.
- Stay on the same sentence until the user can say it naturally enough.
- During the main practice before the quick review test, announce the item number before each new Japanese prompt, for example: "1つめの日本語は...", "4つめの日本語は...".
- Keep the same item number when asking the user to retry the same sentence.
- Increase the item number only after the user has completed that sentence naturally enough.
- Do not introduce a new Japanese sentence immediately after pointing out mistakes.
- Do not say the practice is finished, wrap up, or ask whether to stop before at least 10 completed numbered items.
- Count only numbered items that the user has corrected to a sufficiently natural English version.
- Keep track of the Japanese prompts and accepted English answers during the session.
- For the quick review test, use this exact format: announce the item number before each practiced Japanese prompt, then wait for the user to translate it into English.
- Number each quick review item in Japanese before the prompt, for example: "1つめの日本語は...", "4つめの日本語は...".
- Do not say the English answer before the user attempts the quick review item.
- After feedback on one quick review item, announce the next item number, say the next practiced Japanese prompt, and wait for the user's English translation.
- During the quick review test, give brief Japanese feedback after each answer and correct the user if needed.
- Only wrap up after completing the quick review test.
- Keep the interaction in this Japanese-to-English training loop unless the user clearly wants to stop or change topics.
"""


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
        self.available = platform.system() not in {"Darwin", "Windows"}
        self._mixer_ready = False
        self._pygame = None

    def _load_pygame(self):
        if not self.available:
            return None
        if self._pygame is not None:
            return self._pygame
        try:
            import pygame  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.available = False
            return None
        self._pygame = pygame
        return pygame

    def _ensure_mixer(self):
        pygame = self._load_pygame()
        if pygame is None:
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
        pygame = self._load_pygame()
        if pygame is None:
            return False
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
    WAKE_COMMANDS = {"wake", "mimic_wake", "translate_wake", "no_auto_start_wake"}

    def __init__(
        self,
        model_path,
        wake_word,
        mimic_wake_word,
        translate_wake_word,
        exit_word,
        logger,
        no_auto_start_wake_word=None,
        cooldown_seconds=1.5,
        wake_min_confidence=WAKE_COMMAND_MIN_CONFIDENCE,
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
        self.mimic_wake_word = normalize_phrase(mimic_wake_word)
        self.translate_wake_word = normalize_phrase(translate_wake_word)
        self.exit_word = normalize_phrase(exit_word)
        self.no_auto_start_wake_word = normalize_phrase(no_auto_start_wake_word or "")
        self.cooldown_seconds = cooldown_seconds
        self.wake_min_confidence = wake_min_confidence
        self.last_trigger_at = {
            "wake": 0.0,
            "mimic_wake": 0.0,
            "translate_wake": 0.0,
            "exit": 0.0,
            "no_auto_start_wake": 0.0,
        }
        self.last_phrase_trigger_at = {}
        self.model = vosk.Model(str(resolved))
        self.recognizer = vosk.KaldiRecognizer(self.model, SEND_SAMPLE_RATE)
        self.recognizer.SetWords(True)

    def _should_emit(self, command, now, phrase=None):
        last = self.last_trigger_at.get(command, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        if phrase:
            last_phrase = self.last_phrase_trigger_at.get(phrase, 0.0)
            if now - last_phrase < self.cooldown_seconds:
                return False
        self.last_trigger_at[command] = now
        if phrase:
            self.last_phrase_trigger_at[phrase] = now
        return True

    def reset(self):
        if hasattr(self.recognizer, "Reset"):
            self.recognizer.Reset()

    def _extract_result(self, payload):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return "", False, []
        is_final = "text" in data
        raw_text = (data.get("text") if is_final else data.get("partial")) or ""
        text = normalize_phrase(raw_text)
        return text, is_final, data.get("result") or []

    def _words_with_confidence(self, words):
        normalized_words = []
        for word_info in words:
            tokens = normalize_phrase(word_info.get("word") or "").split()
            if not tokens:
                continue
            try:
                confidence = float(word_info.get("conf", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            normalized_words.extend((token, confidence) for token in tokens)
        return normalized_words

    def _phrase_confident_enough(self, phrase, words):
        normalized_words = self._words_with_confidence(words)
        phrase_tokens = phrase.split()
        window = len(phrase_tokens)
        if not normalized_words or window == 0 or window > len(normalized_words):
            return False

        word_tokens = [token for token, _confidence in normalized_words]
        for index in range(len(word_tokens) - window + 1):
            if word_tokens[index : index + window] != phrase_tokens:
                continue
            confidences = [
                confidence
                for _token, confidence in normalized_words[index : index + window]
            ]
            return min(confidences) >= self.wake_min_confidence
        return False

    def _commands_for_state(self, state):
        if state == "idle":
            return (
                ("mimic_wake", self.mimic_wake_word),
                ("translate_wake", self.translate_wake_word),
                ("no_auto_start_wake", self.no_auto_start_wake_word),
                ("wake", self.wake_word),
            )
        if state in {"connecting", "active"}:
            return (("exit", self.exit_word),)
        return (
            ("exit", self.exit_word),
            ("mimic_wake", self.mimic_wake_word),
            ("translate_wake", self.translate_wake_word),
            ("no_auto_start_wake", self.no_auto_start_wake_word),
            ("wake", self.wake_word),
        )

    def _match_command(self, text, state=None, is_final=False, words=None):
        now = asyncio.get_running_loop().time()
        for command, phrase in self._commands_for_state(state):
            if not phrase_in_text(phrase, text):
                continue
            if command in self.WAKE_COMMANDS:
                if not is_final:
                    continue
                if words and not self._phrase_confident_enough(phrase, words):
                    self.logger.info(
                        "Ignored low-confidence %s phrase from transcript=%r",
                        command,
                        text,
                    )
                    continue
            if self._should_emit(command, now, phrase):
                return command
        return None

    def feed(self, pcm_bytes, state=None):
        payloads = []
        if self.recognizer.AcceptWaveform(pcm_bytes):
            payloads.append(self.recognizer.Result())
        else:
            payloads.append(self.recognizer.PartialResult())

        for payload in payloads:
            text, is_final, words = self._extract_result(payload)
            if not text:
                continue
            command = self._match_command(
                text,
                state=state,
                is_final=is_final,
                words=words,
            )
            if command is None:
                continue
            result_type = "final" if is_final else "partial"
            self.logger.info(
                "Detected %s phrase from %s transcript=%r",
                command,
                result_type,
                text,
            )
            self.reset()
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
        mimic_wake_word=DEFAULT_MIMIC_WAKE_WORD,
        translate_wake_word=DEFAULT_TRANSLATE_WAKE_WORD,
        exit_word=DEFAULT_EXIT_WORD,
        no_auto_start_wake_word=None,
        wake_word_enabled=True,
        stt_model_path=DEFAULT_STT_MODEL_PATH,
        model=DEFAULT_MODEL,
        enable_search=None,
        vad_start_rms=USER_SPEECH_START_RMS,
        vad_end_rms=USER_SPEECH_END_RMS,
        vad_silence_seconds=USER_SPEECH_SILENCE_SECONDS,
        wake_min_confidence=WAKE_COMMAND_MIN_CONFIDENCE,
    ):
        self.video_mode = video_mode
        self.auto_start = auto_start
        self.enable_text_input = enable_text_input
        self.mic_index = mic_index
        self.strict_turns = strict_turns
        self.wake_word_enabled = wake_word_enabled
        self.model = model
        self.enable_search = (
            (not self.auto_start) if enable_search is None else enable_search
        )
        self.vad_start_rms = vad_start_rms
        self.vad_end_rms = vad_end_rms
        self.vad_silence_seconds = vad_silence_seconds
        self.wake_min_confidence = wake_min_confidence
        self.no_auto_start_wake_word = normalize_phrase(no_auto_start_wake_word or "")
        self.prompt_scenarios = load_prompt_scenarios()
        self.daily_prompt = select_daily_prompt(self.prompt_scenarios)
        self.live_config = build_live_config(enable_search=self.enable_search)

        self.logger = configure_logging()
        self.speaker = LocalSpeaker(self.logger)
        self.sound_player = LocalSoundPlayer(self.logger)
        self.detector = LocalCommandDetector(
            model_path=stt_model_path,
            wake_word=wake_word,
            mimic_wake_word=mimic_wake_word,
            translate_wake_word=translate_wake_word,
            exit_word=exit_word,
            logger=self.logger,
            no_auto_start_wake_word=self.no_auto_start_wake_word,
            wake_min_confidence=self.wake_min_confidence,
        )

        self.audio_stream = None
        self.session = None
        self.audio_in_queue = None
        self.out_queue = None
        self.session_task = None
        self.session_stop_event = None
        self.input_task = None
        self.assistant_speaking = False
        self.assistant_audio_streaming = False
        self.user_activity_active = False
        self._user_activity_started_at = 0.0
        self._last_user_voice_at = 0.0
        self.running = True
        self.state = "idle"
        self.control_queue = asyncio.Queue()
        self._last_realtime_send_log = 0.0
        self._last_receive_wait_log = 0.0
        self._last_receive_chunk_log = 0.0
        self._last_audio_play_log = 0.0
        self._last_mic_health_log = 0.0
        self._last_suppressed_speech_log = 0.0
        self._last_out_queue_full_log = 0.0
        self._out_queue_pressure_since = None
        self._session_failure = None
        self._stdin_reader = None
        self._stdin_read_transport = None
        self.local_output_active = False
        self._voice_commands_suppressed_until = 0.0

    def _loop_time(self):
        return asyncio.get_running_loop().time()

    def _debug_every(self, attr_name, interval_seconds):
        now = self._loop_time()
        previous = getattr(self, attr_name, 0.0)
        if now - previous < interval_seconds:
            return False
        setattr(self, attr_name, now)
        return True

    def _status(self, text, level="info"):
        getattr(self.logger, level)("%s", text)
        print(f"[status] {text}")

    def _suppress_voice_commands(self, seconds, reason):
        until = self._loop_time() + seconds
        self._voice_commands_suppressed_until = max(
            self._voice_commands_suppressed_until,
            until,
        )
        self.detector.reset()
        self.logger.info("voice commands suppressed for %.1fs: %s", seconds, reason)

    def _voice_commands_suppressed(self):
        return (
            self.local_output_active
            or self._loop_time() < self._voice_commands_suppressed_until
        )

    def _mark_out_queue_pressure(self):
        if self._out_queue_pressure_since is None:
            self._out_queue_pressure_since = self._loop_time()

    def _clear_out_queue_pressure(self):
        self._out_queue_pressure_since = None

    async def _enqueue_realtime_message(self, message):
        if self.out_queue is None or self.state != "active":
            return False

        while self.out_queue.full():
            self._mark_out_queue_pressure()

            try:
                dropped = self.out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if self._debug_every("_last_out_queue_full_log", 2.0):
                self._status(
                    "realtime input queue full; dropping oldest "
                    f"{dropped.get('kind', 'unknown')} frame to stay responsive",
                    level="warning",
                )

        try:
            self.out_queue.put_nowait(message)
        except asyncio.QueueFull:
            self._mark_out_queue_pressure()
            if self._debug_every("_last_out_queue_full_log", 2.0):
                self._status(
                    "realtime input queue still full; dropping newest "
                    f"{message.get('kind', 'unknown')} frame to stay responsive",
                    level="warning",
                )
            return False
        return True

    def _read_stdin_line_blocking(self):
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        return stream.readline()

    def _close_stdin_reader(self):
        if self._stdin_read_transport is not None:
            with contextlib.suppress(Exception):
                self._stdin_read_transport.close()
        self._stdin_reader = None
        self._stdin_read_transport = None

    def _pcm16_rms(self, data):
        if not data:
            return 0.0
        sample_count = len(data) // 2
        if sample_count == 0:
            return 0.0
        total = 0.0
        for i in range(0, sample_count * 2, 2):
            sample = int.from_bytes(data[i : i + 2], byteorder="little", signed=True)
            total += sample * sample
        return math.sqrt(total / sample_count)

    def _set_state(self, new_state):
        previous_state = self.state
        if previous_state == new_state:
            return
        self.logger.info("state %s -> %s", previous_state, new_state)
        print(f"[state] {previous_state} -> {new_state}")
        self.state = new_state
        if previous_state in {"active", "closing"} and new_state == "idle":
            self._suppress_voice_commands(
                VOICE_COMMAND_ECHO_GRACE_SECONDS,
                "session returned to idle",
            )
            try:
                asyncio.get_running_loop().create_task(self._play_idle_sound())
            except RuntimeError:
                self.logger.warning("idle sound skipped: no running event loop")

    async def _announce(self, text, level="info"):
        getattr(self.logger, level)("%s", text)
        print(text)
        self.local_output_active = True
        self.detector.reset()
        try:
            await self.speaker.speak(text)
        finally:
            self.local_output_active = False
            self._suppress_voice_commands(
                VOICE_COMMAND_ECHO_GRACE_SECONDS,
                "local announcement finished",
            )

    def _show_daily_prompt(self):
        path = self.daily_prompt["path"]
        title = self.daily_prompt["title"]
        message = f"[prompt] {title} ({path.as_posix()})"
        self.logger.info("daily prompt: %s", message)
        print(message)

    def _reload_daily_prompt(self):
        previous_prompt = getattr(self, "daily_prompt", None)
        self.prompt_scenarios = load_prompt_scenarios()
        self.daily_prompt = select_daily_prompt(self.prompt_scenarios)

        if previous_prompt is None:
            return

        if (
            previous_prompt["path"] != self.daily_prompt["path"]
            or previous_prompt["title"] != self.daily_prompt["title"]
            or previous_prompt["prompt"] != self.daily_prompt["prompt"]
        ):
            self._show_daily_prompt()

    def _build_opening_prompt(self, session_mode):
        if session_mode == "silent":
            return None
        if session_mode == "mimic":
            return build_mimic_mode_prompt(self.daily_prompt)
        if session_mode == "translate":
            return build_translate_mode_prompt(self.daily_prompt)
        return self.daily_prompt["prompt"]

    def _get_idle_sound_command(self):
        sound_path = str(IDLE_SOUND_PATH.resolve())
        system_name = platform.system()

        if system_name == "Darwin":
            if shutil.which("afplay") is not None:
                return ["afplay", sound_path]
            return None

        if system_name == "Windows":
            if not IDLE_SOUND_HELPER_PATH.exists():
                return None
            return [sys.executable, str(IDLE_SOUND_HELPER_PATH.resolve()), sound_path]

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
        self.local_output_active = True
        self.detector.reset()
        try:
            if await self.sound_player.play(IDLE_SOUND_PATH):
                self._status("idle sound playback finished")
                return
            command = self._get_idle_sound_command()
            if command is None:
                self._status("idle sound skipped: no supported audio player is available", level="warning")
                return

            system_name = platform.system()
            if system_name == "Windows":
                creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
                subprocess.Popen(
                    command,
                    creationflags=creationflags,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._status("idle sound playback started in detached helper")
                return
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._status(f"idle sound playback started with {command[0]}")
        except Exception as exc:  # pragma: no cover - environment dependent
            self._status(f"idle sound playback failed: {exc}", level="warning")
        finally:
            self.local_output_active = False
            self._suppress_voice_commands(
                VOICE_COMMAND_ECHO_GRACE_SECONDS,
                "local idle sound finished",
            )

    def _background_task_done(self, name, task):
        if task.cancelled():
            self.logger.info("background task cancelled: %s", name)
            return
        exc = task.exception()
        if exc is None:
            self.logger.info("background task finished: %s", name)
            return
        self.logger.exception("background task failed: %s", name, exc_info=exc)
        self._status(f"background task failed: {name}: {exc}", level="error")

    def _start_background_task(self, background_tasks, coro, name):
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(lambda finished: self._background_task_done(name, finished))
        background_tasks.append(task)
        return task

    def _session_task_done(self, name, task):
        if task.cancelled():
            self.logger.info("session task cancelled: %s", name)
            return

        exc = task.exception()
        if exc is None:
            if self.session_stop_event is None or self.session_stop_event.is_set():
                self.logger.info("session task finished: %s", name)
                return
            error = RuntimeError(f"session task stopped unexpectedly: {name}")
            self.logger.error("%s", error)
            self._status(str(error), level="warning")
            if self._session_failure is None:
                self._session_failure = error
        elif self._is_normal_session_close(exc):
            self.logger.info("session task closed normally: %s", name)
        else:
            self.logger.exception("session task failed: %s", name, exc_info=exc)
            self._status(f"session task failed: {name}: {exc}", level="error")
            if self._session_failure is None:
                self._session_failure = exc

        if self.session_stop_event is not None and not self.session_stop_event.is_set():
            self.session_stop_event.set()

    def _start_session_task(self, session_tasks, coro, name):
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(lambda finished: self._session_task_done(name, finished))
        session_tasks.append(task)
        return task

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
        return {
            "kind": "video",
            "data": image_bytes,
            "mime_type": "image/jpeg",
        }

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
                    await self._enqueue_realtime_message(frame)
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
        return {
            "kind": "video",
            "data": image_bytes,
            "mime_type": "image/jpeg",
        }

    async def get_screen(self):
        while self.running and self.state == "active":
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            if self.out_queue is not None and not (
                self.strict_turns and self.assistant_speaking
            ):
                await self._enqueue_realtime_message(frame)

    async def send_realtime(self):
        try:
            while self.running and self.state == "active":
                if self.out_queue is None:
                    await asyncio.sleep(0.01)
                    continue
                msg = await self.out_queue.get()
                if self.state != "active":
                    continue
                if self.session is not None:
                    if self._debug_every("_last_realtime_send_log", 5.0):
                        queue_size = self.out_queue.qsize() if self.out_queue is not None else 0
                        self._status(
                            f"realtime send loop alive; kind={msg.get('kind')} out_queue={queue_size}"
                        )
                    if msg.get("kind") == "audio":
                        await asyncio.wait_for(
                            self.session.send_realtime_input(
                                audio=types.Blob(
                                    data=msg["data"],
                                    mime_type=msg["mime_type"],
                                )
                            ),
                            timeout=REALTIME_SEND_TIMEOUT_SECONDS,
                        )
                    elif msg.get("kind") == "activity_start":
                        await asyncio.wait_for(
                            self.session.send_realtime_input(
                                activity_start=types.ActivityStart()
                            ),
                            timeout=REALTIME_SEND_TIMEOUT_SECONDS,
                        )
                    elif msg.get("kind") == "activity_end":
                        await asyncio.wait_for(
                            self.session.send_realtime_input(
                                activity_end=types.ActivityEnd()
                            ),
                            timeout=REALTIME_SEND_TIMEOUT_SECONDS,
                        )
                    elif msg.get("kind") == "video":
                        await asyncio.wait_for(
                            self.session.send_realtime_input(
                                video=types.Blob(
                                    data=msg["data"],
                                    mime_type=msg["mime_type"],
                                )
                            ),
                            timeout=REALTIME_SEND_TIMEOUT_SECONDS,
                        )
                    else:
                        await asyncio.wait_for(
                            self.session.send_realtime_input(
                                media=types.Blob(
                                    data=msg["data"],
                                    mime_type=msg["mime_type"],
                                )
                            ),
                            timeout=REALTIME_SEND_TIMEOUT_SECONDS,
                        )
                    self._clear_out_queue_pressure()
        except TimeoutError as exc:
            raise RuntimeError(
                f"realtime audio send stalled for more than {REALTIME_SEND_TIMEOUT_SECONDS:.0f}s"
            ) from exc

    async def receive_audio(self):
        try:
            while self.running and self.state == "active":
                if self.session is None:
                    await asyncio.sleep(0.01)
                    continue

                if self._debug_every("_last_receive_wait_log", 5.0):
                    self._status("waiting for Gemini response")

                turn = self.session.receive()
                async for response in turn:
                    if self._debug_every("_last_receive_chunk_log", 2.0):
                        self._status("received Gemini response chunk")
                    for part in iter_response_parts(response):
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            self.assistant_speaking = True
                            self.assistant_audio_streaming = True
                            if self.audio_in_queue is not None:
                                self.audio_in_queue.put_nowait(part.inline_data.data)
                            continue
                        if isinstance(part.text, str) and not getattr(part, "thought", False):
                            print(part.text, end="")

                self._status("Gemini turn complete")

                self.assistant_audio_streaming = False
                if not self.strict_turns and self.audio_in_queue is not None:
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                    self.assistant_speaking = False
                elif self.audio_in_queue is None or self.audio_in_queue.empty():
                    self.assistant_speaking = False
        except Exception as exc:
            if self._is_normal_session_close(exc):
                self.logger.info("receive_audio closed normally: %s", exc)
                return
            raise

    async def play_audio(self):
        self._status("opening local speaker stream")
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
                if self._debug_every("_last_audio_play_log", 2.0):
                    pending = self.audio_in_queue.qsize() if self.audio_in_queue is not None else 0
                    self._status(f"playing model audio locally; pending_audio_chunks={pending}")
                await asyncio.to_thread(stream.write, bytestream)
                if (
                    self.audio_in_queue is not None
                    and self.audio_in_queue.empty()
                    and not self.assistant_audio_streaming
                ):
                    self.assistant_speaking = False
        finally:
            self._status("closing local speaker stream")
            await asyncio.to_thread(stream.close)

    async def listen_microphone(self):
        if self.mic_index is None:
            mic_info = pya.get_default_input_device_info()
            mic_index = mic_info["index"]
        else:
            mic_index = self.mic_index

        self._status(f"opening microphone stream; mic_index={mic_index}")

        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=CHUNK_SIZE,
        )
        self._status("microphone stream opened")

        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while self.running:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            rms = self._pcm16_rms(data)
            if self._debug_every("_last_mic_health_log", 5.0):
                detector_state = "speaking" if self.assistant_speaking else "listening"
                self._status(
                    f"microphone loop alive; state={self.state}; detector={detector_state}; rms={rms:.0f}"
                )

            if self._voice_commands_suppressed():
                self.detector.reset()
                command = None
            else:
                command = self.detector.feed(data, state=self.state)
            if command == "wake":
                if self.state == "idle":
                    session_mode = "prompt" if self.no_auto_start_wake_word and not self.auto_start else "default"
                    self._status(
                        f"wake word detected from voice; session_mode={session_mode}"
                    )
                    await self.control_queue.put(("wake", session_mode, "voice"))
            elif command == "mimic_wake":
                if self.state == "idle":
                    self._status("mimic wake word detected from voice; session_mode=mimic")
                    await self.control_queue.put(("wake", "mimic", "voice"))
            elif command == "translate_wake":
                if self.state == "idle":
                    self._status("translate wake word detected from voice; session_mode=translate")
                    await self.control_queue.put(("wake", "translate", "voice"))
            elif command == "no_auto_start_wake":
                if self.state == "idle" and not self.auto_start:
                    self._status("no-auto-start wake word detected from voice; session_mode=silent")
                    await self.control_queue.put(("wake", "silent", "voice"))
            elif command == "exit":
                self._status("exit word detected from voice")
                await self.control_queue.put(("sleep", None, "voice"))

            can_stream_audio = (
                self.out_queue is not None
                and self.state == "active"
                and not (self.strict_turns and self.assistant_speaking)
            )
            if can_stream_audio:
                now = self._loop_time()
                if not self.user_activity_active and rms >= self.vad_start_rms:
                    self.user_activity_active = True
                    self._user_activity_started_at = now
                    self._last_user_voice_at = now
                    self._status(f"user speech started; rms={rms:.0f}")
                    await self._enqueue_realtime_message({"kind": "activity_start"})
                elif self.user_activity_active and rms >= self.vad_end_rms:
                    self._last_user_voice_at = now

                if (
                    self.user_activity_active
                    and rms < self.vad_end_rms
                    and now - self._last_user_voice_at >= self.vad_silence_seconds
                ):
                    duration = now - self._user_activity_started_at
                    self.user_activity_active = False
                    self._user_activity_started_at = 0.0
                    self._status(
                        f"user speech ended; duration={duration:.1f}s; sending activity_end"
                    )
                    await self._enqueue_realtime_message({"kind": "activity_end"})

                await self._enqueue_realtime_message(
                    {
                        "kind": "audio",
                        "data": data,
                        "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                    }
                )
                continue

            if self.user_activity_active:
                self.user_activity_active = False
                self._user_activity_started_at = 0.0

            if (
                self.state == "active"
                and self.strict_turns
                and self.assistant_speaking
                and rms >= self.vad_start_rms
                and self._debug_every("_last_suppressed_speech_log", 2.0)
            ):
                self._status(
                    "user speech detected while Gemini audio is playing; "
                    "not sending because --strict-turns is enabled",
                    level="warning",
                )

    async def monitor_session_health(self):
        while self.running and self.state == "active":
            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            if self.session_stop_event is None or self.out_queue is None:
                continue
            if self._out_queue_pressure_since is None:
                continue

            stall_for = self._loop_time() - self._out_queue_pressure_since
            if stall_for < OUT_QUEUE_STALL_TIMEOUT_SECONDS:
                continue

            self._status(
                f"realtime send queue stalled for {stall_for:.1f}s; returning to idle to recover",
                level="warning",
            )
            self.session_stop_event.set()
            return

    async def send_text(self):
        while self.running:
            print("message > ", end="", flush=True)
            line = await asyncio.to_thread(self._read_stdin_line_blocking)
            if not line:
                self._status("stdin closed; stopping text input")
                return
            if isinstance(line, bytes):
                text = line.decode(errors="replace").rstrip("\r\n")
            else:
                text = str(line).rstrip("\r\n")
            normalized = normalize_phrase(text)
            if normalized == "q":
                self._status("quit requested from text input")
                await self.control_queue.put(("quit", None, "text"))
                return
            if self.state == "idle" and normalized == self.detector.wake_word:
                session_mode = "prompt" if self.no_auto_start_wake_word and not self.auto_start else "default"
                self._status(
                    f"wake word detected from text; session_mode={session_mode}"
                )
                await self.control_queue.put(("wake", session_mode, "text"))
                continue
            if self.state == "idle" and normalized == self.detector.mimic_wake_word:
                self._status("mimic wake word detected from text; session_mode=mimic")
                await self.control_queue.put(("wake", "mimic", "text"))
                continue
            if self.state == "idle" and normalized == self.detector.translate_wake_word:
                self._status("translate wake word detected from text; session_mode=translate")
                await self.control_queue.put(("wake", "translate", "text"))
                continue
            if (
                self.state == "idle"
                and not self.auto_start
                and normalized == self.detector.no_auto_start_wake_word
            ):
                self._status("no-auto-start wake word detected from text; session_mode=silent")
                await self.control_queue.put(("wake", "silent", "text"))
                continue
            if self.state in {"connecting", "active"} and normalized == self.detector.exit_word:
                self._status("exit word detected from text")
                await self.control_queue.put(("sleep", None, "text"))
                continue
            if self.session is not None and self.state == "active":
                while self.strict_turns and self.assistant_speaking:
                    await asyncio.sleep(0.05)
                self._status("sending text turn to Gemini")
                await self._send_realtime_text_turn(text)
                self._status("text turn sent")

    def _is_recoverable_gemini_error(self, exc):
        message = normalize_phrase(str(exc))
        return any(pattern in message for pattern in RECOVERABLE_ERROR_PATTERNS)

    def _is_quota_or_billing_error(self, exc):
        message = normalize_phrase(str(exc))
        return any(pattern in message for pattern in QUOTA_OR_BILLING_ERROR_PATTERNS)

    def _is_unsupported_live_operation(self, exc):
        message = normalize_phrase(str(exc))
        return any(pattern in message for pattern in UNSUPPORTED_LIVE_OPERATION_PATTERNS)

    def _is_normal_session_close(self, exc):
        if isinstance(exc, asyncio.CancelledError):
            return True
        if isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 1000:
            return True
        message = normalize_phrase(str(exc))
        return message in {"1000 none", "1000 ok"} or "connection closed ok" in message

    async def _send_realtime_text_turn(self, text):
        await self.session.send_realtime_input(activity_start=types.ActivityStart())
        await self.session.send_realtime_input(text=text or ".")
        await self.session.send_realtime_input(activity_end=types.ActivityEnd())

    async def _report_session_error(self, exc):
        message = str(exc).strip() or exc.__class__.__name__
        if self._is_quota_or_billing_error(exc):
            await self._announce(
                "Gemini quota or billing limit was reached. Returning to idle. "
                f"You can say {self.detector.wake_word} to retry later.",
                level="error",
            )
            self.logger.error("Gemini quota or billing error: %s", message)
            return
        if self._is_recoverable_gemini_error(exc):
            await self._announce(
                f"Gemini server error. Returning to idle. You can say {self.detector.wake_word} to reconnect.",
                level="error",
            )
            self.logger.error("recoverable Gemini error: %s", message)
            return
        if self._is_unsupported_live_operation(exc):
            await self._announce(
                "Gemini Live API rejected this model or session configuration. "
                f"Returning to idle. Details: {message}",
                level="error",
            )
            self.logger.error("unsupported Gemini Live operation: %s", message)
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
        self.assistant_audio_streaming = False
        self.user_activity_active = False
        self._user_activity_started_at = 0.0
        self.audio_in_queue = None
        self.out_queue = None
        self._clear_out_queue_pressure()
        self.session = None
        self.session_stop_event = None
        self._session_failure = None
        self.session_task = None
        self._set_state("idle")

    async def _run_session(self, session_mode):
        self._set_state("connecting")
        opening_prompt = self._build_opening_prompt(session_mode)
        self._status(
            "opening Gemini live session; "
            f"model={self.model}; "
            f"session_mode={session_mode}; "
            f"send_opening_prompt={opening_prompt is not None}; "
            f"search_enabled={self.enable_search}"
        )
        self.session_stop_event = asyncio.Event()
        self._session_failure = None
        session_tasks = []
        try:
            async with get_client().aio.live.connect(
                model=self.model,
                config=self.live_config,
            ) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self._clear_out_queue_pressure()
                self._set_state("active")

                self._start_session_task(session_tasks, self.send_realtime(), "send_realtime")
                self._start_session_task(session_tasks, self.receive_audio(), "receive_audio")
                self._start_session_task(session_tasks, self.play_audio(), "play_audio")
                self._start_session_task(
                    session_tasks, self.monitor_session_health(), "monitor_session_health"
                )

                if self.video_mode == "camera":
                    self._start_session_task(session_tasks, self.get_frames(), "get_frames")
                elif self.video_mode == "screen":
                    self._start_session_task(session_tasks, self.get_screen(), "get_screen")

                if opening_prompt is not None:
                    self._status(
                        f"sending opening prompt: {self.daily_prompt['title']} ({session_mode})"
                    )
                    await self._send_realtime_text_turn(opening_prompt)
                    self._status("opening prompt sent")
                else:
                    self._status("session started without opening prompt")

                await self.session_stop_event.wait()
                if self.state == "active":
                    self._set_state("closing")
                if self._session_failure is not None:
                    raise self._session_failure
        except asyncio.CancelledError:
            self.logger.info("session cancelled while state=%s", self.state)
            self._status("session cancelled")
            raise
        except Exception as exc:
            self.logger.exception("session failure")
            self._status(f"session failure: {exc}", level="error")
            if self.state in {"connecting", "active"}:
                self._set_state("closing")
            await self._report_session_error(exc)
        finally:
            await self._cleanup_session(session_tasks)

    async def start_session(self, reason, session_mode):
        if self.state != "idle":
            self.logger.info("start_session ignored in state=%s reason=%s", self.state, reason)
            return

        self._reload_daily_prompt()
        self.logger.info(
            "start_session requested: reason=%s session_mode=%s",
            reason,
            session_mode,
        )
        self._status(
            f"starting session; reason={reason}; session_mode={session_mode}"
        )
        self.session_task = asyncio.create_task(self._run_session(session_mode))

    async def run(self):
        background_tasks = []
        try:
            self.logger.info("application start")
            self._status("application start")
            self._show_daily_prompt()
            if self.no_auto_start_wake_word and not self.auto_start:
                startup_message = (
                    f"Idle. Say {self.detector.wake_word} to start with today's prompt,"
                    f" say {self.detector.mimic_wake_word} for mimic mode,"
                    f" say {self.detector.translate_wake_word} for translate mode,"
                    f" or say {self.detector.no_auto_start_wake_word} to start without it."
                    f" Say {self.detector.exit_word} to return to idle."
                )
            else:
                startup_message = (
                    f"Idle. Say {self.detector.wake_word} to start,"
                    f" say {self.detector.mimic_wake_word} for mimic mode,"
                    f" say {self.detector.translate_wake_word} for translate mode,"
                    f" and say {self.detector.exit_word} to return to idle."
                )
            await self._announce(startup_message)

            self._start_background_task(background_tasks, self.listen_microphone(), "listen_microphone")
            if self.enable_text_input:
                self._start_background_task(background_tasks, self.send_text(), "send_text")

            if not self.wake_word_enabled:
                self._status("wake word disabled at startup; starting session immediately")
                await self.control_queue.put(("wake", "default", "startup"))

            while self.running:
                command, start_mode, reason = await self.control_queue.get()
                self._status(
                    f"received control command={command} session_mode={start_mode} reason={reason}"
                )
                if command == "wake":
                    if start_mode == "default":
                        session_mode = "prompt" if self.auto_start else "silent"
                    else:
                        session_mode = start_mode
                    await self.start_session(reason, session_mode=session_mode)
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
            self._close_stdin_reader()
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
        "--mimic-wake-word",
        type=str,
        default=DEFAULT_MIMIC_WAKE_WORD,
        help="Phrase that starts mimic mode with a model answer and imitation feedback.",
    )
    parser.add_argument(
        "--translate-wake-word",
        type=str,
        default=DEFAULT_TRANSLATE_WAKE_WORD,
        help="Phrase that starts Japanese-to-English translation practice based on today's prompt.",
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
        "--wake-min-confidence",
        type=float,
        default=WAKE_COMMAND_MIN_CONFIDENCE,
        help=(
            "Minimum Vosk word confidence required for wake phrases. "
            "Raise it to reduce false starts; lower it if wake words are missed."
        ),
    )
    parser.add_argument(
        "--stt-model-path",
        type=str,
        default=os.environ.get("VOSK_MODEL_PATH", DEFAULT_STT_MODEL_PATH),
        help="Path to the local Vosk speech recognition model directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Gemini Live API model name. Defaults to GEMINI_LIVE_MODEL or the current Live preview.",
    )
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument(
        "--search",
        action="store_true",
        help="Enable Google Search grounding for Live sessions.",
    )
    search_group.add_argument(
        "--no-search",
        action="store_true",
        help="Explicitly disable Google Search grounding for Live sessions.",
    )
    parser.add_argument(
        "--vad-start-rms",
        type=float,
        default=USER_SPEECH_START_RMS,
        help="RMS level that starts a user speech activity.",
    )
    parser.add_argument(
        "--vad-end-rms",
        type=float,
        default=USER_SPEECH_END_RMS,
        help="RMS level below which silence can end a user speech activity.",
    )
    parser.add_argument(
        "--vad-silence-seconds",
        type=float,
        default=USER_SPEECH_SILENCE_SECONDS,
        help="Seconds of low RMS before ending a user speech activity.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.wake_min_confidence <= 1.0:
        parser.error("--wake-min-confidence must be between 0.0 and 1.0.")

    if args.list_mics:
        list_input_devices()
        raise SystemExit(0)

    enable_search = None
    if args.search:
        enable_search = True
    elif args.no_search:
        enable_search = False

    main = AudioLoop(
        video_mode=args.mode,
        auto_start=(not args.no_auto_start),
        enable_text_input=(not args.no_text),
        mic_index=args.mic_index,
        strict_turns=args.strict_turns,
        wake_word=args.wake_word,
        mimic_wake_word=args.mimic_wake_word,
        translate_wake_word=args.translate_wake_word,
        exit_word=args.exit_word,
        no_auto_start_wake_word=args.no_auto_start_wake_word,
        wake_word_enabled=(not args.no_wake_word),
        stt_model_path=args.stt_model_path,
        model=args.model,
        enable_search=enable_search,
        vad_start_rms=args.vad_start_rms,
        vad_end_rms=args.vad_end_rms,
        vad_silence_seconds=args.vad_silence_seconds,
        wake_min_confidence=args.wake_min_confidence,
    )
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        pass
