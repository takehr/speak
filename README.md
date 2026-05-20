### install

for raspberry pi users:
```text
[Unit]
Description=My Python Script (uv venv)
After=network.target

[Service]
User=t-hara
WorkingDirectory=/home/t-hara/speak
ExecStart=/bin/bash -lc '/home/@@@@_user_name_@@@@/.local/bin/uv run python speak.py --wake-word "gemini" --exit-word "see you" --stt-model-path "/home/@@@@_user_name_@@@@/speak/vosk-model-small-en-us-0.15" --mic-index @@@@_yourmic_index_@@@@ --strict-turns --mode none'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

```

```bash
sudo apt update
sudo apt install portaudio19-dev
sudo systemctl daemon-reload
sudo systemctl enable myscript.service
sudo systemctl start myscript.service
```

common setup:

```bash
uv sync
export $GEMINI_API_KEY="your_gemini_api_key"
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip ./vosk-model-small-en-us-0.15.zip
uv pip install vosk
```

### examples
```bash
uv run python speak.py
uv run python speak.py --list-mics
uv run python speak.py --wake-word "gemini" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
uv run python speak.py --wake-word "gemini" --mimic-wake-word "copy mode" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
uv run python speak.py --no-auto-start --no-auto-start-wake-word "start talking" --wake-word "gemini" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
```

By default the app uses `models/gemini-2.5-flash-native-audio-preview-12-2025`,
because this project uses Google Search grounding in no-auto-start sessions.
To try another Live API model without editing the script, pass `--model ...` or
set `GEMINI_LIVE_MODEL`.

### daily prompts
Auto-start scenarios are loaded from `./prompts/*.md`.
The app picks one markdown file per day based on the current date, so the same day always uses the same scenario.
At startup, the selected scenario is shown in the console as `[prompt] ...`.
The prompt list is also reloaded whenever a session starts, so edits under `./prompts` are picked up without restarting the app.

Current starter set:
- conference discussion / Q&A
- conference small talk
- administrative tasks
- daily life / settling in

To add a new scenario, drop another markdown file into `./prompts`.
The full markdown content is sent as the opening prompt when auto-start is enabled.

### sample for `--no-auto-start`
When you start with `--no-auto-start`, Gemini connects without sending the first turn.
In this mode, the Live session also enables the Google Search tool. Pass
`--no-search` only when you want a plain voice session.
The app uses local RMS-based voice activity detection and sends explicit
activity start/end signals to Gemini. If speech is not detected, lower
`--vad-start-rms` (for example `--vad-start-rms 30`).
If you also set `--no-auto-start-wake-word`, you get two idle startup paths:
- `--wake-word`: start with today's prompt
- `--mimic-wake-word`: start mimic mode with a model answer first
- `--translate-wake-word`: start Japanese-to-English translation practice
- `--no-auto-start-wake-word`: start without today's prompt

### mimic mode
`--mimic-wake-word` starts a shadowing-style session based on today's daily prompt.
In this mode, Gemini should:
- give a polished model answer first
- wait for the user to imitate it
- check how closely the user copied the wording and meaning
- give feedback on pronunciation, rhythm, stress, and missing or changed words
- ask for a retry

Default mimic wake word: `copy mode`

### translate mode
`--translate-wake-word` starts an instant sentence translation session based on today's daily prompt.
In this mode, Gemini should:
- prepare a polished model answer internally
- say the Japanese translation of that answer first
- ask the user to express it in English
- give feedback in Japanese about meaning and naturalness
- keep the user on the same sentence until it becomes natural enough, then move on
- continue for at least 10 completed sentences before wrapping up
- finish with a quick review test where Gemini says a practiced Japanese prompt and the user translates it into English

Default translate wake word: `translate`

````md
# hands-free no-auto-start sample

- idle wake word: `gemini`
- mimic wake word: `copy mode`
- translate wake word: `translate`
- no-auto-start idle wake word: `start talking`
- exit word: `see you`

example:

```bash
uv run python speak.py \
  --no-auto-start \
  --wake-word "gemini" \
  --mimic-wake-word "copy mode" \
  --translate-wake-word "translate" \
  --no-auto-start-wake-word "start talking" \
  --exit-word "see you" \
  --stt-model-path "./vosk-model-small-en-us-0.15" \
  --mic-index 4 \
  --strict-turns \
  --mode none
```
````
