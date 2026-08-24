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
uv run python speak.py --wake-min-confidence 0.85 --wake-min-active-seconds 0.30 --wake-word "hello" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
uv run python speak.py --wake-word "gemini" --mimic-wake-word "copy mode" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
uv run python speak.py --no-auto-start --no-auto-start-wake-word "start talking" --wake-word "gemini" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
```

### hourly afternoon conversations

While the app is running, it automatically starts the normal daily-prompt
conversation every hour at `13:00`, `14:00`, `15:00`, `16:00`, `17:00`,
`18:00`, `19:00`, and `20:00` in the computer's local time. After Gemini
finishes its first message, the app waits up to 60 seconds for sustained
microphone activity. If nobody responds, it returns to idle wake-word mode
automatically.

Once a normal-mode session has a real user response and remains active for at
least 60 seconds, practice is complete for that date and all remaining
automatic starts are skipped. The completion date is saved in
`state/daily_practice_date.txt`, so an application restart does not reset it.

```bash
uv run python speak.py \
  --scheduled-response-timeout 60 \
  --daily-practice-min-seconds 60 \
  --mode none
```

Use `--no-daily-start` to disable the schedule. `--daily-start-time HH:MM` adds
an extra time to the default schedule. A legacy `--daily-start-time 20:00` in
an existing startup command is harmless because 20:00 is already included. If
another conversation is active at a scheduled time, that start is skipped. The
app must stay running (for example, as the systemd service above) for the
schedule to fire.

### Gemini Live connection timeouts

Gemini Live WebSocket connections wait up to 30 seconds and retry up to three
times by default. If regular HTTPS works but Windows repeatedly reports
`timed out during opening handshake`, try bypassing a stale system proxy:

```bat
uv run python speak.py --no-websocket-proxy --mode none
```

If `Test-NetConnection` reports failed connections to addresses beginning with
`2001:`, the machine has a broken IPv6 route. Force the Live WebSocket to IPv4:

```bat
uv run python speak.py --no-websocket-proxy --live-ipv4 --mode none
```

The timeout and retry behavior can also be adjusted with
`--live-open-timeout`, `--live-connect-attempts`, and
`--live-connect-retry-seconds`.

`--wake-min-confidence` controls how confident Vosk must be before an idle wake
phrase is accepted. The default is `0.80`; try `0.85` or higher if the app
starts without you saying the wake word. Wake phrases also require `0.25`
seconds of continuous audio above `--wake-min-rms 45` by default, which rejects
short impact sounds such as keyboard clicks. If clicks still trigger it, try
`--wake-min-active-seconds 0.30`. If softly spoken wake phrases are missed,
lower `--wake-min-rms` or `--wake-min-active-seconds`.

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
- say the numbered Japanese translation of that answer first
- ask the user to express it in English
- give feedback in Japanese about meaning and naturalness
- keep the user on the same sentence until it becomes natural enough, then move on
- continue for at least 10 completed numbered practice items before wrapping up
- finish with a numbered quick review test where Gemini says each practiced Japanese prompt and the user translates it into English

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
