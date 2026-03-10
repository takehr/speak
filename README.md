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
```
