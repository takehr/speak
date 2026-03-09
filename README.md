### install

```bash
uv sync
export $GEMINI_API_KEY="your_gemini_api_key"
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
uv pip install vosk
```

### examples
```bash
uv run python speak.py
uv run python speak.py --list-mics
uv run python speak.py --wake-word "gemini" --exit-word "see you" --stt-model-path "./vosk-model-small-en-us-0.15" --mic-index 4 --strict-turns --mode none
```
