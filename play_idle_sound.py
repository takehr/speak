import sys
import time
from pathlib import Path

try:
    import pygame
except ImportError as exc:
    raise SystemExit(f"pygame is required for idle sound playback: {exc}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: play_idle_sound.py <path>")

    sound_path = Path(sys.argv[1]).resolve()
    if not sound_path.exists():
        raise SystemExit(f"sound file not found: {sound_path}")

    pygame.mixer.init()
    try:
        pygame.mixer.music.load(str(sound_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
    finally:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
