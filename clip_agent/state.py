import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

STATE_FILE = ROOT / "state" / "posted.json"


def load() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(state: dict, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(state: dict, clip, results: dict) -> None:
    state[clip.id] = {
        "posted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "streamer": clip.broadcaster_login,
        "title": clip.title,
        "url": clip.url,
        "posts": results,
    }
