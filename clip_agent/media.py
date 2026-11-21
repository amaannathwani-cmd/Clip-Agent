"""Download a clip and transcribe it with word timestamps."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import yt_dlp


@dataclass
class Word:
    start: float
    end: float
    text: str


def download(url: str, dest_dir: Path, name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(dest_dir / f"{name}.%(ext)s"),
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


_model_cache = {}


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Decode audio with ffmpeg (avoids PyAV version breakage in faster-whisper)."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def transcribe(path: Path, model_name: str = "small") -> list[Word]:
    from faster_whisper import WhisperModel

    if model_name not in _model_cache:
        _model_cache[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = _model_cache[model_name].transcribe(load_audio(path), word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(float(w.start), float(w.end), text))
    return words
