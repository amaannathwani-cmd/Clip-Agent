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
    try:
        audio = load_audio(path)
    except subprocess.CalledProcessError:
        return []  # no audio track
    if audio.size < 16000 // 2:
        return []
    segments, timed = None, True
    for vad, word_ts in ((True, True), (False, True), (False, False)):
        # Word alignment occasionally crashes on odd audio; fall back to segment timings.
        try:
            segments = list(_model_cache[model_name].transcribe(audio, word_timestamps=word_ts, vad_filter=vad)[0])
            timed = word_ts
            break
        except Exception as e:  # a transcription glitch shouldn't throw away a good clip
            print(f"  transcription failed (vad={vad}, words={word_ts}): {e}")
    if segments is None:
        return []
    if not timed:
        return spread_words(segments)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(float(w.start), float(w.end), text))
    return words


def spread_words(segments) -> list[Word]:
    """Give each word of a segment an equal share of the segment's time."""
    words = []
    for seg in segments:
        parts = seg.text.split()
        if not parts:
            continue
        step = (seg.end - seg.start) / len(parts)
        words += [Word(seg.start + i * step, seg.start + (i + 1) * step, p) for i, p in enumerate(parts)]
    return words


def proxy_bytes(path: Path, max_bytes: int = 14 * 1024 * 1024) -> bytes | None:
    """A small 360p copy for the AI to watch (inline uploads must stay under ~20MB)."""
    out = path.with_name("proxy.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", "scale=-2:240,fps=10",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "34", "-c:a", "aac", "-b:a", "48k", "-ac", "1",
         str(out)],
        check=True,
    )
    data = out.read_bytes()
    return data if len(data) <= max_bytes else None


def timed_transcript(words: list[Word]) -> str:
    """Transcript with a [seconds] marker every few words so the AI can pick start/end points."""
    parts = []
    for i, w in enumerate(words):
        if i % 6 == 0:
            parts.append(f"[{w.start:.1f}]")
        parts.append(w.text)
    return " ".join(parts)
