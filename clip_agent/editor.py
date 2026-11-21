"""Turn a landscape clip into a 1080x1920 short with word-by-word captions."""
import subprocess
from pathlib import Path

from .layout import Layout, plan, video_filter

W, H = 1080, 1920
CAPTION_Y = 1390  # default caption height (blur layout)
HIGHLIGHT = "&H0000E5FF&"  # yellow (ASS colours are BGR)


def _ts(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def chunk_words(words, max_words: int = 3, max_gap: float = 0.6):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w.start - cur[-1].end > max_gap):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words, credit: str | None = None, duration: float = 60.0, caption_y: int = CAPTION_Y) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {W}",
        f"PlayResY: {H}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        "Style: Cap,DejaVu Sans,92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1",
        "Style: Credit,DejaVu Sans,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,8,40,40,170,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if credit:
        lines.append(f"Dialogue: 0,{_ts(0)},{_ts(duration)},Credit,,0,0,0,,{_esc(credit)}")
    for chunk in chunk_words(words):
        for i, w in enumerate(chunk):
            start = w.start
            end = chunk[i + 1].start if i + 1 < len(chunk) else max(w.end, w.start + 0.25)
            parts = []
            for j, cw in enumerate(chunk):
                t = _esc(cw.text.upper())
                parts.append(f"{{\\c{HIGHLIGHT}\\fscx110\\fscy110}}{t}{{\\r}}" if j == i else t)
            lines.append(
                f"Dialogue: 1,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{{\\pos({W // 2},{caption_y})}}" + " ".join(parts)
            )
    return "\n".join(lines) + "\n"


def shift_words(words, start: float, end: float):
    """Keep words inside [start, end) and make their times relative to start."""
    from .media import Word
    return [Word(w.start - start, min(w.end, end) - start, w.text) for w in words if w.start >= start and w.start < end]


def render(src: Path, words, out: Path, credit: str | None, max_seconds: float = 60.0,
           start: float = 0.0, end: float | None = None, layout: Layout | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir = src.parent
    end = min(end if end is not None else start + max_seconds, start + max_seconds)
    length = end - start
    layout = layout or plan(src)
    print(f"  layout: {layout.kind}")
    (workdir / "subs.ass").write_text(
        build_ass(shift_words(words, start, end), credit, length, layout.caption_y), encoding="utf-8")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", str(src.resolve()),
        "-filter_complex", video_filter(layout),
        "-map", "[v]", "-map", "0:a?",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-t", f"{length:.2f}",
        "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=workdir)
    return out
