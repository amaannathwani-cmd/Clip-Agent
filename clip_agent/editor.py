"""Turn a landscape clip into a 1080x1920 short with word-by-word captions."""
import subprocess
from pathlib import Path

W, H = 1080, 1920
CAPTION_Y = 1390  # captions sit under the video
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


def build_ass(words, credit: str | None = None, duration: float = 60.0) -> str:
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
                f"Dialogue: 1,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{{\\pos({W // 2},{CAPTION_Y})}}" + " ".join(parts)
            )
    return "\n".join(lines) + "\n"


FILTER = (
    "[0:v]split=2[a][b];"
    f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:3,eq=brightness=-0.12[bg];"
    # crop the centre of the frame to 4:3 so the action is bigger, then fit to full width
    f"[b]crop='min(iw,ih*4/3)':ih,scale={W}:-2[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2-160,ass=subs.ass[v]"
)


def render(src: Path, words, out: Path, credit: str | None, max_seconds: float = 60.0) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir = src.parent
    (workdir / "subs.ass").write_text(build_ass(words, credit, max_seconds), encoding="utf-8")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src.resolve()),
        "-filter_complex", FILTER,
        "-map", "[v]", "-map", "0:a?",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-t", str(max_seconds),
        "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=workdir)
    return out
