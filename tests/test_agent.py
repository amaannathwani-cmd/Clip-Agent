import subprocess
from datetime import datetime, timedelta, timezone

from clip_agent.editor import build_ass, chunk_words, render
from clip_agent.media import Word
from clip_agent.scorer import final_score, heuristic_score, parse_judgement
from clip_agent.twitch import Clip

WORDS = [Word(0.2, 0.5, "no"), Word(0.5, 0.9, "way"), Word(0.9, 1.4, "he"), Word(2.5, 3.0, "did{that}")]


def make_clip(views=5000, hours=2, viewers=10000, duration=30):
    return Clip("abc", "https://clips.twitch.tv/abc", "t", "1", "streamer", "Streamer", "Game", views, duration,
                datetime.now(timezone.utc) - timedelta(hours=hours), channel_viewers=viewers)


def test_chunks_split_on_gap_and_size():
    chunks = chunk_words(WORDS)
    assert [len(c) for c in chunks] == [3, 1]


def test_ass_highlights_each_word_and_escapes():
    ass = build_ass(WORDS, credit="twitch.tv/streamer", duration=5)
    assert ass.count("Style: Cap") == 1
    assert ass.count(",Cap,") == 4
    assert "twitch.tv/streamer" in ass
    assert "DID(THAT)" in ass


def test_heuristic_prefers_fast_clips():
    assert heuristic_score(make_clip(views=20000, hours=1)) > heuristic_score(make_clip(views=500, hours=20))
    assert 0 <= heuristic_score(make_clip()) <= 100


def test_parse_judgement_and_blend():
    j = parse_judgement('```json\n{"score": 80, "title": "He did WHAT", "hashtags": ["#gaming", "funny clip"], "unsafe": false}\n```', "x")
    assert j["score"] == 80 and j["hashtags"] == ["gaming", "funnyclip"]
    assert final_score(50, 80) == 72.5
    assert final_score(50, None) == 50


def test_render_makes_vertical_video(tmp_path):
    src = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "4", "-shortest", str(src)], check=True)
    out = render(src, WORDS, tmp_path / "out.mp4", "twitch.tv/streamer", 4)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert probe.stdout.strip() == "1080,1920"


def test_load_audio_decodes_with_ffmpeg(tmp_path):
    from clip_agent.media import load_audio
    src = tmp_path / "a.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2",
                    str(src)], check=True)
    audio = load_audio(src)
    assert 31000 < len(audio) < 33000 and audio.dtype.name == "float32" and abs(audio).max() <= 1.0


def test_pick_flash_model_prefers_newest_stable():
    from clip_agent.scorer import pick_flash_model
    names = ["models/gemini-2.5-flash", "models/gemini-3.0-flash", "models/gemini-3.5-flash-preview-09",
             "models/gemini-3.0-flash-lite", "models/gemini-3.0-pro", "models/gemini-3.0-flash-image"]
    assert pick_flash_model(names) == "gemini-3.0-flash"
    assert pick_flash_model(["models/gemini-4-flash-preview"]) == "gemini-4-flash-preview"
    assert pick_flash_model(["models/gemini-3.0-pro"]) is None


def test_parse_judgement_trim_rules():
    j = parse_judgement('{"score": 75, "start": 4.0, "end": 19.0}', "x", 30)
    assert j["start"] == 3.7 and j["end"] == 19.5
    j = parse_judgement('{"score": 75, "start": 4.0, "end": 6.0}', "x", 30)   # too short -> whole clip
    assert j["start"] == 0.0 and j["end"] is None
    j = parse_judgement('{"score": 75, "start": 0, "end": 99}', "x", 30)     # past the end -> whole clip
    assert j["end"] is None


def test_shift_words_and_timed_transcript():
    from clip_agent.editor import shift_words
    from clip_agent.media import timed_transcript
    shifted = shift_words(WORDS, 0.5, 2.0)
    assert [w.text for w in shifted] == ["way", "he"] and shifted[0].start == 0.0
    assert timed_transcript(WORDS).startswith("[0.2] no way")


def test_choose_layout():
    from clip_agent.layout import choose_layout
    # small steady face bottom-left -> facecam split
    cam = [(120, 760, 90, 90)] * 10
    lay = choose_layout(1920, 1080, cam)
    assert lay.kind == "split"
    c = lay.cam
    assert c.x <= 120 and c.x + c.w >= 210 and c.y + c.h <= 1080 and c.x >= 0
    # big face -> follow-cam crop
    lay = choose_layout(1920, 1080, [(800, 300, 320, 320)] * 10)
    assert lay.kind == "face" and lay.crop.w == 606 and 600 < lay.crop.x + lay.crop.w / 2 < 1320
    # no / jumpy faces -> blur
    assert choose_layout(1920, 1080, []).kind == "blur"
    jumpy = [(100 + i * 170, 100 + (i % 3) * 300, 80, 80) for i in range(10)]
    assert choose_layout(1920, 1080, jumpy).kind == "blur"


import pytest


@pytest.mark.parametrize("faces", [[(120, 760, 90, 90)] * 10, [(800, 300, 320, 320)] * 10, []])
def test_render_each_layout(tmp_path, faces):
    from clip_agent.layout import choose_layout
    src = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "6", "-shortest", str(src)], check=True)
    out = render(src, WORDS, tmp_path / "out.mp4", "twitch.tv/x", 60, start=1.0, end=4.0,
                 layout=choose_layout(1920, 1080, faces))
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height:format=duration",
                            "-of", "default=nw=1", str(out)], capture_output=True, text=True).stdout
    assert "width=1080" in probe and "height=1920" in probe
    dur = float([l for l in probe.splitlines() if l.startswith("duration=")][0].split("=")[1])
    assert 2.8 < dur < 3.3
