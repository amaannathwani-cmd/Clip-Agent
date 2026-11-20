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
    assert final_score(50, 80) == 68.0
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
