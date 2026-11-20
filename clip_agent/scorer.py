"""Score clips: a fast heuristic on view velocity, then an AI judge on the transcript."""
import json
import math
import re

import requests


def heuristic_score(clip) -> float:
    """0-100. Rewards clips gaining views fast, and clips popular relative to the stream's audience."""
    velocity = clip.view_count / clip.age_hours
    v = min(math.log10(velocity + 1) / 4, 1.0)  # 10k views/hour maxes it out
    rel = min(clip.view_count / max(clip.channel_viewers, 1), 1.0) ** 0.5 if clip.channel_viewers else 0.3
    length = 1.0 if 12 <= clip.duration <= 45 else 0.7
    return round(100 * (0.6 * v + 0.4 * rel) * length, 1)


PROMPT = """You pick Twitch clips to repost as YouTube Shorts / TikToks.
Judge whether this clip would go viral with people who have never heard of the streamer
(funny, shocking, skillful, rage, wholesome, a clear moment within the first 3 seconds).

Streamer: {streamer}
Game/category: {game}
Original clip title: {title}
Views: {views} in {age:.1f} hours
Transcript: {transcript}

Reply with JSON only:
{{"score": 0-100, "reason": "short", "title": "catchy short title under 70 chars, no hashtags, no clickbait lies",
"hashtags": ["5 relevant hashtags without #"], "unsafe": true/false}}
Set unsafe true if it contains slurs, sexual content, real violence, doxxing, or the moment needs lots of context."""


GEMINI = "https://generativelanguage.googleapis.com/v1beta"
_resolved_model: dict[str, str] = {}


def pick_flash_model(names: list[str]) -> str | None:
    """Choose the newest stable general 'flash' model from a ListModels result."""
    skip = ("lite", "image", "tts", "audio", "live", "embedding", "thinking", "exp")
    good = [n.removeprefix("models/") for n in names if "flash" in n and not any(k in n for k in skip)]
    if not good:
        return None
    stable = [n for n in good if "preview" not in n] or good

    def version(n: str):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (float(m.group(1)) if m else 0.0, "latest" in n, -len(n))

    return max(stable, key=version)


def list_models(api_key: str) -> list[str]:
    r = requests.get(f"{GEMINI}/models", params={"key": api_key, "pageSize": 1000}, timeout=30)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def _generate(api_key: str, model: str, prompt: str) -> requests.Response:
    return requests.post(
        f"{GEMINI}/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
        },
        timeout=60,
    )


def gemini_json(api_key: str, wanted: str, prompt: str) -> str:
    """Call Gemini; if the model is gone (404), switch to the newest flash model and retry."""
    model = _resolved_model.get(wanted, wanted)
    r = _generate(api_key, model, prompt)
    if r.status_code == 404:
        names = [n for n in list_models(api_key) if n.removeprefix("models/") != model]
        alt = pick_flash_model(names)
        print(f"  Gemini model '{model}' unavailable ({r.text[:150]!r}); switching to '{alt}'")
        if alt:
            _resolved_model[wanted] = model = alt
            r = _generate(api_key, model, prompt)
    if not r.ok:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def ai_judge(clip, transcript: str, api_key: str | None, model: str) -> dict:
    fallback = {"score": None, "title": clip.title, "hashtags": [], "unsafe": False, "reason": "no AI key"}
    if not api_key:
        return fallback
    prompt = PROMPT.format(
        streamer=clip.broadcaster_name,
        game=clip.game or "unknown",
        title=clip.title,
        views=clip.view_count,
        age=clip.age_hours,
        transcript=(transcript or "(no speech)")[:3000],
    )
    try:
        text = gemini_json(api_key, model, prompt)
        return parse_judgement(text, clip.title)
    except Exception as e:  # never let the AI step kill the run
        print(f"  AI judge failed: {e}")
        return fallback


def parse_judgement(text: str, default_title: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)
    score = data.get("score")
    return {
        "score": float(score) if isinstance(score, (int, float)) else None,
        "title": (data.get("title") or default_title).strip()[:90],
        "hashtags": [h.lstrip("#").replace(" ", "") for h in data.get("hashtags", []) if h][:8],
        "unsafe": bool(data.get("unsafe", False)),
        "reason": data.get("reason", ""),
    }


def final_score(heuristic: float, ai_score: float | None) -> float:
    if ai_score is None:
        return heuristic
    return round(0.4 * heuristic + 0.6 * ai_score, 1)
