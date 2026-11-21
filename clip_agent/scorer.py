"""Score clips: a fast heuristic on view velocity, then an AI judge on the transcript."""
import base64
import json
import math
import re
import time

import requests


def heuristic_score(clip) -> float:
    """0-100. Rewards clips gaining views fast, and clips popular relative to the stream's audience."""
    velocity = clip.view_count / clip.age_hours
    v = min(math.log10(velocity + 1) / 4, 1.0)  # 10k views/hour maxes it out
    rel = min(clip.view_count / max(clip.channel_viewers, 1), 1.0) ** 0.5 if clip.channel_viewers else 0.3
    length = 1.0 if 12 <= clip.duration <= 45 else 0.7
    return round(100 * (0.6 * v + 0.4 * rel) * length, 1)


PROMPT = """You run a viral clips channel that reposts Twitch moments as YouTube Shorts and TikToks.
You are shown the clip itself (video + audio) plus its transcript. Be a harsh judge: most clips are boring.

What goes viral with strangers who have never heard of the streamer:
- genuinely funny moments: big reactions, fails, awkward silences, unexpected one-liners, chat or friends roasting them
- shock / jump scares / rage / "no way" skill moments
- a clear payoff that lands within ~20 seconds, and a hook in the first 2 seconds
What does NOT: giveaways, sub alerts, talking about plans, inside jokes, long setups, menus/loading screens,
clips that are cut off before the payoff, anything that needs context.

Streamer: {streamer}
Game/category: {game}
Original clip title: {title}
Views: {views} in {age:.1f} hours
Clip length: {duration:.0f}s
Transcript (with seconds): {transcript}

Reply with JSON only:
{{"funny": 0-10, "hook": 0-10, "standalone": 0-10,
"score": 0-100 overall chance this goes viral (70+ only if you would genuinely share it),
"reason": "one sentence",
"start": seconds where the short should start (cut dead air before the hook), "end": seconds where it should end (just after the payoff),
"title": "punchy title under 60 chars that teases the moment without spoiling it, no hashtags, no lies",
"hashtags": ["5 relevant hashtags without #"],
"unsafe": true/false}}
Set unsafe true for slurs, sexual content, real violence, gambling promos, doxxing, or anything that needs lots of context."""


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


def _generate(api_key: str, model: str, prompt: str, video: bytes | None = None) -> requests.Response:
    parts = [{"text": prompt}]
    if video:
        parts.insert(0, {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(video).decode()}})
    return requests.post(
        f"{GEMINI}/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4,
                                 **({"mediaResolution": "MEDIA_RESOLUTION_LOW"} if video else {})},
        },
        timeout=180 if video else 60,
    )


def rank_flash_models(names: list[str]) -> list[str]:
    """All usable flash models, best first: newest stable, then previews, then the cheaper 'lite' ones."""
    skip = ("image", "tts", "audio", "live", "embedding", "thinking", "exp")
    good = [n.removeprefix("models/") for n in names if "flash" in n and not any(k in n for k in skip)]

    def key(n: str):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return ("lite" in n, "preview" in n, -(float(m.group(1)) if m else 0.0), len(n))

    return sorted(dict.fromkeys(good), key=key)


def _err(r: requests.Response) -> str:
    try:
        return f"{r.status_code} {r.json()['error'].get('message', '')[:160]}"
    except Exception:
        return f"{r.status_code} {r.text[:160]}"


_model_list: list[str] | None = None


def gemini_json(api_key: str, wanted: str, prompt: str, video: bytes | None = None) -> str:
    """Call Gemini. If a model is retired, rate-limited or overloaded, fall through to the next flash model."""
    global _model_list
    first = _resolved_model.get(wanted, wanted)
    tried, last = [], "no model available"
    order = [first]
    while order:
        model = order.pop(0)
        if video and tried:
            time.sleep(2)
        if model in tried:
            continue
        tried.append(model)
        r = None
        for attempt in range(2):
            try:
                r = _generate(api_key, model, prompt, video)
            except requests.RequestException as e:
                last = f"{model}: {type(e).__name__}"
                break
            if r.ok:
                if model != first:
                    print(f"  using Gemini model '{model}'")
                _resolved_model[wanted] = model
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            last = f"{model}: {_err(r)}"
            if r.status_code in (500, 503) and attempt == 0:
                time.sleep(5)
                continue
            break
        if r is not None and not r.ok and r.status_code not in (404, 429, 500, 503):
            break  # a real error (bad key, bad request): switching models won't help
        if _model_list is None:
            try:
                _model_list = rank_flash_models(list_models(api_key))
            except Exception:
                _model_list = []
        order = [m for m in _model_list if m not in tried] if len(tried) < 4 else []
    raise RuntimeError(f"Gemini failed ({last})")


def ai_judge(clip, transcript: str, api_key: str | None, model: str, video: bytes | None = None) -> dict:
    fallback = {"score": None, "title": clip.title, "hashtags": [], "unsafe": False, "reason": "no AI key",
                "start": 0.0, "end": None}
    if not api_key:
        return fallback
    prompt = PROMPT.format(
        streamer=clip.broadcaster_name,
        game=clip.game or "unknown",
        title=clip.title,
        views=clip.view_count,
        age=clip.age_hours,
        duration=clip.duration,
        transcript=(transcript or "(no speech)")[:3000],
    )
    try:
        try:
            text = gemini_json(api_key, model, prompt, video)
        except Exception as e:
            if not video:
                raise
            print(f"  AI couldn't watch the video ({str(e)[:120]}); judging from the transcript only")
            text = gemini_json(api_key, model, prompt, None)
        return parse_judgement(text, clip.title, clip.duration)
    except Exception as e:  # never let the AI step kill the run
        print(f"  AI judge failed: {e}")
        return fallback


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_judgement(text: str, default_title: str, duration: float | None = None) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)
    score = _num(data.get("score"))
    start, end = _num(data.get("start")) or 0.0, _num(data.get("end"))
    # Only trust a trim that keeps a sensible length; otherwise use the whole clip.
    if end is None or end - start < 6 or start < 0 or (duration and end > duration + 1):
        start, end = 0.0, None
    return {
        "score": score,
        "title": (data.get("title") or default_title).strip()[:90],
        "hashtags": [h.lstrip("#").replace(" ", "") for h in data.get("hashtags", []) if h][:8],
        "unsafe": bool(data.get("unsafe", False)),
        "reason": data.get("reason", ""),
        "start": max(0.0, start - 0.3) if start else 0.0,
        "end": end + 0.5 if end else None,
    }


def final_score(heuristic: float, ai_score: float | None) -> float:
    if ai_score is None:
        return heuristic
    return round(0.25 * heuristic + 0.75 * ai_score, 1)
