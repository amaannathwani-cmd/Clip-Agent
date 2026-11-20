"""One run of the agent: find clips, judge them, edit the best, post it.

Usage:
  python -m clip_agent.main            # normal run (posts)
  python -m clip_agent.main --dry-run  # make the videos but don't post
  python -m clip_agent.main --clip-url https://clips.twitch.tv/... --dry-run   # test one clip
"""
import argparse
import re
from pathlib import Path

from . import state as st
from .config import ROOT, env, load_config
from .editor import render
from .media import download, transcribe
from .scorer import ai_judge, final_score, heuristic_score
from .twitch import Clip, Twitch, find_candidates

WORK = ROOT / "work"
OUT = ROOT / "out"


def describe(clip: Clip, judged: dict) -> tuple[str, str, list[str]]:
    tags = judged.get("hashtags") or []
    base_tags = ["shorts", "twitch", re.sub(r"\W", "", clip.broadcaster_login)]
    if clip.game:
        base_tags.append(re.sub(r"\W", "", clip.game.lower()))
    tags = list(dict.fromkeys([t.lower() for t in tags + base_tags if t]))
    title = f"{judged['title']} #shorts"
    desc = (
        f"{clip.broadcaster_name} on Twitch: https://twitch.tv/{clip.broadcaster_login}\n"
        f"Original clip: {clip.url}\n\n"
        f"All credit to {clip.broadcaster_name}. Go follow them!\n"
        f"Creator who wants this removed? Comment and it will be taken down.\n\n"
        + " ".join(f"#{t}" for t in tags)
    )
    return title, desc, tags


def process(clip: Clip, cfg: dict, gemini_key: str | None) -> dict | None:
    vcfg = cfg.get("video", {})
    print(f"- {clip.broadcaster_login}: '{clip.title}' ({clip.view_count} views, heuristic {clip.heuristic})")
    src = download(clip.url, WORK / clip.id, "source")
    words = transcribe(src, vcfg.get("whisper_model", "small"))
    transcript = " ".join(w.text for w in words)
    judged = ai_judge(clip, transcript, gemini_key, cfg.get("ai", {}).get("gemini_model", "gemini-2.5-flash"))
    score = final_score(clip.heuristic, judged["score"])
    print(f"  AI {judged['score']} -> final {score}; unsafe={judged['unsafe']}; {judged.get('reason', '')}")
    return {"clip": clip, "src": src, "words": words, "judged": judged, "score": score}


def run(dry_run: bool = False, clip_url: str | None = None) -> None:
    cfg = load_config()
    sel = cfg.get("selection", {})
    gemini_key = env("GEMINI_API_KEY", required=False)
    state = st.load()

    if clip_url:
        clip_id = clip_url.rstrip("/").split("/")[-1].split("?")[0]
        from datetime import datetime, timezone
        candidates = [Clip(clip_id, clip_url, "Test clip", "", "streamer", "Streamer", "", 1000, 30,
                           datetime.now(timezone.utc), heuristic=70)]
    else:
        tw = Twitch(env("TWITCH_CLIENT_ID"), env("TWITCH_CLIENT_SECRET"))
        candidates = find_candidates(tw, cfg, set(state))
        for c in candidates:
            c.heuristic = heuristic_score(c)
        candidates.sort(key=lambda c: c.heuristic, reverse=True)
        print(f"Found {len(candidates)} new candidate clips")

    judged = []
    for clip in candidates[: sel.get("judge_top", 4)]:
        try:
            r = process(clip, cfg, gemini_key)
            if r and not r["judged"]["unsafe"]:
                judged.append(r)
        except Exception as e:
            print(f"  skipped: {e}")
    judged.sort(key=lambda r: r["score"], reverse=True)
    winners = [r for r in judged if r["score"] >= sel.get("min_score", 55) or clip_url][: sel.get("posts_per_run", 1)]
    if not winners:
        print("Nothing good enough this run.")
        return

    pub = cfg.get("publish", {})
    for r in winners:
        clip = r["clip"]
        credit = f"twitch.tv/{clip.broadcaster_login}" if cfg.get("video", {}).get("show_credit", True) else None
        out = render(r["src"], r["words"], OUT / f"{clip.id}.mp4", credit,
                     min(clip.duration or 60, cfg.get("twitch", {}).get("max_seconds", 60)))
        title, desc, tags = describe(clip, r["judged"])
        print(f"Rendered {out.name}: {title}")
        if dry_run:
            print("  (dry run, not posting)")
            continue
        results = {}
        if pub.get("youtube", True):
            try:
                from .publishers import youtube
                results["youtube"] = youtube.upload(out, title, desc, tags, env("YOUTUBE_CLIENT_ID"),
                                                    env("YOUTUBE_CLIENT_SECRET"), env("YOUTUBE_REFRESH_TOKEN"))
                print(f"  YouTube: https://youtube.com/shorts/{results['youtube']}")
            except Exception as e:
                print(f"  YouTube failed: {e}")
        if pub.get("tiktok", False):
            try:
                from .publishers import tiktok
                caption = f"{r['judged']['title']} | {credit or clip.broadcaster_name} " + " ".join(f"#{t}" for t in tags)
                results["tiktok"] = tiktok.upload(out, caption, pub.get("tiktok_privacy", "SELF_ONLY"),
                                                  env("TIKTOK_CLIENT_KEY"), env("TIKTOK_CLIENT_SECRET"),
                                                  env("TIKTOK_REFRESH_TOKEN"))
                print(f"  TikTok publish id: {results['tiktok']}")
            except Exception as e:
                print(f"  TikTok failed: {e}")
        if results:
            st.record(state, clip, results)
    st.save(state)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--clip-url")
    a = p.parse_args()
    run(dry_run=a.dry_run, clip_url=a.clip_url)


if __name__ == "__main__":
    main()
