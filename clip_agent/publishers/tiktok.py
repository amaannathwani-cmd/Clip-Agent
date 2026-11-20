"""Post a video with TikTok's Content Posting API (Direct Post, file upload)."""
from pathlib import Path

import requests

API = "https://open.tiktokapis.com/v2"


def access_token(client_key: str, client_secret: str, refresh_token: str) -> str:
    r = requests.post(
        f"{API}/oauth/token/",
        data={"client_key": client_key, "client_secret": client_secret,
              "grant_type": "refresh_token", "refresh_token": refresh_token},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "access_token" not in data:
        raise RuntimeError(f"TikTok token refresh failed: {data}")
    return data["access_token"]


def upload(path: Path, caption: str, privacy: str, client_key: str, client_secret: str, refresh_token: str) -> str:
    token = access_token(client_key, client_secret, refresh_token)
    size = path.stat().st_size
    if size > 64 * 1024 * 1024:
        raise RuntimeError("Video too large for single-chunk upload")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    init = requests.post(
        f"{API}/post/publish/video/init/",
        headers=headers,
        json={
            "post_info": {"title": caption[:2200], "privacy_level": privacy,
                          "disable_duet": False, "disable_comment": False, "disable_stitch": False},
            "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": size, "total_chunk_count": 1},
        },
        timeout=30,
    )
    init.raise_for_status()
    data = init.json()["data"]
    with open(path, "rb") as f:
        put = requests.put(
            data["upload_url"],
            data=f.read(),
            headers={"Content-Type": "video/mp4", "Content-Range": f"bytes 0-{size - 1}/{size}",
                     "Content-Length": str(size)},
            timeout=300,
        )
    put.raise_for_status()
    return data["publish_id"]
