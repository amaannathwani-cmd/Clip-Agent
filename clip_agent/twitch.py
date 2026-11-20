"""Find candidate clips on Twitch using the official Helix API."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests

API = "https://api.twitch.tv/helix"


@dataclass
class Clip:
    id: str
    url: str
    title: str
    broadcaster_id: str
    broadcaster_login: str
    broadcaster_name: str
    game: str
    view_count: int
    duration: float
    created_at: datetime
    channel_viewers: int = 0
    heuristic: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def age_hours(self) -> float:
        return max((datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600, 0.25)


class Twitch:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        r = requests.post(
            "https://id.twitch.tv/oauth2/token",
            data={"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"},
            timeout=30,
        )
        r.raise_for_status()
        self.session = requests.Session()
        self.session.headers.update({"Client-Id": client_id, "Authorization": f"Bearer {r.json()['access_token']}"})
        self._games: dict[str, str] = {}

    def _get(self, path: str, params) -> list[dict]:
        r = self.session.get(f"{API}/{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def top_streams(self, count: int, languages: list[str]) -> list[dict]:
        params = [("first", min(count, 100))] + [("language", lang) for lang in languages]
        return self._get("streams", params)

    def users(self, logins: list[str]) -> list[dict]:
        if not logins:
            return []
        return self._get("users", [("login", login) for login in logins[:100]])

    def game_names(self, ids: set[str]) -> dict[str, str]:
        missing = [i for i in ids if i and i not in self._games]
        if missing:
            for g in self._get("games", [("id", i) for i in missing[:100]]):
                self._games[g["id"]] = g["name"]
        return self._games

    def clips(self, broadcaster_id: str, since: datetime, first: int = 20) -> list[dict]:
        return self._get(
            "clips",
            {"broadcaster_id": broadcaster_id, "started_at": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "first": first},
        )


def find_candidates(tw: Twitch, cfg: dict, already_posted: set[str]) -> list[Clip]:
    tcfg = cfg.get("twitch", {})
    blocklist = {b.lower() for b in tcfg.get("blocklist", [])}
    channels: dict[str, dict] = {}
    for s in tw.top_streams(tcfg.get("top_live_streams", 25), tcfg.get("languages", ["en"])):
        channels[s["user_id"]] = {"login": s["user_login"], "viewers": s.get("viewer_count", 0)}
    for u in tw.users(tcfg.get("always_check", [])):
        channels.setdefault(u["id"], {"login": u["login"], "viewers": 0})

    since = datetime.now(timezone.utc) - timedelta(hours=tcfg.get("lookback_hours", 24))
    out: list[Clip] = []
    raw_all = []
    for bid, info in channels.items():
        if info["login"].lower() in blocklist:
            continue
        for c in tw.clips(bid, since):
            raw_all.append((c, info))
    games = tw.game_names({c.get("game_id", "") for c, _ in raw_all})
    for c, info in raw_all:
        if c["id"] in already_posted:
            continue
        if c["view_count"] < tcfg.get("min_views", 300) or c["duration"] > tcfg.get("max_seconds", 60):
            continue
        out.append(
            Clip(
                id=c["id"],
                url=c["url"],
                title=c["title"],
                broadcaster_id=c["broadcaster_id"],
                broadcaster_login=info["login"],
                broadcaster_name=c["broadcaster_name"],
                game=games.get(c.get("game_id", ""), ""),
                view_count=c["view_count"],
                duration=c["duration"],
                created_at=datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")),
                channel_viewers=info["viewers"],
            )
        )
    return out
