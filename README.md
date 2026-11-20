# Clip Agent

Finds Twitch clips that are blowing up, turns them into vertical 9:16 shorts with word-by-word captions,
and posts them to YouTube Shorts (TikTok and Instagram Reels next). It runs for free on GitHub Actions
every 4 hours. You don't need a server.

## How it works

1. **Find.** Looks at the top live English Twitch streams (plus any channels in `always_check`) and pulls
   the clips viewers made in the last 24 hours.
2. **Score.** Ranks them by how fast they're gaining views and how popular they are compared with the stream's audience.
3. **Judge.** Downloads the top few, transcribes them with Whisper, and asks Gemini (free tier) whether each
   would go viral with strangers. Gemini also writes the title and hashtags, and flags unsafe clips.
4. **Edit.** Blurred background, a zoomed gameplay frame, big captions with the current word highlighted,
   streamer credit, and loudness normalised.
5. **Post.** Uploads to YouTube Shorts (and TikTok when turned on), crediting and linking the streamer.
   `state/posted.json` records what was posted so nothing is posted twice.

Settings live in [`config.yaml`](config.yaml). Add any streamer who complains to `blocklist`.

## Setup (one time)

Every key goes in **GitHub → this repo → Settings → Secrets and variables → Actions → New repository secret**.
Never paste keys anywhere else.

### 1. Twitch (finds the clips)
1. Turn on two-factor authentication on your Twitch account (Twitch requires it for developer apps).
2. Go to https://dev.twitch.tv/console and log in. Click **Register Your Application**.
3. Name: `clip-agent-yourname`. OAuth Redirect URL: `http://localhost`. Category: **Application Integration**.
   Client Type: **Confidential**. Click **Create**.
4. Click **Manage** on the app. Copy the **Client ID**, then click **New Secret** and copy the secret.
5. Add the secrets `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`.

### 2. Gemini (judges clips, writes titles; free)
1. Go to https://aistudio.google.com/apikey and click **Create API key**.
2. Add the secret `GEMINI_API_KEY`.

### 3. Test run (posts nothing)
Go to the **Actions** tab → **Clip Agent** → **Run workflow** (leave "dry run" ticked). When it finishes,
download the **clips** file at the bottom of the run page to watch the video it made.

### 4. YouTube Shorts (posting)
1. Go to https://console.cloud.google.com, create a project called `clip-agent`.
2. **APIs & Services → Library**, search **YouTube Data API v3**, click **Enable**.
3. **OAuth consent screen**: choose **External**, fill in an app name and your email. Under **Audience**,
   add your own Google account as a test user, then click **Publish app**. Otherwise the login stops working after 7 days.
4. **Credentials → Create credentials → OAuth client ID** → type **Web application**. Under
   *Authorized redirect URIs* add `https://developers.google.com/oauthplayground`. Copy the Client ID and Client secret.
5. Open https://developers.google.com/oauthplayground, click the gear icon (top right), tick
   **Use your own OAuth credentials**, and paste the ID and secret.
6. In the box on the left type `https://www.googleapis.com/auth/youtube.upload` and click **Authorize APIs**. Sign in with
   the Google account that owns your YouTube channel. If you see "Google hasn't verified this app", click Advanced → Continue.
7. Click **Exchange authorization code for tokens** and copy the **Refresh token**.
8. Add the secrets `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, and `YOUTUBE_REFRESH_TOKEN`.

**Important:** YouTube makes videos uploaded by new, unaudited API projects **private**. To post publicly, submit
Google's free *YouTube API Services audit* form from the Cloud console. Until then, uploads will show as private on your channel.

### 5. TikTok (optional, later)
Create an app at https://developers.tiktok.com, add **Login Kit** and **Content Posting API** (Direct Post), get a
refresh token for your account, add `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_REFRESH_TOKEN`, and set
`publish.tiktok: true`. Posts stay private (`SELF_ONLY`) until TikTok audits the app.

### 6. Instagram Reels
Not built yet. It needs a Meta developer app and a public video URL. Coming next.

## Limits of the free setup
- YouTube's default quota allows about 6 uploads a day, which is why it runs every 4 hours.
- GitHub Actions is free for public repos; each run takes about 5 to 15 minutes.
- Reposting other people's streams can get copyright strikes. Credit is added automatically. Remove anyone who objects.

## Running locally
```
pip install -r requirements.txt
python -m clip_agent.main --dry-run
pytest
```
