import os
import re
import sys
import time

from datetime import datetime
from typing import List, Dict, Any

from ytmusicapi import YTMusic # type: ignore

# =========================
# CONFIG
# =========================

PLAYLIST_URL = "https://music.youtube.com/playlist?list=PUT_YOUR_PLAYLIST_ID_OR_URL_HERE"
TOP_DOWN = False
SLEEP_SECONDS = 1.5

# insert "Authorization" and "Cookie" values from browser login session (do NOT share!)
AUTH_HEADERS: Dict[str, str] = {
    "Accept": "*/*",
    "Authorization": "SAPISIDHASH_YOUR_REAL_VALUE_HERE",
    "Content-Type": "application/json",
    "X-Goog-AuthUser": "0",
    "x-origin": "https://music.youtube.com",
    "Cookie": "YOUR_FULL_COOKIE_HEADER_HERE"
}

# =========================
# END CONFIG
# =========================

def extract_playlist_id(url_or_id: str) -> str:
    match = re.search(r"[?&]list=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()

def load_playlist(yt: YTMusic, playlist_id: str) -> Dict[str, Any]:
    return yt.get_playlist(playlist_id, limit=None)

def collect_tracks(playlist: Dict[str, Any]):
    tracks = playlist.get("tracks", [])
    canonical_video_ids: List[str] = []
    removable_items: List[Dict[str, Any]] = []

    for t in tracks:
        vid = t.get("videoId")
        set_vid = t.get("setVideoId")

        if not vid or not set_vid:
            title = t.get("title", "<unknown>")
            print(f"WARNING: Skipping track '{title}' (missing videoId or setVideoId).")
            continue

        canonical_video_ids.append(vid)
        removable_items.append({"videoId": vid, "setVideoId": set_vid})

    return canonical_video_ids, removable_items

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:160] if name else "playlist"

def write_playlist_backup_txt(
    playlist: Dict[str, Any],
    playlist_title: str,
    playlist_id: str,
) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_title = sanitize_filename(playlist_title)
    filename = f"{safe_title}__{playlist_id}__pre_edit__{timestamp}.txt"
    filepath = os.path.join(script_dir, filename)

    tracks = playlist.get("tracks", []) or []
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        for t in tracks:
            title = t.get("title") or "<unknown>"
            f.write(str(title).strip() + "\n")

    return filepath

def remove_tracks_individually(
    yt: YTMusic,
    playlist_id: str,
    removable_items: List[Dict[str, Any]],
    sleep_seconds: float = 0.0,
):
    total = len(removable_items)
    print(f"Removing {total} items one-by-one...")

    for i, item in enumerate(removable_items, start=1):
        try:
            yt.remove_playlist_items(playlist_id, [item])
            print(f"[remove] {i}/{total}")
        except Exception as e:
            print(f"ERROR removing item {i}/{total}: {e}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

def readd_tracks_individually(
    yt: YTMusic,
    playlist_id: str,
    add_order: List[str],
    sleep_seconds: float = 0.0,
):
    total = len(add_order)
    print(f"Re-adding {total} items one-by-one...")

    for i, vid in enumerate(add_order, start=1):
        try:
            yt.add_playlist_items(playlist_id, [vid], duplicates=True)
            print(f"[add] {i}/{total}")
        except Exception as e:
            print(f"ERROR adding video {vid} ({i}/{total}): {e}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

def main():
    if "PUT_YOUR_PLAYLIST_ID_OR_URL_HERE" in PLAYLIST_URL:
        print("ERROR: You forgot to set PLAYLIST_URL in the config section at the top of the script.")
        sys.exit(1)

    if not AUTH_HEADERS or "Authorization" not in AUTH_HEADERS or "Cookie" not in AUTH_HEADERS:
        print("ERROR: AUTH_HEADERS is missing or incomplete. Make sure you filled it with your real headers.")
        sys.exit(1)

    playlist_id = extract_playlist_id(PLAYLIST_URL)

    try:
        yt = YTMusic(AUTH_HEADERS)
    except Exception as e:
        print("Failed to initialize YTMusic with the provided auth headers.")
        print("Double-check AUTH_HEADERS and that your session hasn't expired.")
        print(f"Error: {e}")
        sys.exit(1)

    print()
    print(f"Target playlist ID: {playlist_id}")
    print("Loading playlist...")
    playlist = load_playlist(yt, playlist_id)

    title = playlist.get("title", "<unknown title>")

    canonical_video_ids, removable_items = collect_tracks(playlist)

    num_tracks = len(canonical_video_ids)

    if num_tracks == 0:
        print("No usable tracks found in the playlist. Exiting.")
        return

    if TOP_DOWN:
        add_order = canonical_video_ids
    else:
        add_order = list(reversed(canonical_video_ids))

    print()
    print("SUMMARY:")
    print(f"  Playlist: {title}")
    print(f"  Tracks detected: {num_tracks}")
    print(f"  TOP_DOWN: {TOP_DOWN}")
    print(f"  SLEEP_SECONDS: {SLEEP_SECONDS}")
    print()

    confirm = input(
        "This will REMOVE and RE-ADD all tracks in this playlist, "
        "permanently resetting 'recently added' timestamps.\n"
        "Proceed? [y/n]: "
    ).strip().lower()

    if confirm != "y":
        print("Aborted by user.")
        return

    try:
        backup_path = write_playlist_backup_txt(playlist, title, playlist_id)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print("ERROR: Failed to create pre-edit backup file. Aborting to avoid destructive edits.")
        print(f"Error: {e}")
        return

    remove_tracks_individually(
        yt, playlist_id, removable_items, sleep_seconds=SLEEP_SECONDS
    )

    readd_tracks_individually(
        yt, playlist_id, add_order, sleep_seconds=SLEEP_SECONDS
    )

    print("Done!")

if __name__ == "__main__":
    main()


