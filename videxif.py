# Needs: exiftooler.py, exiftool_files, and exiftool.exe in the same directory
# Usage: python exiftooler.py "D:\path\to\directory"
# Flags:
# --overwrite  | Pass -overwrite_original
# --dry-run    | Preview only
# --forcelocal | Fall back to system time zone for unspecified entries

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov"}
TZ_RE = re.compile(r"([+-]\d{2}:\d{2}|Z)$") # "-05:00", "+01:30", "Z"

def exiftool_path() -> Path:
    p = Path(__file__).resolve().with_name("exiftool.exe")
    if not p.exists():
        print(f"[ERROR] exiftool.exe not found next to script: {p}")
        sys.exit(1)
    return p

def read_tags(exiftool: Path, file: Path) -> dict:
    cmd = [str(exiftool), "-j", "-G1", "-s", "-Keys:CreationDate", "-UserData:DateTimeOriginal", str(file)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="replace"))
        return data[0] if data else {}
    except Exception as e:
        print(f"[ERROR] Read failed for {file}: {e}")
        return {}

def has_timezone(s: str) -> bool:
    return bool(TZ_RE.search(s.strip()))

def parse_to_utc_string(dt_str: str, assume_local: bool) -> str | None:
    """
    Accepts exiftool-like date strings 'YYYY:MM:DD HH:MM:SS[.fff][±HH:MM|Z]'
    Returns UTC as 'YYYY:MM:DD HH:MM:SS' (no offset) for QuickTime fields
    If source is naive and assume_local is False -> None
    """
    s = dt_str.strip()
    try:
        y, m, rest = s.split(":", 2)
        d = rest[:2]
        tail = rest[2:].strip() # "HH:MM:SS[.fff][±HH:MM|Z]"
        iso = f"{y}-{m}-{d}T{tail}" # "2024:12:08 19:13:19-05:00" -> "2024-12-08T19:13:19-05:00"

        if has_timezone(s):
            dt = datetime.fromisoformat(iso)
        else:
            if not assume_local:
                return None
            local_dt = datetime.fromisoformat(iso) # naive
            local_tz = datetime.now().astimezone().tzinfo
            dt = local_dt.replace(tzinfo=local_tz)

        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

def write_all(exiftool: Path, file: Path, *,
              utc_qt: str | None,
              raw_src: str,
              dry: bool,
              overwrite: bool) -> bool:
    """
    Build a single exiftool call that:
      - Always sets FileCreateDate & FileModifyDate to raw_src (exact string)
      - If utc_qt is provided, sets QuickTime CreateDate/ModifyDate and Track/MediaCreateDate to utc_qt
    """
    cmd = [str(exiftool)]
    if overwrite:
        cmd.append("-overwrite_original")

    cmd += [
        f"-FileCreateDate={raw_src}",
        f"-FileModifyDate={raw_src}",
    ]

    if utc_qt:
        cmd += [
            f"-CreateDate={utc_qt}",
            f"-ModifyDate={utc_qt}",
            f"-TrackCreateDate={utc_qt}",
            f"-MediaCreateDate={utc_qt}",
        ]

    cmd.append(str(file))

    if dry:
        print("[DRY] " + " ".join(cmd))
        return True

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        print(out.decode("utf-8", errors="replace").strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Write failed for {file}: {e.output.decode(errors='replace').strip()}")
        return False

def main():
    ap = argparse.ArgumentParser(
        description=("Write QuickTime (Create/Modify + Track/Media) in UTC and set filesystem "
                     "FileCreateDate/FileModifyDate from Keys:CreationDate (preferred) or "
                     "UserData:DateTimeOriginal for .mp4/.mov in a folder.")
    )
    ap.add_argument("folder", help="Target folder (non-recursive).")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Pass -overwrite_original to ExifTool (no *._original backups).")
    ap.add_argument("--forcelocal", action="store_true",
                    help=("If source time lacks a timezone, assume system local timezone "
                          "before converting to UTC; otherwise QuickTime/Track/Media writes are skipped "
                          "but filesystem timestamps still update from the raw source."))
    args = ap.parse_args()

    target = Path(args.folder).expanduser().resolve()
    if not target.is_dir():
        print(f"[ERROR] Not a folder: {target}")
        sys.exit(1)

    exif = exiftool_path()
    files = [p for p in target.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not files:
        print("[INFO] No .mp4 or .mov files found.")
        return

    changed = 0
    skipped = [] # list of (filename, reason)

    for f in sorted(files):
        tags = read_tags(exif, f)
        keys = tags.get("Keys:CreationDate")
        ud   = tags.get("UserData:DateTimeOriginal")

        src_label = None
        src_value = None
        if keys:
            src_label, src_value = "Keys:CreationDate", keys
        elif ud:
            src_label, src_value = "UserData:DateTimeOriginal", ud

        if not src_value:
            msg = "no Keys:CreationDate or UserData:DateTimeOriginal found"
            print(f"[SKIP] {f.name}: {msg}.")
            skipped.append((f.name, msg))
            continue

        utc_string = parse_to_utc_string(src_value, assume_local=args.forcelocal) # prepare UTC for QuickTime & Track/Media
        qt_part = "will update QuickTime/Track/Media" if utc_string else "SKIP QuickTime/Track/Media"

        if not utc_string and not has_timezone(src_value) and not args.forcelocal:
            reason = "no timezone under non --forcelocal"
            print(f"[INFO] {f.name}: {src_label}='{src_value}' -> {reason}; "
                  f"filesystem dates will still be set from source.")
            skipped.append((f.name, reason)) # record skip reason for the summary (specific to metadata conversion)
        elif not utc_string:
            reason = "parse error"
            print(f"[INFO] {f.name}: {src_label}='{src_value}' -> {reason}; "
                  f"filesystem dates will still be set from source.")
            skipped.append((f.name, reason))

        print(f"[INFO] {f.name}: using {src_label}='{src_value}' -> {qt_part}; "
              f"FileCreateDate/FileModifyDate will be set from source.")

        ok = write_all(exif, f,
                       utc_qt=utc_string,
                       raw_src=src_value,
                       dry=args.dry_run,
                       overwrite=args.overwrite)

        if ok:
            changed += 1
        else:
            skipped.append((f.name, "write error"))

    print(f"\n[SUMMARY] Changed: {changed}  Skipped: {len(skipped)}") # summary with explicit reasons for skips
    if skipped:
        print("[SKIPPED FILES]")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

if __name__ == "__main__":
    main()

