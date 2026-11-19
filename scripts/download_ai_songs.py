#!/usr/bin/env python3
"""Download the AI half of the benchmark staging set.

This script performs network requests. Do not run it without owner approval.

It reads the AI-tagged songs from v4vmusic, selects a deterministic seeded pool,
downloads each mp3 politely (serial requests with a delay, redirect and byte caps,
bounded retries that honour ``Retry-After``), verifies each file with ffprobe, and
appends successful rows to ``test-audio/test-songs.csv``. Failed URLs are logged to
``test-audio/ai-download-errors.csv`` so a rerun can resume.

The staging CSV is local-only data (git-ignored); the bench manifest is separate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import email.utils
import functools
import ipaddress
import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import IO, Any

# Benchmark constants: CC §7 (single source of truth, no magic numbers in logic).
SEED = 42
CANDIDATE_POOL = 300
TARGET_AI_SONGS = 200
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
MAX_REDIRECTS = 5
CONNECT_TIMEOUT_SECONDS = 15.0
MAX_BYTES = 80 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
MIN_DURATION_SECONDS = 5.0
ALLOWED_SUFFIXES = frozenset({".mp3", ".m4a", ".ogg", ".opus", ".flac", ".wav"})
CSV_FIELDS = ("path", "label", "artist_name", "feed_title", "song_title")
USER_AGENT = "ai-music-detector-benchmark-downloader/0.1"

AddressInfo = tuple[int, int, int, str, tuple[str, int]]
Resolver = Callable[..., Any]

SONGS_SQL = """
SELECT DISTINCT s.id, s."mp3Url", s."artistName", s.title,
       COALESCE(al.title, '') AS feed_title
FROM songs s
JOIN "ItemTag" it ON it."itemId" = s.id AND it."itemType" = 'song'
JOIN tags t ON t.id = it."tagId" AND t.name IN ('ai', 'ai-generated')
LEFT JOIN albums al ON al.id = s."albumId"
WHERE s."mp3Url" IS NOT NULL AND s."mp3Url" <> ''
ORDER BY s.id
"""


class DownloadError(Exception):
    """Base class for a download that cannot produce usable audio."""


class UnsafeUrlError(DownloadError):
    """The URL is not an allowed, publicly-addressable http(s) target."""


class VerifyError(DownloadError):
    """The downloaded file is not usable audio."""


class SkipSong(DownloadError):
    """This song should be skipped (e.g. HTTP 404 or persistent 429)."""


class HttpError(DownloadError):
    """An HTTP error response that may be retryable."""

    def __init__(self, status: int, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class Song:
    song_id: str
    mp3_url: str
    artist_name: str
    song_title: str
    feed_title: str


@dataclass(frozen=True, slots=True)
class Config:
    csv_path: Path
    out_dir: Path
    error_log: Path
    seed: int = SEED
    pool_size: int = CANDIDATE_POOL
    target: int = TARGET_AI_SONGS
    delay: float = REQUEST_DELAY_SECONDS
    max_retries: int = MAX_RETRIES
    max_redirects: int = MAX_REDIRECTS
    max_bytes: int = MAX_BYTES
    timeout: float = CONNECT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class StagingInfo:
    fieldnames: list[str]
    rows: list[dict[str, str]]
    ai_song_ids: set[str]


@dataclass(frozen=True, slots=True)
class Stats:
    succeeded: int
    failed: int
    already_present: int
    shortfall: int


def clean(value: object) -> str:
    """Normalise a metadata value for a single CSV field."""
    return " ".join(str(value or "").split())


def load_songs(database_url: str) -> list[Song]:
    """Read the AI-tagged song pool from v4vmusic (psycopg2 imported lazily)."""
    psycopg2: ModuleType = __import__("psycopg2")
    with closing(
        psycopg2.connect(database_url, connect_timeout=int(CONNECT_TIMEOUT_SECONDS))
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SONGS_SQL)
            rows = cursor.fetchall()
    return [
        Song(
            song_id=clean(row[0]),
            mp3_url=clean(row[1]),
            artist_name=clean(row[2]),
            song_title=clean(row[3]),
            feed_title=clean(row[4]),
        )
        for row in rows
    ]


def read_staging_rows(csv_path: Path) -> StagingInfo:
    """Load the staging CSV and collect song ids already labelled ai_generated."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or CSV_FIELDS)
        rows = [dict(row) for row in reader]
    existing = {
        Path(row.get("path") or "").stem
        for row in rows
        if row.get("label") == "ai_generated" and row.get("path")
    }
    return StagingInfo(fieldnames=fieldnames, rows=rows, ai_song_ids=existing)


def select_candidates(songs: Sequence[Song], seed: int, pool_size: int) -> list[Song]:
    """Return a deterministic, deduplicated seeded sample of the song pool."""
    ordered = sorted(songs, key=lambda song: song.song_id)
    random.Random(seed).shuffle(ordered)
    return ordered[:pool_size]


def validate_url(url: str, resolver: Resolver = socket.getaddrinfo) -> None:
    """Reject non-http(s) and private/loopback/link-local targets (CC §8)."""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeUrlError(f"Unsupported URL scheme {scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("URL has no host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError(f"Invalid URL port: {exc}") from exc
    try:
        infos = resolver(host, port or (443 if scheme == "https" else 80), 0, socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeUrlError(f"Cannot resolve {host}: {exc}") from exc
    if not infos:
        raise UnsafeUrlError(f"No DNS result for {host}")
    for info in infos:
        address = str(info[4][0]).split("%", 1)[0]
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeUrlError(f"Refusing private/special address {ip} for {host}")


def build_download_opener() -> urllib.request.OpenerDirector:
    """Build an opener with no redirect handler so fetch_bytes validates every hop."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.ProxyHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def parse_retry_after(value: str | None, now: Callable[[], float] = time.time) -> float | None:
    """Parse a Retry-After header (seconds or HTTP date), clamped to the cap."""
    if not value:
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        reference = dt.datetime.fromtimestamp(now(), tz=dt.timezone.utc)
        seconds = (when - reference).total_seconds()
    return max(0.0, min(seconds, MAX_BACKOFF_SECONDS))


def _read_capped(response: IO[bytes], max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise DownloadError(f"Response exceeds {max_bytes} byte cap")
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_bytes(
    url: str,
    *,
    opener: Any,
    resolver: Resolver = socket.getaddrinfo,
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int = MAX_BYTES,
    timeout: float = CONNECT_TIMEOUT_SECONDS,
) -> bytes:
    """Fetch one URL with explicit, validated redirect handling and a byte cap."""
    current = url
    for _ in range(max_redirects + 1):
        validate_url(current, resolver)
        request = urllib.request.Request(current, headers={"User-Agent": USER_AGENT})
        try:
            with opener.open(request, timeout=timeout) as response:
                return _read_capped(response, max_bytes)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                location = exc.headers.get("Location") if exc.headers else None
                if not location:
                    raise DownloadError(f"Redirect {exc.code} without Location") from exc
                current = urllib.parse.urljoin(current, location)
                continue
            retry_after = parse_retry_after(exc.headers.get("Retry-After")) if exc.headers else None
            raise HttpError(exc.code, retry_after) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise DownloadError(str(exc)) from exc
    raise DownloadError(f"Exceeded {max_redirects} redirects")


def download_with_retries(
    url: str,
    *,
    fetch: Callable[[str], bytes],
    sleep: Callable[[float], None],
    max_retries: int = MAX_RETRIES,
    base_backoff: float = BASE_BACKOFF_SECONDS,
) -> bytes:
    """Retry rate limits and server errors; skip dead links and client errors."""
    for attempt in range(max_retries + 1):
        try:
            return fetch(url)
        except HttpError as exc:
            retryable = exc.status == 429 or 500 <= exc.status < 600
            if not retryable:
                raise SkipSong(f"HTTP {exc.status}") from exc
            if attempt >= max_retries:
                raise SkipSong(f"HTTP {exc.status} after {max_retries} retries") from exc
            delay = exc.retry_after
            if delay is None:
                delay = base_backoff * (2**attempt)
            sleep(min(delay, MAX_BACKOFF_SECONDS))
        except DownloadError as exc:
            raise SkipSong(str(exc)) from exc
    raise SkipSong("Download failed after retries")


def audio_suffix(url: str) -> str:
    """Derive a whitelisted audio extension from the URL path."""
    suffix = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    suffix = Path(suffix).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise SkipSong(f"Unsupported audio extension {suffix!r}")
    return suffix


def probe_duration(path: Path) -> float:
    """Return the duration in seconds using ffprobe (argument list, no shell)."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, shell=False)
    if result.returncode != 0:
        raise VerifyError(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise VerifyError(f"ffprobe returned no duration for {path.name}") from exc


def write_audio(destination: Path, body: bytes) -> None:
    """Write to a .part file first, then atomically replace the destination."""
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(body)
    os.replace(temporary, destination)


def _ensure_trailing_newline(path: Path) -> None:
    """Avoid merging the first appended row into a CSV that lacks a final newline."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
    with path.open("ab") as handle:
        handle.write(b"\n")


def append_row(csv_path: Path, fieldnames: Sequence[str], song: Song, relative_path: str) -> None:
    """Append one ai_generated row, preserving any extra staging columns."""
    _ensure_trailing_newline(csv_path)
    row = {name: "" for name in fieldnames}
    metadata = {
        "path": relative_path,
        "label": "ai_generated",
        "artist_name": song.artist_name,
        "feed_title": song.feed_title,
        "song_title": song.song_title,
    }
    for key, value in metadata.items():
        if key in row:
            row[key] = value
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writerow(row)


def log_failure(error_log: Path, song: Song, reason: str) -> None:
    """Append a failure record; the log is local-only data."""
    needs_header = not error_log.exists()
    with error_log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if needs_header:
            writer.writerow(["song_id", "url", "reason"])
        writer.writerow([song.song_id, song.mp3_url, clean(reason)])


def run(
    config: Config,
    songs: Sequence[Song],
    *,
    fetch: Callable[[str], bytes],
    sleep: Callable[[float], None],
    probe: Callable[[Path], float],
) -> Stats:
    """Download candidates until the AI target is met, appending verified rows."""
    staging = read_staging_rows(config.csv_path)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.error_log.parent.mkdir(parents=True, exist_ok=True)
    needed = max(0, config.target - len(staging.ai_song_ids))
    succeeded = 0
    failed = 0

    def obtain(destination: Path, song: Song) -> float:
        if destination.is_file():
            try:
                duration = probe(destination)
                if duration >= MIN_DURATION_SECONDS:
                    return duration
                raise VerifyError(f"Duration {duration:.1f}s is below {MIN_DURATION_SECONDS}s")
            except VerifyError:
                destination.unlink(missing_ok=True)
        sleep(config.delay)
        body = download_with_retries(
            song.mp3_url,
            fetch=fetch,
            sleep=sleep,
            max_retries=config.max_retries,
        )
        write_audio(destination, body)
        try:
            duration = probe(destination)
        except VerifyError:
            destination.unlink(missing_ok=True)
            raise
        if duration < MIN_DURATION_SECONDS:
            destination.unlink(missing_ok=True)
            raise VerifyError(f"Duration {duration:.1f}s is below {MIN_DURATION_SECONDS}s")
        return duration

    for song in select_candidates(songs, config.seed, config.pool_size):
        if succeeded >= needed:
            break
        if song.song_id in staging.ai_song_ids:
            continue
        try:
            destination = config.out_dir / f"{song.song_id}{audio_suffix(song.mp3_url)}"
            obtain(destination, song)
        except DownloadError as exc:
            log_failure(config.error_log, song, str(exc))
            failed += 1
            continue
        relative = destination.relative_to(config.csv_path.parent).as_posix()
        append_row(config.csv_path, staging.fieldnames, song, relative)
        succeeded += 1

    total_ai = len(staging.ai_song_ids) + succeeded
    return Stats(
        succeeded=succeeded,
        failed=failed,
        already_present=len(staging.ai_song_ids),
        shortfall=max(0, config.target - total_ai),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--database-url",
        default=os.environ.get("V4VMUSIC_DATABASE_URL"),
        help="v4vmusic DSN (default: V4VMUSIC_DATABASE_URL)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Workspace root (default: parent of scripts/)",
    )
    parser.add_argument("--csv", type=Path, help="Staging CSV (default: <root>/test-audio)")
    parser.add_argument(
        "--out-dir", type=Path, help="AI audio directory (default: <root>/test-audio)"
    )
    parser.add_argument("--error-log", type=Path, help="Failure log (default: <root>/test-audio)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pool", type=int, default=CANDIDATE_POOL)
    parser.add_argument("--target", type=int, default=TARGET_AI_SONGS)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the seeded candidate selection without downloading anything",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    csv_path = args.csv or root / "test-audio" / "test-songs.csv"
    out_dir = args.out_dir or root / "test-audio" / "ai-songs"
    error_log = args.error_log or root / "test-audio" / "ai-download-errors.csv"

    if not csv_path.is_file():
        print(f"error: staging CSV not found: {csv_path}", file=sys.stderr)
        return 2
    if not args.database_url:
        print("error: set V4VMUSIC_DATABASE_URL or pass --database-url", file=sys.stderr)
        return 2

    songs = load_songs(args.database_url)
    config = Config(
        csv_path=csv_path,
        out_dir=out_dir,
        error_log=error_log,
        seed=args.seed,
        pool_size=args.pool,
        target=args.target,
        delay=args.delay,
    )

    if args.dry_run:
        staging = read_staging_rows(csv_path)
        for song in select_candidates(songs, config.seed, config.pool_size):
            state = "already" if song.song_id in staging.ai_song_ids else "new"
            print(f"{song.song_id}\t{state}\t{song.mp3_url}")
        return 0

    opener = build_download_opener()
    fetch = functools.partial(
        fetch_bytes,
        opener=opener,
        resolver=socket.getaddrinfo,
        max_redirects=config.max_redirects,
        max_bytes=config.max_bytes,
        timeout=config.timeout,
    )
    stats = run(
        config,
        songs,
        fetch=fetch,
        sleep=time.sleep,
        probe=probe_duration,
    )
    print(
        f"Appended {stats.succeeded} AI rows; {stats.failed} failed; "
        f"{stats.already_present} already present; shortfall {stats.shortfall}"
    )
    return 0 if stats.shortfall == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
