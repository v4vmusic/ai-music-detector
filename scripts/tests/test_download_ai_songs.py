from __future__ import annotations

import csv
import datetime as dt
import email.message
import ipaddress
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

import pytest

from scripts.download_ai_songs import (
    MAX_BACKOFF_SECONDS,
    AddressInfo,
    Config,
    DownloadError,
    HttpError,
    SkipSong,
    Song,
    UnsafeUrlError,
    append_row,
    audio_suffix,
    build_download_opener,
    download_with_retries,
    fetch_bytes,
    parse_retry_after,
    read_staging_rows,
    run,
    select_candidates,
    validate_url,
)

PUBLIC_IP = "93.184.216.34"


def location_headers(location: str) -> email.message.Message:
    headers = email.message.Message()
    headers["Location"] = location
    return headers


def fake_resolver(host: str, port: int, family: int, socktype: int) -> list[AddressInfo]:
    """Resolve literals to themselves and every hostname to a public address."""
    try:
        address = str(ipaddress.ip_address(host))
    except ValueError:
        address = PUBLIC_IP
    return [(2, 1, 6, "", (address, port or 80))]


class FakeResponse:
    """Minimal context-manager response that yields queued chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""


class FakeOpener:
    """Queue responses/exceptions and record requested URLs."""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[str] = []

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
        self.calls.append(request.full_url)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return cast(FakeResponse, result)


def make_song(index: int) -> Song:
    return Song(
        song_id=f"s{index}",
        mp3_url=f"http://public.example/{index}.mp3",
        artist_name=f"Artist {index}",
        song_title=f"Title {index}",
        feed_title=f"Feed {index}",
    )


def write_csv(path: Path, header: str, body: str = "") -> None:
    path.write_text(header + "\n" + body, encoding="utf-8")


def test_select_candidates_is_seeded_and_capped() -> None:
    songs = [make_song(i) for i in range(500)]
    first = select_candidates(songs, 42, 300)
    second = select_candidates(songs, 42, 300)
    assert first == second
    assert len(first) == 300
    assert len({song.song_id for song in first}) == 300


def test_download_opener_does_not_follow_redirects_automatically() -> None:
    opener = build_download_opener()
    handlers = cast(list[object], getattr(opener, "handlers", []))
    assert not any(isinstance(handler, urllib.request.HTTPRedirectHandler) for handler in handlers)


def test_validate_url_allows_public_http_sources() -> None:
    validate_url("http://public.example/a.mp3", resolver=fake_resolver)
    validate_url("https://public.example:8443/a.mp3", resolver=fake_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/a.mp3",
        "http://10.0.0.1/a.mp3",
        "http://169.254.1.1/a.mp3",
        "http://public.example:not-a-port/a.mp3",
    ],
)
def test_validate_url_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        validate_url(url, resolver=fake_resolver)


def test_validate_url_rejects_unresolvable_host() -> None:
    def no_results(host: str, port: int, family: int, socktype: int) -> list[AddressInfo]:
        return []

    with pytest.raises(UnsafeUrlError):
        validate_url("http://missing.example/a.mp3", resolver=no_results)


def test_fetch_bytes_follows_validated_redirects() -> None:
    redirect = urllib.error.HTTPError(
        "http://public.example/a.mp3", 302, "Found", location_headers("/b.mp3"), None
    )
    opener = FakeOpener(redirect, FakeResponse([b"audio"]))
    body = fetch_bytes("http://public.example/a.mp3", opener=opener, resolver=fake_resolver)
    assert body == b"audio"
    assert opener.calls == ["http://public.example/a.mp3", "http://public.example/b.mp3"]


def test_fetch_bytes_caps_redirects() -> None:
    redirects = [
        urllib.error.HTTPError(
            "http://public.example/a.mp3", 302, "Found", location_headers(f"/{i}.mp3"), None
        )
        for i in range(6)
    ]
    opener = FakeOpener(*redirects)
    with pytest.raises(DownloadError, match="redirects"):
        fetch_bytes(
            "http://public.example/a.mp3",
            opener=opener,
            resolver=fake_resolver,
            max_redirects=2,
        )
    assert len(opener.calls) == 3


def test_fetch_bytes_rejects_unsafe_redirect_target() -> None:
    redirect = urllib.error.HTTPError(
        "http://public.example/a.mp3",
        302,
        "Found",
        location_headers("http://127.0.0.1/x.mp3"),
        None,
    )
    opener = FakeOpener(redirect)
    with pytest.raises(UnsafeUrlError):
        fetch_bytes("http://public.example/a.mp3", opener=opener, resolver=fake_resolver)


def test_fetch_bytes_caps_download_size() -> None:
    opener = FakeOpener(FakeResponse([b"aaa", b"bbb"]))
    with pytest.raises(DownloadError, match="byte cap"):
        fetch_bytes(
            "http://public.example/a.mp3",
            opener=opener,
            resolver=fake_resolver,
            max_bytes=4,
        )


def test_parse_retry_after_handles_seconds_date_and_cap() -> None:
    assert parse_retry_after("7", now=lambda: 0.0) == 7.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("nonsense") is None
    assert parse_retry_after("9999", now=lambda: 0.0) == MAX_BACKOFF_SECONDS
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    header = "Thu, 01 Jan 2026 00:00:02 GMT"
    assert parse_retry_after(header, now=lambda: now) == 2.0


def test_download_with_retries_honours_retry_after() -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def fetch(url: str) -> bytes:
        attempts.append(url)
        if len(attempts) == 1:
            raise HttpError(429, 7.0)
        return b"ok"

    assert download_with_retries("u", fetch=fetch, sleep=sleeps.append) == b"ok"
    assert sleeps == [7.0]


def test_download_with_retries_uses_exponential_backoff() -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def fetch(url: str) -> bytes:
        attempts.append(url)
        if len(attempts) < 3:
            raise HttpError(503)
        return b"ok"

    assert download_with_retries("u", fetch=fetch, sleep=sleeps.append) == b"ok"
    assert sleeps == [2.0, 4.0]


def test_download_with_retries_skips_404_without_sleep() -> None:
    sleeps: list[float] = []

    def fetch(url: str) -> bytes:
        raise HttpError(404)

    with pytest.raises(SkipSong):
        download_with_retries("u", fetch=fetch, sleep=sleeps.append)
    assert sleeps == []


def test_download_with_retries_stops_after_max_retries() -> None:
    sleeps: list[float] = []

    def fetch(url: str) -> bytes:
        raise HttpError(429, 1.0)

    with pytest.raises(SkipSong):
        download_with_retries("u", fetch=fetch, sleep=sleeps.append, max_retries=3)
    assert len(sleeps) == 3


def test_audio_suffix_accepts_audio_and_rejects_other_files() -> None:
    assert audio_suffix("https://host/path/TRACK.MP3") == ".mp3"
    with pytest.raises(SkipSong):
        audio_suffix("https://host/path/track.exe")


def test_read_staging_rows_collects_existing_ai_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "test-songs.csv"
    write_csv(
        csv_path,
        "path,label,artist_name,feed_title,song_title",
        "human-songs/h.mp3,human,A,F,S\nai-songs/existing.mp3,ai_generated,A,F,S\n",
    )
    staging = read_staging_rows(csv_path)
    assert staging.ai_song_ids == {"existing"}
    assert len(staging.rows) == 2


def test_append_row_preserves_extra_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "test-songs.csv"
    write_csv(csv_path, "path,label,note", "human.mp3,human,ok")
    staging = read_staging_rows(csv_path)
    append_row(csv_path, staging.fieldnames, make_song(0), "ai-songs/s0.mp3")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["path"] == "ai-songs/s0.mp3"
    assert rows[-1]["label"] == "ai_generated"
    assert rows[-1]["note"] == ""


def test_run_appends_verified_rows_and_is_resumable(tmp_path: Path) -> None:
    csv_path = tmp_path / "test-songs.csv"
    write_csv(
        csv_path,
        "path,label,artist_name,feed_title,song_title",
        "human-songs/h1.mp3,human,Human,Feed,Song",
    )
    config = Config(
        csv_path=csv_path,
        out_dir=tmp_path / "ai-songs",
        error_log=tmp_path / "errors.csv",
        pool_size=3,
        target=3,
        delay=0.0,
    )
    songs = [make_song(i) for i in range(3)]
    sleeps: list[float] = []

    def fetch(url: str) -> bytes:
        return b"audio"

    def probe(path: Path) -> float:
        return 2.0 if "s1" in path.name else 100.0

    stats = run(config, songs, fetch=fetch, sleep=sleeps.append, probe=probe)
    assert stats.succeeded == 2
    assert stats.failed == 1
    assert stats.shortfall == 1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ai_paths = {row["path"] for row in rows if row["label"] == "ai_generated"}
    assert ai_paths == {"ai-songs/s0.mp3", "ai-songs/s2.mp3"}
    assert rows[0]["label"] == "human"
    assert not (config.out_dir / "s1.mp3").exists()
    assert not (config.out_dir / "s1.mp3.part").exists()

    with config.error_log.open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 1
    assert failures[0]["song_id"] == "s1"

    again = run(config, songs, fetch=fetch, sleep=sleeps.append, probe=probe)
    assert again.succeeded == 0
    assert again.already_present == 2
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
