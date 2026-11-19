from __future__ import annotations

import email.message
import ipaddress
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

from aidetect.errors import NetworkAudioError
from aidetect.url import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_AUDIO_BYTES,
    USER_AGENT,
    fetch_audio_bytes,
    validate_url,
)

WriteWav = Callable[[Path, float], Path]

PUBLIC_IP = "93.184.216.34"
AddressInfo = tuple[int, int, int, str, tuple[str, int]]
ResolverResult = list[AddressInfo]


def resolver_for(address: str) -> Callable[..., ResolverResult]:
    def resolve(host: str, port: int, family: int, kind: int) -> ResolverResult:
        del host, family, kind
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    return resolve


PUBLIC_RESOLVER = resolver_for(PUBLIC_IP)


def literal_or_public_resolver(host: str, port: int, family: int, kind: int) -> ResolverResult:
    del family, kind
    try:
        address = str(ipaddress.ip_address(host))
    except ValueError:
        address = PUBLIC_IP
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]


class _Response:
    """Minimal context-managed response with chunked reads."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = email.message.Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, size: int) -> bytes:
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    """Returns canned responses without any network; records every hop and timeout."""

    def __init__(self, responses: dict[str, bytes | str | Exception]) -> None:
        self.responses = responses
        self.urls: list[str] = []
        self.timeouts: list[float] = []
        self.user_agents: list[str | None] = []

    def open(self, request: urllib.request.Request, timeout: float) -> _Response:
        url = request.full_url
        self.urls.append(url)
        self.timeouts.append(timeout)
        self.user_agents.append(request.get_header("User-agent"))
        outcome = self.responses.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            headers = email.message.Message()
            headers["Location"] = outcome
            raise urllib.error.HTTPError(url, 302, "Found", headers, None)
        if outcome is None:
            raise AssertionError(f"unexpected URL {url}")
        return _Response(outcome)


def http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", email.message.Message(), None)


def tiny_wav_bytes(tmp_path: Path, write_wav: WriteWav) -> bytes:
    return write_wav(tmp_path / "tiny.wav", 0.2).read_bytes()


def test_validate_url_accepts_public_http_and_https() -> None:
    validate_url("http://example.com/a.mp3", resolver=PUBLIC_RESOLVER)
    validate_url("https://example.com/a.mp3", resolver=PUBLIC_RESOLVER)


@pytest.mark.parametrize(
    "url", ["ftp://example.com/a.mp3", "file:///etc/passwd", "example.com/a.mp3"]
)
def test_validate_url_rejects_non_http_schemes(url: str) -> None:
    with pytest.raises(NetworkAudioError, match="scheme"):
        validate_url(url, resolver=PUBLIC_RESOLVER)


def test_validate_url_rejects_missing_host() -> None:
    with pytest.raises(NetworkAudioError, match="host"):
        validate_url("https:///a.mp3", resolver=PUBLIC_RESOLVER)


def test_validate_url_rejects_invalid_port() -> None:
    with pytest.raises(NetworkAudioError, match="port"):
        validate_url("https://example.com:not-a-port/a.mp3", resolver=PUBLIC_RESOLVER)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.10",
        "169.254.1.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "224.0.0.1",
        "0.0.0.0",
        "240.0.0.1",
        "::ffff:127.0.0.1",
    ],
)
def test_validate_url_rejects_private_and_special_addresses(address: str) -> None:
    with pytest.raises(NetworkAudioError, match="address"):
        validate_url("https://example.com/a.mp3", resolver=resolver_for(address))


def test_validate_url_rejects_multiple_dns_results_when_one_is_private() -> None:
    def mixed(host: str, port: int, family: int, kind: int) -> ResolverResult:
        del host, family, kind
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    with pytest.raises(NetworkAudioError, match="address"):
        validate_url("https://example.com/a.mp3", resolver=mixed)


def test_validate_url_rejects_dns_failure_and_empty_result() -> None:
    def failing(host: str, port: int, family: int, kind: int) -> ResolverResult:
        del host, port, family, kind
        raise OSError("dns down")

    def empty(host: str, port: int, family: int, kind: int) -> ResolverResult:
        del host, port, family, kind
        return []

    with pytest.raises(NetworkAudioError, match="resolve"):
        validate_url("https://example.com/a.mp3", resolver=failing)
    with pytest.raises(NetworkAudioError, match="DNS"):
        validate_url("https://example.com/a.mp3", resolver=empty)


def test_fetch_audio_bytes_returns_validated_audio(tmp_path: Path, write_wav: WriteWav) -> None:
    payload = tiny_wav_bytes(tmp_path, write_wav)
    opener = FakeOpener({"https://example.com/a.mp3": payload})

    fetched = fetch_audio_bytes(
        "https://example.com/a.mp3", opener=opener, resolver=PUBLIC_RESOLVER
    )
    assert fetched == payload
    assert opener.urls == ["https://example.com/a.mp3"]
    assert opener.timeouts == [CONNECT_TIMEOUT_SECONDS]
    assert opener.user_agents == [USER_AGENT]


def test_fetch_audio_bytes_rejects_non_audio_payload() -> None:
    opener = FakeOpener({"https://example.com/a.mp3": b"<html>not audio</html>"})
    with pytest.raises(NetworkAudioError, match="audio"):
        fetch_audio_bytes("https://example.com/a.mp3", opener=opener, resolver=PUBLIC_RESOLVER)


def test_fetch_audio_bytes_caps_body_size(tmp_path: Path, write_wav: WriteWav) -> None:
    payload = tiny_wav_bytes(tmp_path, write_wav)
    opener = FakeOpener({"https://example.com/a.mp3": payload})
    with pytest.raises(NetworkAudioError, match="byte cap"):
        fetch_audio_bytes(
            "https://example.com/a.mp3",
            opener=opener,
            resolver=PUBLIC_RESOLVER,
            max_bytes=len(payload) - 1,
        )


def test_fetch_audio_bytes_caps_content_length_before_reading() -> None:
    response = _Response(b"whatever", {"Content-Length": str(MAX_AUDIO_BYTES + 1)})

    class LengthOpener(FakeOpener):
        def open(self, request: urllib.request.Request, timeout: float) -> _Response:
            self.urls.append(request.full_url)
            self.timeouts.append(timeout)
            return response

    with pytest.raises(NetworkAudioError, match="byte cap"):
        fetch_audio_bytes(
            "https://example.com/a.mp3", opener=LengthOpener({}), resolver=PUBLIC_RESOLVER
        )


def test_fetch_audio_bytes_validates_redirect_targets() -> None:
    opener = FakeOpener({"https://example.com/a.mp3": "http://127.0.0.1/a.mp3"})
    with pytest.raises(NetworkAudioError, match="address"):
        fetch_audio_bytes(
            "https://example.com/a.mp3", opener=opener, resolver=literal_or_public_resolver
        )
    assert opener.urls == ["https://example.com/a.mp3"]


def test_fetch_audio_bytes_resolves_relative_redirects(tmp_path: Path, write_wav: WriteWav) -> None:
    payload = tiny_wav_bytes(tmp_path, write_wav)
    opener = FakeOpener(
        {
            "https://example.com/feed/a.mp3": "/audio/b.mp3",
            "https://example.com/audio/b.mp3": payload,
        }
    )
    assert (
        fetch_audio_bytes("https://example.com/feed/a.mp3", opener=opener, resolver=PUBLIC_RESOLVER)
        == payload
    )
    assert opener.urls == ["https://example.com/feed/a.mp3", "https://example.com/audio/b.mp3"]


def test_fetch_audio_bytes_caps_redirects() -> None:
    url = "https://example.com/a.mp3"
    opener = FakeOpener({url: "/next.mp3", "https://example.com/next.mp3": "/next.mp3"})
    with pytest.raises(NetworkAudioError, match="redirect"):
        fetch_audio_bytes(url, opener=opener, resolver=PUBLIC_RESOLVER, max_redirects=2)
    assert len(opener.urls) == 3


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (TimeoutError("slow"), "Timed out"),
        (urllib.error.URLError("offline"), "fetch"),
        (http_error("https://example.com/a.mp3", 404), "404"),
    ],
)
def test_fetch_audio_bytes_wraps_network_failures(error: Exception, message: str) -> None:
    opener = FakeOpener({"https://example.com/a.mp3": error})
    with pytest.raises(NetworkAudioError, match=message):
        fetch_audio_bytes("https://example.com/a.mp3", opener=opener, resolver=PUBLIC_RESOLVER)


def test_resolver_receives_explicit_port() -> None:
    ports: list[int] = []

    def resolver(host: str, port: int, family: int, kind: int) -> ResolverResult:
        del host, family, kind
        ports.append(port)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port))]

    payload = b"RIFF____WAVE"
    opener = FakeOpener({"http://example.com:8080/a.mp3": payload})
    fetch_audio_bytes("http://example.com:8080/a.mp3", opener=opener, resolver=resolver)
    assert ports == [8080]
    assert ipaddress.ip_address(PUBLIC_IP).is_global
