from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, Final, Protocol, TypeAlias

from aidetect.audio import is_audio_payload
from aidetect.errors import NetworkAudioError

__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "MAX_AUDIO_BYTES",
    "MAX_REDIRECTS",
    "USER_AGENT",
    "AudioFetcher",
    "fetch_audio_bytes",
    "validate_url",
]

ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
MAX_REDIRECTS: Final = 5
CONNECT_TIMEOUT_SECONDS: Final = 15.0
MAX_AUDIO_BYTES: Final = 80 * 1024 * 1024
CHUNK_BYTES: Final = 64 * 1024
USER_AGENT: Final = "aidetect/0.1"

# Any: getaddrinfo's overloads return several address-tuple shapes; only the IP string is read.
Resolver: TypeAlias = Callable[..., Any]
AudioFetcher: TypeAlias = Callable[[str], bytes]


class _ReadableResponse(Protocol):
    def read(self, size: int) -> bytes: ...


def _reject_special_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, host: str) -> None:
    mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
    candidates = (ip, mapped) if mapped is not None else (ip,)
    for candidate in candidates:
        if (
            candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_reserved
            or candidate.is_multicast
            or candidate.is_unspecified
        ):
            raise NetworkAudioError(f"Refusing private/special address {candidate} for {host}")


def validate_url(url: str, *, resolver: Resolver = socket.getaddrinfo) -> None:
    """Reject non-http(s) schemes and any resolved private/special address (CC §8)."""

    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise NetworkAudioError(f"Unsupported URL scheme {scheme!r}; expected http or https")
    host = parsed.hostname
    if not host:
        raise NetworkAudioError("URL has no host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise NetworkAudioError(f"Invalid URL port: {exc}") from exc
    try:
        infos = resolver(host, port or (443 if scheme == "https" else 80), 0, socket.SOCK_STREAM)
    except OSError as exc:
        raise NetworkAudioError(f"Cannot resolve {host}: {exc}") from exc
    if not infos:
        raise NetworkAudioError(f"No DNS result for {host}")
    for info in infos:
        address = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise NetworkAudioError(f"Invalid resolved address for {host}: {address}") from exc
        _reject_special_address(ip, host)


def _build_opener() -> urllib.request.OpenerDirector:
    """Build an opener with no redirect handler so every hop is validated.

    Proxies are disabled: a proxy would connect somewhere other than the validated host.
    """

    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.ProxyHandler({}))
    opener.add_handler(urllib.request.UnknownHandler())
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def _reject_declared_oversize(response: _ReadableResponse, max_bytes: int) -> None:
    # Any: urllib exposes response.headers as an email.message.Message with a .get method.
    headers: Any = getattr(response, "headers", None)
    declared = headers.get("Content-Length") if headers is not None else None
    if not declared:
        return
    try:
        length = int(declared)
    except (TypeError, ValueError):
        return  # A malformed header is not trusted; the stream cap still applies.
    if length > max_bytes:
        raise NetworkAudioError(f"Response exceeds the {max_bytes} byte cap (Content-Length)")


def _read_capped(response: _ReadableResponse, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise NetworkAudioError(f"Response exceeds the {max_bytes} byte cap")
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_audio_bytes(
    url: str,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    resolver: Resolver = socket.getaddrinfo,
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int = MAX_AUDIO_BYTES,
    timeout: float = CONNECT_TIMEOUT_SECONDS,
) -> bytes:
    """Fetch remote audio with scheme/address checks, redirect and size caps and a timeout."""

    if max_redirects < 0:
        raise ValueError("max_redirects must be non-negative")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    active_opener = opener if opener is not None else _build_opener()
    current = url
    for _ in range(max_redirects + 1):
        validate_url(current, resolver=resolver)
        request = urllib.request.Request(current, headers={"User-Agent": USER_AGENT})
        try:
            with active_opener.open(request, timeout=timeout) as response:
                _reject_declared_oversize(response, max_bytes)
                data = _read_capped(response, max_bytes)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                location = exc.headers.get("Location") if exc.headers else None
                if not location:
                    raise NetworkAudioError(f"Redirect {exc.code} without Location") from exc
                current = urllib.parse.urljoin(current, location)
                continue
            raise NetworkAudioError(f"HTTP {exc.code} fetching {current}") from exc
        except TimeoutError as exc:
            raise NetworkAudioError(f"Timed out fetching {current}: {exc}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise NetworkAudioError(f"Cannot fetch {current}: {exc}") from exc
        if not is_audio_payload(data):
            raise NetworkAudioError(f"Downloaded bytes from {current} are not recognized audio")
        return data
    raise NetworkAudioError(f"Exceeded {max_redirects} redirects fetching {url}")
