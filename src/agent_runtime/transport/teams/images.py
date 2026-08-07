"""Authenticated download helper for Teams inline image attachments.

Inline images (camera captures, pasted/shared photos) live on the Bot
Framework attachment store and require a Bot Framework connector token to
read back — unlike ``FileAttachment``, whose ``download_url`` is
pre-authenticated. This module owns the ONLY place a connector token is
attached to an outbound request built from model-external input
(``InlineImageAttachment.content_url`` rides the inbound activity payload),
so the host allowlist below is a security boundary, not a convenience check.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from agent_runtime.transport.teams._msal import BoundedAppCredentials

if TYPE_CHECKING:
    from agent_runtime.transport.teams.events import InlineImageAttachment

# Bot Framework's public-cloud attachment host. The suffix rule in
# `_host_is_allowed` additionally admits subdomains under it (e.g. a future
# regional or gov-cloud host) without widening the check to arbitrary hosts.
_DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset({"smba.trafficmanager.net"})

_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # phone photos are typically 1-5 MiB

# T-115l -- explicit timeouts for the client this module OWNS. A bare
# `httpx.AsyncClient()` gets httpx's `DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=5.0)`
# (httpx 0.28.1, `_config.py:246`) -- 5 s on connect, read, write AND pool. httpx's
# `read` is re-armed for every 64 KiB socket read and for the response headers
# (httpcore `_async/http11.py:172-176`, `:195-206`, `READ_NUM_BYTES` `:44`), so 5 s was
# at once TOO TIGHT for time-to-first-byte on a multi-MiB attachment (the download
# fails fast having transferred nothing) and NO BOUND at all on the transfer as a
# whole (~160 reads at the 10 MiB cap => ~13 min of legal stalling).
#
# Only the owned client is configured. A caller who injects a client -- documented as
# being for connection-pool reuse -- owns its per-phase settings; the wall-clock
# deadline below is what applies to every caller.
_DOWNLOAD_TIMEOUT = httpx.Timeout(10.0, read=15.0)

# The bound that makes raising `read` safe, and the ONLY total-duration limit httpx can
# express (it has no whole-request timeout). Without it, `read=15` would push the
# pathological-stall worst case from ~13 min to ~40 min -- on a path that sits outside
# every deadline the Teams consumer has: it downloads BEFORE the turn starts, inline
# inside the Bot Framework inbound-activity webhook request, which the Connector retries
# after ~15 s. So this is calibrated as low as it defensibly goes, not as high as an
# image might want: 30 s clears the 10 MiB cap at ~340 KB/s and a typical 1-5 MiB photo
# at ~170 KB/s. `read` is kept strictly BELOW it so a stalled socket still raises the
# more diagnostic ReadTimeout rather than an anonymous deadline expiry.
# Read from the module global at call time so tests can patch it.
_DOWNLOAD_DEADLINE_SECONDS = 30.0

_MAGIC_SNIFFS: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_image_mime(data: bytes) -> str | None:
    """Return the image MIME type from magic bytes, or None if not a known image.

    Teams' attachment CDN serves inline images as ``application/octet-stream``
    (TBP T-084 Issue 4 — live receipt rejected), so the declared Content-Type
    cannot be trusted to say "not an image". The four types here are exactly
    Anthropic's supported image media types. WebP is RIFF-framed
    (``RIFF<size>WEBP``), hence the offset check.
    """
    for magic, mime in _MAGIC_SNIFFS:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":  # noqa: PLR2004
        return "image/webp"
    return None


class InlineImageDownloadError(Exception):
    """An inline image could not be downloaded or failed validation.

    Raised for every failure mode: a non-allowlisted host, connector-token
    acquisition failure, a non-200 HTTP response, an oversize body, a
    non-image response Content-Type, a transport error, or the wall-clock
    download deadline (T-115l). No partial bytes are ever returned to the
    caller. The consumer decides retry/UX from the message -- catching this
    one type is sufficient to degrade gracefully, which is why the httpx
    exceptions are converted rather than allowed to escape.
    ``asyncio.CancelledError`` is NOT converted and propagates untouched.
    """


@dataclass(frozen=True, slots=True)
class BotFrameworkCredentials:
    """Bot Framework app credentials used to mint the connector token.

    Bundled into one value (rather than three loose keyword params) to keep
    ``download_inline_image``'s signature under the project's max-arguments
    lint threshold.
    """

    app_id: str
    app_password: str = field(repr=False)  # secret — keep out of repr/tracebacks
    tenant_id: str


@dataclass(frozen=True, slots=True)
class DownloadedImage:
    """The downloaded bytes and the response-declared mime type."""

    data: bytes
    mime: str  # from the response Content-Type, e.g. "image/jpeg"


def _host_is_allowed(url: str, allowed_hosts: frozenset[str]) -> bool:
    """True if ``url`` is ``https`` and its host is in, or a subdomain of, ``allowed_hosts``."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in allowed_hosts)


# T-115m -- credentials live for the process, keyed by the (frozen, hashable) value.
# The SDK builds its MSAL app lazily INSIDE the credentials object and discards it with
# them, so the previous fresh-per-call construction meant MSAL's token cache ALWAYS
# missed: every inline image paid a full discovery GET + token POST, up to 3 a turn. A
# warm entry now serves subsequent images and turns with no network at all until the
# ~1 h expiry. Keying on the credentials means a secret rotation lands on a new key --
# no invalidation logic, no stale-credential window. The cached object is shared across
# worker threads, which is safe: MSAL guards its token cache with an RLock and documents
# the app as a process-lifetime singleton.
_credentials_cache: dict[BotFrameworkCredentials, BoundedAppCredentials] = {}
_cache_lock = threading.Lock()


def _cached_credentials(credentials: BotFrameworkCredentials) -> BoundedAppCredentials:
    """Return the process-wide ``BoundedAppCredentials`` for ``credentials``.

    Construction is network-free (MSAL is built lazily on first token request), so
    this is safe to call from anywhere; the lock only serialises the miss path.
    """
    cached = _credentials_cache.get(credentials)
    if cached is not None:
        return cached
    with _cache_lock:
        cached = _credentials_cache.get(credentials)
        if cached is None:
            cached = BoundedAppCredentials(
                credentials.app_id,
                credentials.app_password,
                channel_auth_tenant=credentials.tenant_id,
            )
            _credentials_cache[credentials] = cached
        return cached


async def _acquire_token(credentials: BotFrameworkCredentials) -> str:
    """Fetch a Bot Framework connector token, wrapping the blocking SDK call.

    ``get_access_token`` is synchronous and raises ``PermissionError`` on failure
    (any other SDK/MSAL exception is also caught broadly, since the underlying
    library exposes no single narrow type) — both surface here as
    ``InlineImageDownloadError``.

    The credentials carry a bounded MSAL app (T-115m), and the whole call runs in
    one ``to_thread``: the MSAL constructor does network I/O on first use, so
    building it on the event loop would block the process on an unbounded GET.
    """
    app_credentials = _cached_credentials(credentials)
    try:
        return await asyncio.to_thread(app_credentials.get_access_token)
    except Exception as exc:
        msg = f"Failed to acquire Bot Framework connector token: {exc}"
        raise InlineImageDownloadError(msg) from exc


async def _stream_download(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    max_bytes: int,
) -> DownloadedImage:
    """Stream the GET response, enforcing the size cap and image Content-Type."""
    # follow_redirects pinned False even if an injected client enables it: a
    # redirect must never carry the fetch off the allowlisted host.
    async with client.stream("GET", url, headers=headers, follow_redirects=False) as response:
        if response.status_code != httpx.codes.OK:
            status = response.status_code
            msg = f"Inline image download failed with HTTP status {status}"
            raise InlineImageDownloadError(msg)
        content_type = response.headers.get("content-type", "")
        declared_image = content_type.startswith("image/")
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                msg = f"Inline image exceeded the {max_bytes}-byte download cap"
                raise InlineImageDownloadError(msg)
            chunks.append(chunk)
        data = b"".join(chunks)
        if declared_image:
            return DownloadedImage(data=data, mime=content_type)
        # T-084b: a non-image Content-Type no longer rejects up front — Teams'
        # CDN serves real images as application/octet-stream. The magic bytes
        # decide; the max_bytes cap above still bounds the speculative read,
        # and a genuinely non-image payload still raises the same error.
        sniffed = _sniff_image_mime(data)
        if sniffed is None:
            msg = f"Inline image download returned a non-image Content-Type: {content_type!r}"
            raise InlineImageDownloadError(msg)
        return DownloadedImage(data=data, mime=sniffed)


async def download_inline_image(
    att: InlineImageAttachment,
    credentials: BotFrameworkCredentials,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    allowed_hosts: frozenset[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> DownloadedImage:
    """Download an inline image from the Bot Framework attachment store.

    Refuses to attach a connector token unless ``att.content_url`` is
    ``https`` and its host is allowlisted (``allowed_hosts`` overrides the
    module default — a consumer may widen it from the session's own
    ``ConversationRef.service_url`` host); this check runs before any token
    acquisition or HTTP call. The response is streamed with a hard
    ``max_bytes`` cap. A declared ``Content-Type`` starting with ``"image/"``
    is trusted as-is (byte-identical path). Any other declared type (Teams'
    CDN serves real images as ``application/octet-stream`` — T-084b) is still
    streamed under the same ``max_bytes`` cap, then sniffed by magic bytes
    (PNG/JPEG/GIF/WebP); ``DownloadedImage.mime`` carries the sniffed type in
    that case. Oversize responses, or non-image responses whose bytes don't
    match a known image signature, raise with the partial bytes discarded.

    ``client`` is an injectable ``httpx.AsyncClient`` for tests / connection
    pool reuse; when omitted, a client is created and closed for this call,
    carrying ``_DOWNLOAD_TIMEOUT``. An injected client keeps its own per-phase
    timeouts, but the ``_DOWNLOAD_DEADLINE_SECONDS`` wall-clock bound applies
    either way -- like ``max_bytes``, it is a limit this module owns for every
    caller. Timeouts and transport errors surface as
    ``InlineImageDownloadError`` (T-115l), so one ``except`` clause is enough
    for a consumer to skip a failed image instead of failing the whole turn.
    """
    hosts = allowed_hosts if allowed_hosts is not None else _DEFAULT_ALLOWED_HOSTS
    if not _host_is_allowed(att.content_url, hosts):
        host = urlsplit(att.content_url).hostname
        msg = f"Refusing to attach a connector token to a non-allowlisted host: {host!r}"
        raise InlineImageDownloadError(msg)

    token = await _acquire_token(credentials)
    headers = {"Authorization": f"Bearer {token}"}

    owns_client = client is None
    http_client = client if client is not None else httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT)
    try:
        async with asyncio.timeout(_DOWNLOAD_DEADLINE_SECONDS):
            result = await _stream_download(http_client, att.content_url, headers, max_bytes)
    except TimeoutError as exc:
        # Our own deadline. httpx's timeouts derive from httpx.HTTPError, not from
        # builtins.TimeoutError, so this handler is unambiguous. CancelledError is a
        # BaseException and matches neither handler -- it propagates (T-089/T-090).
        msg = f"Inline image download exceeded the {_DOWNLOAD_DEADLINE_SECONDS}-second deadline"
        raise InlineImageDownloadError(msg) from exc
    except httpx.HTTPError as exc:
        # The whole transport family (ConnectTimeout/ReadTimeout/ConnectError/...). Before
        # T-115l these escaped raw, bypassing consumers that catch InlineImageDownloadError
        # only -- so one slow image failed the entire turn. Type name only, never str(exc):
        # httpx text embeds connection details (connectors/base.py:177, SEC-2/SEC-3). The
        # original is chained, so exc_info=True logging still gets everything.
        msg = f"Inline image download failed at the transport layer: {type(exc).__name__}"
        raise InlineImageDownloadError(msg) from exc
    else:
        # `return` lives in `else`, not in `try`: with handlers present, a return in the
        # try body trips ruff TRY300 (select = ["ALL"]). `finally` still runs first.
        return result
    finally:
        if owns_client:
            await http_client.aclose()
