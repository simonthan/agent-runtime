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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
from botframework.connector.auth import MicrosoftAppCredentials

if TYPE_CHECKING:
    from agent_runtime.transport.teams.events import InlineImageAttachment

# Bot Framework's public-cloud attachment host. The suffix rule in
# `_host_is_allowed` additionally admits subdomains under it (e.g. a future
# regional or gov-cloud host) without widening the check to arbitrary hosts.
_DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset({"smba.trafficmanager.net"})

_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # phone photos are typically 1-5 MiB

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
    acquisition failure, a non-200 HTTP response, an oversize body, or a
    non-image response Content-Type. No partial bytes are ever returned to
    the caller. The consumer decides retry/UX from the message.
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


async def _acquire_token(credentials: BotFrameworkCredentials) -> str:
    """Fetch a Bot Framework connector token, wrapping the blocking SDK call.

    ``MicrosoftAppCredentials.get_access_token`` is synchronous and raises
    ``PermissionError`` on failure (also catches any other SDK/MSAL exception
    broadly, since the underlying library exposes no single narrow type) —
    both surface here as ``InlineImageDownloadError``.
    """
    app_credentials = MicrosoftAppCredentials(
        credentials.app_id,
        credentials.app_password,
        channel_auth_tenant=credentials.tenant_id,
    )
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
    pool reuse; when omitted, a client is created and closed for this call.
    """
    hosts = allowed_hosts if allowed_hosts is not None else _DEFAULT_ALLOWED_HOSTS
    if not _host_is_allowed(att.content_url, hosts):
        host = urlsplit(att.content_url).hostname
        msg = f"Refusing to attach a connector token to a non-allowlisted host: {host!r}"
        raise InlineImageDownloadError(msg)

    token = await _acquire_token(credentials)
    headers = {"Authorization": f"Bearer {token}"}

    owns_client = client is None
    http_client = client if client is not None else httpx.AsyncClient()
    try:
        return await _stream_download(http_client, att.content_url, headers, max_bytes)
    finally:
        if owns_client:
            await http_client.aclose()
