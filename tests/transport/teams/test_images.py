"""Download-helper tests for Teams inline image attachments (T-067a).

Uses ``httpx.MockTransport`` to back a real ``httpx.AsyncClient`` with a
handler function, so ``download_inline_image``'s streaming/status/Content-Type
logic runs against genuine httpx machinery rather than a hand-rolled mock.
``MicrosoftAppCredentials.get_access_token`` is patched at the class level
(sync method, wrapped in ``asyncio.to_thread`` by the implementation).
"""

from unittest.mock import patch

import httpx
import pytest
from botframework.connector.auth import MicrosoftAppCredentials

from agent_runtime.transport.teams.images import (
    BotFrameworkCredentials,
    DownloadedImage,
    InlineImageDownloadError,
    download_inline_image,
)
from agent_runtime.transport.teams.testing import make_inline_image

_CREDENTIALS = BotFrameworkCredentials(app_id="aid", app_password="pwd", tenant_id="tid")
_JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["authorization"] == "Bearer test-token"
    return httpx.Response(200, content=_JPEG_BYTES, headers={"content-type": "image/jpeg"})


@pytest.fixture(autouse=True)
def _mock_token():
    with patch.object(
        MicrosoftAppCredentials, "get_access_token", return_value="test-token"
    ) as mock:
        yield mock


async def test_happy_path_returns_bytes_and_mime():
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_ok_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert isinstance(result, DownloadedImage)
    assert result.data == _JPEG_BYTES
    assert result.mime == "image/jpeg"


async def test_non_allowlisted_host_raises_without_any_http_call():
    def _boom_handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("HTTP call issued for a non-allowlisted host")

    att = make_inline_image(content_url="https://evil.example/steal")
    client = _client_with_handler(_boom_handler)
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_allowed_hosts_override_honored():
    att = make_inline_image(content_url="https://custom.internal.example/x")
    client = _client_with_handler(_ok_handler)
    result = await download_inline_image(
        att, _CREDENTIALS, client=client, allowed_hosts=frozenset({"custom.internal.example"})
    )
    assert result.data == _JPEG_BYTES


async def test_subdomain_of_allowlisted_host_is_allowed():
    att = make_inline_image(content_url="https://gov.smba.trafficmanager.net/x")
    client = _client_with_handler(_ok_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert result.data == _JPEG_BYTES


async def test_http_scheme_rejected():
    att = make_inline_image(content_url="http://smba.trafficmanager.net/insecure")
    client = _client_with_handler(lambda _r: pytest.fail("must not call http"))
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_oversize_body_raises_with_partial_discarded():
    def _big_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 100, headers={"content-type": "image/jpeg"})

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_big_handler)
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client, max_bytes=10)


async def test_non_image_content_type_raises():
    def _html_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_html_handler)
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_401_raises():
    def _unauthorized_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"")

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_unauthorized_handler)
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_404_raises():
    def _not_found_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"")

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_not_found_handler)
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_default_client_created_and_closed_when_not_injected():
    """When client is omitted, download_inline_image owns the client's lifecycle."""
    owned_client = _client_with_handler(_ok_handler)
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    with patch("agent_runtime.transport.teams.images.httpx.AsyncClient", return_value=owned_client):
        result = await download_inline_image(att, _CREDENTIALS)
    assert result.data == _JPEG_BYTES
    assert owned_client.is_closed


async def test_permission_error_from_token_acquisition_surfaces_as_download_error():
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(lambda _r: pytest.fail("must not call HTTP without a token"))
    with (
        patch.object(
            MicrosoftAppCredentials,
            "get_access_token",
            side_effect=PermissionError("token denied"),
        ),
        pytest.raises(InlineImageDownloadError),
    ):
        await download_inline_image(att, _CREDENTIALS, client=client)


def test_credentials_repr_excludes_password():
    assert "pwd" not in repr(_CREDENTIALS)
    assert "aid" in repr(_CREDENTIALS)


async def test_octet_stream_with_png_magic_accepted():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 16

    def _octet_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=png_bytes, headers={"content-type": "application/octet-stream"}
        )

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_octet_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert result.data == png_bytes
    assert result.mime == "image/png"


async def test_octet_stream_with_jpeg_magic_accepted():
    def _octet_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_JPEG_BYTES, headers={"content-type": "application/octet-stream"}
        )

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_octet_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert result.data == _JPEG_BYTES
    assert result.mime == "image/jpeg"


async def test_octet_stream_non_image_still_rejected():
    def _octet_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"%PDF-1.7 not an image at all",
            headers={"content-type": "application/octet-stream"},
        )

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_octet_handler)
    with pytest.raises(InlineImageDownloadError, match="non-image Content-Type"):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_declared_image_content_type_unchanged():
    """A declared image/* type is trusted as-is — no sniff, byte-identical path."""
    arbitrary_bytes = b"not-actually-a-jpeg-but-declared-as-one"

    def _declared_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=arbitrary_bytes, headers={"content-type": "image/jpeg"})

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_declared_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert result.data == arbitrary_bytes
    assert result.mime == "image/jpeg"


async def test_octet_stream_still_respects_size_cap():
    def _big_octet_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"x" * 100, headers={"content-type": "application/octet-stream"}
        )

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_big_octet_handler)
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client, max_bytes=10)


async def test_redirect_not_followed_off_allowlist():
    def _redirect_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "smba.trafficmanager.net":
            return httpx.Response(302, headers={"location": "https://evil.example/x"})
        pytest.fail("redirect was followed off the allowlisted host")

    att = make_inline_image(content_url="https://smba.trafficmanager.net/r")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_redirect_handler), follow_redirects=True
    )
    with pytest.raises(InlineImageDownloadError):
        await download_inline_image(att, _CREDENTIALS, client=client)
