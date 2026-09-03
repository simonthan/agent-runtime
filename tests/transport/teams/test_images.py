"""Download-helper tests for Teams inline image attachments (T-067a).

Uses ``httpx.MockTransport`` to back a real ``httpx.AsyncClient`` with a
handler function, so ``download_inline_image``'s streaming/status/Content-Type
logic runs against genuine httpx machinery rather than a hand-rolled mock.
``MicrosoftAppCredentials.get_access_token`` is patched at the class level
(sync method, wrapped in ``asyncio.to_thread`` by the implementation).
"""

import asyncio
import threading
from unittest.mock import patch

import httpx
import pytest
from botframework.connector.auth import MicrosoftAppCredentials

from agent_runtime.transport.teams import _msal as _msal_module
from agent_runtime.transport.teams import images
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
    """Patch the token call AND the MSAL app construction (T-115m).

    `_build_msal_app` must be patched or every test in this file makes a real
    tenant-discovery GET to login.microsoftonline.com — MSAL's constructor does
    network I/O (`authority.py:96-99`). Patching the PARENT's `get_access_token`
    still works: `BoundedAppCredentials` overrides it but delegates via `super()`,
    which resolves through the MRO to this patch. The process-wide credentials
    cache is cleared around each test so cache assertions stay independent.
    """
    images._credentials_cache.clear()
    with (
        patch.object(
            MicrosoftAppCredentials, "get_access_token", return_value="test-token"
        ) as mock,
        patch.object(_msal_module, "_build_msal_app", return_value=object()),
    ):
        yield mock
    images._credentials_cache.clear()


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
    with pytest.raises(InlineImageDownloadError, match="non-image payload"):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_declared_image_type_over_unsniffable_bytes_is_rejected():
    """T-134 (replaces `test_declared_image_content_type_unchanged`, which asserted the
    bug): a declared `image/png` over HEIC or other bytes used to sail through as a valid
    `LLMImage` and 400 at the Anthropic API mid-turn, failing the whole turn. It must fail
    HERE, as an ordinary skip-one-image download error."""
    arbitrary_bytes = b"not-actually-a-jpeg-but-declared-as-one"

    def _declared_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=arbitrary_bytes, headers={"content-type": "image/jpeg"})

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_declared_handler)
    with pytest.raises(InlineImageDownloadError, match="non-image payload"):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_content_type_parameters_do_not_lose_a_valid_image():
    """T-134: `image/jpeg; charset=utf-8` is a legal header. Returned verbatim it did not
    equal `"image/jpeg"`, so `LLMImage.__post_init__` raised and the consumer dropped a
    perfectly good photo. The sniffed type replaces the header outright."""

    def _param_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_JPEG_BYTES, headers={"content-type": "image/jpeg; charset=utf-8"}
        )

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_param_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert result.mime == "image/jpeg"
    assert result.data == _JPEG_BYTES


async def test_mislabelled_image_gets_the_sniffed_type_not_the_declared_one():
    """T-134: Teams mislabels. Declared `image/png` over real JPEG bytes previously became
    an `LLMImage(media_type="image/png")` carrying JPEG data — an API 400. The bytes win,
    and the image survives."""

    def _mislabel_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_JPEG_BYTES, headers={"content-type": "image/png"})

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_mislabel_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
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


async def test_owned_client_is_built_with_explicit_timeouts():
    """Regression on the bug itself: the owned client used to be a bare
    `httpx.AsyncClient()`, i.e. httpx's Timeout(5.0) on every phase -- including the
    `read` that also covers time-to-first-byte for a multi-MiB attachment."""
    owned = _client_with_handler(_ok_handler)
    captured: dict[str, object] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return owned

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    with patch("agent_runtime.transport.teams.images.httpx.AsyncClient", _factory):
        result = await download_inline_image(att, _CREDENTIALS)

    assert result.data == _JPEG_BYTES
    timeout = captured["timeout"]
    assert timeout == httpx.Timeout(10.0, read=15.0)
    assert timeout != httpx.Timeout(5.0)  # the pre-T-115l default
    assert timeout.read == 15.0
    assert timeout.connect == 10.0
    # `read` must stay strictly below the deadline, or a stalled socket loses the more
    # diagnostic ReadTimeout to an anonymous deadline expiry (Design 1).
    assert timeout.read < images._DOWNLOAD_DEADLINE_SECONDS
    assert owned.is_closed  # ownership/close-out unchanged


async def test_injected_client_keeps_its_own_per_phase_timeouts():
    """The boundary Design 2 / Open Q4 / the docstring / the CHANGELOG all commit to:
    `_DOWNLOAD_TIMEOUT` configures the client this module OWNS, never an injected one --
    a consumer injecting a pooled client keeps its per-phase settings. (The wall-clock
    deadline still applies to them; that is a module-owned limit like `max_bytes`.)"""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_ok_handler), timeout=httpx.Timeout(1.0)
    )
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    result = await download_inline_image(att, _CREDENTIALS, client=client)

    assert result.data == _JPEG_BYTES
    assert client.timeout == httpx.Timeout(1.0)  # untouched
    assert not client.is_closed  # injected client is NOT closed by the callee
    await client.aclose()


async def test_transport_error_surfaces_as_download_error():
    """A raw httpx error used to escape. The only production consumer
    (teams-bot-platform dispatcher.py:963) catches InlineImageDownloadError ONLY, and
    nothing between it and the Bot Framework adapter has a try/except -- so a single
    slow image failed the entire turn instead of being skipped."""

    def _timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated read timeout")

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_timeout_handler)
    with pytest.raises(InlineImageDownloadError, match="transport layer") as excinfo:
        await download_inline_image(att, _CREDENTIALS, client=client)
    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)  # original chained
    assert "simulated read timeout" not in str(excinfo.value)  # SEC-2/3: no httpx text


async def test_connect_error_also_surfaces_as_download_error():
    """httpx.HTTPError is caught as the BASE class, so the whole transport family is
    covered by construction -- not just the timeout subtree."""

    def _connect_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connect failure")

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_connect_handler)
    with pytest.raises(InlineImageDownloadError, match="transport layer"):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_wall_clock_deadline_bounds_a_stalled_download(monkeypatch):
    """httpx's `read` is re-armed per 64 KiB socket read, so it never bounds the
    download as a whole; the deadline does. It applies to an INJECTED client too --
    unlike `_DOWNLOAD_TIMEOUT`, this is a module-owned limit like `max_bytes`."""

    async def _slow_handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.5)
        return httpx.Response(200, content=_JPEG_BYTES, headers={"content-type": "image/jpeg"})

    monkeypatch.setattr(images, "_DOWNLOAD_DEADLINE_SECONDS", 0.05)
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_slow_handler)
    with pytest.raises(InlineImageDownloadError, match="deadline"):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_cancellation_is_not_converted_to_download_error():
    """CancelledError is a BaseException in 3.12 and must propagate untouched -- the new
    handlers name TimeoutError and httpx.HTTPError only. T-089/T-090 precedent: a
    swallowed CancelledError bypasses audit and the user reply."""

    async def _cancel_handler(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_cancel_handler)
    with pytest.raises(asyncio.CancelledError):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_connector_credentials_are_cached_across_calls():
    """MSAL's token cache lives on the credentials' app. Building fresh credentials per
    call meant acquire_token_silent ALWAYS missed — a full discovery + token round trip
    per inline image."""
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    await download_inline_image(att, _CREDENTIALS, client=_client_with_handler(_ok_handler))
    first = images._credentials_cache[_CREDENTIALS]
    await download_inline_image(att, _CREDENTIALS, client=_client_with_handler(_ok_handler))

    assert images._credentials_cache[_CREDENTIALS] is first
    assert len(images._credentials_cache) == 1


async def test_credentials_cache_is_keyed_by_credentials():
    """A rotated secret must land on a new key rather than reuse stale credentials."""
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    rotated = BotFrameworkCredentials(app_id="aid", app_password="new-pwd", tenant_id="tid")
    await download_inline_image(att, _CREDENTIALS, client=_client_with_handler(_ok_handler))
    await download_inline_image(att, rotated, client=_client_with_handler(_ok_handler))

    assert images._credentials_cache[_CREDENTIALS] is not images._credentials_cache[rotated]


async def test_token_acquisition_runs_off_the_event_loop():
    """MSAL's constructor does a tenant-discovery GET, so the token call must not run on
    the loop — an unbounded GET there blocks the whole process, not just a worker."""
    threads: list[str] = []

    def _record(_self=None, *_args, **_kwargs):
        threads.append(threading.current_thread().name)
        return "test-token"

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    with patch.object(MicrosoftAppCredentials, "get_access_token", _record):
        await download_inline_image(att, _CREDENTIALS, client=_client_with_handler(_ok_handler))

    assert threads and all(name != threading.main_thread().name for name in threads)


async def test_token_failure_message_carries_the_type_not_the_exception_text():
    """T-119: newly reachable now that T-115m makes MSAL time out. A `requests` timeout
    string embeds the tenant id and the token endpoint; a PermissionError carries AAD's
    error_description. The chained original still reaches `exc_info=True` logging."""

    class ConnectTimeout(OSError):  # noqa: N818 -- mirrors requests' real class name
        """Stand-in for requests.exceptions.ConnectTimeout — same str() shape."""

    def _boom(_self=None, *_args, **_kwargs):
        msg = (
            "HTTPSConnectionPool(host='login.microsoftonline.com', port=443): Max retries "
            "exceeded with url: /tid-guid/oauth2/v2.0/token"
        )
        raise ConnectTimeout(msg)

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(lambda _r: pytest.fail("must not reach HTTP"))
    with (
        patch.object(MicrosoftAppCredentials, "get_access_token", _boom),
        pytest.raises(InlineImageDownloadError) as excinfo,
    ):
        await download_inline_image(att, _CREDENTIALS, client=client)

    # Exact equality, not a substring check: it proves nothing ELSE leaked into the message.
    assert str(excinfo.value) == "Failed to acquire Bot Framework connector token: ConnectTimeout"
    assert isinstance(excinfo.value.__cause__, ConnectTimeout)


# ---------------------------------------------------------------------------
# T-134-b: HEIC→JPEG transcoding
# ---------------------------------------------------------------------------


def _real_heic_bytes() -> bytes:
    """A genuine tiny HEIC file, encoded at test time (no binary fixture in git)."""
    pillow_heif = pytest.importorskip("pillow_heif")
    from io import BytesIO

    from PIL import Image

    pillow_heif.register_heif_opener()
    buf = BytesIO()
    # format="HEIF" produces a generic HEVC-coded HEIF container — a superset
    # variant of iPhone HEIC (which adds Apple metadata boxes). Detection is
    # brand-based (`ftyp` + brand in _HEIF_BRANDS) so the variant difference is
    # immaterial; the assert pins the property the branch actually keys on.
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, format="HEIF")
    data = buf.getvalue()
    assert data[4:8] == b"ftyp"
    return data


async def test_heic_payload_is_transcoded_to_jpeg():
    heic = _real_heic_bytes()

    def _heic_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=heic, headers={"content-type": "application/octet-stream"}
        )

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_heic_handler)
    result = await download_inline_image(att, _CREDENTIALS, client=client)
    assert result.mime == "image/jpeg"
    assert result.data.startswith(b"\xff\xd8\xff")  # re-sniffed real JPEG bytes


async def test_corrupt_heic_container_raises_skip_error():
    """HEIF magic over garbage must degrade to the ordinary skip contract.

    The missing-dependency case (ImportError from `import pillow_heif`) and a
    corrupt payload (PIL raising on garbage) both land in the same `except
    Exception` arm — one test covers both failure classes without simulating an
    uninstalled package.
    """
    corrupt = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=corrupt)

    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_handler)
    with pytest.raises(InlineImageDownloadError, match="HEIC transcode failed"):
        await download_inline_image(att, _CREDENTIALS, client=client)


async def test_oversize_transcode_output_raises(monkeypatch):
    """A transcode that inflates past max_bytes must not return oversize bytes."""
    heic = _real_heic_bytes()

    def _heic_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=heic)

    monkeypatch.setattr(
        images, "transcode_heif_to_jpeg", lambda _data: b"\xff\xd8\xff" + b"\x00" * 512
    )
    att = make_inline_image(content_url="https://smba.trafficmanager.net/x")
    client = _client_with_handler(_heic_handler)
    with pytest.raises(InlineImageDownloadError, match="exceeded the"):
        await download_inline_image(att, _CREDENTIALS, client=client, max_bytes=256)


def test_transcode_helper_is_public_with_private_alias():
    """T-338: consumers import the public name; the private name survives one release."""
    assert callable(images.transcode_heif_to_jpeg)
    assert images._transcode_heif_to_jpeg is images.transcode_heif_to_jpeg
