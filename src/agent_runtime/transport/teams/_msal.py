"""Bot Framework credentials whose MSAL app carries an HTTP timeout (T-115m).

`requests` has NO default timeout, and neither the Bot Framework SDK nor anything
in this package ever passed one to MSAL — so a hung socket blocked token
acquisition forever. That cannot be fixed from the caller: the inline-image path
wraps the call in `asyncio.to_thread`, which is uncancellable (an outer
`asyncio.timeout` frees the coroutine while the worker stays hung), and the
adapter path runs it synchronously inside msrest's async pipeline, i.e. directly
on the event loop. `msal.ClientApplication` accepts `timeout=` and threads it into
its `requests.Session` (msal 1.37.0, `application.py:659-662`) — the only layer
that can bound this.
"""

from __future__ import annotations

import threading

from botframework.connector.auth import MicrosoftAppCredentials
from msal import ConfidentialClientApplication

# (connect, read), per `requests`' convention. NOT a 15 s ceiling: MSAL issues up to
# two requests per cold acquisition (tenant-discovery GET during construction, token
# POST) and mounts `HTTPAdapter(max_retries=1)`, so the arithmetic bound is
# ~2 x 2 x 15 = 60 s. Finite instead of infinite is the property bought here.
_MSAL_HTTP_TIMEOUT: tuple[float, float] = (5.0, 10.0)


def _build_msal_app(app_credentials: MicrosoftAppCredentials) -> ConfidentialClientApplication:
    """Build the MSAL app the SDK would have built, with ``timeout=`` threaded in.

    Mirrors ``MicrosoftAppCredentials.__get_msal_app`` (botframework-connector 4.x)
    argument for argument, reading the values off the SDK object rather than
    re-deriving them — that keeps the SDK's default-tenant fallback and authority
    prefix authoritative here. The only difference is ``timeout``.

    BLOCKING: the constructor performs tenant discovery (a real
    ``GET .../v2.0/.well-known/openid-configuration`` — msal ``authority.py:96-99``),
    so callers must keep it off the event loop wherever they can. A module-level
    function so tests can patch one seam instead of the network.
    """
    return ConfidentialClientApplication(
        client_id=app_credentials.microsoft_app_id,
        client_credential=app_credentials.microsoft_app_password,
        authority=app_credentials.oauth_endpoint,
        timeout=_MSAL_HTTP_TIMEOUT,
    )


class BoundedAppCredentials(MicrosoftAppCredentials):
    """``MicrosoftAppCredentials`` whose MSAL app carries ``_MSAL_HTTP_TIMEOUT``.

    ``self.app`` is a public attribute the SDK checks before building its own
    (timeout-less) app, so seeding it first is the whole mechanism. Seeding is
    LAZY — done on the first token request, not in ``__init__`` — because
    ``teams-bot-platform`` constructs the adapter inside an ``@lru_cache``'d
    provider documented as network-free (``backend/app/deps.py:488``). Construction
    here stays pure attribute assignment.
    """

    def __init__(
        self,
        app_id: str,
        password: str,
        channel_auth_tenant: str | None = None,
        oauth_scope: str | None = None,
    ) -> None:
        # The vendored SDK's stub types these params `str` with a `None` default
        # (`microsoft_app_credentials.py:20-21`), so ty flags the accurate `str | None`
        # signature above as a mismatch against the SDK's own inaccurate one below.
        super().__init__(
            app_id,
            password,
            channel_auth_tenant=channel_auth_tenant,  # ty: ignore[invalid-argument-type]
            oauth_scope=oauth_scope,  # ty: ignore[invalid-argument-type]
        )
        self._app_lock = threading.Lock()

    def get_access_token(self, force_refresh: bool = False) -> str:  # noqa: FBT001, FBT002
        # Double-checked locking around CONSTRUCTION ONLY -- it never wraps
        # `acquire_token_*`, so the warm path takes the outer `if` and no lock.
        # Needed because this object is now shared across worker threads.
        if not self.app:
            with self._app_lock:
                if not self.app:
                    self.app = _build_msal_app(self)
        return super().get_access_token(force_refresh)
