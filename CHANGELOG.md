# Changelog

## 0.20.0 — 2026-08-06

### Added
- **`ToolRoundContext` / `current_tool_round()` / `bind_tool_round()`
  (`agent_runtime.llm`)** — an executor can now read which `ToolUseLoop` round it is
  in (`round_index`, 1-based), the cap the consumer supplied (`max_rounds`), and how
  many tool rounds remain after this one (`rounds_remaining`, clamped at 0;
  `is_final_round`). `ToolUseLoop` binds it around every executor invocation on both
  the `run()` and `resume()` paths.
  **Backward-compatible by construction:** `ToolExecutor` is unchanged — still
  `Callable[[str, dict[str, Any]], Awaitable[ToolResult]]`. Delivery is via a
  contextvar defaulting to `None`, so an executor that never reads it behaves
  byte-for-byte as before and needs no edit. A reader outside a loop round gets
  `None`. `bind_tool_round` restores the previous value on exit (token-based), so a
  nested loop does not clear the outer round's budget, and contextvars are
  per-asyncio-task so sibling turns cannot cross-talk.
  The loop stays policy-free: this publishes NUMBERS only — no advisory text, no
  proximity threshold. The consumer decides what, if anything, to tell the model
  (see teams-bot-platform T-115j-b).

## v0.19.1 — 2026-08-06

### Fixed
- **Teams inline-image downloads ran on httpx's 5-second default and could take down the whole
  turn.** `download_inline_image` built its owned client as a bare `httpx.AsyncClient()`, so
  `Timeout(5.0)` governed connect, read, write and pool. httpx applies `read` per 64 KiB socket
  read (and to the response headers), so 5 s was at once too tight for time-to-first-byte on a
  multi-MiB phone photo — the download failed fast having transferred nothing — and no bound at
  all on the transfer as a whole (~160 reads under the 10 MiB cap). The owned client now carries
  `httpx.Timeout(10.0, read=15.0)`, and the streamed **download** is bounded by a 30-second
  wall-clock deadline that applies to injected clients too (like `max_bytes`, it is a module-owned
  limit). An injected client keeps its own per-phase timeouts. Note the scope precisely: the
  deadline covers the HTTP transfer only — connector-token acquisition
  (`MicrosoftAppCredentials.get_access_token`, wrapped in an uncancellable `asyncio.to_thread`)
  runs before it and remains unbounded.
- **Timeouts and transport errors now raise `InlineImageDownloadError`** instead of escaping as raw
  `httpx` exceptions. Consumers catch that one type to degrade to a skipped image; a raw
  `httpx.ReadTimeout` bypassed the handler and failed the turn. The message names the exception
  type only — never httpx's text, which embeds connection details (SEC-2/SEC-3) — and chains the
  original via `raise ... from`, so `exc_info=True` logging is unchanged.
  `asyncio.CancelledError` is deliberately **not** converted and still propagates.

## v0.19.0 — 2026-08-04

### Added
- **`OutboundChannel.update_activity(activity_id, text) -> bool`** — edit a previously sent
  bot message in place, backed by Bot Framework's Connector `update_activity`. Returns
  `False` (never raises) on an empty id or a failed edit, so callers degrade on a return
  value instead of branching on channel type. Works unchanged on a detached-turn context:
  the adapter resolves the same cached `ConnectorClient` from `turn_state` that
  `send_activities` uses. `CancelledError` deliberately propagates.
- **`FakeOutboundChannel.updates` / `supports_update`** — the test double records edits and
  can model an edit-incapable channel. `clear()` resets `updates`; `supports_update` is an
  injected capability and survives it.

### Changed
- **`OutboundChannel.send_text` now returns `str | None`** (was `None`) — the
  channel-assigned activity id, which is what `update_activity` targets. Source-compatible
  for every existing caller (all ignore the return). Send failures still raise, unchanged.
  An empty id is normalised to `None` (botbuilder substitutes `ResourceResponse(id="")` on a
  falsy Connector response), so `if not activity_id` is a complete degrade check.
  `FakeOutboundChannel.send_text` returns stable synthetic ids (`activity-1`, …).

### Fixed
- **Version drift:** `__init__.py` still reported `0.17.0` after the v0.18.0 release bumped
  only `pyproject.toml` (and `uv.lock`). All sites now read `0.19.0`. This also turns
  `tests/unit/test_version.py` — the dual-source guard added after the v0.6.2/v0.6.3 drift,
  and **red since v0.18.0 shipped** — back to green.

## v0.18.0 — 2026-08-02

### Fixed
- **`SessionManager` silently reset a live conversation every `idle_timeout`.** The
  `(user_id, bot_id)` reverse index was written with `ex=idle_timeout` exactly ONCE, in
  `create_session`, and never refreshed — while the session key's own lease *was*
  extended on every turn. So the index expired `idle_timeout` after session CREATION even
  on a continuously active conversation. The next turn missed the `Active` hot path, fell
  into cold-cache rehydration, and that path built `SessionData` with a hard-coded empty
  `conversation_history`; `_save_session` then overwrote the still-intact Redis blob with
  it. Net effect: the bot forgot the entire conversation while the session kept its id and
  `status="active"`, so the user got no notice at all. Observed live in TBP 2026-08-02 — a
  10-turn session created 00:12:32Z lost all 18 messages on the 00:43:49Z turn (77 s after
  the index's 30-minute expiry); token telemetry showed `cache_read=0` and a
  `cache_creation` byte-identical to the session's cold start.
  - **Fix A** — `_rearm_active_index`: the index lease is now extended on the same beat as
    the session lease, on every `Active` decision. `xx=True`, so a session closed by
    `end_session` (which deletes the key) is never resurrected.
  - **Fix B** — `_load_durable_history`: both rebuild-from-Postgres paths
    (`get_or_prompt_resume` cold-cache rehydration and `_resume_from_db`) now hydrate the
    transcript from the durable store instead of hard-coding `[]`. Requires a repo
    advertising `supports_durable_history = True` (T-036); without one, behaviour is
    unchanged. Best-effort — a durable-read failure logs a warning and degrades to empty
    history rather than denying the user their session.

  Fix B is the safety net: it makes *any* future loss of the Redis hot cache
  (eviction, restart, flush) non-destructive, not just the index-expiry path in Fix A.

## v0.17.0 — 2026-07-31

### Changed
- `SessionManager._persist_resume_to_db` / `_persist_message_to_db` failures now log at
  `error` with `exc_type` + `exc_info=True` (was `warning`, message-only). Still swallowed —
  persist failure never fails the turn. Surfaced TBP T-084's 40-turn silent transcript loss.
- `download_inline_image` (Teams transport): a non-`image/*` Content-Type no longer rejects
  up front. The body is streamed (still bounded by `max_bytes`) and sniffed by magic bytes
  (PNG/JPEG/GIF/WebP — exactly Anthropic's supported set); `DownloadedImage.mime` carries the
  sniffed type. Teams' CDN serves real images as `application/octet-stream` (TBP T-084
  Issue 4 — live receipt rejected). Genuinely non-image payloads raise the same error.
- `resolve_identity` (Teams transport): `TeamsInfo.get_member` gets one immediate retry
  before the no-email fallback/drop path — a transient Graph error no longer silently
  discards the user's message (TBP T-084 Issue 5). Worst case 2 Graph calls per activity.

## v0.16.0 — 2026-07-31

### Added
- `max_result_chars: int | None = None` keyword on `ToolUseLoop.run` AND
  `ToolUseLoop.resume` (`llm/tool_loop.py`). When set, every EXECUTOR-produced tool
  result longer than the cap is truncated to the first `max_result_chars` characters
  and an explicit `_TRUNCATION_MARKER` is appended, so the model is told the result is
  partial (silent clipping would make a knowledge bot summarise a fragment as if it were
  whole). `is_error` is preserved; the `InjectResultDecision` (consumer-supplied)
  content is not capped. A `tool_loop_result_truncated` warning is emitted to the
  consumer's `AuditLogger`. Measurement is on characters (exact, zero-cost,
  deterministic — keeps the multi-round cache prefix byte-stable); the marker reports an
  estimated token figure via `estimate_tokens`. Default `None` = no cap = byte-for-byte
  identical to v0.15.0 (regression guarantee for ithelpdesk and all existing callers).
  The loop owns no default ceiling; the consumer supplies it, exactly as with
  `max_rounds`. Fixes the unbounded-tool-result class behind TBP T-081 (a live
  `copilot_retrieval` returned 163,563 tokens into a single turn).

### Notes
- The cap is per result, not per turn: a round with several large tools can still stack
  multiple capped results. The per-turn aggregate remains bounded only by `max_rounds`;
  a per-turn aggregate cap is a possible future hardening.
- `_resolve_round` changed from a static to an instance method (internal) so it can
  reach the loop's `AuditLogger`. No public API impact.

## v0.15.0 — 2026-07-30

### Added
- `SignInResource` frozen dataclass + `OutboundChannel.get_sign_in_resource(*, connection_name)`
  (`transport/teams/outbound.py`, re-exported from `agent_runtime.transport.teams`).
  `BotFrameworkOutboundChannel` implements it via
  `BotFrameworkAdapter.get_sign_in_resource_from_user`, returning the signed
  token-service sign-in link + token-exchange URI, or `None` when no resource can be
  obtained (empty connection, missing user id, non-BotFramework turn context, or no
  sign-in link). Enables consumers to build a Teams-renderable OAuthCard with a valid
  `signin` button instead of a hand-authored `api://` value (unblocks TBP T-075).
  `FakeOutboundChannel` gains an injectable `sign_in_resource` field. Additive —
  existing `OutboundChannel` implementations gain one Protocol method; default `None`
  return keeps all current send paths byte-identical.

## v0.14.0 — 2026-07-29

### Added
- `LLMImage` frozen dataclass + `ANTHROPIC_IMAGE_MEDIA_TYPES` (`llm/models.py`, re-exported
  from `agent_runtime.llm`): base64 image content blocks with fail-fast media-type
  validation; `from_bytes()` / `to_block()` helpers (T-067d).
- `images: tuple[LLMImage, ...] = ()` keyword on `ToolUseLoop.run` AND
  `AnthropicClient.complete` — vision passthrough. Image blocks are inserted between the
  cached retrieval block (breakpoint #2 untouched) and the user text block. Default `()`
  is byte-for-byte identical to v0.13.0 (regression guarantee). With images present and an
  empty `user_message`, the empty text block is omitted (Anthropic rejects empty text).
  Note: image blocks live in the uncached suffix and are re-billed each tool round —
  consumers bound cost via round caps; a cache_control on the last image block is a
  possible future consumer-level hardening.

### Fixed
- Compaction folding is now content-block-aware (`_content_to_text`): a block-shaped
  history turn folds its text blocks and collapses non-text blocks to `[<type>]`
  placeholders instead of dumping raw base64 reprs into the merge prompt and the token
  estimate (defense-in-depth — consumers are contracted to persist text-only history).

## v0.13.0 — 2026-07-28

### Added
- `InlineImageAttachment` dataclass + `InboundMessage.images: tuple[InlineImageAttachment, ...]`
  (default empty tuple) — surfaces Teams **inline images** (camera captures, pasted/shared
  photos), which arrive with contentType `image/*` (or a concrete `image/*` mime) and a
  `contentUrl` on the Bot Framework attachment store, distinct from the paperclip
  file-upload path (`FileAttachment`/`.attachments`). Additive and dormant for existing
  consumers: images land on the **new** `.images` field, never mixed into `.attachments`
  (T-067a).
- `agent_runtime.transport.teams.images.download_inline_image` — authenticated download
  helper for inline images. Refuses to attach a Bot Framework connector token unless the
  attachment's `content_url` is `https` and its host is allowlisted (default
  `smba.trafficmanager.net` + subdomains; override via `allowed_hosts`) — this check runs
  before any token acquisition or HTTP call, since `content_url` is model-external input.
  Streams the response with a hard `max_bytes` cap (default 10 MiB) and validates the
  response `Content-Type` starts with `image/`; oversize or non-image responses raise
  `InlineImageDownloadError` with partial bytes discarded. Token acquisition uses
  `botframework-connector`'s `MicrosoftAppCredentials` (blocking call wrapped in
  `asyncio.to_thread`); `PermissionError` on token failure surfaces as
  `InlineImageDownloadError`. New `BotFrameworkCredentials` + `DownloadedImage` dataclasses.
  No new dependency (`httpx` is a core dep; `botframework-connector` ships via the `teams`
  extra).
- `make_inline_image` test double + `make_inbound_message(images=...)` param in
  `agent_runtime.transport.teams.testing`.
- All new symbols exported from `agent_runtime.transport.teams`.

## v0.12.0 — 2026-07-19

### Added
- `agent_runtime.observability` — correlation-ID + error-envelope primitives (T-060a).
  - `RequestIDMiddleware`: pure-ASGI (no new dependency) middleware that reuses an inbound
    `X-Request-ID` header or generates a UUID4, binds it to a request contextvar for the
    request's duration, and echoes it on the response `X-Request-ID` header. Register via
    `app.add_middleware(RequestIDMiddleware)` on any Starlette/FastAPI app.
  - `get_request_id` / `set_request_id` / `generate_request_id` / `clear_request_id` /
    `get_or_create_request_id`: contextvar accessors that thread the id across async
    boundaries so the runtime's OWN log lines carry it.
  - `request_id_log_fields()`: audit-inject hook returning `{"request_id": <id>}` when set,
    `{}` when unset (no-op safe) — consumers fold it into log kwargs.
  - `error_envelope(...)`: pure-data helper producing
    `{error, error_code, detail, request_id, timestamp}` (request_id defaults to the
    contextvar; timestamp to UTC ISO-8601).
  - Re-exported at top level (`from agent_runtime import ...`) and from
    `agent_runtime.observability`. Opt-in: middleware must be registered by the consumer;
    existing consumers that don't register it are byte-identical.

### Fixed
- Circuit breaker (`resilience.circuit_breaker`): `asyncio.CancelledError` raised inside an
  `async with breaker:` block is now classified as **neither success nor failure** and re-raised
  (T-063a). A caller-side deadline/timeout cancellation no longer counts toward the failure
  threshold (can't trip the SHARED breaker) nor resets it. Ports ithelpdesk's T-617 rule.

## v0.11.0 — 2026-06-28

### Added
- `FileAttachment` dataclass + `InboundMessage.attachments: tuple[FileAttachment, ...]`
  (default empty tuple) — surfaces Teams file uploads. The adapter parses
  `turn_context.activity.attachments`, keeping only attachments with
  `contentType == "application/vnd.microsoft.teams.file.download.info"` and a
  non-empty `content.uniqueId` (the OneDrive driveItem id used for read-on-demand);
  everything else (inline images, cards) is ignored. Exported from
  `agent_runtime.transport.teams`; `make_inbound_message(attachments=…)` +
  `make_file_attachment` added to the testing helpers. Byte-identical for any
  message without a readable file attachment (empty tuple). Enables consumer
  file-capture into user Projects (tbp T-037c).

## v0.10.0 — 2026-06-27

### Added
- `AnthropicClient.complete(..., cache_history=False)` and
  `ToolUseLoop.run(..., cache_history=False)` — opt-in third `cache_control` ephemeral
  breakpoint on the last conversation-history message. When `True` and history is
  non-empty, Anthropic incrementally caches the stable history prefix across turns
  (~20–30% input-token cut on 3+ turn sessions). Default `False` keeps message assembly
  byte-identical for existing callers (ithelpdesk). New exported helper
  `assemble_history_messages(history, *, cache_history)`. Reopens ARCHITECTURE.md §4 #5
  ("two breakpoints") to permit a third; the 5-min TTL choice is unchanged.

## v0.9.0 — 2026-06-26

### Added
- `ConversationRef.conversation_type` (`"personal"` | `"channel"` | `"groupChat"`, default
  `"personal"`) — captured in `resolve_identity` from `activity.conversation.conversation_type`,
  serialized by `conversation_ref_to_dict`/`from_dict` (missing key → `"personal"`, forward-compat
  for rows persisted before the field). Lets a consumer route channel turns differently from 1:1
  DMs (T-031). Additive: existing construction and durable rows are unaffected.
- `_EventDispatchingHandler.on_message_activity` now strips the bot's own `<at>…</at>`
  recipient-mention from inbound text via `TurnContext.remove_recipient_mention`, so a channel
  `"@Bot summarize this"` arrives as `"summarize this"`. No-op in 1:1 DMs (no recipient-mention
  entity), so the personal-chat path — and ithelpdesk — is byte-equivalent.

## v0.8.0 — 2026-06-26

### Added
- Durable per-session conversation history (T-036). New `DurableHistoryRepository`
  Protocol (`append_message`, `get_conversation_history`, `list_sessions`). A repository
  opts in by setting `supports_durable_history = True`; `SessionManager` gates durable
  writes/reads on that explicit flag (NOT method-name `isinstance`, which would collide
  with unrelated `list_sessions` methods). Repos that don't opt in (e.g. ithelpdesk) keep
  the prior Redis-only behaviour unchanged.
- `SessionManager.update_session` now best-effort writes each message through to the
  durable store. Best-effort by design: a store outage degrades to a gap in the cold
  transcript, never a failed turn (chat availability over completeness; a warning is
  logged). Redis is source-of-truth while warm; durable once cold.
- `SessionManager.fork_session(*, source_session_id, user_id, bot_id, ...)` — creates a
  new active session seeded with a past session's durable transcript (Redis context only;
  not re-persisted). Ownership scoped by user+bot. The caller must durably end any active
  session first (else `SessionAlreadyActive`).
- `SessionManager.list_sessions(*, user_id, bot_id, limit=20, before)` — paginated recent
  sessions for a history UI; returns `SessionSummaryRow`s. `[]` without a durable repo.
- `SessionManager.create_session(..., initial_history=)` — seed Redis conversation
  context (used by `fork_session`); respects `max_history`.
- New wire type `SessionSummaryRow`; both it and `DurableHistoryRepository` exported
  from `agent_runtime.session`.

## v0.7.0 — 2026-06-25

### Added
- `agent_runtime.llm.compaction` module (T-028): in-session conversation compaction primitive.
  Exposes `CompactionConfig`, `WorkingMemory`, `CompactionResult`, `CompactionEngine`, and
  `estimate_tokens`. `CompactionEngine.maybe_compact()` folds the oldest turns into a running
  prose summary (via `AnthropicClient.complete()`) while keeping the most recent `keep_k` turns
  verbatim. Triggered when estimated live-prompt tokens cross a configurable fraction of the model
  window. Failed merge calls are best-effort — turns are never dropped; a `memory_compaction_failed`
  audit event is emitted instead. All types exported from `agent_runtime.llm`.

## v0.6.8 — 2026-06-24

### Added
- `TeamsAdapter.send_proactive(ref, *, bot_app_id, text=, card=)` (T-029a) — send an
  unsolicited (proactive) message into an existing 1:1 Teams chat. Reconstructs a canonical
  botbuilder `ConversationReference` from a stored `ConversationRef` and drives
  `BotFrameworkAdapter.continue_conversation`; the callback reuses `BotFrameworkOutboundChannel`
  so proactive text/cards render identically to solicited ones.
- `ConversationRef` gains `user_channel_id` (the `29:…` sender) and `recipient_id`
  (the `28:<appid>` bot) channel-account ids, captured in `identity.py` on inbound, so the
  reconstructed proactive reference is byte-canonical (not OID-synthesized). Both default `= ""`
  (additive; existing construction + equality unchanged).
- `conversation_ref_to_dict` / `conversation_ref_from_dict` — flat str->str (de)serializers for
  durable storage of a `ConversationRef`; `from_dict` tolerates missing/unknown keys for schema
  evolution. Consumer wiring is TBP T-029a-b.

## v0.6.7 — 2026-06-24

### Added
- `agent_runtime.llm.build_anthropic_sdk_client(provider=, api_key=, foundry_resource=,
  foundry_base_url=)` — provider factory returning the SDK client to inject into
  `AnthropicClient`. `provider="anthropic"` → `AsyncAnthropic` (public api.anthropic.com);
  `provider="foundry"` → `AsyncAnthropicFoundry` against the in-tenant Azure AI Foundry
  `/anthropic/` passthrough (data stays in the Azure tenant). Foundry auth is API-key only
  (the SDK sets the Foundry auth header(s) automatically); requires exactly one of
  `foundry_resource` or `foundry_base_url`.
  `AsyncAnthropicFoundry` is an `AsyncAnthropic` subclass, so the two-cache-breakpoint contract
  and native model IDs are unchanged — a transport change, not a contract change. **Additive**;
  no existing call path changes. Consumer wiring is TBP T-034a-b.

### Changed
- `llm` extra floor bumped `anthropic>=0.42` → `>=0.102,<1.0` (the `AsyncAnthropicFoundry`
  client the factory imports).

## v0.6.6 — 2026-06-23

### Added
- `ToolUseLoop` confirm-before-dispatch (T-025a): optional `confirm(name, input) -> bool`
  predicate on `run()`. A flagged tool SUSPENDS the loop instead of executing — `run()`
  returns a `ToolLoopResult.pending_confirmation` (`PendingConfirmation`) carrying the
  proposed call + an opaque, JSON-serializable `state`. The consumer persists `state`
  (surviving an async approval round-trip across processes) and calls the new
  `ToolUseLoop.resume(state=, decision=)` to continue. Decisions are policy-free:
  `ExecuteDecision(tool_input=None|dict)` runs the tool (Send/Edit); `InjectResultDecision`
  feeds a synthetic tool_result without executing (Discard). New public exports:
  `PendingConfirmation`, `ExecuteDecision`, `InjectResultDecision`, `ResumeDecision`,
  `ConfirmPredicate`. **Purely additive** — `confirm=None` (default) is byte-for-byte the
  prior behaviour.

## v0.6.5 — 2026-06-21

### Added
- `agent_runtime.safety.mask_telemetry` — masks free-text exception/telemetry bodies with the default
  secret/PII patterns PLUS Entra OID/GUID + AAD tenant-URL redaction. Kept separate from the default
  `PATTERNS` so `mask_string`/`mask_dict` default behaviour is unchanged. Wired into the library's
  connector, circuit-breaker, identity, and session telemetry so an OID-bearing Graph URL / tenant
  token-endpoint URL never reaches an audit sink raw (TBP T-021a; consumer wiring T-021a-b).

### Changed
- **Prompt sanitizers (SEC-1, SEC-7) now alter outputs they previously passed through.**
  `sanitize_for_llm_prompt` strips role sentinels case-INsensitively (was case-sensitive); both
  sanitizers NFKC-normalize and strip zero-width/format chars. Consumers (ithelpdesk) will see more
  aggressive neutralization of confusable/lowercase/full-width injection markers — behavioural, not
  purely additive.

### Security (SEC-1..7, 2026-06-20 audit)
- SEC-1 case-insensitive role-sentinel strip; SEC-2 mask exception text in connector/breaker logs;
  SEC-3 mask `ConnectorResult.data._internal_error`; SEC-4 `mask_dict` non-str-key + depth-cap
  hardening; SEC-5 whitespace-only auth-header bypass guard; SEC-6 optional `max_history` cap
  (default `None` = prior unbounded behaviour); SEC-7 NFKC + zero-width normalization in sanitizers.

## v0.6.4 — 2026-06-20

### Added
- `agent_runtime.safety.mask_string` / `mask_dict` — generic PII/secret masking
  (ssn, credit_card, email, phone, otp, api_key, password regex patterns; key-name
  redaction + recursive dict/list traversal for `mask_dict`). Lifted from ithelpdesk's
  `DataMasker` and reshaped to the `safety/` free-function convention. A masking
  *primitive* applied before a consumer's own audit sink — consistent with the
  "agent_runtime does not own the sink" stance. Additive; no consumer wired until
  teams-bot-platform task T-015a-b.

## v0.6.3 — 2026-06-18

### Removed
- `agent_runtime.safety.InjectionDetector` / `DetectionResult` / `PatternMatch` — log-only
  injection detector that was never wired into any consumer's live path (its only effect was an
  `AuditLogger.security` event, and no concrete `AuditLogger` sink exists). Removed to eliminate
  false-coverage in `safety/`. `safety/` now exposes only the on-path primitives
  `sanitize_for_llm_prompt` + `sanitize_tool_result`. No consumer affected (ithelpdesk uses a
  local copy; teams-bot-platform never imported it). See teams-bot-platform task T-012b.

## v0.6.1 — 2026-06-17

### Added
- `OutboundChannel.send_oauth_card(card: dict)` + `BotFrameworkOutboundChannel`
  impl — sends a Bot Framework OAuthCard (`application/vnd.microsoft.card.oauth`)
  to trigger Teams SSO token exchange. `FakeOutboundChannel` records sends in
  `sent_oauth_cards`. Additive; no behavior change to existing methods.

## v0.6.0 — 2026-06-17

### Added
- `ToolUseLoop` — generic, policy-free fenced model-driven tool-use loop. Drives
  Anthropic tool-use over a caller-supplied tool set + executor bounded by a
  caller-supplied round cap. Returns `ToolLoopResult` with final text, aggregate
  token usage, per-round trace (`ToolLoopStep`/`ToolCall`), and `cap_exhausted` flag.
- `complete_messages(*, system_blocks, messages, tools=None)` low-level method on
  `AnthropicClient` — caller-assembled message list entry point used by `ToolUseLoop`
  for multi-round calls; `complete()` wraps it.
- `tools=` parameter on `AnthropicClient.complete()` — passed verbatim to the SDK.
  `tools=None` (default) omits the param entirely (D5 — byte-identical to v0.5.0).
- `ClaudeResponse.tool_use: tuple[ToolUseBlock, ...]` field (defaults to `()`);
  `ToolUseBlock(id, name, input)` frozen dataclass.
- New public types in `agent_runtime.llm`: `ToolResult`, `ToolCall`, `ToolLoopStep`,
  `ToolLoopResult`, `ToolExecutor`, `ToolUseBlock`.

### Notes
- Policy-free: `ToolUseLoop` owns no cap value, no PATH dispatch, no MCP knowledge.
  Consumer (teams-bot-platform T-011d-c) supplies the cap, tools, and executor and
  classifies `ToolLoopResult` into PATH A/B.
- Two-breakpoint cache contract preserved: `static_system_prefix` (breakpoint 1) and
  `retrieval_block` (breakpoint 2) still carry `cache_control: {type: ephemeral}` in
  both `complete()` and `ToolUseLoop.run()`.
- `complete(tools=None)` is byte-identical to v0.5.0 behavior — single-shot callers
  are unaffected.
- `ToolResult` field names are the cross-plan duck-type contract with T-011d-b's local
  `ToolResult`: `content: str`, `is_error: bool`. The loop reads by attribute access
  only (`.content`, `.is_error`) — no `isinstance` check, no import cycle.
- `cap_exhausted=True` CONTRACT: the consumer MUST route to PATH B regardless of
  `final_text` (which may be empty when the model wanted a tool it can't call).
- `max_rounds=N` issues up to N+1 SDK calls (N tool rounds + 1 final no-tools call).
- `llm_request_start` debug event kwargs changed: `has_retrieval_block`/`history_len`
  → `has_tools`/`n_messages`. Event is now emitted in `complete_messages` (not
  `complete`) so every loop round gets a paired start/response event without
  double-logging single-shot calls.
- `llm_unexpected_extra_blocks` warning `count` kwarg semantics changed: was "total
  extra blocks" (v0.5.0), now "count of unknown-typed blocks" (blocks of type other
  than `text` or `tool_use`). Downstream alerting keyed on `count` should note this
  shift.

## v0.5.0 — 2026-06-02

### Added
- `agent_runtime.session` subpackage (optional extras `[redis]`, `[postgres]`):
  Redis-backed conversation session store with Postgres durable resume fallback.
  Lift from ithelpdesk's `app.core.session_*` family with three new surfaces:
  (1) `(user_id, bot_id)` keying per teams-bot-platform `ARCHITECTURE.md` §4 #4;
  (2) `get_or_prompt_resume(...) -> ResumeDecision` sealed-union API for T-008f
  Resume-card UX; (3) `SessionAlreadyActive` typed exception for dispatcher
  pattern-matching when concurrent panes are attempted in v1.
- `pydantic >= 2.6` is now a base runtime dep (was only required by `[llm]`).

### Notes
- ORM-free posture at the library boundary: `ResumeRow` is a Pydantic model;
  no SQLAlchemy mixin. Consumers own their `sessions` table schema, FKs, and RLS.
- Redis key prefix is consumer-configurable via `SessionManager(key_prefix=...)`.
  Resume tokens are scoped by user OID + token, mirroring T-512's SQL-layer fix.
- Atomic lease extension uses `SET ... EX ttl XX` to prevent TOCTOU resurrection
  when a key Redis-evicts between read and write.
- Cold-cache rehydration: a Redis miss + Postgres hit within the idle window
  silently repopulates the cache and returns `Resumable`. Redis restarts no
  longer lose session continuity (same `session_id`, same Resume-card UX).
- **Cold-cache history limitation (v1)**: `ResumeRow` carries metadata only
  (`id`, `user_id`, `bot_id`, `status`, `last_message_at`, `client_context`).
  `data` and `conversation_history` live in Redis only. A cold-cache rehydration
  therefore restores session identity but presents the LLM with an empty turn
  history — the user resumes the same logical session but the model has no
  recall of prior turns. Acceptable for v1 because (a) Redis evictions are rare
  in the 30-min window, (b) the retrieval-snapshot store (teams-bot-platform
  T-008i) is the durable record for replay/audit. v2 will add durable turn
  history to `ResumeRow` if telemetry shows evictions are user-visible.
- IT-specific extension pattern preserved — `session_state_ihd.py` in ithelpdesk
  continues to subclass `ConversationState`, demonstrating the consumer-extension
  contract.

### Breaking changes
- `SessionRepositoryProtocol.upsert_resume_data` and `get_session_for_resume` now
  require `bot_id: str` kwarg. v0.4.0 consumers (none yet for sessions) must
  update their concrete repositories before pinning v0.5.0.
- `SessionManager.update_session` no longer applies an internal model-filter to
  `data` (the IHD consumer model round-trip is gone). Consumers that depended
  on the filter must apply it upstream before calling `update_session`.

## v0.4.0 — 2026-06-02

### Added
- `agent_runtime.transport.teams` subpackage (optional extra `[teams]`): framework-agnostic
  Bot Framework SDK wrapper providing `TeamsAdapter`, `TeamsHandler` Protocol,
  `OutboundChannel` Protocol, `InboundMessage`/`InboundMembersAdded`/`InboundInvoke`
  event dataclasses, and `ConversationRef`. Public test helpers in
  `agent_runtime.transport.teams.testing` (`FakeOutboundChannel` + event factories).
- Optional dependencies: `botbuilder-core>=4.15,<5`, `botbuilder-schema>=4.15,<5`,
  `aiohttp>=3.9,<4` (required by botbuilder's async connector).

### Notes
- Fresh-write subpackage — no per-file ruff ignores added. Code passes `select = ["ALL"]`
  cleanly. Future changes should preserve this property.
- Identity resolution fails closed: inbound activities from users with no resolvable
  email are dropped with a WARNING log; handler is not invoked.

## v0.3.0 — 2026-05-31 (backfilled)

### Added
- `agent_runtime.connectors` — `BaseConnector` ABC, `ConnectorResult`, `RetryMixin`,
  throttle mechanism (lifted from ithelpdesk `service_registry.py` family; T-490a).
- `agent_runtime.protocol` — `NodeResult`, `NodeHandler` / `TemplateResolver` /
  `NodeExecutor` Protocols.

### Notes
- Pre-existing gap — v0.3.0 was released without a CHANGELOG entry. Backfilled here
  for completeness; see git commit `016207f` for the canonical commit history.

## v0.2.0 — 2026-05-30

### Added
- `agent_runtime.llm` subpackage (install via extras: `agent-runtime[llm]`)
- `AnthropicClient` — async wrapper around `anthropic` SDK with opinionated
  two-`cache_control`-breakpoint contract (static system prefix + per-turn
  retrieval block) per teams-bot-platform `ARCHITECTURE.md` §4 decision #5
- `ClaudeResponse` — frozen dataclass with token-usage and cache statistics
- `Message` / `History` — `TypedDict`-based conversation history types
- `LLMError` / `LLMRateLimitError` / `LLMAPIError` / `LLMResponseError` —
  exception hierarchy wrapping SDK exceptions (consumers never import from
  `anthropic` to catch wrapper errors)
- Post-call cache-write detection: AuditLogger `llm_cache_not_written`
  WARNING when `cache_creation_input_tokens == 0` despite caching being
  requested (catches the silent-failure trap when cached blocks are below
  the model's min-cache threshold)

### Notes
- Runtime dep on `anthropic >= 0.42` is **optional** — `agent_runtime.llm`
  is the first subpackage with an external runtime dep; install with
  `pip install agent-runtime[llm]` to opt in
- SDK client is constructor-injected (DI of `AsyncAnthropic`) so consumers
  can share one `httpx` connection pool across many wrapper instances and
  tests can inject a fake without monkeypatching the SDK
- Wrapper is **bot-agnostic** — no `bot_id`/`user_id` knowledge; per-tenant
  budget enforcement happens at the service-layer call site
- v0.1.0 surface unchanged; this is a fully additive release

## v0.1.0 — 2026-05-29

Initial release. Extracted from `ithelpdesk` per
[teams-bot-platform/docs/extraction-inventory-review.md](https://github.com/simonthan/teams-bot-platform/blob/master/docs/extraction-inventory-review.md) (T-001).

### Added
- `agent_runtime.logging.AuditLogger` Protocol + `NullAuditLogger` no-op default
- `agent_runtime.safety` — `sanitize_for_llm_prompt`, `InjectionDetector`
- `agent_runtime.resilience` — `CircuitBreaker` + registry
- `agent_runtime.flows` — `MessageRouter`
- `agent_runtime.context` — `PluginExecutionContext` (without ihd's `.state` property; deferred until `session_state` lifts)

### Notes
- Zero runtime deps; pure stdlib + `typing.Protocol`
- Consumer logger injected via `AuditLogger` Protocol; default `NullAuditLogger` discards events
- ihd's `app.utils.audit_logger.AuditLogger` satisfies the Protocol structurally
