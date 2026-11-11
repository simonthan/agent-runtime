"""Early stop of a runaway tool-call argument (v0.47.0, TBP T-70172 / T-70157).

A model call can run away while it writes a tool call's arguments: TBP saw one write 16,000
output tokens (127 s, the whole output cap) into a call whose parsed input was 170 characters,
because the argument still being written was one endless id or a run of blank space. The
call never runs (it is cut at the cap), so all that time is lost.

``AnthropicClient(tool_input_limit=ToolInputLimit(...))`` watches the raw JSON of every
``tool_use`` block of a STREAMED call that offered tools. When the argument breaks a bound,
the client stops reading the stream (closing the response ends generation) and returns what
was generated so far with ``stop_reason="max_tokens"``, exactly as if the call had hit the
output cap there: every consumer path for a cut call applies unchanged. The response's
``tool_input_report`` says why (``stopped``) and what the argument looked like.

The bounds are the consumer's (this module has no opinion on what a real argument looks
like); each is off when None:

- ``max_whitespace_run``: consecutive blank units -- a space, tab, newline or carriage return
  between JSON tokens, or inside a string a space, a tab, or the escapes ``\\n`` ``\\r`` ``\\t``.
- ``max_key_chars``: the length of one object key.
- ``string_limit(tool, key)``: the bound for a string value under ``key`` (the innermost
  object key, escapes decoded; an array's items take the array's key), or None for no bound.
  Called once when each string value starts. Through ``AnthropicClient``, a callback that
  raises or returns a non-int means no bound for that value (``llm_tool_input_limit_failed``,
  once per call); the call never fails on it.

``ToolInputScanner`` is a single pass over the characters (no regex, no re-scan of what it
has seen), so its cost is linear in the argument's length whatever the chunking.

A stopped call's ``output_tokens`` is an estimate (streamed text and argument characters / 4;
thinking deltas are not counted): no ``message_delta`` arrives once the client stops reading.
Closing the response ends generation on api.anthropic.com; a proxy or passthrough in between
may keep generating (and billing) for a while, though the caller no longer waits for it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "KEY_TOO_LONG",
    "STRING_TOO_LONG",
    "WHITESPACE_RUN",
    "StringLimit",
    "ToolInputLimit",
    "ToolInputReport",
    "ToolInputScanner",
]

WHITESPACE_RUN = "whitespace_run"
STRING_TOO_LONG = "string_too_long"
KEY_TOO_LONG = "key_too_long"

_TAIL_CHARS = 256  # the tail whose distinct characters the report counts
_KEY_KEEP = 64  # characters of a key kept (string_limit's argument and the report)
_BLANK_OUTSIDE = frozenset(" \t\n\r")
_BLANK_INSIDE = frozenset(" \t")
_BLANK_ESCAPES = frozenset("nrt")
_HEX = frozenset("0123456789abcdefABCDEF")
_UHEX_DIGITS = 4
# a key's escapes decoded, so string_limit sees the real key (iav review F2)
_KEY_ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

StringLimit = Callable[[str, str], int | None]


@dataclass(frozen=True, slots=True)
class ToolInputLimit:
    """Bounds on a streamed tool call's arguments; each is off when None (see module doc)."""

    max_whitespace_run: int | None = None
    max_key_chars: int | None = None
    string_limit: StringLimit | None = None


@dataclass(frozen=True, slots=True)
class ToolInputReport:
    """What one streamed tool call's arguments looked like when the call ended in them.

    ``stopped`` is the bound that stopped the stream (``WHITESPACE_RUN`` /
    ``STRING_TOO_LONG`` / ``KEY_TOO_LONG``), or None when the call ran to the output cap.
    ``tool`` and ``key`` are model-written: sanitize them before logging. ``key`` is the key
    whose value was being written (for ``KEY_TOO_LONG``: the start of the key itself), "" at
    the top level. ``chars`` = raw JSON characters streamed for this call;
    ``open_string_chars`` = the length of the string still open at the end (0 = none);
    ``max_whitespace_run`` = the longest blank run seen; ``tail_distinct`` = distinct
    characters among the last 256 (a small number on a long argument = it was repeating)."""

    tool: str
    stopped: str | None
    key: str
    chars: int
    open_string_chars: int
    max_whitespace_run: int
    tail_distinct: int


class ToolInputScanner:
    """Incremental scan of ONE tool call's raw JSON arguments, fed in stream chunks.

    ``feed(chunk)`` returns the name of the bound the arguments broke (and stops scanning), or
    None. Malformed JSON never raises: the scan only tracks strings, keys and nesting."""

    __slots__ = (
        "_esc",
        "_in_str",
        "_is_key",
        "_key_buf",
        "_limit",
        "_stack",
        "_str_cap",
        "_str_key",
        "_str_len",
        "_tail",
        "_uhex",
        "_ws_run",
        "chars",
        "max_ws_run",
        "stopped",
        "stopped_key",
        "tool",
    )

    def __init__(self, tool: str, limit: ToolInputLimit) -> None:
        self.tool = tool
        self._limit = limit
        # one frame per open container: [is_object, key, expecting_key]. An object's key is
        # the last key read in it; an array's is the key the array is the value of.
        self._stack: list[list[object]] = []
        self._in_str = False
        self._esc = False
        self._is_key = False
        self._key_buf: list[str] = []
        self._str_len = 0
        self._str_cap: int | None = None
        self._str_key = ""
        self._uhex: str | None = None  # the hex digits of a \uXXXX escape in a key
        self._ws_run = 0
        self._tail = ""
        self.chars = 0
        self.max_ws_run = 0
        self.stopped: str | None = None
        self.stopped_key = ""

    def _value_key(self) -> str:
        return str(self._stack[-1][1]) if self._stack else ""

    def _trip(self, reason: str, key: str) -> str:
        self.stopped = reason
        self.stopped_key = key
        return reason

    def feed(self, chunk: str) -> str | None:  # noqa: C901, PLR0912, PLR0915 -- one state machine
        if self.stopped is not None or not chunk:
            return self.stopped
        self.chars += len(chunk)
        tail = self._tail + chunk
        self._tail = tail[-_TAIL_CHARS:]
        limit = self._limit
        max_ws = limit.max_whitespace_run
        max_key = limit.max_key_chars
        for c in chunk:
            if self._in_str:
                hex_digit = False
                if self._uhex is not None:  # only ever set inside a key
                    if c in _HEX:
                        hex_digit = True
                        self._uhex += c
                        if len(self._uhex) == _UHEX_DIGITS:
                            self._key_add(chr(int(self._uhex, 16)))
                            self._uhex = None
                    else:
                        self._uhex = None  # malformed: this character is read as usual
                if hex_digit:
                    self._ws_run = 0
                elif self._esc:
                    self._esc = False
                    self._ws_run = self._ws_run + 1 if c in _BLANK_ESCAPES else 0
                    if self._is_key:
                        if c == "u":
                            self._uhex = ""
                        else:
                            self._key_add(_KEY_ESCAPES.get(c, c))
                elif c == "\\":
                    self._esc = True
                    continue  # the escape's second character decides
                elif c == '"':
                    self._in_str = False
                    self._ws_run = 0
                    if self._is_key and self._stack:
                        frame = self._stack[-1]
                        frame[1] = "".join(self._key_buf)
                    continue
                else:
                    self._ws_run = self._ws_run + 1 if c in _BLANK_INSIDE else 0
                    if self._is_key:
                        self._key_add(c)
                self._str_len += 1
                if self._is_key:
                    if max_key is not None and self._str_len > max_key:
                        return self._trip(KEY_TOO_LONG, "".join(self._key_buf))
                elif self._str_cap is not None and self._str_len > self._str_cap:
                    return self._trip(STRING_TOO_LONG, self._str_key)
            elif c in _BLANK_OUTSIDE:
                self._ws_run += 1
            else:
                self._ws_run = 0
                if c == '"':
                    self._in_str = True
                    self._str_len = 0
                    top = self._stack[-1] if self._stack else None
                    self._is_key = bool(top is not None and top[0] and top[2])
                    if self._is_key:
                        self._key_buf = []
                        self._str_cap = None
                    else:
                        self._str_key = self._value_key()
                        fn = limit.string_limit
                        self._str_cap = fn(self.tool, self._str_key) if fn is not None else None
                elif c == "{":
                    self._stack.append([True, "", True])
                elif c == "[":
                    self._stack.append([False, self._value_key(), False])
                elif c in "}]":
                    if self._stack:
                        self._stack.pop()
                elif c == ":":
                    if self._stack:
                        self._stack[-1][2] = False
                elif c == "," and self._stack and self._stack[-1][0]:
                    self._stack[-1][2] = True
            if self._ws_run > self.max_ws_run:
                self.max_ws_run = self._ws_run
                if max_ws is not None and self._ws_run > max_ws:
                    return self._trip(WHITESPACE_RUN, self._open_key())
        return None

    def _key_add(self, text: str) -> None:
        if len(self._key_buf) < _KEY_KEEP:
            self._key_buf.append(text)

    def _open_key(self) -> str:
        """The key of what is being written: inside a key, that key's start (R3 F3)."""
        if self._in_str:
            return "".join(self._key_buf) if self._is_key else self._str_key
        return self._value_key()

    def report(self) -> ToolInputReport:
        key = self.stopped_key if self.stopped is not None else self._open_key()
        return ToolInputReport(
            tool=self.tool,
            stopped=self.stopped,
            key=key,
            chars=self.chars,
            open_string_chars=self._str_len if self._in_str else 0,
            max_whitespace_run=self.max_ws_run,
            tail_distinct=len(set(self._tail)),
        )
