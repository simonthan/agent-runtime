"""Magic-byte image type detection, shared by every path that builds an ``LLMImage``.

Deliberately dependency-free and NOT under ``agent_runtime.llm``: importing any
``agent_runtime.llm`` submodule executes ``llm/__init__.py``, which imports the
``anthropic`` SDK (the ``[llm]`` extra). ``transport.teams`` must be able to sniff
without that dependency, so the implementation lives here and ``agent_runtime.llm``
re-exports it for consumers that already depend on the LLM surface.
"""

from __future__ import annotations

_MAGIC_SNIFFS: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_mime(data: bytes) -> str | None:
    """Return the image MIME type from magic bytes, or None if not a known image.

    A declared content type cannot be trusted in either direction. Teams' attachment
    CDN serves inline images as ``application/octet-stream`` (TBP T-084 Issue 4), and
    MCP tool servers declare a supported type over unsupported bytes (TBP T-134-c) —
    the latter reaches the Anthropic API and 400s the whole turn. The four types here
    are exactly Anthropic's supported image media types, so a non-None return is
    always a member of ``ANTHROPIC_IMAGE_MEDIA_TYPES``. WebP is RIFF-framed
    (``RIFF<size>WEBP``), hence the offset check.
    """
    for magic, mime in _MAGIC_SNIFFS:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":  # noqa: PLR2004
        return "image/webp"
    return None
