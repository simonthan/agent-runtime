"""Magic-byte image type detection, shared by every path that builds an ``LLMImage``.

Deliberately dependency-free and NOT under ``agent_runtime.llm``: importing any
``agent_runtime.llm`` submodule executes ``llm/__init__.py``, which imports the
``anthropic`` SDK (the ``[llm]`` extra). ``transport.teams`` must be able to sniff
without that dependency, so the implementation lives here and ``agent_runtime.llm``
re-exports it for consumers that already depend on the LLM surface.
``sniff_heif`` is the one deliberate exception: it detects a NON-supported container so
the Teams transport can transcode it (T-134-b).
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


# ISO-BMFF major brands that identify a HEIF/HEIC still-image container (iPhone
# camera output and its HEIF siblings). `mif1`/`msf1` are the generic HEIF
# structural brands. AVIF (`avif`/`avis`) is deliberately EXCLUDED: same box
# structure, different codec, out of this task's scope.
_HEIF_BRANDS: frozenset[bytes] = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"}
)


def sniff_heif(data: bytes) -> bool:
    """True if ``data`` is an ISO-BMFF HEIF/HEIC container (e.g. an iPhone photo).

    Unlike ``sniff_image_mime`` this does NOT return a mime type: HEIF is not an
    Anthropic-supported media type, so a positive here never feeds ``LLMImage``
    directly — it gates the transcode-to-JPEG recovery path in
    ``transport.teams.images`` (T-134-b). Detection is the standard ftyp-box
    check: bytes 4-8 are ``ftyp`` and the major brand at 8-12 is a HEIF brand.
    """
    if len(data) < 12 or data[4:8] != b"ftyp":  # noqa: PLR2004
        return False
    return data[8:12] in _HEIF_BRANDS
