from pathlib import Path

import agent_runtime.image_sniff
from agent_runtime.image_sniff import sniff_image_mime
from agent_runtime.llm import ANTHROPIC_IMAGE_MEDIA_TYPES
from agent_runtime.llm import sniff_image_mime as llm_sniff_image_mime

_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF"
_GIF87A = b"GIF87a\x01\x00\x01\x00"
_GIF89A = b"GIF89a\x01\x00\x01\x00"
_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP"


def test_sniffs_png() -> None:
    assert sniff_image_mime(_PNG) == "image/png"


def test_sniffs_jpeg() -> None:
    assert sniff_image_mime(_JPEG) == "image/jpeg"


def test_sniffs_gif87a() -> None:
    assert sniff_image_mime(_GIF87A) == "image/gif"


def test_sniffs_gif89a() -> None:
    assert sniff_image_mime(_GIF89A) == "image/gif"


def test_sniffs_webp() -> None:
    assert sniff_image_mime(_WEBP) == "image/webp"


def test_empty_bytes_returns_none() -> None:
    assert sniff_image_mime(b"") is None


def test_pdf_bytes_returns_none() -> None:
    assert sniff_image_mime(b"%PDF-1.7") is None


def test_short_rif_does_not_indexerror() -> None:
    """The WebP branch must not IndexError on a payload shorter than 12 bytes."""
    assert sniff_image_mime(b"RIF") is None


def test_riff_container_that_is_not_webp_returns_none() -> None:
    assert sniff_image_mime(b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE") is None


def test_every_non_none_sniff_is_an_anthropic_supported_type() -> None:
    """Contract test: the tbp consumer relies on this invariant to guarantee an
    LLMImage built from a sniffed type can never raise."""
    samples = (_PNG, _JPEG, _GIF87A, _GIF89A, _WEBP)
    for sample in samples:
        sniffed = sniff_image_mime(sample)
        assert sniffed is not None
        assert sniffed in ANTHROPIC_IMAGE_MEDIA_TYPES


def test_reexport_is_the_same_object() -> None:
    assert llm_sniff_image_mime is sniff_image_mime


def test_image_sniff_module_has_no_llm_dependency() -> None:
    """A `[teams]`-only install must be able to sniff. `agent_runtime.llm` imports the
    `anthropic` SDK (the `[llm]` extra), so the sniffer must not live under it."""
    src = Path(agent_runtime.image_sniff.__file__).read_text(encoding="utf-8")
    assert "import anthropic" not in src
    assert "agent_runtime.llm" not in src.split('"""', 2)[-1]  # docstring may name it


_HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
_HEIF_MIF1 = b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00mif1heic"
_AVIF = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"


def test_sniff_heif_detects_heic_brand() -> None:
    from agent_runtime.image_sniff import sniff_heif

    assert sniff_heif(_HEIC) is True


def test_sniff_heif_detects_generic_mif1_brand() -> None:
    from agent_runtime.image_sniff import sniff_heif

    assert sniff_heif(_HEIF_MIF1) is True


def test_sniff_heif_rejects_avif_and_non_heif() -> None:
    from agent_runtime.image_sniff import sniff_heif

    assert sniff_heif(_AVIF) is False
    assert sniff_heif(_JPEG) is False
    assert sniff_heif(b"") is False
    assert sniff_heif(b"\x00\x00\x00\x18fty") is False  # shorter than 12 bytes


def test_heic_bytes_still_return_none_from_sniff_image_mime() -> None:
    """sniff_image_mime's contract is untouched: HEIC is not an Anthropic type."""
    assert sniff_image_mime(_HEIC) is None


def test_sniff_heif_reexport_is_the_same_object() -> None:
    """Mirror of test_reexport_is_the_same_object (line ~62) for the new symbol."""
    from agent_runtime.image_sniff import sniff_heif
    from agent_runtime.llm import sniff_heif as llm_sniff_heif

    assert llm_sniff_heif is sniff_heif
