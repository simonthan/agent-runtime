"""Unit tests for ``agent_runtime.llm.models.LLMImage`` (T-067d)."""

from __future__ import annotations

import base64

import pytest

from agent_runtime.llm.models import LLMImage


def test_llm_image_from_bytes_roundtrip() -> None:
    data = b"\x89PNG\r\n\x1a\nfake-bytes"
    img = LLMImage.from_bytes(data, "image/png")
    assert base64.b64decode(img.data_b64) == data


def test_llm_image_to_block_shape() -> None:
    img = LLMImage(media_type="image/jpeg", data_b64="Zm9v")
    assert img.to_block() == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "Zm9v",
        },
    }


def test_llm_image_rejects_unsupported_media_type() -> None:
    with pytest.raises(ValueError, match="unsupported image media_type"):
        LLMImage(media_type="image/bmp", data_b64="Zm9v")
    with pytest.raises(ValueError, match="unsupported image media_type"):
        LLMImage(media_type="application/pdf", data_b64="Zm9v")
