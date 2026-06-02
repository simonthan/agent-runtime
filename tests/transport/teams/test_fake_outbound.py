"""FakeOutboundChannel test double behavior."""

import pytest

from agent_runtime.transport.teams.testing import FakeOutboundChannel


async def test_fake_records_text_sends_in_order():
    fake = FakeOutboundChannel()
    await fake.send_text("a")
    await fake.send_text("b")
    assert fake.sent_texts == ["a", "b"]


async def test_fake_records_cards():
    fake = FakeOutboundChannel()
    await fake.send_card({"type": "AdaptiveCard"})
    assert len(fake.sent_cards) == 1


async def test_fake_counts_typing():
    fake = FakeOutboundChannel()
    await fake.send_typing()
    await fake.send_typing()
    assert fake.sent_typing_count == 2


async def test_fake_clear_resets_state():
    fake = FakeOutboundChannel()
    await fake.send_text("x")
    await fake.send_card({})
    await fake.send_typing()
    fake.clear()
    assert fake.sent_texts == []
    assert fake.sent_cards == []
    assert fake.sent_typing_count == 0
