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
    await fake.send_oauth_card({})
    fake.clear()
    assert fake.sent_texts == []
    assert fake.sent_cards == []
    assert fake.sent_typing_count == 0
    assert fake.sent_oauth_cards == []


async def test_fake_records_oauth_cards():
    fake = FakeOutboundChannel()
    await fake.send_oauth_card({"tokenExchangeResource": {"uri": "api://x"}})
    assert len(fake.sent_oauth_cards) == 1


async def test_fake_get_sign_in_resource_returns_injected():
    from agent_runtime.transport.teams.outbound import SignInResource

    ch = FakeOutboundChannel(
        sign_in_resource=SignInResource(sign_in_link="L", token_exchange_uri="U")
    )
    r = await ch.get_sign_in_resource(connection_name="c")
    assert r is not None
    assert r.sign_in_link == "L"
    assert r.token_exchange_uri == "U"


async def test_fake_get_sign_in_resource_defaults_none():
    ch = FakeOutboundChannel()
    assert await ch.get_sign_in_resource(connection_name="c") is None
