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


async def test_fake_send_text_returns_incrementing_activity_ids():
    fake = FakeOutboundChannel()
    assert await fake.send_text("one") == "activity-1"
    assert await fake.send_text("two") == "activity-2"


async def test_fake_update_activity_records_and_clears():
    fake = FakeOutboundChannel()
    aid = await fake.send_text("working…")
    assert await fake.update_activity(aid, "working… 12s") is True
    assert fake.updates == [("activity-1", "working… 12s")]
    fake.clear()
    assert fake.updates == []
    assert fake.supports_update is True  # capability survives clear()


async def test_fake_update_activity_unsupported_returns_false():
    fake = FakeOutboundChannel(supports_update=False)
    assert await fake.update_activity("activity-1", "x") is False
    assert fake.updates == []


async def test_fake_send_card_returns_incrementing_activity_ids():
    fake = FakeOutboundChannel()
    card1 = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "one"}]}
    card2 = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "two"}]}
    assert await fake.send_card(card1) == "card-activity-1"
    assert await fake.send_card(card2) == "card-activity-2"
    assert fake.sent_cards == [card1, card2]
