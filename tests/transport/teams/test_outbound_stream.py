"""T-316b -- BotFrameworkOutboundChannel.send_stream_activity: the Teams streaming wire format.

Teams streams a bot reply in a 1:1 chat as TYPING activities carrying a `streaminfo` entity
(`streamType` informative / streaming, `streamSequence` 1, 2, ...), closed by a MESSAGE
activity with `streamType: final` and no sequence. The first response id is the stream id.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from botbuilder.schema import ActivityTypes, ResourceResponse
from botframework.connector import models as connector_models
from msrest.serialization import Serializer

from agent_runtime.transport.teams import (
    STREAM_TYPE_FINAL,
    STREAM_TYPE_INFORMATIVE,
    STREAM_TYPE_STREAMING,
    TeamsStreamRejected,
)
from agent_runtime.transport.teams.outbound import BotFrameworkOutboundChannel


@pytest.fixture
def turn_context():
    tc = MagicMock()
    tc.send_activity = AsyncMock(return_value=ResourceResponse(id="stream-1"))
    return tc


def _wire(activity) -> dict:
    """The JSON the Connector client would POST (same Serializer setup as botframework's)."""
    types = {k: v for k, v in connector_models.__dict__.items() if isinstance(v, type)}
    return Serializer(types).body(activity, "Activity")


async def test_first_streaming_request_is_a_typing_activity_with_sequence(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    stream_id = await channel.send_stream_activity(
        text="The answer", stream_type=STREAM_TYPE_STREAMING, sequence=1
    )
    assert stream_id == "stream-1"
    activity = turn_context.send_activity.await_args.args[0]
    assert activity.type == ActivityTypes.typing
    body = _wire(activity)
    assert body["text"] == "The answer"
    assert body["entities"] == [
        {"type": "streaminfo", "streamType": "streaming", "streamSequence": 1}
    ]


async def test_later_request_carries_the_stream_id(turn_context):
    turn_context.send_activity.return_value = ResourceResponse(id="")
    channel = BotFrameworkOutboundChannel(turn_context)
    out = await channel.send_stream_activity(
        text="The answer is", stream_type=STREAM_TYPE_STREAMING, stream_id="stream-1", sequence=2
    )
    assert out is None  # 202 {} -> no new id
    body = _wire(turn_context.send_activity.await_args.args[0])
    assert body["entities"] == [
        {
            "type": "streaminfo",
            "streamId": "stream-1",
            "streamType": "streaming",
            "streamSequence": 2,
        }
    ]


async def test_informative_request(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    await channel.send_stream_activity(
        text="Searching...", stream_type=STREAM_TYPE_INFORMATIVE, sequence=1
    )
    body = _wire(turn_context.send_activity.await_args.args[0])
    assert body["type"] == "typing"
    assert body["entities"][0]["streamType"] == "informative"


async def test_final_request_is_a_message_without_sequence(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    await channel.send_stream_activity(
        text="The answer is 42.", stream_type=STREAM_TYPE_FINAL, stream_id="stream-1"
    )
    activity = turn_context.send_activity.await_args.args[0]
    assert activity.type == ActivityTypes.message
    body = _wire(activity)
    assert body["text"] == "The answer is 42."
    assert body["entities"] == [
        {"type": "streaminfo", "streamId": "stream-1", "streamType": "final"}
    ]


@pytest.mark.parametrize(
    ("stream_type", "sequence"),
    [
        ("bogus", 1),
        (STREAM_TYPE_FINAL, 3),
        (STREAM_TYPE_STREAMING, None),
        (STREAM_TYPE_STREAMING, 0),
    ],
)
async def test_bad_requests_raise_before_sending(turn_context, stream_type, sequence):
    channel = BotFrameworkOutboundChannel(turn_context)
    with pytest.raises(ValueError, match="bad stream request"):
        await channel.send_stream_activity(text="x", stream_type=stream_type, sequence=sequence)
    turn_context.send_activity.assert_not_awaited()


class _ConnectorError(Exception):
    """The msrest ErrorResponseException shape: `.response.status_code`, `.error.error`."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status)
        self.error = SimpleNamespace(error=SimpleNamespace(code=code, message=message))


async def test_user_stop_maps_to_a_cancelled_rejection(turn_context):
    turn_context.send_activity.side_effect = _ConnectorError(
        403, "ContentStreamNotAllowed", "Content stream was canceled by user."
    )
    channel = BotFrameworkOutboundChannel(turn_context)
    with pytest.raises(TeamsStreamRejected) as info:
        await channel.send_stream_activity(
            text="abc", stream_type=STREAM_TYPE_STREAMING, stream_id="s", sequence=4
        )
    assert info.value.status == 403
    assert info.value.code == "ContentStreamNotAllowed"
    assert info.value.user_cancelled is True


async def test_time_limit_rejection_is_not_a_user_stop(turn_context):
    turn_context.send_activity.side_effect = _ConnectorError(
        403, "ContentStreamNotAllowed", "Content stream finished due to exceeded streaming time."
    )
    channel = BotFrameworkOutboundChannel(turn_context)
    with pytest.raises(TeamsStreamRejected) as info:
        await channel.send_stream_activity(text="abc", stream_type=STREAM_TYPE_FINAL, stream_id="s")
    assert info.value.user_cancelled is False


async def test_non_http_failure_propagates_unchanged(turn_context):
    turn_context.send_activity.side_effect = ConnectionError("reset")
    channel = BotFrameworkOutboundChannel(turn_context)
    with pytest.raises(ConnectionError):
        await channel.send_stream_activity(
            text="abc", stream_type=STREAM_TYPE_STREAMING, sequence=1
        )
