"""Delta-channel reconstruction tests over a live Tigris bucket (sync + async).

`DeltaChannel` stores only a sentinel in each checkpoint blob; the live value is
rebuilt by replaying ancestor writes back to the nearest snapshot. That makes it
a sharp probe for two object-store correctness requirements the published
conformance suite (0.0.2) does not yet cover:

* a by-`checkpoint_id` `get_tuple` lookup must be a direct read, because the
  default `get_delta_channel_history` walks the parent chain one checkpoint at a
  time — a broken by-id lookup reconstructs every delta channel as empty,
  silently;
* `copy_thread` must copy the *whole* ancestor chain, not just the head, or the
  copied thread's delta channels reconstruct as empty.

We force replay (rather than per-step snapshots) with a high `snapshot_frequency`
and verify reconstruction from cold savers, after copy, and across the sync/async
boundary. Requires a live Tigris account; auto-skipped without credentials.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Annotated

import pytest
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client
from langgraph.checkpoint.tigris.aio import AsyncTigrisSaver

from .conftest import requires_tigris

pytestmark = requires_tigris

_ENDPOINT = os.getenv("TIGRIS_ENDPOINT_URL", DEFAULT_ENDPOINT)
_STEPS = 6
_EXPECTED = [f"x{i}" for i in range(_STEPS)]


def _reduce(state: list[str] | None, writes: Sequence[list[str]]) -> list[str]:
    out = list(state or [])
    for w in writes:
        out.extend(w)
    return out


class _State(TypedDict):
    # High snapshot_frequency => intermediate checkpoints store only the
    # sentinel, so reconstruction must replay the full ancestor write chain.
    log: Annotated[list[str], DeltaChannel(_reduce, snapshot_frequency=1000)]
    n: int


def _step(state: _State) -> dict:
    return {"log": [f"x{state['n']}"], "n": state["n"] + 1}


def _continue(state: _State) -> str:
    return "step" if state["n"] < _STEPS else END


def _build(checkpointer):
    return (
        StateGraph(_State)
        .add_node("step", _step)
        .add_edge(START, "step")
        .add_conditional_edges("step", _continue, ["step", END])
        .compile(checkpointer=checkpointer)
    )


@pytest.fixture
def async_saver() -> AsyncIterator[AsyncTigrisSaver]:
    bucket = os.environ["TIGRIS_TEST_BUCKET"]
    prefix = f"test/delta-async/{uuid.uuid4().hex}/"
    yield AsyncTigrisSaver(bucket, prefix=prefix, endpoint_url=_ENDPOINT)
    TigrisSaver(
        bucket, client=make_client(endpoint_url=_ENDPOINT), prefix=prefix
    )._delete_prefix(prefix)


def test_sync_delta_reconstructs_from_cold_saver_and_after_copy(
    saver: TigrisSaver,
) -> None:
    cfg = {"configurable": {"thread_id": "delta-sync"}}
    final = _build(saver).invoke({"log": [], "n": 0}, cfg)
    assert final["log"] == _EXPECTED

    # Deep ancestor chain: more than one checkpoint, so replay actually happens.
    assert len(list(_build(saver).get_state_history(cfg))) > _STEPS

    # Cold saver bound to the same bucket/prefix has no in-memory channel cache,
    # so this reconstruction runs entirely through get_delta_channel_history ->
    # get_tuple's by-id parent walk. Wrong-by-id lookup => empty log here.
    cold = TigrisSaver(saver.bucket, client=saver.client, prefix=saver.prefix)
    assert _build(cold).get_state(cfg).values["log"] == _EXPECTED

    # copy_thread must carry the full ancestor chain, not just the head.
    saver.copy_thread("delta-sync", "delta-sync-copy")
    copy_cfg = {"configurable": {"thread_id": "delta-sync-copy"}}
    fresh = TigrisSaver(saver.bucket, client=saver.client, prefix=saver.prefix)
    assert _build(fresh).get_state(copy_cfg).values["log"] == _EXPECTED


async def test_async_delta_reconstructs_from_cold_saver_and_after_copy(
    async_saver: AsyncTigrisSaver,
) -> None:
    await async_saver.setup()
    cfg = {"configurable": {"thread_id": "delta-async"}}
    final = await _build(async_saver).ainvoke({"log": [], "n": 0}, cfg)
    assert final["log"] == _EXPECTED

    history = [s async for s in _build(async_saver).aget_state_history(cfg)]
    assert len(history) > _STEPS

    cold = AsyncTigrisSaver(
        async_saver.bucket, prefix=async_saver.prefix, endpoint_url=_ENDPOINT
    )
    snap = await _build(cold).aget_state(cfg)
    assert snap.values["log"] == _EXPECTED

    await async_saver.acopy_thread("delta-async", "delta-async-copy")
    copy_cfg = {"configurable": {"thread_id": "delta-async-copy"}}
    fresh = AsyncTigrisSaver(
        async_saver.bucket, prefix=async_saver.prefix, endpoint_url=_ENDPOINT
    )
    snap = await _build(fresh).aget_state(copy_cfg)
    assert snap.values["log"] == _EXPECTED
