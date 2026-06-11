"""End-to-end tests: compile a real StateGraph on the Tigris saver and assert
resume + time-travel work over a live bucket (sync and async).

Requires a live Tigris account; auto-skipped without credentials.
"""

from __future__ import annotations

import operator
import os
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client
from langgraph.checkpoint.tigris.aio import AsyncTigrisSaver

from .conftest import requires_tigris

pytestmark = requires_tigris

_ENDPOINT = os.getenv("TIGRIS_ENDPOINT_URL", DEFAULT_ENDPOINT)


class _State(TypedDict):
    value: int
    trail: Annotated[list[str], operator.add]


def _node_a(state: _State) -> dict:
    return {"value": state["value"] + 1, "trail": ["a"]}


def _node_b(state: _State) -> dict:
    return {"value": state["value"] + 10, "trail": ["b"]}


def _build(checkpointer):
    return (
        StateGraph(_State)
        .add_node("a", _node_a)
        .add_node("b", _node_b)
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile(checkpointer=checkpointer)
    )


@pytest.fixture
def async_saver() -> AsyncIterator[AsyncTigrisSaver]:
    bucket = os.environ["TIGRIS_TEST_BUCKET"]
    prefix = f"test/e2e-async/{uuid.uuid4().hex}/"
    yield AsyncTigrisSaver(bucket, prefix=prefix, endpoint_url=_ENDPOINT)
    TigrisSaver(
        bucket, client=make_client(endpoint_url=_ENDPOINT), prefix=prefix
    )._delete_prefix(prefix)


def test_sync_resume_and_time_travel(saver: TigrisSaver) -> None:
    graph = _build(saver)
    cfg = {"configurable": {"thread_id": "e2e-sync"}}

    final = graph.invoke({"value": 0, "trail": []}, cfg)
    assert final["value"] == 11
    assert final["trail"] == ["a", "b"]

    # Resume: state persisted across a fresh graph bound to the same saver.
    fresh = _build(saver)
    snap = fresh.get_state(cfg)
    assert snap.values["value"] == 11
    assert snap.next == ()

    # Time travel: find the checkpoint taken just before node "b" ran and
    # branch from it. trail == ["a"] and the pending node is "b".
    history = list(graph.get_state_history(cfg))
    before_b = next(s for s in history if s.next == ("b",))
    assert before_b.values["trail"] == ["a"]

    resumed = graph.invoke(None, before_b.config)
    assert resumed["value"] == 11
    assert resumed["trail"] == ["a", "b"]


async def test_async_resume_and_time_travel(async_saver: AsyncTigrisSaver) -> None:
    await async_saver.setup()
    graph = _build(async_saver)
    cfg = {"configurable": {"thread_id": "e2e-async"}}

    final = await graph.ainvoke({"value": 0, "trail": []}, cfg)
    assert final["value"] == 11
    assert final["trail"] == ["a", "b"]

    snap = await graph.aget_state(cfg)
    assert snap.values["value"] == 11
    assert snap.next == ()

    history = [s async for s in graph.aget_state_history(cfg)]
    before_b = next(s for s in history if s.next == ("b",))
    assert before_b.values["trail"] == ["a"]

    resumed = await graph.ainvoke(None, before_b.config)
    assert resumed["value"] == 11
    assert resumed["trail"] == ["a", "b"]
