"""Asynchronous integration tests against a live Tigris bucket."""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client
from langgraph.checkpoint.tigris.aio import AsyncTigrisSaver

from .conftest import requires_tigris

pytestmark = requires_tigris

_ENDPOINT = os.getenv("TIGRIS_ENDPOINT_URL", DEFAULT_ENDPOINT)


def _cfg(thread_id: str, ns: str = "", checkpoint_id: str | None = None) -> dict:
    configurable = {"thread_id": thread_id, "checkpoint_ns": ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


@pytest.fixture
async def async_saver():
    bucket = os.environ["TIGRIS_TEST_BUCKET"]
    prefix = f"test/async/{uuid.uuid4().hex}/"
    saver = AsyncTigrisSaver(bucket, prefix=prefix, endpoint_url=_ENDPOINT)
    await saver.setup()
    try:
        yield saver
    finally:
        make_client(endpoint_url=_ENDPOINT)
        TigrisSaver(
            bucket, client=make_client(endpoint_url=_ENDPOINT), prefix=prefix
        )._delete_prefix(prefix)


@pytest.mark.asyncio
async def test_async_put_get(async_saver) -> None:
    cp = create_checkpoint(empty_checkpoint(), {}, 1)
    saved = await async_saver.aput(_cfg("at1"), cp, {"step": 1}, {})
    assert saved["configurable"]["checkpoint_id"] == cp["id"]

    tup = await async_saver.aget_tuple(_cfg("at1"))
    assert tup is not None
    assert tup.checkpoint["id"] == cp["id"]


@pytest.mark.asyncio
async def test_async_list(async_saver) -> None:
    parent = None
    ids = []
    for step in range(3):
        cp = create_checkpoint(empty_checkpoint(), {}, step)
        await async_saver.aput(
            _cfg("at2", checkpoint_id=parent), cp, {"step": step}, {}
        )
        ids.append(cp["id"])
        parent = cp["id"]

    listed = [t async for t in async_saver.alist(_cfg("at2"))]
    assert [t.checkpoint["id"] for t in listed] == list(reversed(ids))


@pytest.mark.asyncio
async def test_async_writes_and_delete(async_saver) -> None:
    cp = create_checkpoint(empty_checkpoint(), {}, 1)
    cfg = await async_saver.aput(_cfg("at3"), cp, {"step": 1}, {})
    await async_saver.aput_writes(cfg, [("ch", "v")], task_id="task-1")

    tup = await async_saver.aget_tuple(_cfg("at3"))
    assert tup is not None
    assert ("task-1", "ch", "v") in tup.pending_writes

    await async_saver.adelete_thread("at3")
    assert await async_saver.aget_tuple(_cfg("at3")) is None


@pytest.mark.asyncio
async def test_async_copy_thread(async_saver) -> None:
    parent = None
    ids = []
    for step in range(3):
        cp = create_checkpoint(empty_checkpoint(), {}, step)
        await async_saver.aput(
            _cfg("at4", checkpoint_id=parent), cp, {"step": step}, {}
        )
        ids.append(cp["id"])
        parent = cp["id"]
    await async_saver.aput_writes(
        _cfg("at4", checkpoint_id=ids[-1]), [("ch", "v")], task_id="task-1"
    )

    await async_saver.acopy_thread("at4", "at4-copy")

    copied = [t async for t in async_saver.alist(_cfg("at4-copy"))]
    assert [t.checkpoint["id"] for t in copied] == list(reversed(ids))
    head = await async_saver.aget_tuple(_cfg("at4-copy"))
    assert ("task-1", "ch", "v") in head.pending_writes

    # Source untouched after copying.
    src = [t async for t in async_saver.alist(_cfg("at4"))]
    assert len(src) == 3
