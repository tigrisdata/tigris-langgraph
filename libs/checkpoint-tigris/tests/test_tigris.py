"""Synchronous integration tests against a live Tigris bucket."""

from __future__ import annotations

from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from .conftest import requires_tigris

pytestmark = requires_tigris


def _cfg(thread_id: str, ns: str = "", checkpoint_id: str | None = None) -> dict:
    configurable = {"thread_id": thread_id, "checkpoint_ns": ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def test_put_get_roundtrip(saver) -> None:
    cfg = _cfg("t1")
    chkpt = create_checkpoint(empty_checkpoint(), {}, 1)
    saved = saver.put(cfg, chkpt, {"source": "input", "step": 1}, {})
    assert saved["configurable"]["checkpoint_id"] == chkpt["id"]

    tup = saver.get_tuple(_cfg("t1"))
    assert tup is not None
    assert tup.checkpoint["id"] == chkpt["id"]
    assert tup.metadata["step"] == 1


def test_latest_is_returned(saver) -> None:
    first = create_checkpoint(empty_checkpoint(), {}, 1)
    saver.put(_cfg("t2"), first, {"step": 1}, {})
    second = create_checkpoint(empty_checkpoint(), {}, 2)
    saver.put(_cfg("t2", checkpoint_id=first["id"]), second, {"step": 2}, {})

    tup = saver.get_tuple(_cfg("t2"))
    assert tup is not None
    assert tup.checkpoint["id"] == second["id"]
    assert tup.parent_config["configurable"]["checkpoint_id"] == first["id"]


def test_list_orders_and_limits(saver) -> None:
    ids = []
    parent = None
    for step in range(3):
        cp = create_checkpoint(empty_checkpoint(), {}, step)
        saver.put(_cfg("t3", checkpoint_id=parent), cp, {"step": step}, {})
        ids.append(cp["id"])
        parent = cp["id"]

    listed = list(saver.list(_cfg("t3")))
    assert [t.checkpoint["id"] for t in listed] == list(reversed(ids))

    limited = list(saver.list(_cfg("t3"), limit=2))
    assert len(limited) == 2


def test_put_writes_and_pending(saver) -> None:
    cp = create_checkpoint(empty_checkpoint(), {}, 1)
    cfg = saver.put(_cfg("t4"), cp, {"step": 1}, {})
    saver.put_writes(cfg, [("channel_a", "value_a")], task_id="task-1")

    tup = saver.get_tuple(_cfg("t4"))
    assert tup is not None
    assert ("task-1", "channel_a", "value_a") in tup.pending_writes


def test_delete_thread(saver) -> None:
    cp = create_checkpoint(empty_checkpoint(), {}, 1)
    saver.put(_cfg("t5"), cp, {"step": 1}, {})
    assert saver.get_tuple(_cfg("t5")) is not None
    saver.delete_thread("t5")
    assert saver.get_tuple(_cfg("t5")) is None


def test_copy_thread(saver) -> None:
    parent = None
    ids = []
    for step in range(3):
        cp = create_checkpoint(empty_checkpoint(), {}, step)
        saver.put(_cfg("t6", checkpoint_id=parent), cp, {"step": step}, {})
        ids.append(cp["id"])
        parent = cp["id"]
    saver.put_writes(_cfg("t6", checkpoint_id=ids[-1]), [("ch", "v")], task_id="task-1")

    saver.copy_thread("t6", "t6-copy")

    # Whole lineage copied, newest-first order preserved, writes carried over.
    copied = list(saver.list(_cfg("t6-copy")))
    assert [t.checkpoint["id"] for t in copied] == list(reversed(ids))
    head = saver.get_tuple(_cfg("t6-copy"))
    assert ("task-1", "ch", "v") in head.pending_writes

    # Source untouched; the two threads are independent.
    saver.delete_thread("t6-copy")
    assert len(list(saver.list(_cfg("t6")))) == 3
    assert saver.get_tuple(_cfg("t6-copy")) is None
