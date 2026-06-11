"""Fork branching tests: a fork inherits prior checkpoints and is then isolated.

Requires a live Tigris account; creates and deletes temporary fork buckets.
"""

from __future__ import annotations

import os
import uuid

import pytest
from botocore.exceptions import ClientError
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client
from langgraph.checkpoint.tigris._fork import create_snapshot_bucket

from .conftest import requires_tigris

pytestmark = requires_tigris

_ENDPOINT = os.getenv("TIGRIS_ENDPOINT_URL", DEFAULT_ENDPOINT)


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> dict:
    configurable = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


@pytest.fixture
def source_bucket():
    """A dedicated, snapshot-enabled source bucket so we can fork it cleanly."""
    client = make_client(endpoint_url=_ENDPOINT)
    name = f"lg-cptest-src-{uuid.uuid4().hex[:12]}"
    create_snapshot_bucket(client, name)
    try:
        yield name
    finally:
        _drop_bucket(client, name)


def _drop_bucket(client, name: str) -> None:
    # Empty every object version + delete marker, then drop the bucket. Delete a
    # fork before its source — a source cannot be removed while a fork depends on
    # it. Snapshot-enabled buckets retain snapshots that the S3 API cannot delete
    # directly (the t3.storage.dev endpoint has no force-delete), so the final
    # DeleteBucket is best-effort: a lingering snapshot leaves the temp bucket to
    # be reaped by Tigris / the dashboard rather than failing the test.
    paginator = client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=name):
        objs = [
            {"Key": o["Key"], "VersionId": o["VersionId"]}
            for o in page.get("Versions", []) + page.get("DeleteMarkers", [])
        ]
        if objs:
            client.delete_objects(Bucket=name, Delete={"Objects": objs})
    try:
        client.delete_bucket(Bucket=name)
    except ClientError as exc:  # noqa: PERF203
        if exc.response.get("Error", {}).get("Code") != "BucketNotEmpty":
            raise


def test_fork_inherits_then_isolates(source_bucket) -> None:
    client = make_client(endpoint_url=_ENDPOINT)
    src = TigrisSaver(source_bucket, client=client)

    base = create_checkpoint(empty_checkpoint(), {}, 1)
    src.put(_cfg("thread"), base, {"step": 1}, {})

    fork_name = f"lg-cptest-fork-{uuid.uuid4().hex[:12]}"
    fork = src.fork(fork_name)
    try:
        # Fork inherits the pre-fork checkpoint.
        inherited = fork.get_tuple(_cfg("thread"))
        assert inherited is not None
        assert inherited.checkpoint["id"] == base["id"]

        # Diverge: write only to the fork.
        forked_cp = create_checkpoint(empty_checkpoint(), {}, 2)
        fork.put(_cfg("thread", checkpoint_id=base["id"]), forked_cp, {"step": 2}, {})

        # Source is unchanged (isolation).
        src_latest = src.get_tuple(_cfg("thread"))
        assert src_latest.checkpoint["id"] == base["id"]

        # Fork sees the new head.
        fork_latest = fork.get_tuple(_cfg("thread"))
        assert fork_latest.checkpoint["id"] == forked_cp["id"]
    finally:
        _drop_bucket(client, fork_name)
