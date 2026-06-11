"""Run the shared LangGraph checkpointer conformance suite against Tigris.

This is the primary correctness gate. `passed_all_base()` must be True.
Extended capabilities we do not implement are auto-skipped by the suite.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client
from langgraph.checkpoint.tigris.aio import AsyncTigrisSaver

from .conftest import requires_tigris

_ENDPOINT = os.getenv("TIGRIS_ENDPOINT_URL", DEFAULT_ENDPOINT)


def _assert_copy_thread(report) -> None:
    """Tigris implements copy_thread/acopy_thread, so it must be detected and pass."""
    result = report.results.get("copy_thread")
    assert result is not None, "copy_thread capability was not reported"
    assert result.detected, "copy_thread capability was not detected"
    assert result.passed is True, "copy_thread capability did not pass"


@checkpointer_test(name="TigrisSaver")
async def _sync_factory():
    bucket = os.environ["TIGRIS_TEST_BUCKET"]
    prefix = f"conformance/sync/{uuid.uuid4().hex}/"
    client = make_client(endpoint_url=_ENDPOINT)
    saver = TigrisSaver(bucket, client=client, prefix=prefix)
    saver.setup()
    try:
        yield saver
    finally:
        saver._delete_prefix(prefix)


@checkpointer_test(name="AsyncTigrisSaver")
async def _async_factory():
    bucket = os.environ["TIGRIS_TEST_BUCKET"]
    prefix = f"conformance/async/{uuid.uuid4().hex}/"
    saver = AsyncTigrisSaver(bucket, prefix=prefix, endpoint_url=_ENDPOINT)
    await saver.setup()
    try:
        yield saver
    finally:
        # reuse the sync client for cleanup
        client = make_client(endpoint_url=_ENDPOINT)
        TigrisSaver(bucket, client=client, prefix=prefix)._delete_prefix(prefix)


@requires_tigris
@pytest.mark.asyncio
async def test_sync_conformance() -> None:
    report = await validate(_sync_factory)
    report.print_report()
    assert report.passed_all_base()
    _assert_copy_thread(report)


@requires_tigris
@pytest.mark.asyncio
async def test_async_conformance() -> None:
    report = await validate(_async_factory)
    report.print_report()
    assert report.passed_all_base()
    _assert_copy_thread(report)
