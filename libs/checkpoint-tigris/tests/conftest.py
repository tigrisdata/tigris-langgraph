"""Shared fixtures for Tigris checkpointer tests.

Integration and fork tests require a live Tigris account. They are skipped
unless the following environment variables are set:

* `TIGRIS_TEST_BUCKET` — an existing Single-region or Multi-region bucket.
* `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — Tigris access keys.
* `TIGRIS_ENDPOINT_URL` — optional, defaults to https://t3.storage.dev.

Each test run uses a unique key prefix and deletes it on teardown so runs stay
isolated and idempotent. Fork tests additionally create and delete fork buckets.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client

_HAS_CREDS = bool(
    os.getenv("TIGRIS_TEST_BUCKET")
    and os.getenv("AWS_ACCESS_KEY_ID")
    and os.getenv("AWS_SECRET_ACCESS_KEY")
)

requires_tigris = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="Set TIGRIS_TEST_BUCKET + AWS credentials to run live Tigris tests.",
)


@pytest.fixture
def endpoint_url() -> str:
    return os.getenv("TIGRIS_ENDPOINT_URL", DEFAULT_ENDPOINT)


@pytest.fixture
def test_bucket() -> str:
    return os.environ["TIGRIS_TEST_BUCKET"]


@pytest.fixture
def unique_prefix() -> str:
    return f"test/{uuid.uuid4().hex}/"


@pytest.fixture
def saver(
    test_bucket: str, unique_prefix: str, endpoint_url: str
) -> Iterator[TigrisSaver]:
    client = make_client(endpoint_url=endpoint_url)
    s = TigrisSaver(test_bucket, client=client, prefix=unique_prefix)
    s.setup()
    try:
        yield s
    finally:
        s._delete_prefix(unique_prefix)
