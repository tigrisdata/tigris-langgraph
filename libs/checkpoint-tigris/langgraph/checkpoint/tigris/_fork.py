"""Tigris zero-copy bucket-fork helper.

A Tigris fork creates a new bucket that shares all objects with the source by
reference (no data copy), created instantly, with writes isolated from the
source. Over the S3 API this is a `CreateBucket` call carrying the
`X-Tigris-Fork-Source-Bucket` header.

A bucket can only be forked if it is snapshot-enabled. Enable snapshots at
creation with the `X-Tigris-Enable-Snapshot: true` header (see
`create_snapshot_bucket`), or convert an existing bucket in the Tigris
dashboard. Forking a plain bucket fails with `Source bucket must be of snapshot
type`.

We inject these headers via a boto3 event handler so we do not depend on
`tigris-boto3-ext`. If you prefer that package, it offers equivalent helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Enable snapshots/forking on a new bucket. Required for the bucket to be a
# valid fork source.
ENABLE_SNAPSHOT_HEADER = "X-Tigris-Enable-Snapshot"
# Source bucket to fork from. Without a snapshot header, Tigris snapshots the
# source at request time and forks from that.
FORK_SOURCE_HEADER = "X-Tigris-Fork-Source-Bucket"
# Optional: pin the fork to a specific snapshot of the source. The value is a
# UNIX nanosecond-precision timestamp (e.g. "1751631910140685342").
FORK_SNAPSHOT_HEADER = "X-Tigris-Fork-Source-Bucket-Snapshot"


@contextmanager
def _create_bucket_headers(s3_client: Any, headers: dict[str, str]) -> Iterator[None]:
    """Inject custom headers onto the next CreateBucket call on this client."""

    def _add_headers(request: Any, **_: Any) -> None:
        for name, value in headers.items():
            request.headers[name] = value

    event = "before-sign.s3.CreateBucket"
    s3_client.meta.events.register(event, _add_headers)
    try:
        yield
    finally:
        s3_client.meta.events.unregister(event, _add_headers)


def create_snapshot_bucket(s3_client: Any, bucket: str) -> None:
    """Create `bucket` with snapshots enabled, making it a valid fork source."""
    with _create_bucket_headers(s3_client, {ENABLE_SNAPSHOT_HEADER: "true"}):
        s3_client.create_bucket(Bucket=bucket)


def create_bucket_fork(
    s3_client: Any,
    target_bucket: str,
    source_bucket: str,
    *,
    snapshot_version: str | None = None,
) -> None:
    """Create `target_bucket` as a zero-copy fork of `source_bucket`.

    `source_bucket` must be snapshot-enabled (see `create_snapshot_bucket`).
    """
    headers = {FORK_SOURCE_HEADER: source_bucket}
    if snapshot_version:
        headers[FORK_SNAPSHOT_HEADER] = snapshot_version
    with _create_bucket_headers(s3_client, headers):
        s3_client.create_bucket(Bucket=target_bucket)
