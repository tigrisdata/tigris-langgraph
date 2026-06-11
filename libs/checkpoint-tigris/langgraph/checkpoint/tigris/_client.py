"""S3 client construction targeting the Tigris endpoint.

Credentials follow the standard AWS resolution chain (env vars
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, shared config, etc.). The
default endpoint is the public Tigris endpoint.
"""

from __future__ import annotations

from typing import Any

DEFAULT_ENDPOINT = "https://t3.storage.dev"
# Tigris ignores region for addressing but the SDK requires one to be set.
DEFAULT_REGION = "auto"


def make_client(
    *,
    endpoint_url: str = DEFAULT_ENDPOINT,
    region_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Build a synchronous boto3 S3 client pointed at Tigris."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name or DEFAULT_REGION,
        **kwargs,
    )


def make_async_session() -> Any:
    """Build an aioboto3 session for the async saver.

    aioboto3 clients are async context managers, so the async saver creates a
    client per operation from this session rather than holding one open.
    """
    import aioboto3

    return aioboto3.Session()
