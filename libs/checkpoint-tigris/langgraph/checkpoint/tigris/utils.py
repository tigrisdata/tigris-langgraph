"""Shared (de)serialization helpers for the Tigris checkpointer."""

from __future__ import annotations

import json
from typing import Any

# Manifest schema version, so the layout can evolve safely.
MANIFEST_VERSION = 1


def build_manifest(
    *,
    checkpoint_id: str,
    parent_checkpoint_id: str | None,
    checkpoint_type: str,
    metadata: dict[str, Any],
    ts: str | None,
    channel_versions: dict[str, Any],
) -> bytes:
    """Serialize a checkpoint manifest (small JSON sidecar) to bytes."""
    return json.dumps(
        {
            "v": MANIFEST_VERSION,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": checkpoint_type,
            "metadata": metadata,
            "ts": ts,
            "channel_versions": channel_versions,
        },
        ensure_ascii=False,
    ).encode("utf-8", "ignore")


def parse_manifest(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def matches_filter(metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Subset match: every key/value in `filter` must be present in metadata."""
    if not filter:
        return True
    return all(metadata.get(k) == v for k, v in filter.items())
