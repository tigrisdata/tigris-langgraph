"""Object-key layout helpers for the Tigris checkpointer.

All keys live under an optional `prefix` and the `checkpoints/` root:

    {prefix}checkpoints/{thread}/{ns}/{checkpoint_id}/manifest.json
    {prefix}checkpoints/{thread}/{ns}/{checkpoint_id}/checkpoint.bin
    {prefix}checkpoints/{thread}/{ns}/{checkpoint_id}/writes/{task_id}/{idx}.bin

Notes:
* `checkpoint_ns` may be empty; it is encoded as the literal `__default__`
  segment so we never emit an empty path component.
* `checkpoint_id` values are time-sortable, so lexical listing order matches
  chronological order and "latest" needs no mutable pointer.
* Write indices may be negative (see `WRITES_IDX_MAP`); we offset them so the
  zero-padded filename remains lexically sortable and dash-free.

These functions are pure (no network) and are unit-tested in `test_keys.py`.
"""

from __future__ import annotations

from urllib.parse import quote, unquote

ROOT = "checkpoints"
DEFAULT_NS = "__default__"
MANIFEST = "manifest.json"
BLOB = "checkpoint.bin"
WRITES = "writes"
# Offset write indices so negative special-channel indices stay non-negative and
# lexically sortable. Must exceed the largest expected number of writes per task.
IDX_OFFSET = 1_000_000


def _enc(segment: str) -> str:
    """URL-encode a single path segment (so `/` etc. cannot break the layout)."""
    return quote(str(segment), safe="")


def _dec(segment: str) -> str:
    return unquote(segment)


def _ns(checkpoint_ns: str) -> str:
    return _enc(checkpoint_ns) if checkpoint_ns else DEFAULT_NS


def _root(prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{ROOT}/" if prefix else f"{ROOT}/"


def thread_prefix(prefix: str, thread_id: str) -> str:
    """Prefix covering every checkpoint/ns/write under a thread (used by delete)."""
    return f"{_root(prefix)}{_enc(thread_id)}/"


def ns_prefix(prefix: str, thread_id: str, checkpoint_ns: str) -> str:
    return f"{thread_prefix(prefix, thread_id)}{_ns(checkpoint_ns)}/"


def checkpoint_prefix(
    prefix: str, thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return f"{ns_prefix(prefix, thread_id, checkpoint_ns)}{_enc(checkpoint_id)}/"


def manifest_key(
    prefix: str, thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return f"{checkpoint_prefix(prefix, thread_id, checkpoint_ns, checkpoint_id)}{MANIFEST}"


def blob_key(
    prefix: str, thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return f"{checkpoint_prefix(prefix, thread_id, checkpoint_ns, checkpoint_id)}{BLOB}"


def writes_prefix(
    prefix: str, thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return (
        f"{checkpoint_prefix(prefix, thread_id, checkpoint_ns, checkpoint_id)}{WRITES}/"
    )


def write_key(
    prefix: str,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int,
) -> str:
    name = f"{idx + IDX_OFFSET:09d}.bin"
    return f"{writes_prefix(prefix, thread_id, checkpoint_ns, checkpoint_id)}{_enc(task_id)}/{name}"


def is_manifest_key(key: str) -> bool:
    return key.endswith(f"/{MANIFEST}")


def checkpoint_id_from_manifest_key(key: str) -> str:
    """Extract and decode the checkpoint id from a `.../{id}/manifest.json` key."""
    parts = key.split("/")
    # parts[-1] == MANIFEST, parts[-2] == encoded checkpoint id
    return _dec(parts[-2])


def ns_from_manifest_key(key: str) -> str:
    """Extract the (decoded) checkpoint_ns from a manifest key; '' for default."""
    parts = key.split("/")
    raw = parts[-3]
    return "" if raw == DEFAULT_NS else _dec(raw)


def parse_write_key(key: str) -> tuple[str, int]:
    """Return `(task_id, idx)` parsed from a write object key."""
    parts = key.split("/")
    task_id = _dec(parts[-2])
    idx = int(parts[-1].removesuffix(".bin")) - IDX_OFFSET
    return task_id, idx
