"""Tigris object-storage checkpoint saver for LangGraph (synchronous).

Stores each checkpoint as immutable, uniquely-keyed objects in a Tigris bucket
over the S3 API, plus a zero-copy `fork()` helper for instant branching of an
entire thread lineage.

> **Bucket requirement:** use a **Single-region** or **Multi-region** Tigris
> bucket. These provide strong, globally consistent reads/lists/conditional
> operations, which this saver relies on to find the latest checkpoint without a
> mutable pointer. Global/Dual-region buckets give only eventual cross-region
> consistency.

Example:

```python
from langgraph.checkpoint.tigris import TigrisSaver

with TigrisSaver.from_conn_string("my-bucket") as checkpointer:
    checkpointer.setup()
    # use as the `checkpointer=` for any compiled LangGraph graph
```
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph.checkpoint.tigris import _keys
from langgraph.checkpoint.tigris._client import DEFAULT_ENDPOINT, make_client
from langgraph.checkpoint.tigris._fork import create_bucket_fork
from langgraph.checkpoint.tigris.utils import (
    build_manifest,
    matches_filter,
    parse_manifest,
)

__all__ = ["TigrisSaver"]

META_CHANNEL = "channel"
META_STYPE = "stype"


class TigrisSaver(BaseCheckpointSaver[str]):
    """A LangGraph checkpoint saver backed by a Tigris bucket (S3 API)."""

    def __init__(
        self,
        bucket: str,
        *,
        client: Any | None = None,
        prefix: str = "",
        endpoint_url: str = DEFAULT_ENDPOINT,
        region_name: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.bucket = bucket
        self.prefix = prefix
        self.client = client or make_client(
            endpoint_url=endpoint_url, region_name=region_name
        )
        self.is_setup = False

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str = DEFAULT_ENDPOINT,
        region_name: str | None = None,
    ) -> Iterator[TigrisSaver]:
        """Create a saver using credentials from the standard AWS env chain."""
        yield cls(
            bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

    def setup(self) -> None:
        """Verify the bucket is reachable. Idempotent; safe to call repeatedly.

        This does not attempt to detect the bucket's consistency class: that is
        not exposed via the S3 API, and within a region every bucket type gives
        the read-after-write and list-after-write consistency this saver needs.
        Cross-region eventual consistency is the only caveat — see the module
        docstring's bucket requirement.
        """
        if self.is_setup:
            return
        self.client.head_bucket(Bucket=self.bucket)
        self.is_setup = True

    # ------------------------------------------------------------------ fork
    def fork(
        self, target_bucket: str, *, snapshot_version: str | None = None
    ) -> TigrisSaver:
        """Instantly fork the underlying bucket (zero-copy) and return a saver
        bound to the new fork. The fork starts as a full, isolated copy of every
        thread and checkpoint in this bucket.

        The underlying bucket must be snapshot-enabled to be forkable (create it
        with the `X-Tigris-Enable-Snapshot: true` header, or enable snapshots in
        the Tigris dashboard); otherwise Tigris rejects the fork with `Source
        bucket must be of snapshot type`. Pass `snapshot_version` (a UNIX
        nanosecond timestamp) to fork from a specific snapshot instead of a
        fresh one taken now."""
        create_bucket_fork(
            self.client,
            target_bucket,
            self.bucket,
            snapshot_version=snapshot_version,
        )
        return TigrisSaver(
            target_bucket,
            client=self.client,
            prefix=self.prefix,
            serde=self.serde,
        )

    # ----------------------------------------------------------------- writes
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        type_, blob = self.serde.dumps_typed(checkpoint)
        # Checkpoint blob (overwrite allowed: re-puts for the same id are updates).
        self._put_object(
            _keys.blob_key(self.prefix, thread_id, checkpoint_ns, checkpoint_id),
            blob,
        )
        self._put_object(
            _keys.manifest_key(self.prefix, thread_id, checkpoint_ns, checkpoint_id),
            build_manifest(
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_id,
                checkpoint_type=type_,
                metadata=dict(get_checkpoint_metadata(config, metadata)),
                ts=checkpoint.get("ts"),
                channel_versions=dict(new_versions),
            ),
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            widx = WRITES_IDX_MAP.get(channel, idx)
            type_, blob = self.serde.dumps_typed(value)
            self._put_object(
                _keys.write_key(
                    self.prefix,
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    widx,
                ),
                blob,
                metadata={META_CHANNEL: _keys._enc(channel), META_STYPE: type_},
            )

    # ----------------------------------------------------------------- reads
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            checkpoint_id = self._latest_checkpoint_id(thread_id, checkpoint_ns)
            if checkpoint_id is None:
                return None

        return self._load_tuple(thread_id, checkpoint_ns, checkpoint_id)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns")

        before_id = get_checkpoint_id(before) if before else None
        prefix = (
            _keys.ns_prefix(self.prefix, thread_id, checkpoint_ns)
            if checkpoint_ns is not None
            else _keys.thread_prefix(self.prefix, thread_id)
        )

        # Manifest keys sorted descending by checkpoint id (lexical == chrono).
        manifests = sorted(
            (k for k in self._list_keys(prefix) if _keys.is_manifest_key(k)),
            key=_keys.checkpoint_id_from_manifest_key,
            reverse=True,
        )

        count = 0
        for key in manifests:
            cp_id = _keys.checkpoint_id_from_manifest_key(key)
            if before_id is not None and cp_id >= before_id:
                continue
            ns = _keys.ns_from_manifest_key(key)
            tup = self._load_tuple(thread_id, ns, cp_id)
            if tup is None:
                continue
            if not matches_filter(dict(tup.metadata), filter):
                continue
            yield tup
            count += 1
            if limit is not None and count >= limit:
                return

    def delete_thread(self, thread_id: str) -> None:
        prefix = _keys.thread_prefix(self.prefix, str(thread_id))
        self._delete_prefix(prefix)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy every checkpoint, write, and namespace of a thread to a new one.

        A thread's objects all live under a single key prefix, so this is a
        batch of server-side S3 copies (no download/re-serialize) that rewrites
        only the thread segment of each key. The whole parent chain is copied,
        so the target thread is independently resumable. The source is left
        untouched. (See `fork()` for whole-bucket, cross-thread branching.)
        """
        src_prefix = _keys.thread_prefix(self.prefix, str(source_thread_id))
        dst_prefix = _keys.thread_prefix(self.prefix, str(target_thread_id))
        for key in self._list_keys(src_prefix):
            self._copy_object(key, dst_prefix + key[len(src_prefix) :])

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # ------------------------------------------------------------------- async
    # The sync saver is built on the blocking boto3 client. These async methods
    # run the sync implementations in a worker thread so the saver can also be
    # driven from async LangGraph runs. (boto3 clients are safe to share across
    # threads.) Use `AsyncTigrisSaver` for a native non-blocking aioboto3 path.
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: list(self.list(config, filter=filter, before=before, limit=limit)),
        )
        for item in items:
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self.delete_thread, thread_id
        )

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self.copy_thread, source_thread_id, target_thread_id
        )

    # --------------------------------------------------------------- internals
    def _load_tuple(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> CheckpointTuple | None:
        manifest_raw = self._get_object(
            _keys.manifest_key(self.prefix, thread_id, checkpoint_ns, checkpoint_id)
        )
        if manifest_raw is None:
            return None
        manifest = parse_manifest(manifest_raw)
        blob = self._get_object(
            _keys.blob_key(self.prefix, thread_id, checkpoint_ns, checkpoint_id)
        )
        if blob is None:
            return None

        checkpoint = self.serde.loads_typed((manifest["type"], blob))
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_id = manifest.get("parent_checkpoint_id")
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )
        pending_writes = self._load_writes(thread_id, checkpoint_ns, checkpoint_id)
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=manifest.get("metadata", {}),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _load_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        prefix = _keys.writes_prefix(
            self.prefix, thread_id, checkpoint_ns, checkpoint_id
        )
        keys = sorted(self._list_keys(prefix), key=_keys.parse_write_key)
        writes: list[tuple[str, str, Any]] = []
        for key in keys:
            task_id, _ = _keys.parse_write_key(key)
            body, meta = self._get_object_with_meta(key)
            if body is None:
                continue
            channel = _keys._dec(meta.get(META_CHANNEL, ""))
            stype = meta.get(META_STYPE, "")
            value = self.serde.loads_typed((stype, body))
            writes.append((task_id, channel, value))
        return writes

    def _latest_checkpoint_id(self, thread_id: str, checkpoint_ns: str) -> str | None:
        prefix = _keys.ns_prefix(self.prefix, thread_id, checkpoint_ns)
        manifests = [k for k in self._list_keys(prefix) if _keys.is_manifest_key(k)]
        if not manifests:
            return None
        latest = max(manifests, key=_keys.checkpoint_id_from_manifest_key)
        return _keys.checkpoint_id_from_manifest_key(latest)

    # --------------------------------------------------------- S3 primitives
    def _put_object(
        self, key: str, body: bytes, metadata: dict[str, str] | None = None
    ) -> None:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key, "Body": body}
        if metadata:
            kwargs["Metadata"] = metadata
        self.client.put_object(**kwargs)

    def _get_object(self, key: str) -> bytes | None:
        body, _ = self._get_object_with_meta(key)
        return body

    def _copy_object(self, source_key: str, dest_key: str) -> None:
        # Server-side copy; preserves user metadata (channel/stype on writes) by
        # default (MetadataDirective=COPY).
        self.client.copy_object(
            Bucket=self.bucket,
            Key=dest_key,
            CopySource={"Bucket": self.bucket, "Key": source_key},
        )

    def _get_object_with_meta(self, key: str) -> tuple[bytes | None, dict[str, str]]:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None, {}
        except Exception as exc:  # noqa: BLE001
            if _is_not_found(exc):
                return None, {}
            raise
        return resp["Body"].read(), resp.get("Metadata", {})

    def _list_keys(self, prefix: str) -> Iterator[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def _delete_prefix(self, prefix: str) -> None:
        batch: list[dict[str, str]] = []
        for key in self._list_keys(prefix):
            batch.append({"Key": key})
            if len(batch) == 1000:
                self.client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": batch}
                )
                batch = []
        if batch:
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})


def _is_not_found(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    code = resp.get("Error", {}).get("Code")
    return code in {"NoSuchKey", "404", "NotFound"}
