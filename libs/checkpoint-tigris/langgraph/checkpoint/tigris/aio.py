"""Asynchronous Tigris checkpoint saver (aioboto3).

Mirrors :class:`langgraph.checkpoint.tigris.TigrisSaver` but uses a real async
S3 client so it works under LangGraph's async runs. aioboto3 clients are async
context managers, so a fresh client is created per operation from a shared
session.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
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
from langgraph.checkpoint.tigris._client import (
    DEFAULT_ENDPOINT,
    DEFAULT_REGION,
    make_async_session,
)
from langgraph.checkpoint.tigris.utils import (
    build_manifest,
    matches_filter,
    parse_manifest,
)

__all__ = ["AsyncTigrisSaver"]

META_CHANNEL = "channel"
META_STYPE = "stype"


class AsyncTigrisSaver(BaseCheckpointSaver[str]):
    """Async LangGraph checkpoint saver backed by a Tigris bucket."""

    def __init__(
        self,
        bucket: str,
        *,
        session: Any | None = None,
        prefix: str = "",
        endpoint_url: str = DEFAULT_ENDPOINT,
        region_name: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint_url = endpoint_url
        self.region_name = region_name or DEFAULT_REGION
        self.session = session or make_async_session()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str = DEFAULT_ENDPOINT,
        region_name: str | None = None,
    ) -> AsyncIterator[AsyncTigrisSaver]:
        yield cls(
            bucket, prefix=prefix, endpoint_url=endpoint_url, region_name=region_name
        )

    @asynccontextmanager
    async def _s3(self) -> AsyncIterator[Any]:
        async with self.session.client(
            "s3", endpoint_url=self.endpoint_url, region_name=self.region_name
        ) as client:
            yield client

    async def setup(self) -> None:
        async with self._s3() as s3:
            await s3.head_bucket(Bucket=self.bucket)

    # ----------------------------------------------------------------- writes
    async def aput(
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
        manifest = build_manifest(
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=parent_id,
            checkpoint_type=type_,
            metadata=dict(get_checkpoint_metadata(config, metadata)),
            ts=checkpoint.get("ts"),
            channel_versions=dict(new_versions),
        )
        async with self._s3() as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=_keys.blob_key(
                    self.prefix, thread_id, checkpoint_ns, checkpoint_id
                ),
                Body=blob,
            )
            await s3.put_object(
                Bucket=self.bucket,
                Key=_keys.manifest_key(
                    self.prefix, thread_id, checkpoint_ns, checkpoint_id
                ),
                Body=manifest,
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        async with self._s3() as s3:
            for idx, (channel, value) in enumerate(writes):
                widx = WRITES_IDX_MAP.get(channel, idx)
                type_, blob = self.serde.dumps_typed(value)
                await s3.put_object(
                    Bucket=self.bucket,
                    Key=_keys.write_key(
                        self.prefix,
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        widx,
                    ),
                    Body=blob,
                    Metadata={META_CHANNEL: _keys._enc(channel), META_STYPE: type_},
                )

    # ----------------------------------------------------------------- reads
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        async with self._s3() as s3:
            if checkpoint_id is None:
                checkpoint_id = await self._latest_checkpoint_id(
                    s3, thread_id, checkpoint_ns
                )
                if checkpoint_id is None:
                    return None
            return await self._load_tuple(s3, thread_id, checkpoint_ns, checkpoint_id)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
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
        async with self._s3() as s3:
            keys = [
                k async for k in self._list_keys(s3, prefix) if _keys.is_manifest_key(k)
            ]
            keys.sort(key=_keys.checkpoint_id_from_manifest_key, reverse=True)
            count = 0
            for key in keys:
                cp_id = _keys.checkpoint_id_from_manifest_key(key)
                if before_id is not None and cp_id >= before_id:
                    continue
                ns = _keys.ns_from_manifest_key(key)
                tup = await self._load_tuple(s3, thread_id, ns, cp_id)
                if tup is None or not matches_filter(dict(tup.metadata), filter):
                    continue
                yield tup
                count += 1
                if limit is not None and count >= limit:
                    return

    async def adelete_thread(self, thread_id: str) -> None:
        prefix = _keys.thread_prefix(self.prefix, str(thread_id))
        async with self._s3() as s3:
            batch: list[dict[str, str]] = []
            async for key in self._list_keys(s3, prefix):
                batch.append({"Key": key})
                if len(batch) == 1000:
                    await s3.delete_objects(
                        Bucket=self.bucket, Delete={"Objects": batch}
                    )
                    batch = []
            if batch:
                await s3.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy every checkpoint, write, and namespace of a thread to a new one.

        A thread's objects share one key prefix, so this is a batch of
        server-side S3 copies (no download/re-serialize) that rewrites only the
        thread segment of each key. The full parent chain is copied, so the
        target thread is independently resumable; the source is untouched.
        """
        src_prefix = _keys.thread_prefix(self.prefix, str(source_thread_id))
        dst_prefix = _keys.thread_prefix(self.prefix, str(target_thread_id))
        async with self._s3() as s3:
            async for key in self._list_keys(s3, src_prefix):
                await s3.copy_object(
                    Bucket=self.bucket,
                    Key=dst_prefix + key[len(src_prefix) :],
                    CopySource={"Bucket": self.bucket, "Key": key},
                )

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    # --------------------------------------------------------------- internals
    async def _load_tuple(
        self, s3: Any, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> CheckpointTuple | None:
        manifest_raw, _ = await self._get_object(
            s3, _keys.manifest_key(self.prefix, thread_id, checkpoint_ns, checkpoint_id)
        )
        if manifest_raw is None:
            return None
        manifest = parse_manifest(manifest_raw)
        blob, _ = await self._get_object(
            s3, _keys.blob_key(self.prefix, thread_id, checkpoint_ns, checkpoint_id)
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
        parent_config = (
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
        pending_writes = await self._load_writes(
            s3, thread_id, checkpoint_ns, checkpoint_id
        )
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=manifest.get("metadata", {}),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def _load_writes(
        self, s3: Any, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        prefix = _keys.writes_prefix(
            self.prefix, thread_id, checkpoint_ns, checkpoint_id
        )
        keys = [k async for k in self._list_keys(s3, prefix)]
        keys.sort(key=_keys.parse_write_key)
        writes: list[tuple[str, str, Any]] = []
        for key in keys:
            task_id, _ = _keys.parse_write_key(key)
            body, meta = await self._get_object(s3, key)
            if body is None:
                continue
            channel = _keys._dec(meta.get(META_CHANNEL, ""))
            value = self.serde.loads_typed((meta.get(META_STYPE, ""), body))
            writes.append((task_id, channel, value))
        return writes

    async def _latest_checkpoint_id(
        self, s3: Any, thread_id: str, checkpoint_ns: str
    ) -> str | None:
        prefix = _keys.ns_prefix(self.prefix, thread_id, checkpoint_ns)
        manifests = [
            k async for k in self._list_keys(s3, prefix) if _keys.is_manifest_key(k)
        ]
        if not manifests:
            return None
        latest = max(manifests, key=_keys.checkpoint_id_from_manifest_key)
        return _keys.checkpoint_id_from_manifest_key(latest)

    async def _get_object(
        self, s3: Any, key: str
    ) -> tuple[bytes | None, dict[str, str]]:
        try:
            resp = await s3.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _is_not_found(exc):
                return None, {}
            raise
        async with resp["Body"] as stream:
            data = await stream.read()
        return data, resp.get("Metadata", {})

    async def _list_keys(self, s3: Any, prefix: str) -> AsyncIterator[str]:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]


def _is_not_found(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    code = resp.get("Error", {}).get("Code")
    return code in {"NoSuchKey", "404", "NotFound"}
