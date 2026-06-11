# LangGraph Tigris Checkpoint

Implementation of a LangGraph `BaseCheckpointSaver` that persists agent state in
[Tigris](https://www.tigrisdata.com/) globally-distributed object storage over
the S3 API (sync and async), plus a **zero-copy bucket-fork** helper for instant
branching of an entire thread's checkpoint lineage.

## Why Tigris

- **Durable, global, cheap state** — checkpoints live in globally-replicated
  object storage with no egress fees.
- **Instant branching** — `fork()` creates a zero-copy fork of the bucket, so an
  agent's entire checkpoint history can be branched in O(1) for parallel
  experiments, evals, or what-if exploration. Writes to the fork are isolated
  from the source.
- **Pure object store** — no separate database to operate. "Latest checkpoint"
  is found via a sorted list under strong consistency; conditional writes are
  available for idempotency/CAS.

## Bucket requirement

> Use a **Single-region** or **Multi-region** Tigris bucket. These provide
> strong, globally consistent reads, lists, and conditional operations, which
> this saver relies on. **Global** and **Dual-region** buckets give only
> eventual cross-region consistency and can return stale "latest" results.

## Install

```bash
pip install langgraph-checkpoint-tigris
```

## Usage

```python
from langgraph.checkpoint.tigris import TigrisSaver

# Credentials from the standard AWS env chain
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), endpoint defaults to Tigris.
with TigrisSaver.from_conn_string("my-bucket") as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [...]}, {"configurable": {"thread_id": "1"}})
```

Async:

```python
from langgraph.checkpoint.tigris.aio import AsyncTigrisSaver

async with AsyncTigrisSaver.from_conn_string("my-bucket") as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    await graph.ainvoke({"messages": [...]}, {"configurable": {"thread_id": "1"}})
```

## Instant branching with forks

```python
with TigrisSaver.from_conn_string("prod-agent-state") as checkpointer:
    # Branch the ENTIRE bucket (all threads/checkpoints) instantly.
    experiment = checkpointer.fork("experiment-run-42")
    # `experiment` is an isolated TigrisSaver; writes here never touch prod.
```

## Object layout

```
{prefix}checkpoints/{thread}/{ns}/{checkpoint_id}/manifest.json
{prefix}checkpoints/{thread}/{ns}/{checkpoint_id}/checkpoint.bin
{prefix}checkpoints/{thread}/{ns}/{checkpoint_id}/writes/{task_id}/{idx}.bin
```

Checkpoints are immutable and uniquely keyed; `checkpoint_id`s are time-sortable
so the latest is `max(...)` of a prefix listing — no mutable HEAD pointer.

## Development

```bash
cd libs/checkpoint-tigris
uv sync
make format && make lint && make test         # unit tests run without creds

# Full suite (conformance + integration + fork) needs a live Tigris bucket:
TIGRIS_TEST_BUCKET=my-bucket \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
make test
```

Integration, conformance, and fork tests are skipped automatically when those
environment variables are absent. The pure key-layout unit tests
(`tests/test_keys.py`) always run.

## Status

The synchronous and asynchronous savers, the `copy_thread`/`acopy_thread`
branching capability, and the zero-copy `fork()` helper are implemented and
pass the LangGraph checkpointer conformance suite
(`langgraph-checkpoint-conformance`) against a live Tigris bucket. The pure
key-layout and manifest logic are additionally unit-tested offline.
