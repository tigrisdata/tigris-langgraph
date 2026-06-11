# AGENTS Instructions — checkpoint-tigris

`langgraph-checkpoint-tigris`: a `BaseCheckpointSaver` backed by Tigris object
storage (S3 API), with a zero-copy bucket-fork helper for instant branching.

## What this package is

- A standalone checkpointer backend, parallel to `checkpoint-postgres` and
  `checkpoint-sqlite`. It depends only on `langgraph-checkpoint` (the base
  interface). Core `langgraph` is not modified.
- Sync saver: `langgraph.checkpoint.tigris.TigrisSaver`.
- Async saver: `langgraph.checkpoint.tigris.aio.AsyncTigrisSaver`.
- Fork helper: `TigrisSaver.fork(target_bucket)` →
  `langgraph/checkpoint/tigris/_fork.py`.

## Layout

```
langgraph/checkpoint/tigris/
  __init__.py   # TigrisSaver (sync) + fork()
  aio.py        # AsyncTigrisSaver (aioboto3)
  _keys.py      # pure object-key layout helpers (unit-tested, no network)
  _client.py    # boto3 / aioboto3 client construction (endpoint: t3.storage.dev)
  _fork.py      # X-Tigris-Fork-Source-Bucket fork helper
  utils.py      # manifest (de)serialization + filter matching
tests/          # test_keys (offline) + conformance/integration/fork (live)
```

## Design invariants (don't break these)

- Checkpoints are immutable and uniquely keyed by
  `(thread_id, checkpoint_ns, checkpoint_id)`; `checkpoint_id` is time-sortable
  so "latest" = `max()` over a prefix listing (no mutable HEAD pointer).
- Requires a **Single-region or Multi-region** bucket for strong consistency.
- Conditional writes (`If-Match` / `If-None-Match`) are for idempotency/CAS only.

## Workflow (run inside this dir)

```bash
uv sync
make format
make lint        # ruff + ty
make test        # offline unit tests; live tests auto-skip without creds
```

Live conformance/integration/fork tests need `TIGRIS_TEST_BUCKET`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (optional `TIGRIS_ENDPOINT_URL`).

## Source of truth

The LangGraph checkpointer conformance suite (`langgraph-checkpoint-conformance`,
installed from PyPI) is the primary correctness gate: `report.passed_all_base()`
must be True, and the `copy_thread` capability must be detected and pass. The
live conformance/integration/fork tests live in `tests/` and run only when
Tigris credentials are present in the environment.

Formatting: use single backticks for inline code in docstrings/comments (no
Sphinx-style double backticks).
