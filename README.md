# tigris-langgraph

[LangGraph](https://github.com/langchain-ai/langgraph) integrations for
[Tigris](https://www.tigrisdata.com/) — globally distributed, S3-compatible
object storage.

## Packages

| Package | Description | PyPI |
|---------|-------------|------|
| [`langgraph-checkpoint-tigris`](libs/checkpoint-tigris) | LangGraph checkpointer backed by a Tigris bucket, with `copy_thread` branching and a zero-copy bucket-`fork()` helper. | [![PyPI](https://img.shields.io/pypi/v/langgraph-checkpoint-tigris.svg)](https://pypi.org/project/langgraph-checkpoint-tigris/) |

More integrations may be added here over time (this repo is an umbrella, in the
same spirit as other provider `*-langgraph` repos).

## Quick start

```bash
pip install -U langgraph-checkpoint-tigris
```

```python
from langgraph.checkpoint.tigris import TigrisSaver

# Credentials come from the standard AWS env chain
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY); endpoint defaults to Tigris.
with TigrisSaver.from_conn_string("my-bucket") as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [...]}, {"configurable": {"thread_id": "1"}})
```

See [`libs/checkpoint-tigris`](libs/checkpoint-tigris) for full usage, async
support, branching, and development instructions.

## License

MIT — see [LICENSE](LICENSE).
</content>
