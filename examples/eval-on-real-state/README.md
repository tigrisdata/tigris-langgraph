# Eval your agent against real state, not fixtures

> Stop shipping prompt changes that pass your three hand-written test threads and
> regress in production. **Fork prod, replay real conversations through the
> change, judge it head-to-head, drop the fork.** Prod is never touched.

A LangGraph checkpointer keyed in a Tigris bucket can `fork()` your agent's
**entire** state — every thread, every checkpoint — instantly and zero-copy. So
"evaluate this change against the real, accumulated world my users actually
built up" goes from impractical (a `pg_dump` whose cost scales with how valuable
your history is) to a loop you can run on every PR.

## What this demo proves

It evaluates a one-line prompt tweak — "use what you already know about this
customer" — that is **invisible to fixtures**. Toy test threads have no memory,
so the change looks like a no-op. Against four real customer threads (an allergy,
a plan + timezone, a beginner, a past billing credit) the candidate's win shows
up clearly, and a deterministic recall check confirms *why*: it actually used the
context the fixtures lacked.

```
            prod bucket (real state — never written to during the eval)
               │  fork()                    fork()
        ┌──────┴───────┐            ┌────────┴────────┐
   baseline fork                candidate fork              ← isolated, zero-copy
   (current prompt)             (new prompt)
        │                            │
   replay real threads          replay real threads
        └─────────────┬──────────────┘
              pairwise judge (order-swapped)
                       │
            VERDICT: ship / hold  +  memory-recall signal
                       │
               drop both forks  (no residue)
```

## Files

| File | What it is |
|------|------------|
| `agent.py` | The support agent (parameterized only by system prompt) + the realistic prod threads and their held-out probes. |
| `harness.py` | `run_variant_on_fork()` (fork prod, replay threads, drop fork) and `judge_report()` (order-swapped pairwise judge + recall). Bring your own judge or metric. |
| `run.py` | The end-to-end demo: seed prod, fork baseline vs candidate, judge, print a verdict. |

## Run it

Needs a **Single-region** or **Multi-region** Tigris bucket's credentials and an
Anthropic key.

```bash
export AWS_ACCESS_KEY_ID=...        # Tigris access key  (console.tigris.dev)
export AWS_SECRET_ACCESS_KEY=...    # Tigris secret key
export ANTHROPIC_API_KEY=...

pip install -r requirements.txt
python run.py            # add --keep to leave the forks behind and inspect them
```

The script creates its own snapshot-enabled prod bucket, seeds it, forks it, and
cleans everything up at the end.

## Adapt it to your agent

Two seams, both swappable without touching the fork machinery:

- **The candidate.** Change `CANDIDATE_PROMPT` in `agent.py`, or pass a different
  model to `build_agent`, to A/B anything — a prompt, a model, a tool.
- **The scorer.** `judge_report` takes any `judge(probe, A, B) -> 'A'|'B'|'tie'`.
  The shipped one is an LLM judge; drop in an objective metric (a unit-test pass
  count, a regex/constraint check) for fully reproducible scoring.

To eval **N** variants instead of two, call `run_variant_on_fork` in a loop —
each variant gets its own zero-copy fork of prod, isolated from the rest.

## Why only object storage makes this practical

A relational checkpointer has to **read rows to copy them**, so branching the
whole agent costs more the more history you've accumulated — exactly backwards.
Tigris forks by **sharing immutable blocks** and layering new writes on top, so a
fork is a pointer, not a duplication, and a dropped fork leaves nothing behind.

See the announcement,
[*Clone your production agent in one API call*](https://www.tigrisdata.com/blog/langgraph-checkpointer/),
and the [bucket forking docs](https://www.tigrisdata.com/docs/buckets/snapshots-and-forks/).

**Next level:** the same fork primitive powers *speculative branching* — forking
mid-run to explore several agent trajectories at once and keep the best. That's a
follow-up; this harness is the everyday win.
