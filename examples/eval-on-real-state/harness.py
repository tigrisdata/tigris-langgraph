"""Fork-prod-and-eval harness.

The move that only object storage makes practical: to test a change, fork your
production agent's ENTIRE state (every thread, every checkpoint) by reference —
instant, zero-copy — point the candidate at the fork, replay real threads
through it, and score. Prod is never written to. Drop the forks when you're
done; nothing lingers.

On Postgres this means `pg_dump`/restore (cost scales with how much valuable
history you've got) or evaluating against toy fixtures that don't resemble real
state. So in practice the eval stays fake. This makes it real.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from botocore.exceptions import ClientError

from langgraph.checkpoint.tigris import TigrisSaver

from agent import Thread, build_agent


# --------------------------------------------------------------------- buckets
def drop_bucket(client: Any, name: str) -> None:
    """Empty all object versions + delete markers, then drop the fork bucket.

    A fork shares immutable blocks with its source, so this never touches prod's
    data. The final DeleteBucket is best-effort: a snapshot-enabled bucket keeps
    snapshots the S3 API can't force-delete, so a lingering empty bucket is left
    for Tigris to reap rather than failing the run.
    """
    paginator = client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=name):
        objs = [
            {"Key": o["Key"], "VersionId": o["VersionId"]}
            for o in page.get("Versions", []) + page.get("DeleteMarkers", [])
        ]
        if objs:
            client.delete_objects(Bucket=name, Delete={"Objects": objs})
    try:
        client.delete_bucket(Bucket=name)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "BucketNotEmpty":
            raise


# ----------------------------------------------------------------- run variant
@dataclass
class VariantRun:
    name: str
    bucket: str  # the fork this variant ran in
    responses: dict[str, str]  # thread_id -> final assistant text


def _last_text(state: Any) -> str:
    msg = state["messages"][-1]
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def run_variant_on_fork(
    prod: TigrisSaver,
    system_prompt: str,
    threads: list[Thread],
    *,
    name: str,
    run_id: str,
    keep: bool = False,
) -> VariantRun:
    """Fork prod, run one variant's probe on every real thread, return replies.

    The fork inherits all of prod's accumulated state, so each probe is answered
    with the customer's real history in context. Writes land in the fork only.
    """
    fork = prod.fork(f"{prod.bucket}-eval-{name}-{run_id}")
    graph = build_agent(system_prompt).compile(checkpointer=fork)

    responses: dict[str, str] = {}
    for t in threads:
        out = graph.invoke(
            {"messages": [{"role": "user", "content": t.probe}]},
            {"configurable": {"thread_id": t.thread_id}},
        )
        responses[t.thread_id] = _last_text(out)

    if not keep:
        drop_bucket(prod.client, fork.bucket)
    return VariantRun(name, fork.bucket, responses)


# ----------------------------------------------------------------- scoring
JudgeFn = Callable[[str, str, str], str]  # (probe, answer_A, answer_B) -> 'A'|'B'|'tie'


@dataclass
class ThreadResult:
    thread_id: str
    winner: str  # 'baseline' | 'candidate' | 'tie'
    baseline_recall: bool
    candidate_recall: bool


@dataclass
class Report:
    results: list[ThreadResult]

    @property
    def candidate_wins(self) -> int:
        return sum(r.winner == "candidate" for r in self.results)

    @property
    def baseline_wins(self) -> int:
        return sum(r.winner == "baseline" for r in self.results)

    @property
    def ties(self) -> int:
        return sum(r.winner == "tie" for r in self.results)

    @property
    def baseline_recall(self) -> int:
        return sum(r.baseline_recall for r in self.results)

    @property
    def candidate_recall(self) -> int:
        return sum(r.candidate_recall for r in self.results)

    @property
    def verdict(self) -> str:
        if self.candidate_wins > self.baseline_wins:
            return "SHIP — candidate beats baseline on real state"
        if self.candidate_wins < self.baseline_wins:
            return "HOLD — candidate regresses vs baseline"
        return "TOSS-UP — no clear winner"


def _recalled(answer: str, markers: list[str]) -> bool:
    low = answer.lower()
    return all(m in low for m in markers) if markers else False


def judge_report(
    threads: list[Thread],
    baseline: VariantRun,
    candidate: VariantRun,
    judge: JudgeFn,
) -> Report:
    """Pairwise verdict per thread, order-swapped to cancel position bias.

    A side only "wins" a thread if it's preferred in BOTH orderings; otherwise
    the judge is being swayed by position and we call it a tie. Plus a cheap,
    deterministic memory-recall check: did each reply actually use the fact we
    know lives in that thread's history?
    """
    results: list[ThreadResult] = []
    for t in threads:
        a = baseline.responses[t.thread_id]
        c = candidate.responses[t.thread_id]

        v1 = judge(t.probe, a, c)  # A=baseline, B=candidate
        v2 = judge(t.probe, c, a)  # A=candidate, B=baseline (swapped)

        if v1 == "B" and v2 == "A":
            winner = "candidate"
        elif v1 == "A" and v2 == "B":
            winner = "baseline"
        else:
            winner = "tie"

        results.append(
            ThreadResult(
                thread_id=t.thread_id,
                winner=winner,
                baseline_recall=_recalled(a, t.recall_markers),
                candidate_recall=_recalled(c, t.recall_markers),
            )
        )
    return Report(results)


# ----------------------------------------------------- a concrete pairwise judge
_PICK = re.compile(r"\b([AB])\b|\b(tie)\b", re.IGNORECASE)


def make_llm_judge(temperature: float = 0.0) -> JudgeFn:
    """A pairwise judge backed by the same model family. Swap in your own."""
    from langchain.chat_models import init_chat_model

    model = init_chat_model("claude-haiku-4-5-20251001", temperature=temperature)
    rubric = (
        "You are grading two customer-support replies, A and B, to the same "
        "question. Prefer the reply that is more helpful and that correctly uses "
        "what is already known about this specific customer; penalize replies "
        "that ignore known context or ask the customer to repeat themselves. "
        "Answer with exactly one token: A, B, or tie."
    )

    def judge(probe: str, answer_a: str, answer_b: str) -> str:
        out = model.invoke(
            [
                {"role": "system", "content": rubric},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{probe}\n\n"
                        f"REPLY A:\n{answer_a}\n\nREPLY B:\n{answer_b}\n\n"
                        "Which is better? A, B, or tie."
                    ),
                },
            ]
        )
        text = out.content if isinstance(out.content, str) else str(out.content)
        # Take the LAST token so "between A and B, B wins" resolves to B.
        matches = _PICK.findall(text)
        if not matches:
            return "tie"
        g1, _g2 = matches[-1]
        return g1.upper() if g1 else "tie"

    return judge


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]
