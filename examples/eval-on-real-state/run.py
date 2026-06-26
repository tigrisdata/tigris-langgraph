"""Eval a change against REAL accumulated state — a runnable demo.

The story, end to end:

  1. A prod support agent accumulates real memory across several customer
     threads (plan, allergies, timezone, past issues).
  2. You're considering a one-line prompt change. The classic move is to eval it
     against a handful of toy fixtures — which have no memory, so the change
     looks like a no-op.
  3. Instead: FORK prod twice (baseline + candidate), replay the real threads
     through each fork in isolation, and judge them head to head. Prod is never
     touched; the forks are dropped at the end.
  4. Out comes a ship/hold verdict grounded in the real world — plus a recall
     signal showing the candidate actually used the memory the fixtures lacked.

Run it:

    export AWS_ACCESS_KEY_ID=...        # Tigris access key  (console.tigris.dev)
    export AWS_SECRET_ACCESS_KEY=...    # Tigris secret key
    export ANTHROPIC_API_KEY=...
    pip install -r requirements.txt
    python run.py                       # add --keep to leave the forks behind
"""

from __future__ import annotations

import argparse
import uuid

from langgraph.checkpoint.tigris import TigrisSaver
from langgraph.checkpoint.tigris._client import make_client
from langgraph.checkpoint.tigris._fork import create_snapshot_bucket

from agent import BASELINE_PROMPT, CANDIDATE_PROMPT, THREADS, build_agent
from harness import (
    drop_bucket,
    judge_report,
    make_llm_judge,
    new_run_id,
    run_variant_on_fork,
)


def seed_prod(prod: TigrisSaver) -> None:
    """Build up realistic per-customer memory on the prod bucket.

    We run the *baseline* agent here — prod has been running the old prompt all
    along. Each history turn produces an assistant reply, so every thread ends up
    with genuine back-and-forth that both variants will later inherit.
    """
    graph = build_agent(BASELINE_PROMPT).compile(checkpointer=prod)
    for t in THREADS:
        cfg = {"configurable": {"thread_id": t.thread_id}}
        for turn in t.history:
            graph.invoke({"messages": [{"role": "user", "content": turn}]}, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="don't delete the forks")
    args = ap.parse_args()

    client = make_client()
    prod_bucket = f"prod-agent-{uuid.uuid4().hex[:10]}"
    run_id = new_run_id()

    print(f"\n→ creating snapshot-enabled prod bucket: {prod_bucket}")
    create_snapshot_bucket(client, prod_bucket)
    prod = TigrisSaver(prod_bucket, client=client)
    prod.setup()

    try:
        print(f"→ seeding {len(THREADS)} customer threads with real memory...")
        seed_prod(prod)

        # --- Fork prod per variant; run the same real threads in isolation. ---
        print("\n→ forking prod twice and replaying real threads:")
        print("   • baseline  (current prompt)")
        print("   • candidate (use-what-you-know prompt)\n")

        baseline = run_variant_on_fork(
            prod, BASELINE_PROMPT, THREADS,
            name="baseline", run_id=run_id, keep=args.keep,
        )
        candidate = run_variant_on_fork(
            prod, CANDIDATE_PROMPT, THREADS,
            name="candidate", run_id=run_id, keep=args.keep,
        )

        # --- Judge head to head, order-swapped. -------------------------------
        print("→ judging head-to-head (order-swapped to cancel position bias)...\n")
        report = judge_report(THREADS, baseline, candidate, make_llm_judge())

        # --- Results. ---------------------------------------------------------
        print(f"   {'thread':<12} {'winner':<11} {'baseline recall':<17} candidate recall")
        print(f"   {'-'*12} {'-'*11} {'-'*17} {'-'*16}")
        for r in report.results:
            print(
                f"   {r.thread_id:<12} {r.winner:<11} "
                f"{('yes' if r.baseline_recall else 'no'):<17} "
                f"{'yes' if r.candidate_recall else 'no'}"
            )

        print(
            f"\n   head-to-head: candidate {report.candidate_wins}"
            f" / baseline {report.baseline_wins} / ties {report.ties}"
        )
        print(
            f"   used real memory: candidate {report.candidate_recall}/{len(THREADS)}"
            f"  vs  baseline {report.baseline_recall}/{len(THREADS)}"
        )
        print(f"\n   VERDICT: {report.verdict}")
        print(
            "\n   Note: against empty fixtures, recall would be 0/0 for both and "
            "the\n   verdict a wash — the difference only exists because we "
            "evaluated\n   against prod's real accumulated state.\n"
        )

        if args.keep:
            print(
                f"--keep set: forks left in place "
                f"({baseline.bucket}, {candidate.bucket}).\n"
                f"Prod bucket '{prod_bucket}' was never written to during the "
                "eval — verify with `aws s3 ls`.\n"
            )

    finally:
        if not args.keep:
            print("→ cleaning up buckets...")
            for name in (
                f"{prod_bucket}-eval-baseline-{run_id}",
                f"{prod_bucket}-eval-candidate-{run_id}",
                prod_bucket,
            ):
                try:
                    drop_bucket(client, name)
                except Exception:
                    pass
            print("  done.\n")


if __name__ == "__main__":
    main()
