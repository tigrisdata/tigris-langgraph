# Hand-off brief: verify the eval-on-real-state example

Paste the prompt below into Claude Code, run from the repo root of
`tigris-langgraph` (it has the Tigris + Anthropic creds and network this sandbox
lacked). The creds-free logic is already verified here: `py_compile` is clean and
a stubbed test confirms the pairwise tally, recall counting, SHIP/HOLD verdicts,
the position-bias guard (an always-"A" judge collapses to all ties), and the
judge-output parser. What's left needs live creds.

---

## Prompt for Claude Code

> You're in the `tigris-langgraph` repo. A runnable example lives in
> `examples/eval-on-real-state/` (`agent.py`, `harness.py`, `run.py`,
> `requirements.txt`, `README.md`). It demonstrates the headline use case for the
> Tigris LangGraph checkpointer: to test an agent change, **fork prod, replay
> real threads through baseline vs candidate in isolation, judge head-to-head,
> drop the forks** — prod is never written to. Core helpers are
> `run_variant_on_fork()` and `judge_report()` in `harness.py`.
>
> Do three things and report back tersely:
>
> **1. Install + static.**
> - `cd examples/eval-on-real-state && python -m py_compile *.py`
> - `pip install ../../libs/checkpoint-tigris && pip install -r requirements.txt`
>
> **2. Live end-to-end (needs creds).**
> - Confirm `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are a Tigris
>   Single-region or Multi-region key and `ANTHROPIC_API_KEY` is set.
> - `python run.py`. Expect: a seeded prod bucket, two forks (baseline +
>   candidate), a per-thread table, a head-to-head tally, a memory-recall line
>   (candidate should recall noticeably more than baseline), and a SHIP/HOLD/
>   TOSS-UP verdict. The candidate *should* win or tie — if it loses, the prompt
>   or judge needs a look; say so.
>
> **3. Prove the fork is load-bearing (the whole point).**
> - `python run.py --keep`, then with
>   `aws s3 ls --endpoint-url https://t3.storage.dev` compare the **prod** bucket
>   against a **fork**: the fork has new checkpoint objects under the threads from
>   the eval; prod has **no** new objects written during the eval. That's the
>   isolation claim, proven. Clean up the `--keep` buckets afterward.
>
> **Report:** PASS/FAIL for each, plus the verdict line, the recall numbers, and
> wall-clock time. Don't change behavior unless something fails — if so, fix the
> bug, note it, re-run. Specifically flag: (a) any boto3 thread-safety error
> (n/a here, runs are sequential), (b) judge flakiness (re-run twice; is the
> verdict stable?), (c) whether Haiku's baseline already recalls memory well
> enough that the gap is small — if so, suggest a starker `BASELINE_PROMPT`.

---

## What "good" looks like

- Live run prints a per-thread table, a head-to-head tally, recall numbers where
  **candidate > baseline**, and a SHIP verdict.
- `--keep` + `s3 ls` shows the forks carry new checkpoints while the **prod
  bucket has none** added during the eval.
- Re-running gives a stable verdict (judge is at temperature 0).

## Known caveats to check

- The recall check is deliberately simple (lowercased substring of a known
  marker). It's an illustration of *why* the candidate wins, not the verdict
  itself — the verdict is the pairwise judge. Don't over-tune it.
- Fork buckets are snapshot-backed; `drop_bucket`'s final `DeleteBucket` is
  best-effort (a lingering snapshot may leave an empty bucket for Tigris to
  reap). Expected, not a failure.
- If the model reliably uses memory even under the baseline prompt, the gap
  shrinks. That's a realistic outcome worth reporting — and an argument for
  testing a bigger change (e.g., a model swap) rather than evidence of a bug.
