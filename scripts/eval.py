"""
Runs a small labeled evaluation set against the live RAG graph and reports pass/fail.

This exists because manually spot-checking a couple of queries is not enough to trust
a RAG system — it's exactly what let a routing bug (see README) go unnoticed until it
was hit by chance. Re-run this any time the prompts, routing logic, or extraction
pipeline change, to catch regressions automatically instead of by accident.

Usage (from project root):
    python scripts/eval.py
"""

import sys
import os

# allow running as `python scripts/run_eval.py` from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.graph import rag_app

EVAL_SET = [
    {
        "query": "Which invoice has the highest total?",
        "expected_answer_contains": "860.38",
    },
    {
        "query": "What is the total on Adrian Hane's invoice?",
        "expected_answer_contains": "860.38",
    },
    {
        "query": "What items did Aaron Bergman order?",
        "expected_answer_contains": "Akro Stacking Bins",
    },
    {
        "query": "Which invoice has the lowest total?",
        "expected_answer_contains": "22.17",
    },
    # Add more cases here as the project grows. Prefer facts you can verify by hand
    # against the source PDFs, so a failing test always means something real broke.
]


def run_eval():
    passed = 0
    failures = []

    for case in EVAL_SET:
        result = rag_app.invoke({"query": case["query"], "attempts": 0})
        answer = result.get("answer", "")
        success = case["expected_answer_contains"] in answer

        status = "PASS" if success else "FAIL"
        print(f"{status} — {case['query']}")

        if success:
            passed += 1
        else:
            failures.append(
                {
                    "query": case["query"],
                    "expected_to_contain": case["expected_answer_contains"],
                    "got": answer,
                    "route": result.get("route"),
                }
            )
            print(f"   expected to contain: {case['expected_answer_contains']!r}")
            print(f"   got: {answer!r}")
            print(f"   route: {result.get('route')!r}")

    total = len(EVAL_SET)
    print(f"\n{passed}/{total} passed")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f['query']}")

    return passed, total, failures


if __name__ == "__main__":
    passed, total, failures = run_eval()
    # non-zero exit code on failure — makes this usable in CI later
    sys.exit(0 if not failures else 1)