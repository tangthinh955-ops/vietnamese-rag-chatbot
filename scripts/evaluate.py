"""Evaluate the RAG pipeline on a labeled testset.

Metrics
-------
- hit_rate@k : share of questions whose expected source doc appears in the
  top-k retrieved chunks (retrieval quality).
- kw_coverage: share of `must_contain` strings present in the generated answer
  (cheap, deterministic answer-quality proxy — works even with MockLLM).
- abstention : out-of-scope questions should produce the "not found" answer.

Usage:
    python scripts/evaluate.py                 # uses eval/testset.json
    python scripts/evaluate.py --testset eval/testset.json --top-k 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import build_pipeline  # noqa: E402


def evaluate(testset_path: str, top_k: int | None) -> dict:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    k = top_k or settings.top_k

    cases = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    if not cases:
        raise ValueError(f"Empty testset: {testset_path}")
    rows, hits, kw_hit, kw_total, abstain_ok, abstain_total = [], 0, 0, 0, 0, 0

    for case in cases:
        result = pipeline.chat(case["question"], top_k=k)
        sources = [s.source for s in result.sources]

        expected = case["expected_source"]
        hit = expected in sources if expected else None
        if hit:
            hits += 1

        keywords = case.get("must_contain", [])
        kw_ok = all(kw.lower() in result.answer.lower() for kw in keywords)
        if keywords:
            kw_total += 1
            kw_hit += kw_ok

        if expected is None:
            abstain_total += 1
            abstain_ok += kw_ok  # "không tìm thấy" present

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected": expected or "(out-of-scope)",
                "hit": "—" if hit is None else ("✔" if hit else "✘"),
                "retrieved": ", ".join(dict.fromkeys(sources)) or "(none)",
                "kw": "✔" if kw_ok else "✘",
            }
        )

    in_scope = [c for c in cases if c["expected_source"]]
    metrics = {
        "hit_rate@k": hits / len(in_scope) if in_scope else 0.0,
        "kw_coverage": kw_hit / kw_total if kw_total else 0.0,
        "abstention": abstain_ok / abstain_total if abstain_total else 0.0,
        "n_cases": len(cases),
        "k": k,
        "model": result.model,
    }
    return {"metrics": metrics, "rows": rows}


def render_report(report: dict) -> str:
    m = report["metrics"]
    lines = [
        f"# Evaluation report — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        f"- cases: {m['n_cases']} | top_k: {m['k']} | llm: {m['model']}",
        f"- **hit_rate@{m['k']}: {m['hit_rate@k']:.2%}**",
        f"- keyword coverage: {m['kw_coverage']:.2%}",
        f"- out-of-scope abstention: {m['abstention']:.2%}",
        "",
        "| id | hit | question | expected | retrieved | kw |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['id']} | {r['hit']} | {r['question']} | {r['expected']} "
            f"| {r['retrieved']} | {r['kw']} |"
        )
    return "\n".join(lines)


def main() -> int:
    # Windows consoles default to cp1252 — the report uses ✔/✘
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="eval/testset.json")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Report path (default: eval/reports/<timestamp>.md)",
    )
    args = parser.parse_args()

    report = evaluate(args.testset, args.top_k)
    text = render_report(report)
    print(text)

    out = Path(
        args.out
        or f"eval/reports/report-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
