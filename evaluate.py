"""Task 13: LLM-as-judge evaluation (MOCK_LLM) on 15 queries: Accuracy, Grounding, Completeness, Safety."""
import config  # noqa: F401,E402

import json

from judge import JUDGE_PROMPT, judge
from pipeline import SupportPipeline
from queries import EVAL_SET

METRICS = ["accuracy", "grounding", "completeness", "safety"]

if __name__ == "__main__":
    p = SupportPipeline()
    rows = []
    print("Judge prompt used (payload filled per query):\n" + JUDGE_PROMPT.split("PAYLOAD_JSON")[0].strip() + "\n")
    print(f"{'#':>2} {'kind':<13} {'acc':>5} {'grnd':>5} {'comp':>5} {'safe':>5}  query")
    for i, (q, kind, expected_doc, key_terms) in enumerate(EVAL_SET, 1):
        r = p.ask(q, f"eval-{i}", "evaluate")
        s = judge(query=q, answer=r.answer, context=r.context_snippets, sources=r.sources, refused=r.refused,
                  kind=kind, expected_doc=expected_doc, key_terms=key_terms, pii_masked=r.pii_masked)
        rows.append({"query": q, "kind": kind, "answer": r.answer, "refused": r.refused, "scores": s})
        print(f"{i:>2} {kind:<13} {s['accuracy']:>5.2f} {s['grounding']:>5.2f} {s['completeness']:>5.2f} {s['safety']:>5.2f}  {q[:60]}")
    avgs = {m: round(sum(r["scores"][m] for r in rows) / len(rows), 3) for m in METRICS}
    print("\nAVERAGES over", len(rows), "queries:", json.dumps(avgs))
    print("\nper-query answers:")
    for i, r in enumerate(rows, 1):
        print(f"[{i:02d}] {r['query']}\n     -> {r['answer']}")
    (__import__("pathlib").Path(__file__).parent / "eval_results.json").write_text(json.dumps({"rows": rows, "averages": avgs}, indent=2))
