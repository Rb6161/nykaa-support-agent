"""Fill the 'Measured results' section of README.md from the JSON files the demos wrote."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load(name):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = []
    cal, chunk, ev, gov = load("calibration.json"), load("chunk_eval.json"), load("eval_results.json"), load("governance_results.json")
    if cal:
        out.append("### Measured similarity threshold (Task 4)\n")
        for s, c in cal.items():
            out.append(f"**{s} collection** - in-scope top-1 similarities: "
                       + ", ".join(f"{v:.3f}" for v in c["in_scope"].values())
                       + f" (min {c['min_in_scope']:.3f}); out-of-scope: "
                       + ", ".join(f"{v:.3f}" for v in c["out_of_scope"].values())
                       + f" (max {c['max_out_of_scope']:.3f}). **Chosen threshold = {c['threshold']}** (midpoint of the two clusters).\n")
    if chunk:
        out.append("### Chunking comparison on 25 queries (Task 5)\n\n| strategy | chunks | mean precision | mean recall | F1 |\n|---|---|---|---|---|")
        for s in ("fixed", "sentence"):
            r = chunk[s]
            out.append(f"| {s} | {r['chunks']} | {r['mean_precision']:.3f} | {r['mean_recall']:.3f} | {r['f1']:.3f} |")
        out.append("\n" + chunk["recommendation"]["text"] + "\n\nPer-query arithmetic: `transcripts/04_chunking_comparison.txt`.\n")
    if ev:
        a = ev["averages"]
        out.append("### Evaluation, 15 queries under MOCK_LLM (Task 13)\n\n| metric | average |\n|---|---|")
        for m in ("accuracy", "grounding", "completeness", "safety"):
            out.append(f"| {m} | {a[m]:.3f} |")
        out.append("\nPer-query scores: `transcripts/09_evaluation_15_queries.txt`.\n")
    if gov:
        c = gov["cache"]
        out.append(f"### Cache evidence (Task 16)\n\nFirst call {c['first_ms']} ms (generation calls {c['counters_after_first']['generation_calls']}); "
                   f"repeated query {c['second_ms']} ms with generation calls still {c['counters_after_second']['generation_calls']}.\n")
    text = "\n".join(out) if out else "_Run `python run_all.py` to generate these numbers._"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pre, rest = readme.split(START, 1)
    _, post = rest.split(END, 1)
    (ROOT / "README.md").write_text(pre + START + "\n" + text + "\n" + END + post, encoding="utf-8")
    print("README results section updated")


if __name__ == "__main__":
    main()
