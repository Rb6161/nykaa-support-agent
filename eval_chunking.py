"""Task 5: compare the two chunking strategies at the DOCUMENT level on the same 25 queries.

Per query: retrieve top-k chunks, map each chunk back to its parent document, DEDUP, then
    precision = |retrieved_docs ∩ relevant_docs| / |retrieved_docs|
    recall    = |retrieved_docs ∩ relevant_docs| / |relevant_docs|"""
import config  # noqa: F401,E402

import json

import rag
from queries import IN_SCOPE_25


def evaluate(strategy: str, eng, verbose=True):
    rows = []
    if verbose:
        print(f"\n===== collection: {strategy} (top-k = {rag.TOP_K} chunks) =====")
    for i, (q, gold) in enumerate(IN_SCOPE_25, 1):
        hits = eng.retrieve(q, strategy, rag.TOP_K)
        docs = list(dict.fromkeys(h["doc_id"] for h in hits))         # map chunks -> parent docs, dedup
        tp = len(set(docs) & gold)
        p, r = tp / len(docs), tp / len(gold)
        rows.append({"query": q, "retrieved_docs": docs, "relevant": sorted(gold), "tp": tp,
                     "precision": p, "recall": r})
        if verbose:
            print(f"Q{i:02d} {q}\n     {len(hits)} chunks -> docs {docs}  | relevant {sorted(gold)}  | TP={tp}\n"
                  f"     precision = {tp}/{len(docs)} = {p:.3f}   recall = {tp}/{len(gold)} = {r:.3f}")
    n = len(rows)
    avg_p, avg_r = sum(r["precision"] for r in rows) / n, sum(r["recall"] for r in rows) / n
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if avg_p + avg_r else 0.0
    if verbose:
        print(f"MEAN precision = {avg_p:.3f}   MEAN recall = {avg_r:.3f}   F1 = {f1:.3f}")
    return {"rows": rows, "mean_precision": avg_p, "mean_recall": avg_r, "f1": f1,
            "chunks": rag.get_engine().collections[strategy].count()}


if __name__ == "__main__":
    eng = rag.get_engine()
    results = {s: evaluate(s, eng) for s in rag.COLLECTIONS}
    fx, sn = results["fixed"], results["sentence"]
    best = max(results, key=lambda s: (round(results[s]["f1"], 6), results[s]["mean_recall"], -results[s]["chunks"]))
    other = "fixed" if best == "sentence" else "sentence"
    b, o = results[best], results[other]
    rec = (f"I recommend deploying the {best}-based chunking. It scored mean precision {b['mean_precision']:.3f} and "
           f"mean recall {b['mean_recall']:.3f} (F1 {b['f1']:.3f}) over the 25 queries, against precision "
           f"{o['mean_precision']:.3f}, recall {o['mean_recall']:.3f} (F1 {o['f1']:.3f}) for the {other} strategy. "
           f"It indexes {b['chunks']} chunks versus {o['chunks']}.")
    print("\nRECOMMENDATION:\n" + rec)
    results["recommendation"] = {"strategy": best, "text": rec}
    (rag.ROOT / "chunk_eval.json").write_text(json.dumps(results, indent=2))
    rag.RECOMMENDATION_PATH.write_text(json.dumps({"strategy": best, "text": rec}, indent=2))
    print(f"wrote chunk_eval.json and {rag.RECOMMENDATION_PATH.name} (the pipeline now uses '{best}')")
