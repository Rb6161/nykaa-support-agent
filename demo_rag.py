"""Task 4 demo: grounded generation on 25 in-scope queries + 1 out-of-scope query that must trigger
the "I don't know" fallback. Uses the recommended chunking strategy."""
import config  # noqa: F401,E402

import rag
from queries import IN_SCOPE_25, OUT_OF_SCOPE_DEMO

if __name__ == "__main__":
    strategy = rag.recommended_strategy()
    print(f"strategy = {strategy}   threshold = {rag.threshold_for(strategy)}   top-k = {rag.TOP_K}\n")
    answered = 0
    for i, (q, gold) in enumerate(IN_SCOPE_25, 1):
        r = rag.grounded_answer(q, strategy, use_cache=False)
        answered += r["grounded"]
        print(f"[{i:02d}] Q: {q}\n     top-1 sim {r['top1_similarity']:.3f} | grounded={r['grounded']} | sources={r['sources']}\n     A: {r['answer']}\n")
    print(f"in-scope answered (grounded): {answered}/{len(IN_SCOPE_25)}\n")
    r = rag.grounded_answer(OUT_OF_SCOPE_DEMO, strategy, use_cache=False)
    print(f"[OUT-OF-SCOPE] Q: {OUT_OF_SCOPE_DEMO}\n     top-1 sim {r['top1_similarity']:.3f} (< threshold {r['threshold']}) | grounded={r['grounded']}\n     A: {r['answer']}")
    assert not r["grounded"], "the out-of-scope query must trigger the fallback"
    print("\nOK: out-of-scope query triggered the 'I don't know' fallback.")
