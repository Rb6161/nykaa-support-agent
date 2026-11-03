"""Task 14 demo: Autogen RoundRobinGroupChat review stage on TWO different sample queries:
  query 1 -> faithful draft -> approved unchanged
  query 2 -> draft with a deliberately injected ungrounded claim -> revised"""
import config  # noqa: F401,E402

import rag
from review import review

if __name__ == "__main__":
    # ---------------- sample query 1: approve unchanged
    q1 = "How long does a COD refund take?"
    r1 = rag.grounded_answer(q1, use_cache=False)
    print("##### SAMPLE QUERY 1:", q1, "#####\nretrieved context:\n  - " + "\n  - ".join(r1["chunks"]))
    draft1 = r1["answer"]
    v1 = review(draft1, r1["chunks"])
    print("\ndraft  :", draft1, "\nverdict:", v1.model_dump_json(indent=2))
    assert v1.approved and v1.final_answer == draft1

    # ---------------- sample query 2: revise (ungrounded claim injected on purpose)
    q2 = "What is the return window for beauty products?"
    r2 = rag.grounded_answer(q2, use_cache=False)
    print("\n##### SAMPLE QUERY 2:", q2, "#####\nretrieved context:\n  - " + "\n  - ".join(r2["chunks"]))
    injected = " Every return also earns a free Nykaa gift hamper worth 5000 rupees."
    draft2 = r2["answer"] + injected
    v2 = review(draft2, r2["chunks"])
    print("\ndraft (with injected ungrounded claim):", draft2, "\nverdict:", v2.model_dump_json(indent=2))
    assert not v2.approved and "hamper" not in v2.final_answer and v2.final_answer.startswith(r2["answer"][:30])
    print("\nOK: the review stage approved query 1's draft unchanged and revised query 2's draft (ungrounded claim removed).")
