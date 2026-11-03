"""Task 4 helper: empirically calibrate the "I don't know" threshold (no untested 0.5/0.6/0.7 presets).

For EACH collection we measure the top-1 cosine similarity of >=3 in-scope and >=2 deliberately
out-of-scope queries, then place the threshold midway between the two clusters we actually observe:
    threshold = (min(in-scope top-1) + max(out-of-scope top-1)) / 2"""
import config  # noqa: F401,E402

import json

import rag
from queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE

if __name__ == "__main__":
    eng = rag.get_engine()
    result = {}
    for strategy in rag.COLLECTIONS:
        ins = {q: eng.retrieve(q, strategy, 1)[0]["similarity"] for q in CALIBRATION_IN_SCOPE}
        outs = {q: eng.retrieve(q, strategy, 1)[0]["similarity"] for q in CALIBRATION_OUT_OF_SCOPE}
        lo_in, hi_out = min(ins.values()), max(outs.values())
        threshold = round((lo_in + hi_out) / 2, 4)
        result[strategy] = {"in_scope": ins, "out_of_scope": outs, "min_in_scope": lo_in,
                            "max_out_of_scope": hi_out, "separated": lo_in > hi_out, "threshold": threshold}
        print(f"\n=== collection: {strategy} ===")
        print("in-scope top-1 cosine similarity:")
        for q, s in ins.items():
            print(f"  {s:.4f}  {q}")
        print("out-of-scope top-1 cosine similarity:")
        for q, s in outs.items():
            print(f"  {s:.4f}  {q}")
        print(f"cluster edges: min(in-scope)={lo_in:.4f}  max(out-of-scope)={hi_out:.4f}  separated={lo_in > hi_out}")
        print(f"CHOSEN THRESHOLD = ({lo_in:.4f} + {hi_out:.4f}) / 2 = {threshold}")
        if lo_in <= hi_out:
            print("WARNING: clusters overlap; add more/clearer calibration queries before trusting this threshold.")
    rag.CALIBRATION_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {rag.CALIBRATION_PATH.name}")
