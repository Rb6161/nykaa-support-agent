"""HELD-OUT retrieval test + threshold sensitivity.

The 25 demo queries and 5+4 calibration queries were written together with the system, so they are optimistic.
This script uses NEW queries (paraphrases, typos, Hinglish, multi-topic) and a bigger in-scope / out-of-scope
set to show how the calibrated threshold really behaves.

    python robustness_eval.py
Needs `python run_all.py` to have run once (calibration.json)."""
import config  # noqa: F401,E402

import json

import rag

# (query, gold documents). Written independently of the demo queries; expectations noted BEFORE running.
HELD_OUT = [
    ("can I send back a lipstick I bought last week?", {"01_return_window"}),
    ("my hair straightener stopped working after 8 months, is it covered?", {"05_warranty"}),
    ("refnd for cod ordr how long", {"02_cod_refund"}),
    ("mera refund kab aayega cash on delivery wala", {"02_cod_refund"}),                   # Hinglish: expected to be hard
    ("is it possible to change my kurta size?", {"09_size_exchange"}),
    ("the bottle leaked in transit, what do I do?", {"10_damaged_item"}),
    ("my UPI payment went through but the order was not placed", {"08_payment_failure"}),
    ("how many days to deliver to a small town?", {"03_delivery_sla"}),
    ("can I use my reward points to pay the full amount?", {"07_loyalty_points"}),
    ("how do I escalate if support does not respond in a day?", {"12_escalation_matrix"}),
    ("can I order lipstick to be delivered in the USA?", {"11_international_shipping"}),
    ("I clicked cancel but the courier already picked up my parcel", {"06_cancellation"}),
    ("will you pick up the product from my house for return?", {"04_reverse_pickup"}),
    ("what does 'Returned' mean on my order?", {"13_order_status_meanings"}),
    ("I need a tax invoice with my GSTIN", {"14_gift_wrap_and_invoice"}),
    ("how long before a delivery counts as late?", {"03_delivery_sla"}),
    ("which items have no warranty?", {"05_warranty"}),
    ("can I return a phone charger after 10 days?", {"01_return_window"}),
    ("I received the wrong product and want my money back, I paid cash", {"10_damaged_item", "02_cod_refund"}),
    ("my order is delayed and support did not help for two days, who is next?", {"12_escalation_matrix", "03_delivery_sla"}),
    ("is gift wrapping refundable?", {"14_gift_wrap_and_invoice"}),
    ("return a shoe and exchange it for another size", {"09_size_exchange", "01_return_window"}),
    ("how long do my loyalty points last before expiring?", {"07_loyalty_points"}),
    ("can I cancel an international order?", {"11_international_shipping"}),
]

IN_SCOPE_SHORT = ["return window", "cod refund time", "delivery time metro city", "warranty on trimmer", "cancel my order",
                  "loyalty points", "payment failed", "size exchange", "damaged parcel", "international shipping",
                  "escalation matrix", "order status meaning", "gst invoice", "reverse pickup", "reward points expiry",
                  "refund after return"]
OUT_OF_SCOPE_NEAR_MISS = ["What is Amazon's return policy?", "How do I return a library book?",
                          "What is the best lipstick shade for dark skin?", "Where is the Nykaa office located?",
                          "How do I become a Nykaa seller?", "What is the stock price of Nykaa?",
                          "Can I get a job at Nykaa?", "How do I apply for a credit card?",
                          "What is the weather in Mumbai today?", "Who is the CEO of Nykaa?",
                          "recommend a good moisturiser for oily skin", "how do I track a flight status",
                          "what is the refund policy for flight tickets", "how do I reset my email password",
                          "translate hello to French", "how many calories are in a banana"]


def top1(eng, q, strategy):
    return eng.retrieve(q, strategy, 1)[0]["similarity"]


if __name__ == "__main__":
    eng = rag.get_engine()
    out = {}
    for strategy in rag.COLLECTIONS:
        thr = rag.threshold_for(strategy)
        print(f"\n{'=' * 78}\nCOLLECTION: {strategy}   calibrated threshold = {thr}\n{'=' * 78}")

        # ---- A. held-out retrieval quality (document level, same method as eval_chunking.py)
        print("\nA. HELD-OUT QUERIES (document-level precision / recall, top-4 chunks)")
        rows, misses = [], []
        for q, gold in HELD_OUT:
            hits = eng.retrieve(q, strategy, rag.TOP_K)
            docs = list(dict.fromkeys(h["doc_id"] for h in hits))
            tp = len(set(docs) & gold)
            p, r = tp / len(docs), tp / len(gold)
            answered = hits[0]["similarity"] >= thr
            rows.append({"query": q, "precision": p, "recall": r, "answered": answered, "top1": hits[0]["similarity"]})
            flag = "" if (r == 1.0 and answered) else "   <-- check"
            print(f"  P={p:.2f} R={r:.2f} top1={hits[0]['similarity']:.3f} answered={str(answered):5}  {q}{flag}")
            if r < 1.0 or not answered:
                misses.append(q)
        n = len(rows)
        mp, mr = sum(r["precision"] for r in rows) / n, sum(r["recall"] for r in rows) / n
        refused = sum(not r["answered"] for r in rows)
        print(f"  MEAN precision {mp:.3f} | MEAN recall {mr:.3f} | wrongly refused {refused}/{n}")

        # ---- B. threshold sensitivity
        ins = [top1(eng, q, strategy) for q in IN_SCOPE_SHORT]
        outs = [top1(eng, q, strategy) for q in OUT_OF_SCOPE_NEAR_MISS]
        print(f"\nB. THRESHOLD SENSITIVITY  ({len(ins)} short in-scope, {len(outs)} out-of-scope near-miss queries)")
        print(f"   in-scope top-1  : min {min(ins):.3f}  mean {sum(ins) / len(ins):.3f}")
        print(f"   out-of-scope top-1: max {max(outs):.3f}  mean {sum(outs) / len(outs):.3f}")
        print("   threshold | false refusals (in-scope refused) | false answers (out-of-scope answered)")
        sweep = []
        for i in range(10, 71, 5):
            t = i / 100
            frr = sum(s < t for s in ins) / len(ins)
            far = sum(s >= t for s in outs) / len(outs)
            sweep.append({"threshold": t, "false_refusal_rate": frr, "false_answer_rate": far})
            mark = "  <-- chosen" if abs(t - thr) < 0.025 else ""
            print(f"     {t:.2f}    |        {frr:6.1%}                  |         {far:6.1%}{mark}")
        wrong_ans = [q for q, s in zip(OUT_OF_SCOPE_NEAR_MISS, outs) if s >= thr]
        wrong_ref = [q for q, s in zip(IN_SCOPE_SHORT, ins) if s < thr]
        print(f"   at the CHOSEN threshold {thr}:")
        print(f"     out-of-scope questions that would be ANSWERED: {wrong_ans or 'none'}")
        print(f"     in-scope questions that would be REFUSED     : {wrong_ref or 'none'}")
        out[strategy] = {"threshold": thr, "heldout_mean_precision": mp, "heldout_mean_recall": mr,
                         "heldout_wrongly_refused": refused, "problem_queries": misses, "sweep": sweep,
                         "false_answers_at_threshold": wrong_ans, "false_refusals_at_threshold": wrong_ref}
    (rag.ROOT / "robustness_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote robustness_results.json  (queries marked '<-- check' are your real weak spots)")
