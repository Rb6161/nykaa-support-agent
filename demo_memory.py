"""Task 8 demo: LangChain session memory.

    python demo_memory.py multi    -> ONE multi-turn conversation where later turns rely on earlier ones
    python demo_memory.py fresh    -> a SEPARATE fresh conversation (new process, no earlier turns): state absent
    python demo_memory.py          -> both, one after the other (handy for a quick look)

run_all.py runs `multi` and `fresh` as two separate processes and saves two separate transcripts."""
import config  # noqa: F401,E402

import sys

from dataset import ORDERS
from orders import check_order_status
from pipeline import SupportPipeline


def say(p, sid, text):
    r = p.ask(text, sid, endpoint="demo_memory")
    print(f"USER      : {text}\nASSISTANT : {r.answer}\n            (tools={r.tools_used}, refused={r.refused}, reason={r.refusal_reason})\n")
    return r


def multi_turn(p):
    oid = next(o["record_id"] for o in ORDERS if check_order_status(o["record_id"])["escalation_recommended"])
    print("##### CONVERSATION A (session 'multi-turn'): the order id is given in turn 1 ONLY #####\n")
    say(p, "multi-turn", f"Hi, my order id is {oid}. Can you check it?")
    r2 = say(p, "multi-turn", "What is the status of my order?")            # no id: must come from memory
    say(p, "multi-turn", "How long does a COD refund take?")                 # policy turn in the same session
    say(p, "multi-turn", "Should it be escalated?")                          # 'it' = the remembered order
    assert r2.order and r2.order.record_id == oid, "memory failed to carry the order id"
    print("stored history for session 'multi-turn':")
    for line in p.memory.history("multi-turn"):
        print("  ", line[:110])
    print("\nOK: state (the order id) was carried across turns.")


def fresh(p):
    print("##### CONVERSATION B (fresh session, new process): the same follow-up with NO earlier turns #####\n")
    rb = say(p, "fresh", "What is the status of my order?")
    assert rb.order is None and rb.refusal_reason == "missing_order_id"
    print("stored history for session 'fresh':", len(p.memory.history("fresh")), "messages (only this turn)")
    print("\nOK: no state was available, so the agent correctly asked for the order id instead of guessing.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    pipeline = SupportPipeline()
    if mode in ("multi", "both"):
        multi_turn(pipeline)
    if mode == "both":
        print("\n" + "=" * 70 + "\n")
    if mode in ("fresh", "both"):
        fresh(pipeline if mode == "both" else SupportPipeline())
