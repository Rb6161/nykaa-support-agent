"""Task 7/9 demo: run the CrewAI crew via kickoff() on different queries and PROVE which tool ran,
and validate every response against the CrewAnswer Pydantic schema."""
import config  # noqa: F401,E402

import json

from pydantic import ValidationError

from crew import run_crew
from dataset import ORDERS
from orders import check_order_status
from schemas import CrewAnswer
from tools import TOOL_CALL_LOG


def show(title, question, order_id=None):
    TOOL_CALL_LOG.clear()
    ans = run_crew(question, order_id)                      # crew.kickoff() inside
    print(f"\n=== {title} ===\nquestion : {question}\norder_id : {order_id}")
    print("tools actually invoked:", [c["tool"] for c in TOOL_CALL_LOG])
    print("tool call log:", json.dumps(TOOL_CALL_LOG))
    print("validated CrewAnswer:", ans.model_dump_json(indent=2))
    return ans


if __name__ == "__main__":
    escalate = next(o["record_id"] for o in ORDERS if check_order_status(o["record_id"])["escalation_recommended"])
    calm = next(o["record_id"] for o in ORDERS if not check_order_status(o["record_id"])["escalation_recommended"] and o["status"] == "Shipped")

    a = show("Query 1: policy question -> RAG tool", "How long does a COD refund take?")
    assert a.used_tools == ["rag_lookup"], a.used_tools

    b = show("Query 2: order question -> order lookup tool (+ RAG)", "Where is my order and what is its status?", calm)
    assert "check_order_status" in b.used_tools

    c = show("Query 3: order flagged for escalation", "Please check this order's status.", escalate)
    assert c.order and c.order.escalation_recommended

    print("\n=== schema enforcement ===")
    try:
        CrewAnswer.model_validate({"answer": "x", "grounded": "not-a-bool-at-all"})
    except ValidationError as e:
        print("a malformed crew response is rejected by the schema:", e.errors()[0]["msg"])
    print("\nOK: both tools were invoked on different queries and every response validated against CrewAnswer.")
