"""Tests the mock LLM's ReAct behaviour. If crewai is not installed, a minimal stand-in BaseLLM is
injected so the parsing logic (both silent pitfalls from the brief) is still verified."""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import crewai.llms.base_llm  # noqa: F401
except Exception:  # pragma: no cover
    pkg = types.ModuleType("crewai"); llms = types.ModuleType("crewai.llms"); base = types.ModuleType("crewai.llms.base_llm")

    class BaseLLM:
        def __init__(self, model, **kw):
            self.model = model

    base.BaseLLM = BaseLLM
    sys.modules.update({"crewai": pkg, "crewai.llms": llms, "crewai.llms.base_llm": base})

from mock_llm import MockCrewLLM  # noqa: E402

TEMPLATE = (
    "Use the following format:\nThought: you should always think about what to do\nAction: the action to take\n"
    "Action Input: the input to the action\nObservation: the result of the action\n"
)


def sys_prompt(role, tool_name, arg):
    return (f"You are {role}. You do things.\nYour personal goal is: x\n"
            f"Tool Name: {tool_name}\nTool Arguments: {{'{arg}': {{'description': 'd', 'type': 'str'}}}}\n"
            f"Tool Description: some tool\n{TEMPLATE}")


def test_retrieval_calls_tool_then_returns_result_not_template():
    llm = MockCrewLLM()
    msgs = [{"role": "system", "content": sys_prompt("Retrieval Agent", "rag_lookup", "query")},
            {"role": "user", "content": "Current Task: x\nUSER_QUESTION: How long is a COD refund? END_QUESTION"}]
    first = llm.call(msgs)
    # pitfall 1: template text in the system prompt must NOT be mistaken for a tool result
    assert first.startswith("Thought:") and "Action: rag_lookup" in first and "Final Answer" not in first
    assert json.loads(first.split("Action Input:")[1])["query"] == "How long is a COD refund?"
    tool_json = json.dumps({"tool": "rag_lookup", "answer": "5 to 7 working days.", "grounded": True})
    msgs.append({"role": "assistant", "content": first + f"\nObservation: {tool_json}"})
    final = llm.call(msgs)
    assert "Final Answer:" in final and "5 to 7 working days" in final


def test_dispatch_uses_declared_argument_schema_not_tool_name():
    # tool named "rag_lookup" contains the substring "lookup" but takes `query`, not `record_id`
    llm = MockCrewLLM()
    sys_text = sys_prompt("Lookup Agent", "rag_lookup", "query")
    msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": "ORDER_ID: NYK-0001"}]
    out = llm.call(msgs)
    assert "Final Answer" in out and "no tool declares an argument named record_id" in out  # no misclassification
    sys_text = sys_prompt("Lookup Agent", "check_order_status", "record_id")
    out = llm.call([{"role": "system", "content": sys_text}, {"role": "user", "content": "ORDER_ID: NYK-0001"}])
    assert "Action: check_order_status" in out and '"record_id": "NYK-0001"' in out


def test_lookup_skips_when_no_order_id():
    llm = MockCrewLLM()
    out = llm.call([{"role": "system", "content": sys_prompt("Lookup Agent", "check_order_status", "record_id")},
                    {"role": "user", "content": "ORDER_ID: NONE"}])
    assert "Final Answer" in out and '"skipped": true' in out


def test_composer_merges_rag_and_order():
    llm = MockCrewLLM()
    rag = {"tool": "rag_lookup", "answer": "Beauty items can be returned within 15 days.", "sources": ["01_return_window"],
           "grounded": True, "chunks": ["Beauty items can be returned within 15 days of delivery."]}
    order = {"tool": "check_order_status", "record_id": "NYK-0001", "found": True, "status": "Shipped",
             "order_value_inr": 1888, "escalation_score": 0.1867, "escalation_recommended": False,
             "category": "Beauty", "delayed_shipment": False, "days_since_created": 14}
    ctx = "This is the context you're working with:\n" + json.dumps(rag) + "\n" + json.dumps(order)
    out = llm.call([{"role": "system", "content": "You are Response Composer. x"}, {"role": "user", "content": ctx}])
    payload = json.loads(out.split("Final Answer:")[1])
    assert payload["grounded"] and payload["used_tools"] == ["rag_lookup", "check_order_status"]
    assert "NYK-0001" in payload["answer"] and "15 days" in payload["answer"]
    assert payload["order"]["escalation_score"] == 0.1867


def test_composer_ungrounded_gives_idk():
    out = MockCrewLLM().call([{"role": "system", "content": "You are Response Composer. x"},
                              {"role": "user", "content": json.dumps({"tool": "rag_lookup", "grounded": False, "answer": "x", "chunks": []})}])
    payload = json.loads(out.split("Final Answer:")[1])
    assert payload["grounded"] is False and "don't know" in payload["answer"]


def test_composer_unknown_order_is_a_grounded_not_found_answer():
    out = MockCrewLLM().call([{"role": "system", "content": "You are Response Composer. x"},
                              {"role": "user", "content": json.dumps({"tool": "rag_lookup", "grounded": False, "answer": "x", "chunks": []})
                               + "\n" + json.dumps({"tool": "check_order_status", "record_id": "NYK-9999", "found": False,
                                                    "error": "No order found with id NYK-9999."})}])
    payload = json.loads(out.split("Final Answer:")[1])
    assert payload["grounded"] and payload["order_not_found"] and payload["order"] is None
    assert "NYK-9999" in payload["answer"] and payload["context_snippets"] == ["No order found with id NYK-9999."]
