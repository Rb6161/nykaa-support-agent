"""Deterministic MOCK_LLM for CrewAI (no API key, no network).

Extends crewai.llms.base_llm.BaseLLM, CrewAI's documented extension point for a non-litellm LLM.
CrewAI drives agents with a ReAct text protocol, so this mock speaks it:

    Thought: ...
    Action: <tool name>
    Action Input: {"arg": "value"}                      <- first call of an agent that has a tool
    ...
    Thought: I now know the final answer
    Final Answer: <text>                                <- after the tool result ("Observation:") arrives

TWO SILENT PITFALLS handled here (both give wrong output with no crash):
 (1) CrewAI's ReAct system prompt template itself contains the literal text
     "Observation: the result of the action". We therefore NEVER search the system prompt; we only
     read the model-conversation messages after the system message, and we ignore any
     "Observation:" followed by that template placeholder.
 (2) Tool arguments are chosen from the tool's own DECLARED ARGUMENT SCHEMA ("Tool Arguments: {...}"
     in the prompt), never by checking whether a word such as "lookup" is a substring of the tool
     name (a tool called rag_lookup would be misclassified).
"""
from __future__ import annotations

import config  # noqa: F401

import json
import re
from typing import Any, Dict, List, Optional

from crewai.llms.base_llm import BaseLLM

from orders import order_sentence
from textutil import IDK_MESSAGE

TEMPLATE_PLACEHOLDER = "the result of the action"


def _messages_to_pairs(messages) -> List[tuple]:
    if isinstance(messages, str):
        return [("user", messages)]
    pairs = []
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content", "")
            if isinstance(c, list):  # content parts
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            pairs.append((m.get("role", "user"), str(c)))
        else:
            pairs.append((getattr(m, "role", "user"), str(getattr(m, "content", m))))
    return pairs


def parse_declared_tools(system_text: str) -> Dict[str, List[str]]:
    """{tool_name: [declared argument names]} read from the prompt's own tool declarations."""
    tools: Dict[str, List[str]] = {}
    for m in re.finditer(r"Tool Name:\s*([\w\-]+)(.*?)(?=Tool Name:|\Z)", system_text, re.S):
        name, body = m.group(1), m.group(2)
        am = re.search(r"Tool Arguments:\s*(\{.*?\})\s*(?:\n|$)", body, re.S)
        args = re.findall(r"""['"]([A-Za-z_]\w*)['"]\s*:\s*\{""", am.group(1)) if am else []
        if not args and am:
            args = re.findall(r"""['"]([A-Za-z_]\w*)['"]\s*:""", am.group(1))
        tools[name] = args
    return tools


def last_observation(pairs: List[tuple]) -> Optional[str]:
    """Latest tool result, read ONLY from the model-conversation turns (never the system prompt)."""
    result = None
    for role, content in pairs:
        if role == "system":
            continue
        for m in re.finditer(r"Observation:\s*", content):
            rest = content[m.end():]
            if rest.lower().startswith(TEMPLATE_PLACEHOLDER):
                continue  # template text, not a real tool result
            result = rest.strip()
    return result


def _decode_tool_jsons(text: str) -> List[dict]:
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        i = text.find('{"tool":', i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text[i:])
            out.append(obj)
            i += end
        except ValueError:
            i += 1


def first_json_object(text: str) -> Optional[dict]:
    """First balanced JSON object in text (observations may carry trailing ReAct text)."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except ValueError:
        return None


class MockCrewLLM(BaseLLM):
    """Rule-based stand-in for the three agents' language model."""

    def __init__(self):
        super().__init__(model="mock-llm")
        self.calls: List[str] = []

    # ---- BaseLLM interface -------------------------------------------------------------------
    def supports_function_calling(self) -> bool:
        return False  # force CrewAI's text (ReAct) tool protocol

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, **kwargs) -> Any:
        pairs = _messages_to_pairs(messages)
        system = "\n".join(c for r, c in pairs if r == "system")
        convo = "\n".join(c for r, c in pairs if r != "system")
        role_match = re.search(r"You are ([^.\n]+?)\.", system) or re.search(r"You are ([^.\n]+?)\.", convo)
        role = role_match.group(1).strip() if role_match else ""
        self.calls.append(role)

        if role == "Retrieval Agent":
            return self._tool_agent(pairs, system, convo, arg_hint="query", marker=r"USER_QUESTION:\s*(.*?)\s*END_QUESTION")
        if role == "Lookup Agent":
            return self._lookup_agent(pairs, system, convo)
        if role == "Response Composer":
            return self._composer(convo)
        return self._fallback(convo)

    # ---- agents ------------------------------------------------------------------------------
    def _react_call(self, tool: str, args: dict) -> str:
        return f"Thought: I should use the {tool} tool.\nAction: {tool}\nAction Input: {json.dumps(args)}"

    @staticmethod
    def _final(text: str) -> str:
        # Tool results are JSON objects; strip any trailing ReAct text CrewAI may have appended.
        obj = first_json_object(text)
        if obj is not None:
            text = json.dumps(obj)
        return f"Thought: I now know the final answer\nFinal Answer: {text}"

    def _pick_tool(self, system: str, wanted_arg: str) -> Optional[tuple]:
        """Choose the tool whose DECLARED ARGUMENT SCHEMA contains `wanted_arg` (pitfall 2)."""
        for name, args in parse_declared_tools(system).items():
            if wanted_arg in args:
                return name, args
        return None

    def _tool_agent(self, pairs, system, convo, arg_hint: str, marker: str):
        obs = last_observation(pairs)
        if obs is not None:
            return self._final(obs)
        q = re.search(marker, convo, re.S)
        question = q.group(1).strip() if q else convo.strip()[-200:]
        tool = self._pick_tool(system, arg_hint)
        if tool is None:
            return self._final(json.dumps({"tool": "none", "error": f"no tool declares an argument named {arg_hint}"}))
        return self._react_call(tool[0], {arg_hint: question})

    def _lookup_agent(self, pairs, system, convo):
        obs = last_observation(pairs)
        if obs is not None:
            return self._final(obs)
        m = re.search(r"ORDER_ID:\s*(NYK-\d{4}|NONE)", convo)
        order_id = m.group(1) if m else "NONE"
        if order_id == "NONE":
            return self._final(json.dumps({"tool": "check_order_status", "skipped": True, "found": False,
                                           "reason": "no order id supplied"}))
        tool = self._pick_tool(system, "record_id")
        if tool is None:
            return self._final(json.dumps({"tool": "none", "error": "no tool declares an argument named record_id"}))
        return self._react_call(tool[0], {"record_id": order_id})

    def _composer(self, convo: str) -> str:
        results = {r.get("tool"): r for r in _decode_tool_jsons(convo)}
        rag = results.get("rag_lookup", {})
        order = results.get("check_order_status", {})
        parts, tools, snippets = [], [], list(rag.get("chunks", [])) if rag.get("grounded") else []
        sources = list(rag.get("sources", [])) if rag.get("grounded") else []
        if rag:
            tools.append("rag_lookup")
        order_info, not_found = None, False
        if order.get("found"):
            tools.append("check_order_status")
            order_info = {k: order[k] for k in ("record_id", "status", "order_value_inr", "escalation_score",
                                                "escalation_recommended", "category", "delayed_shipment",
                                                "days_since_created")}
            sentence = order_sentence(order_info)
            parts.append(sentence)
            snippets.append(sentence)
        elif order.get("error"):
            tools.append("check_order_status")
            not_found = True
            parts.append(order["error"])
            snippets.append(order["error"])      # the tool's own result is part of the grounding context
        if rag.get("grounded"):
            parts.append(rag.get("answer", ""))
        grounded = bool(rag.get("grounded")) or order_info is not None or not_found
        answer = " ".join(p for p in parts if p) or IDK_MESSAGE
        payload = {"answer": answer, "sources": sources, "grounded": grounded, "used_tools": tools,
                   "order": order_info, "order_not_found": not_found, "context_snippets": snippets}
        return self._final(json.dumps(payload))

    def _fallback(self, convo: str) -> str:
        """Unknown caller (e.g. CrewAI's own structured-output conversion prompt): return the last
        JSON answer object found in the text."""
        for m in reversed(list(re.finditer(r'\{"answer"', convo))):
            try:
                obj, _ = json.JSONDecoder().raw_decode(convo[m.start():])
                return json.dumps(obj)
            except ValueError:
                continue
        return self._final(IDK_MESSAGE)


def get_llm():
    """MOCK_LLM=true (default): deterministic mock. MOCK_LLM=false: a real LLM named in LLM_MODEL
    (optional; every acceptance criterion is demonstrated under the mock)."""
    import os

    if config.mock_llm_enabled():
        return MockCrewLLM()
    from crewai import LLM

    return LLM(model=os.environ["LLM_MODEL"])
