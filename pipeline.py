"""End-to-end support pipeline.

request -> input guardrails (injection, PII masking) -> budget cap -> session memory ->
CrewAI crew (RAG tool + order tool + composer) -> output groundedness guardrail ->
Autogen review team -> structured response + one JSONL log line."""
from __future__ import annotations

import config  # noqa: F401

import re
import time
import uuid
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

import rag
from crew import run_crew
from governance import BudgetExceeded, enforce_budget
from guardrails import check_grounded, detect_injection, mask_pii, strip_mask_tokens
from logger import log_request
from memory import ConversationMemory
from review import review
from schemas import AskResponse, CrewAnswer, OrderInfo, Verdict
from textutil import BUDGET_MESSAGE, IDK_MESSAGE, INJECTION_MESSAGE

ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b", re.I)
NEED_ORDER_ID_MESSAGE = "Please share your order id (for example NYK-0012) so I can check it."
ORDER_WORDS = re.compile(r"\b(status of (my|the|this) order|where is my order|track (my|the) order|order status|check (my|the) order|is my order)\b", re.I)


class SupportPipeline:
    def __init__(self):
        self.memory = ConversationMemory(self._turn)

    # ------------------------------------------------------------------ one conversational turn
    def _turn(self, inputs: dict) -> AIMessage:
        question: str = strip_mask_tokens(inputs["input"])   # memory/logs keep the masked text; retrieval gets the clean question
        history: List = inputs.get("history", [])

        order_id = self._find_order_id(question, history)
        wants_order = bool(ORDER_WORDS.search(question)) or bool(ORDER_ID_RE.search(question))
        if wants_order and not order_id:
            payload = {"answer": NEED_ORDER_ID_MESSAGE, "sources": [], "grounded": False, "refused": True,
                       "reason": "missing_order_id", "tools": [], "order": None, "review": None, "snippets": [],
                       "cache_hit": False, "crew_ms": 0.0, "review_ms": 0.0}
            return AIMessage(content=payload["answer"], additional_kwargs={"payload": payload})

        hits0 = rag.CACHE.hits
        t0 = time.perf_counter()
        crew_answer: CrewAnswer = run_crew(question, order_id)           # kickoff() + schema validation
        crew_ms = (time.perf_counter() - t0) * 1000
        cache_hit = rag.CACHE.hits > hits0

        threshold = rag.threshold_for(rag.recommended_strategy())
        order_ctx = crew_answer.context_snippets
        # ---- output-side groundedness guardrail
        tool_verdict = crew_answer.order is not None or crew_answer.order_not_found
        top1 = 1.0 if tool_verdict else self._top1(question)
        guard = check_grounded(crew_answer.answer, order_ctx, top1 if crew_answer.grounded else 0.0, threshold)
        if not crew_answer.grounded or not guard["ok"]:
            reason = "ungrounded" if not crew_answer.grounded else "unsupported_claims"
            payload = {"answer": IDK_MESSAGE, "sources": [], "grounded": False, "refused": True, "reason": reason,
                       "tools": crew_answer.used_tools, "order": crew_answer.order, "review": None, "snippets": order_ctx,
                       "cache_hit": cache_hit, "crew_ms": crew_ms, "review_ms": 0.0, "guard": guard}
            return AIMessage(content=IDK_MESSAGE, additional_kwargs={"payload": payload})

        # ---- Autogen review stage
        t1 = time.perf_counter()
        verdict: Verdict = review(crew_answer.answer, order_ctx)
        review_ms = (time.perf_counter() - t1) * 1000
        payload = {"answer": verdict.final_answer, "sources": crew_answer.sources, "grounded": True, "refused": False,
                   "reason": None, "tools": crew_answer.used_tools, "order": crew_answer.order, "review": verdict,
                   "snippets": order_ctx, "cache_hit": cache_hit, "crew_ms": crew_ms, "review_ms": review_ms}
        return AIMessage(content=verdict.final_answer, additional_kwargs={"payload": payload})

    @staticmethod
    def _find_order_id(question: str, history: List) -> Optional[str]:
        m = ORDER_ID_RE.search(question)
        if m:
            return m.group(0).upper()
        for msg in reversed(history):                      # session memory: reuse an earlier order id
            if isinstance(msg, HumanMessage):
                m = ORDER_ID_RE.search(str(msg.content))
                if m:
                    return m.group(0).upper()
        return None

    @staticmethod
    def _top1(question: str) -> float:
        return rag.grounded_answer(question)["top1_similarity"]  # cached: no redundant retrieval

    # ------------------------------------------------------------------ public API
    def ask(self, query: str, session_id: Optional[str] = None, endpoint: str = "/ask") -> AskResponse:
        trace_id = uuid.uuid4().hex[:12]
        session_id = session_id or uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        masked, pii = mask_pii(query)                        # the model, memory and logs only ever see `masked`
        resp = AskResponse(trace_id=trace_id, session_id=session_id, answer="", pii_masked=pii)
        timings = {}
        try:
            injected, which = detect_injection(masked)
            if injected:
                resp.answer, resp.refused, resp.refusal_reason = INJECTION_MESSAGE, True, f"prompt_injection:{which}"
            else:
                budget = enforce_budget(masked)
                resp.estimated_tokens = budget["estimated_tokens"]
                msg = self.memory.invoke(masked, session_id)
                p = msg.additional_kwargs["payload"]
                resp.answer, resp.sources, resp.grounded = p["answer"], p["sources"], p["grounded"]
                resp.refused, resp.refusal_reason = p["refused"], p["reason"]
                resp.tools_used, resp.order, resp.review = p["tools"], p["order"], p["review"]
                resp.cache_hit, resp.context_snippets = p["cache_hit"], p["snippets"]
                timings = {"crew_ms": round(p["crew_ms"], 2), "review_ms": round(p["review_ms"], 2)}
        except BudgetExceeded as e:
            resp.answer, resp.refused, resp.refusal_reason = BUDGET_MESSAGE, True, f"budget_exceeded:{e}"
            resp.estimated_tokens = e.estimated
        finally:
            resp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_request({
                "trace_id": trace_id, "endpoint": endpoint, "session_id": session_id, "query": masked,
                "pii_masked": pii, "refused": resp.refused, "refusal_reason": resp.refusal_reason,
                "grounded": resp.grounded, "tools_used": resp.tools_used, "cache_hit": resp.cache_hit,
                "estimated_tokens": resp.estimated_tokens, "latency_ms": resp.latency_ms, **timings,
            })
        return resp
