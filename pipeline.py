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
from logger import log_escalation, log_request
from memory import ConversationMemory
from review import review
from schemas import AskResponse, CrewAnswer, OrderInfo, PublicOrder, Verdict
from textutil import BUDGET_MESSAGE, content_tokens, CARD_SAFETY_MESSAGE, IDK_MESSAGE, INJECTION_MESSAGE, SENSITIVE_CARD_RE

ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b", re.I)
NEED_ORDER_ID_MESSAGE = "Please share your order id (for example NYK-0012) so I can check it."
ORDER_REFERENCE = re.compile(r"\b((my|this|that|the|same) (order|parcel|package|shipment)|escalat\w*|delayed)\b", re.I)
ORDER_FILLER = set("order orders id check status track tracking where delayed escalate escalated escalation hi hello hey thanks thank "
                   "please share details detail update latest current".split())
ORDER_WORDS = re.compile(r"\b(status of (my|the|this) order|where is my order|track (my|the) order|order status|check (my|the) order|is my order)\b", re.I)


class SupportPipeline:
    def __init__(self):
        self.memory = ConversationMemory(self._turn)

    # ------------------------------------------------------------------ one conversational turn
    def _turn(self, inputs: dict) -> AIMessage:
        question: str = strip_mask_tokens(inputs["input"])   # memory/logs keep the masked text; retrieval gets the clean question
        history: List = inputs.get("history", [])

        # 1) Does the question carry content beyond "an order request"? ("cancel", "return window", ... = policy content)
        leftover = ORDER_REFERENCE.sub(" ", ORDER_WORDS.sub(" ", ORDER_ID_RE.sub(" ", question)))
        has_policy_content = bool([t for t in content_tokens(leftover) if t not in ORDER_FILLER])
        # 2) Which order id applies? An id typed now always wins. A REMEMBERED id (from an earlier turn) is used only
        #    for a pure order follow-up ("what is the status of my order?", "should it be escalated?"), never for a
        #    policy question such as "can I cancel my order after it is placed?".
        explicit = ORDER_ID_RE.search(question)
        follow_up = (bool(ORDER_WORDS.search(question)) or bool(ORDER_REFERENCE.search(question))) and not has_policy_content
        order_id = explicit.group(0).upper() if explicit else (self._find_order_id(question, history) if follow_up else None)
        # 3) An order id is DEMANDED only when the customer clearly asks about an order (not for policy questions).
        wants_order = bool(explicit) or bool(ORDER_WORDS.search(question))
        if wants_order and not order_id:
            payload = {"answer": NEED_ORDER_ID_MESSAGE, "sources": [], "grounded": False, "refused": True,
                       "reason": "missing_order_id", "tools": [], "order": None, "review": None, "snippets": [],
                       "cache_hit": False, "crew_ms": 0.0, "review_ms": 0.0}
            return AIMessage(content=payload["answer"], additional_kwargs={"payload": payload})

        hits0 = rag.CACHE.hits
        t0 = time.perf_counter()
        # An order-only request ("check NYK-0003") has no policy question: do not pad the answer with policy text.
        order_only = bool(order_id) and not has_policy_content
        crew_answer: CrewAnswer = run_crew(question, order_id, policy=not order_only)   # kickoff() + schema validation
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
        internal_order = None
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
                resp.tools_used, resp.review = p["tools"], p["review"]
                if p["order"] is not None:
                    internal_order = p["order"]                    # keeps the escalation fields: backend only
                    resp.order = PublicOrder(**internal_order.model_dump(exclude={"escalation_score", "escalation_recommended"}))
                resp.cache_hit, resp.context_snippets = p["cache_hit"], p["snippets"]
                timings = {"crew_ms": round(p["crew_ms"], 2), "review_ms": round(p["review_ms"], 2)}
        except BudgetExceeded as e:
            resp.answer, resp.refused, resp.refusal_reason = BUDGET_MESSAGE, True, f"budget_exceeded:{e}"
            resp.estimated_tokens = e.estimated
        finally:
            if ("card" in pii or SENSITIVE_CARD_RE.search(query)) and CARD_SAFETY_MESSAGE not in resp.answer:
                resp.answer = f"{resp.answer} {CARD_SAFETY_MESSAGE}".strip()   # warn the customer, whatever else was answered
                log_escalation(trace_id, session_id, "sensitive_card_data_shared", masked)   # real hand-off record
            resp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_request({
                "trace_id": trace_id, "endpoint": endpoint, "session_id": session_id, "query": masked,
                "pii_masked": pii, "refused": resp.refused, "refusal_reason": resp.refusal_reason,
                "grounded": resp.grounded, "tools_used": resp.tools_used, "cache_hit": resp.cache_hit,
                "estimated_tokens": resp.estimated_tokens, "latency_ms": resp.latency_ms, **timings,
                **({"order_id": internal_order.record_id, "escalation_score": internal_order.escalation_score,
                    "escalation_recommended": internal_order.escalation_recommended} if internal_order else {}),
            })
            if internal_order is not None and internal_order.escalation_recommended:
                log_escalation(trace_id, session_id, "order_escalation_recommended", masked)
        return resp
