"""CrewAI tools. Each tool records its invocations in TOOL_CALL_LOG so demos can PROVE which tool was
actually invoked by which query."""
from __future__ import annotations

import config  # noqa: F401

import json
from typing import List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import rag
from orders import check_order_status

NO_POLICY_QUESTION = "__NO_POLICY_QUESTION__"   # set when the customer only asked about an order
TOOL_CALL_LOG: List[dict] = []


class RagLookupInput(BaseModel):
    query: str = Field(..., description="The customer's policy question, in plain words.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = (
        "Searches Nykaa's policy knowledge base and returns a grounded answer, its source documents and "
        "the retrieved passages. Use for any policy question (returns, refunds, delivery, warranty, ...)."
    )
    args_schema: Type[BaseModel] = RagLookupInput

    def _run(self, query: str) -> str:
        if query.strip() == NO_POLICY_QUESTION:      # order-only request: nothing to look up in the policy KB
            return json.dumps({"tool": "rag_lookup", "answer": "", "sources": [], "grounded": False,
                               "top1_similarity": 0.0, "chunks": []})
        result = rag.grounded_answer(query)  # cached grounded-generation step
        TOOL_CALL_LOG.append({"tool": self.name, "input": query, "cache_hit": result["cache_hit"]})
        payload = {"tool": "rag_lookup", "answer": result["answer"], "sources": result["sources"],
                   "grounded": result["grounded"], "top1_similarity": result["top1_similarity"],
                   "chunks": result["chunks"] if result["grounded"] else []}
        return json.dumps(payload)


class CheckOrderStatusInput(BaseModel):
    record_id: str = Field(..., description="The order id, for example NYK-0012.")


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = (
        "Looks up one order by id and returns its status, order value in INR and an escalation score "
        "in [0, 1] with a recommendation on whether to escalate."
    )
    args_schema: Type[BaseModel] = CheckOrderStatusInput

    def _run(self, record_id: str) -> str:
        result = check_order_status(record_id)
        TOOL_CALL_LOG.append({"tool": self.name, "input": record_id, "found": result["found"]})
        return json.dumps({"tool": "check_order_status", **result})
