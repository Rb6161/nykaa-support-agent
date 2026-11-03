"""Governance layers (my four layers: Data, Model, Application, Runtime).

  Data        : only fabricated orders; phone / card-last-4 masked before the model AND before logs.
  Model       : MOCK_LLM deterministic models only by default; no keys, no network.
  Application : least autonomy (TOOL_ACL), structured outputs (Pydantic), independent review stage.
  Runtime     : per-request token/cost budget cap, response cache, JSONL audit logging.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------- Application layer
# Least autonomy: each tool is callable only by the agent roles listed here.
TOOL_ACL = {
    "check_order_status": {"Lookup Agent"},
    "rag_lookup": {"Retrieval Agent"},
}


def authorize_tool(agent_role: str, tool_name: str) -> None:
    allowed = TOOL_ACL.get(tool_name)
    if allowed is not None and agent_role not in allowed:
        raise PermissionError(
            f"Least-autonomy violation: agent '{agent_role}' may not use tool '{tool_name}' "
            f"(allowed: {sorted(allowed)})"
        )


def audit_tool_wiring(agent_tools: dict) -> dict:
    """agent_tools: {role: [tool names]}. Raises if any tool is wired to a non-allowed role."""
    for role, tools in agent_tools.items():
        for t in tools:
            authorize_tool(role, t)
    return agent_tools


RISK_CLASSIFICATION = {
    "level": "Medium",
    "justification": (
        "This system is a customer-support assistant that answers policy questions and reads order "
        "status, which falls in the Medium tier (customer support tickets). It is not Low, because it "
        "touches customer order records and receives phone numbers and card last-4 digits, which are "
        "personal data. It is not High, because it handles no medical data, makes no hiring or credit "
        "decisions, cannot move money (it only quotes refund policy), and has read-only access to "
        "fabricated order data. Its answers are grounded in a fixed knowledge base, reviewed by a "
        "second agent team, and low-confidence cases are refused or escalated to a human."
    ),
}

# ----------------------------------------------------------------- Runtime layer
MAX_REQUEST_TOKENS = 1000          # hard cap per request (input estimate + fixed prompt overhead)
PROMPT_OVERHEAD_TOKENS = 700       # three agent system prompts + retrieved context, estimated
PRICE_PER_1K_TOKENS_USD = 0.0006   # notional price used for the cost estimate


class BudgetExceeded(Exception):
    def __init__(self, estimated: int, cap: int):
        super().__init__(f"Estimated {estimated} tokens exceeds the per-request cap of {cap}.")
        self.estimated = estimated
        self.cap = cap


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def enforce_budget(text: str, cap: int = MAX_REQUEST_TOKENS) -> dict:
    est = estimate_tokens(text) + PROMPT_OVERHEAD_TOKENS
    if est > cap:
        raise BudgetExceeded(est, cap)
    return {"estimated_tokens": est, "cap": cap, "estimated_cost_usd": round(est / 1000 * PRICE_PER_1K_TOKENS_USD, 6)}
