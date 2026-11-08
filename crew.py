"""CrewAI crew: Retrieval Agent (RAG tool), Lookup Agent (order tool), Response Composer.

Least autonomy is enforced at build time: make_agent() refuses to wire a tool to any role that is not
on that tool's allow-list (governance.TOOL_ACL)."""
from __future__ import annotations

import config  # noqa: F401  (disables CrewAI telemetry BEFORE crewai is imported)

import json
from typing import List, Optional

from crewai import Agent, Crew, Process, Task

from governance import audit_tool_wiring, authorize_tool
from mock_llm import get_llm
from schemas import CrewAnswer
from tools import NO_POLICY_QUESTION, CheckOrderStatusTool, RagLookupTool


def make_agent(role: str, goal: str, backstory: str, tools: list, llm) -> Agent:
    for t in tools:
        authorize_tool(role, t.name)  # raises PermissionError on a least-autonomy violation
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm,
                 allow_delegation=False, verbose=False)


def build_crew(question: str, order_id: Optional[str], policy: bool = True) -> Crew:
    llm = get_llm()
    retrieval = make_agent(
        "Retrieval Agent", "Find the policy passages that answer the customer's question.",
        "You search Nykaa's policy knowledge base and never answer from memory.", [RagLookupTool()], llm)
    lookup = make_agent(
        "Lookup Agent", "Look up the customer's order when an order id is given.",
        "You are the only agent allowed to read order records.", [CheckOrderStatusTool()], llm)
    composer = make_agent(
        "Response Composer", "Combine the findings into one final, grounded answer.",
        "You write the final customer answer using only what the other agents found.", [], llm)

    t_retrieve = Task(
        description=f"Search the Nykaa policy knowledge base with the rag_lookup tool.\nUSER_QUESTION: {question if policy else NO_POLICY_QUESTION} END_QUESTION",
        expected_output="The tool's JSON result.", agent=retrieval)
    t_lookup = Task(
        description=f"If an order id is given, call the check_order_status tool.\nORDER_ID: {order_id or 'NONE'}",
        expected_output="The tool's JSON result.", agent=lookup)
    t_compose = Task(
        description="Combine the retrieval and lookup findings into one final answer matching the CrewAnswer schema.",
        expected_output="A JSON object matching the CrewAnswer schema.", agent=composer,
        context=[t_retrieve, t_lookup], output_pydantic=CrewAnswer)

    audit_tool_wiring({a.role: [t.name for t in a.tools] for a in (retrieval, lookup, composer)})
    return Crew(agents=[retrieval, lookup, composer], tasks=[t_retrieve, t_lookup, t_compose],
                process=Process.sequential, verbose=False)


def run_crew(question: str, order_id: Optional[str] = None, policy: bool = True) -> CrewAnswer:
    """kickoff() the crew and validate the response against the CrewAnswer Pydantic schema."""
    result = build_crew(question, order_id, policy).kickoff()
    raw = getattr(result, "raw", None) or str(result)
    pyd = getattr(result, "pydantic", None)
    if isinstance(pyd, CrewAnswer):
        return CrewAnswer.model_validate(pyd.model_dump())   # re-validate in code
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"Crew output is not JSON: {raw[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    return CrewAnswer.model_validate(obj)
