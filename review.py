"""Autogen review stage: a 2-agent RoundRobinGroupChat (Policy-Compliance-Reviewer, Final-Editor).

 * team = RoundRobinGroupChat([reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])
 * the Final-Editor has output_content_type=Verdict (Pydantic: approved, final_answer, reason)
 * input = the CrewAI Composer's draft + the original retrieved context
 * output = approve the draft unchanged, or revise it

MOCK_LLM: the two agents' "models" are deterministic replay clients whose replies are computed by
review_rules.py (rule-based: sentences not supported by the retrieved context are removed). The agents,
team, turn limit and structured message types are the real Autogen objects; no key or network is used."""
from __future__ import annotations

import config  # noqa: F401

import asyncio
import json
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.replay import ReplayChatCompletionClient

from review_rules import build_verdict, reviewer_text
from schemas import Verdict

MODEL_INFO = {"vision": False, "function_calling": False, "json_output": True,
              "family": "unknown", "structured_output": True}


def _build_team(draft: str, context_snippets: List[str]) -> RoundRobinGroupChat:
    verdict = build_verdict(draft, context_snippets)
    reviewer_client = ReplayChatCompletionClient([reviewer_text(draft, context_snippets)], model_info=MODEL_INFO)
    editor_client = ReplayChatCompletionClient([verdict.model_dump_json()], model_info=MODEL_INFO)

    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=reviewer_client,
        system_message="Check every sentence of the draft against the retrieved context and flag unsupported claims.")
    editor = AssistantAgent(
        "Final_Editor", model_client=editor_client, output_content_type=Verdict,
        system_message="Return a verdict: approve the draft unchanged, or revise it to remove unsupported claims.")
    return RoundRobinGroupChat(
        [reviewer, editor], max_turns=2,                       # both agents speak exactly once
        custom_message_types=[StructuredMessage[Verdict]],      # required for the structured verdict
    )


async def review_async(draft: str, context_snippets: List[str]) -> Verdict:
    team = _build_team(draft, context_snippets)
    task = "DRAFT ANSWER:\n" + draft + "\n\nRETRIEVED CONTEXT:\n" + "\n".join(f"- {c}" for c in context_snippets)
    result = await team.run(task=task)
    last = result.messages[-1]
    content = getattr(last, "content", last)
    if isinstance(content, Verdict):
        return content
    if isinstance(content, str):
        return Verdict.model_validate(json.loads(content))
    return Verdict.model_validate(content)


def review(draft: str, context_snippets: List[str]) -> Verdict:
    """Synchronous wrapper (call from a thread without a running event loop)."""
    return asyncio.run(review_async(draft, context_snippets))
