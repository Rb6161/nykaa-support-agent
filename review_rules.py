"""Deterministic 'mock model' logic for the Autogen review team (dependency-free, unit-testable).

reviewer_text() plays the Policy-Compliance-Reviewer, build_verdict() plays the Final-Editor.
review.py feeds their output through real Autogen agents (RoundRobinGroupChat) via a replay model client."""
from __future__ import annotations

from typing import List

from guardrails import sentence_support
from schemas import Verdict
from textutil import IDK_MESSAGE, split_sentences


def find_ungrounded(draft: str, context_snippets: List[str], min_support: float = 0.6) -> List[str]:
    ctx = " ".join(context_snippets)
    return [s for s in split_sentences(draft) if sentence_support(s, ctx) < min_support]


def reviewer_text(draft: str, context_snippets: List[str]) -> str:
    bad = find_ungrounded(draft, context_snippets)
    if not bad:
        return "COMPLIANCE REVIEW: every sentence in the draft is supported by the retrieved context. No changes needed."
    listing = " | ".join(bad)
    return f"COMPLIANCE REVIEW: {len(bad)} sentence(s) are NOT supported by the retrieved context: {listing}. Remove them."


def build_verdict(draft: str, context_snippets: List[str]) -> Verdict:
    bad = find_ungrounded(draft, context_snippets)
    if not bad:
        return Verdict(approved=True, final_answer=draft, reason="All claims are supported by the retrieved context; approved unchanged.")
    kept = [s for s in split_sentences(draft) if s not in bad]
    final = " ".join(kept) if kept else IDK_MESSAGE
    return Verdict(
        approved=False,
        final_answer=final,
        reason=f"Removed {len(bad)} ungrounded sentence(s): {' | '.join(bad)}",
    )
