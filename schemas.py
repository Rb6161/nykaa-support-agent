"""All Pydantic models: crew structured output, review verdict, and API request/response models."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class OrderInfo(BaseModel):
    record_id: str
    status: str
    order_value_inr: int
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_recommended: bool
    category: Optional[str] = None
    delayed_shipment: Optional[bool] = None
    days_since_created: Optional[int] = None


class CrewAnswer(BaseModel):
    """Every crew response must conform to this schema (validated in code after kickoff)."""

    answer: str
    sources: List[str] = Field(default_factory=list)
    grounded: bool
    used_tools: List[str] = Field(default_factory=list)
    order: Optional[OrderInfo] = None
    order_not_found: bool = False   # the lookup tool ran and the order id does not exist
    context_snippets: List[str] = Field(default_factory=list)


class Verdict(BaseModel):
    """Structured output of the Autogen Final-Editor."""

    approved: bool
    final_answer: str
    reason: str


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    grounded: bool = False
    refused: bool = False
    refusal_reason: Optional[str] = None
    pii_masked: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    order: Optional[OrderInfo] = None
    review: Optional[Verdict] = None
    cache_hit: bool = False
    estimated_tokens: int = 0
    latency_ms: float = 0.0
    context_snippets: List[str] = Field(default_factory=list)


class AddDocumentRequest(BaseModel):
    title: str
    text: str


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_fixed: int
    chunks_sentence: int
