"""Tiny shared text helpers (no third-party imports)."""
import re

IDK_MESSAGE = (
    "I don't know the answer to that from Nykaa's policy knowledge base. "
    "I can raise it with a human support agent if you like."
)
INJECTION_MESSAGE = (
    "I can't help with that request. I only answer Nykaa order and policy questions."
)
BUDGET_MESSAGE = "Your request is too large for the per-request token budget, so it was rejected."

STOPWORDS = set(
    "a an the is are was were be been being of to in on for and or but if then so as at by with from "
    "it its this that these those i you we they he she my your our their me us them do does did can "
    "could should would will shall may might have has had not no yes how what when where which who "
    "whom why there here also any all each every than into about over after before up out".split()
)


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def content_tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
