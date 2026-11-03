"""Input-side guardrails (PII masking, prompt-injection detection) and the output-side
groundedness check. Pure Python / regex only: works with no API key and no network.

PII in scope for masking (fixed-format fields):
  * phone number (Indian 10-digit mobile, optional +91 / 91 prefix, spaces or dashes)
  * payment-card last-4 digits (and full 16-digit card numbers)
Out of scope (free text, no reliable pattern for a keyless masker): customer name, delivery address.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from textutil import content_tokens, split_sentences

PHONE_TOKEN = "[PHONE_MASKED]"
CARD_TOKEN = "[CARD_LAST4_MASKED]"

PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)")

# Patterns whose capture group 1 is the 4 digits to mask.
_CARD_LAST4_PATTERNS = [
    re.compile(
        r"\b(?:card|visa|mastercard|rupay|amex|debit|credit)\b[^.\n]{0,25}?"
        r"\b(?:ending|ends|last\s*4|last\s*four)(?:\s*digits)?(?:\s*(?:in|is|are|with|:))?\s*(\d{4})\b",
        re.I,
    ),
    re.compile(r"\blast\s*(?:4|four)\s*digits?(?:\s*(?:of|on)\s*(?:my|the)\s*card)?\s*(?:is|are|:)?\s*(\d{4})\b", re.I),
    re.compile(r"(?:[Xx*•]{4}[\s\-]?){2,3}(\d{4})\b"),
]
FULL_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)")


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """Return (masked_text, list_of_field_types_found)."""
    found: List[str] = []
    out = text

    def _full_card(_m):
        found.append("card")
        return CARD_TOKEN

    out = FULL_CARD_RE.sub(_full_card, out)

    for pat in _CARD_LAST4_PATTERNS:

        def _sub(m, _pat=pat):
            found.append("card")
            s, e = m.start(1) - m.start(0), m.end(1) - m.start(0)
            g = m.group(0)
            return g[:s] + CARD_TOKEN + g[e:]

        out = pat.sub(_sub, out)

    def _phone(_m):
        found.append("phone")
        return PHONE_TOKEN

    out = PHONE_RE.sub(_phone, out)
    return out, sorted(set(found))


def mask_text(text: str) -> str:
    return mask_pii(text)[0]


def contains_raw_pii(text: str) -> bool:
    return bool(mask_pii(text)[1])


# ---------------------------------------------------------------- prompt injection
_INJECTION_PATTERNS = [
    ("ignore_instructions", r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?)\b"),
    ("reveal_prompt", r"\b(reveal|show|print|repeat|leak|tell me)\b[^.\n]{0,30}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b"),
    ("role_override", r"\byou are now\b|\bact as (an? |the )?(admin|administrator|root|superuser|developer|system|unrestricted|different|another (ai|assistant|bot))\b|\bpretend (to be|you are)\b|\bfrom now on you\b"),
    ("jailbreak", r"\bjailbreak\b|\bdeveloper mode\b|\bdan mode\b|\bno restrictions\b"),
    ("exfiltrate", r"\b(api[\s_-]?key|password|secret token)\b[^.\n]{0,20}\b(show|give|send|reveal)\b|\b(show|give|send|reveal)\b[^.\n]{0,20}\b(api[\s_-]?key|password|secret token)\b"),
]
_COMPILED_INJECTION = [(n, re.compile(p, re.I)) for n, p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> Tuple[bool, str]:
    for name, pat in _COMPILED_INJECTION:
        if pat.search(text):
            return True, name
    return False, ""


# ---------------------------------------------------------------- groundedness
def sentence_support(sentence: str, context: str) -> float:
    """Fraction of the sentence's content tokens that appear in the context. Any number in the
    sentence that is missing from the context forces the score to 0 (a made-up figure)."""
    toks = content_tokens(sentence)
    if not toks:
        return 1.0
    ctx = set(content_tokens(context))
    nums = [t for t in toks if t.isdigit()]
    if any(n not in ctx for n in nums):
        return 0.0
    return sum(t in ctx for t in toks) / len(toks)


def check_grounded(answer: str, context_snippets: List[str], top1_similarity: float, threshold: float,
                   min_support: float = 0.6) -> dict:
    """Output-side groundedness guardrail.

    Passes only if (a) retrieval similarity cleared the calibrated threshold OR order data supplied
    the context, and (b) every answer sentence is supported by the context."""
    context = " ".join(context_snippets)
    unsupported = [s for s in split_sentences(answer) if sentence_support(s, context) < min_support]
    similarity_ok = top1_similarity >= threshold
    return {
        "ok": similarity_ok and not unsupported,
        "similarity_ok": similarity_ok,
        "top1_similarity": round(top1_similarity, 4),
        "threshold": threshold,
        "unsupported_sentences": unsupported,
    }


# ---------------------------------------------------------------- query clean-up for retrieval
_MASK_RE = re.compile(r"\[(?:PHONE|CARD_LAST4)_MASKED\]")
_ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b", re.I)


def strip_mask_tokens(text: str) -> str:
    """Query used for RETRIEVAL. The mask placeholders carry no meaning but add noise to the embedding and
    can push a valid question below the similarity threshold. Sentences that only supplied PII (e.g.
    'My card is [CARD_LAST4_MASKED].') are dropped when other sentences remain; a sentence with an order id is
    always kept. The masked text itself is still what memory and logs store."""
    sentences = split_sentences(text)
    kept = [s for s in sentences if not _MASK_RE.search(s) or _ORDER_ID_RE.search(s)]
    chosen = " ".join(kept) if kept else text
    return re.sub(r"\s+", " ", _MASK_RE.sub(" ", chosen)).strip()
