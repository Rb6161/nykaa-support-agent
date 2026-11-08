"""Tiny shared text helpers (no third-party imports)."""
import re

IDK_MESSAGE = (
    "I don't know the answer to that from Nykaa's policy knowledge base. "
    "I can raise it with a human support agent if you like."
)
INJECTION_MESSAGE = (
    "I can't help with that request. I only answer Nykaa order and policy questions."
)
CARD_SAFETY_MESSAGE = (
    "For your safety: please never share your card number, PIN, CVV or expiry date on public platforms, "
    "websites or chats. Nykaa support will never ask you for them. "
    "This conversation has also been flagged to our backend support engineers for review."
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


# ---- coverage check: does the retrieved text actually mention what the customer asked about? ------
# Words that never name a product or topic, so they are never reported as "not covered".
GENERIC_WORDS = set(
    "get got give take taken make made need want know tell help please still possible able allowed "
    "mean means meaning long many much days day hours hour time times work works working use using used "
    "check status policy order orders item items product products customer support happen happens "
    "covered cover covers apply applies eligible available offer offered raise ticket thing things "
    "way another different bigger smaller new old today now back money amount charge charged "
    "come comes like hair city cities deducted debited received receive stopped".split()
)
# A product word counts as covered when the text names its category.
CATEGORY_ALIASES = {
    "footwear": "shoe shoes sneaker sneakers sandal sandals heel heels boot boots slipper slippers",
    "apparel": "kurta shirt shirts dress dresses jeans top tops tshirt saree jacket skirt",
    "beauty": "lipstick lipsticks serum shampoo moisturiser moisturizer lotion cream makeup perfume "
              "fragrance mascara foundation kajal conditioner sunscreen",
    "electronics": "trimmer trimmers dryer dryers straightener straighteners",
}
_ALIAS = {w: cat for cat, ws in CATEGORY_ALIASES.items() for w in ws.split()}


def uncovered_terms(query: str, context: str):
    """Content words of the query that the retrieved text never mentions (not even by stem or category).
    A non-empty result means the knowledge base does not clearly cover what was asked."""
    ctx = context.lower()
    ctx_words = set(re.findall(r"[a-z0-9]+", ctx))
    out = []
    for t in content_tokens(query):
        if t.isdigit() or len(t) < 4 or t in GENERIC_WORDS:
            continue
        if t in ctx_words or any(w.startswith(t[:5]) or t.startswith(w[:5]) for w in ctx_words if len(w) >= 4):
            continue
        if _ALIAS.get(t) and _ALIAS[t] in ctx_words:
            continue
        out.append(t)
    return out


# Mentions of secret card credentials (a bare "pin code" is a delivery term, so it is NOT matched).
SENSITIVE_CARD_RE = re.compile(r"\b(cvv|cvc|(card|atm|debit|credit|upi) pin|pin number|expiry|expiration|expires)\b", re.I)
