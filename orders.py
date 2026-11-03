"""Order lookup + designed escalation score (no CrewAI imports, so it is unit-testable).

ESCALATION SCORE (in [0, 1]):

    escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)

  * delayed_shipment is 0/1. A delayed shipment is the strongest escalation signal (weight 0.6).
  * days_since_created / 30 is a normalised "how long has this order been open" signal in [0, 1]
    (weight 0.4). Older orders that are still unresolved are more urgent than brand-new ones.
  * Max possible score = 1.0 (delayed and 30 days old); min = 0.0.

THRESHOLD: ESCALATION_THRESHOLD is the 80th percentile of escalation_score over the 240
generated orders, so roughly the top 20% most urgent orders are flagged for escalation.
It is computed from the data in code (see describe_threshold()), not hard-coded.
"""
from __future__ import annotations

from dataset import ORDERS, get_order

W_DELAY = 0.6
W_AGE = 0.4
MAX_DAYS = 30
PERCENTILE = 80


def escalation_score(order: dict) -> float:
    score = W_DELAY * (1.0 if order["delayed_shipment"] else 0.0) + W_AGE * (
        min(max(order["days_since_created"], 0), MAX_DAYS) / MAX_DAYS
    )
    return round(score, 4)


def percentile(values, p: float) -> float:
    """Linear-interpolation percentile (same as numpy's default)."""
    xs = sorted(values)
    if not xs:
        raise ValueError("empty")
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


SCORES = [escalation_score(o) for o in ORDERS]
ESCALATION_THRESHOLD = round(percentile(SCORES, PERCENTILE), 4)


def describe_threshold() -> dict:
    days = [o["days_since_created"] for o in ORDERS]
    flagged = sum(s >= ESCALATION_THRESHOLD for s in SCORES)
    return {
        "formula": "0.6*delayed_shipment + 0.4*(days_since_created/30)",
        "threshold": ESCALATION_THRESHOLD,
        "threshold_percentile_of_scores": PERCENTILE,
        "p80_days_since_created": round(percentile(days, 80), 2),
        "flagged_orders": flagged,
        "flagged_pct": round(100.0 * flagged / len(ORDERS), 2),
        "delayed_orders_flagged": sum(
            1 for o, s in zip(ORDERS, SCORES) if o["delayed_shipment"] and s >= ESCALATION_THRESHOLD
        ),
        "delayed_orders_total": sum(o["delayed_shipment"] for o in ORDERS),
        "max_score_if_not_delayed": W_AGE,
    }


def check_order_status(record_id: str) -> dict:
    """Look up one order. Returns status, order_value_inr and escalation_score (+ extras)."""
    rid = (record_id or "").strip().upper()
    order = get_order(rid)
    if order is None:
        return {"record_id": rid, "found": False, "error": f"No order found with id {rid}."}
    score = escalation_score(order)
    return {
        "record_id": rid,
        "found": True,
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": score,
        "escalation_recommended": score >= ESCALATION_THRESHOLD,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "category": order["category"],
        "delayed_shipment": order["delayed_shipment"],
        "days_since_created": order["days_since_created"],
    }


def order_sentence(info: dict) -> str:
    """One human-readable sentence about an order. The SAME sentence is added to the review
    context, so the reviewer treats it as grounded."""
    verdict = "escalation is recommended" if info["escalation_recommended"] else "no escalation is needed right now"
    return (
        f"Order {info['record_id']} is currently {info['status']}, with an order value of INR "
        f"{info['order_value_inr']}; its escalation score is {info['escalation_score']:.2f}, so {verdict}."
    )


if __name__ == "__main__":
    import json

    print(json.dumps(describe_threshold(), indent=2))
    print(json.dumps(check_order_status("NYK-0001"), indent=2))
    print(json.dumps(check_order_status("NYK-9999"), indent=2))
