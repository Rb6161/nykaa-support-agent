"""Seeded, deterministic order dataset for the Nykaa support agent.

Design choices (also stated in README.md):
  SEED = 2026
  category weights : Beauty 0.30, Apparel 0.25, Footwear 0.15, Home 0.15, Electronics 0.15
  status weights   : Delivered 0.45, Shipped 0.20, Placed 0.15, Returned 0.12, Refunded 0.08
  order_value_inr  : per-category uniform range, rounded to whole rupees, overall 199-14999
  days_since_created : integer 0-30 (status-aware, see _days_for_status)
  delayed_shipment : Bernoulli, probability depends on status (0.12-0.30)

Run `python dataset.py` to print the validation report.
Records are never hand-edited: if the delayed % falls outside 10-30%, change SEED or weights.
"""
from __future__ import annotations

import random
from collections import Counter

SEED = 2026
N_RECORDS = 240

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]

CATEGORY_WEIGHTS = {"Beauty": 0.30, "Apparel": 0.25, "Footwear": 0.15, "Home": 0.15, "Electronics": 0.15}
STATUS_WEIGHTS = {"Delivered": 0.45, "Shipped": 0.20, "Placed": 0.15, "Returned": 0.12, "Refunded": 0.08}

# Realistic INR price bands per category. Reasoning (one sentence): Nykaa sells mostly beauty
# and fashion items that are priced from a few hundred rupees up to a few thousand, with
# occasional higher-ticket electronics (hair tools, wearables, grooming devices).
VALUE_RANGE_INR = {
    "Beauty": (199, 3499),
    "Apparel": (499, 5999),
    "Footwear": (799, 6999),
    "Home": (399, 7999),
    "Electronics": (999, 14999),
}

# Chance a shipment is delayed, by status (an order only "ships late" once it has moved).
DELAY_PROB = {"Placed": 0.12, "Shipped": 0.30, "Delivered": 0.18, "Returned": 0.20, "Refunded": 0.20}


def _days_for_status(rng: random.Random, status: str) -> int:
    """Newer orders are more likely to still be Placed/Shipped; older ones Delivered/Returned/Refunded."""
    if status == "Placed":
        return rng.randint(0, 6)
    if status == "Shipped":
        return rng.randint(1, 14)
    if status == "Delivered":
        return rng.randint(3, 30)
    if status == "Returned":
        return rng.randint(8, 30)
    return rng.randint(10, 30)  # Refunded


def generate_orders(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    cats, cat_w = zip(*CATEGORY_WEIGHTS.items())
    stats, stat_w = zip(*STATUS_WEIGHTS.items())
    orders = []
    for i in range(1, n + 1):
        category = rng.choices(cats, weights=cat_w)[0]
        status = rng.choices(stats, weights=stat_w)[0]
        lo, hi = VALUE_RANGE_INR[category]
        orders.append(
            {
                "record_id": f"NYK-{i:04d}",
                "category": category,
                "status": status,
                "order_value_inr": int(round(rng.uniform(lo, hi))),
                "days_since_created": _days_for_status(rng, status),
                "delayed_shipment": rng.random() < DELAY_PROB[status],
            }
        )
    return orders


ORDERS: list[dict] = generate_orders()


def validate(orders: list[dict] = None) -> dict:
    orders = orders if orders is not None else ORDERS
    assert len(orders) == N_RECORDS, f"expected {N_RECORDS} records"
    by_cat = Counter(o["category"] for o in orders)
    by_status = Counter(o["status"] for o in orders)
    delayed_pct = 100.0 * sum(o["delayed_shipment"] for o in orders) / len(orders)
    for c in CATEGORIES:
        assert by_cat[c] >= 3, f"category {c} has only {by_cat[c]} records"
    for s in STATUSES:
        assert by_status[s] >= 1, f"status {s} missing"
    assert 10.0 <= delayed_pct <= 30.0, f"delayed % {delayed_pct:.1f} outside 10-30 band; change SEED/weights"
    assert all(0 <= o["days_since_created"] <= 30 for o in orders)
    assert len({o["record_id"] for o in orders}) == len(orders)
    return {"by_category": dict(by_cat), "by_status": dict(by_status), "delayed_pct": delayed_pct}


def get_order(record_id: str) -> dict | None:
    return next((o for o in ORDERS if o["record_id"] == record_id), None)


if __name__ == "__main__":
    report = validate()
    print(f"SEED={SEED}  records={len(ORDERS)}")
    print("Count per category:")
    for c in CATEGORIES:
        print(f"  {c:<12}{report['by_category'][c]}")
    print("Count per status:")
    for s in STATUSES:
        print(f"  {s:<12}{report['by_status'][s]}")
    print(f"delayed_shipment=True: {report['delayed_pct']:.2f}% (required band 10-30%)")
    values = [o["order_value_inr"] for o in ORDERS]
    print(f"order_value_inr range actually generated: {min(values)} - {max(values)}")
    print("Sample records:")
    for o in ORDERS[:3]:
        print(" ", o)
    print("VALIDATION PASSED")
