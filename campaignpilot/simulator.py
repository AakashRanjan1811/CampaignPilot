"""Outcome simulator: a stand-in for the real market.

Customers respond according to *hidden* persona behaviour that no agent can
see. Response depends on the persona, the offer, the channel engagement of
each customer and contact fatigue. Profit counts gross margin minus the cost of
incentives and messages, so giving discounts to people who would buy anyway is
penalised - exactly the trade-off a real campaign faces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import CHANNEL_COST_INR

BASE_RESPONSE = {"champion": 0.14, "loyal": 0.10, "new": 0.07, "at_risk": 0.06, "hibernating": 0.02, "bargain": 0.05}
DISCOUNT_SENSITIVITY = {"champion": 0.8, "loyal": 1.2, "new": 2.0, "at_risk": 3.0, "hibernating": 3.5, "bargain": 4.0}
OFFER_LIFT = {
    "none": {},
    "free_shipping": {"new": 1.6, "bargain": 1.5, "_": 1.2},
    "loyalty_points": {"loyal": 1.5, "champion": 1.4, "_": 1.1},
    "early_access": {"champion": 1.6, "loyal": 1.3, "_": 1.0},
    "bundle": {"bargain": 1.4, "loyal": 1.2, "_": 1.1},
    "percent_discount": {},
}
FREE_SHIPPING_COST = 60.0
FATIGUE_PENALTY = 0.5  # contacted again within a week -> half as likely to respond

# Used when a dataset has no ground truth (e.g. UCI data): infer persona from the segment name.
SEGMENT_TO_PERSONA = {"Champions": "champion", "Loyal": "loyal", "New": "new", "At Risk": "at_risk",
                      "Hibernating": "hibernating", "Bargain": "bargain"}


def persona_for_segment(name: str) -> str:
    for prefix, persona in SEGMENT_TO_PERSONA.items():
        if name.startswith(prefix):
            return persona
    return "hibernating"


def response_probability(persona: str, offer: str, discount_pct: float, engagement: float,
                         recently_contacted: bool) -> float:
    p = BASE_RESPONSE[persona]
    lifts = OFFER_LIFT[offer]
    p *= lifts.get(persona, lifts.get("_", 1.0))
    if offer == "percent_discount":
        p *= 1 + DISCOUNT_SENSITIVITY[persona] * discount_pct / 100
    p *= 0.4 + 2.0 * engagement
    if recently_contacted:
        p *= FATIGUE_PENALTY
    return float(min(p, 0.95))


def incentive_cost(offer: str, discount_pct: float, order_value: float) -> float:
    if offer == "percent_discount":
        return order_value * discount_pct / 100
    if offer == "free_shipping":
        return FREE_SHIPPING_COST
    if offer == "loyalty_points":
        return order_value * 0.03
    if offer == "bundle":
        return order_value * 0.05
    return 0.0


def simulate(sends: list[dict], customers: pd.DataFrame, rfm: pd.DataFrame,
             personas: pd.Series, margin: float, seed: int = 0, frequency_cap_days: int = 7) -> dict:
    """``sends``: list of {segment_id, segment_name, customer_ids, channel, offer, discount_pct}."""
    rng = np.random.default_rng(seed)
    cust = customers.set_index("customer_id")
    aov = rfm.set_index("customer_id")["avg_order_value"]
    outcomes = []
    for s in sends:
        ids = s["customer_ids"]
        sent = len(ids)
        conv = revenue = inc_cost = 0.0
        if sent:
            rows = cust.loc[ids]
            eng = rows[f"{s['channel']}_engagement"].to_numpy()
            recent = (rows["days_since_last_contact"] < frequency_cap_days).to_numpy()
            probs = np.array([
                response_probability(personas.get(cid, "hibernating"), s["offer"], s["discount_pct"], e, r)
                for cid, e, r in zip(ids, eng, recent)
            ])
            bought = rng.random(sent) < probs
            values = aov.loc[ids].to_numpy()[bought]
            conv = int(bought.sum())
            gross = float(values.sum())
            inc_cost = float(sum(incentive_cost(s["offer"], s["discount_pct"], x) for x in values))
            revenue = gross - (inc_cost if s["offer"] == "percent_discount" else 0.0)
        channel_cost = sent * CHANNEL_COST_INR[s["channel"]]
        gross_value = revenue + (inc_cost if s["offer"] == "percent_discount" else 0.0)
        profit = gross_value * margin - inc_cost - channel_cost
        outcomes.append({
            "segment_id": s["segment_id"], "segment_name": s["segment_name"], "channel": s["channel"],
            "offer": s["offer"] if s["offer"] != "percent_discount" else f"{s['discount_pct']:g}% off",
            "sent": sent, "conversions": int(conv), "revenue": round(revenue, 2),
            "incentive_cost": round(inc_cost, 2), "channel_cost": round(channel_cost, 2),
            "profit": round(profit, 2),
        })
    sent = sum(o["sent"] for o in outcomes)
    conversions = sum(o["conversions"] for o in outcomes)
    channel_cost = sum(o["channel_cost"] for o in outcomes)
    profit = sum(o["profit"] for o in outcomes)
    spend = channel_cost + sum(o["incentive_cost"] for o in outcomes)
    return {
        "outcomes": outcomes, "sent": sent, "conversions": conversions,
        "conversion_rate": round(conversions / sent, 4) if sent else 0.0,
        "revenue": round(sum(o["revenue"] for o in outcomes), 2),
        "channel_cost": round(channel_cost, 2), "profit": round(profit, 2),
        "incentive_cost": round(spend - channel_cost, 2),
        # Return on total campaign spend (messages + incentives).
        "roi": round(profit / spend, 2) if spend else 0.0,
    }
