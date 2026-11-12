"""Channel & Budget Agent (deterministic optimiser).

For every segment it
1. builds the *eligible* audience per channel (consent + frequency cap),
2. estimates expected value per contact for each channel,
3. keeps the strategist's preferred channel unless another channel is clearly
   better (>25% higher expected value) or the preferred one reaches too few
   customers,
4. allocates the budget greedily to the segments with the highest expected
   value per rupee, funding the most engaged customers first.

The propensity estimate here is a transparent heuristic computed only from
observable data (RFM + engagement); it never sees the simulator's ground truth.
"""

from __future__ import annotations

import math

import pandas as pd

from ..catalog import CHANNEL_COST_INR
from ..config import Settings
from ..schemas import ChannelPlan, SegmentAllocation, StrategyPlan

SWITCH_THRESHOLD = 1.25


def estimated_base_propensity(profile: dict) -> float:
    """Chance a contacted customer of this segment buys, before channel effects."""
    orders_per_week = profile["median_orders"] / 52
    recency_decay = math.exp(-profile["median_recency_days"] / 120)
    return min(0.30, 0.01 + 0.6 * orders_per_week * recency_decay + 0.03 * recency_decay)


def offer_cost_fraction(offer: str, discount_pct: float) -> float:
    return {"percent_discount": discount_pct / 100, "loyalty_points": 0.03, "bundle": 0.05}.get(offer, 0.0)


def expected_value(profile: dict, channel: str, offer: str, discount_pct: float, margin: float) -> float:
    engagement = profile["channels"][channel]["avg_engagement_opted_in"]
    p = estimated_base_propensity(profile) * (0.4 + 2.0 * engagement)
    p *= 1 + 1.5 * discount_pct / 100  # incentives raise response...
    value = profile["avg_order_value_inr"] * (margin - offer_cost_fraction(offer, discount_pct))  # ...but cost margin
    if offer == "free_shipping":
        value -= 60
    return p * value - CHANNEL_COST_INR[channel]


def eligible_audience(members: pd.DataFrame, channel: str, cap_days: int) -> tuple[pd.DataFrame, int, int]:
    consented = members[members[f"{channel}_opt_in"]]
    no_consent = len(members) - len(consented)
    rested = consented[consented["days_since_last_contact"] >= cap_days]
    return rested, no_consent, len(consented) - len(rested)


def run_channel_agent(plan: StrategyPlan, profiles: list[dict], rfm: pd.DataFrame,
                      customers: pd.DataFrame, settings: Settings,
                      blocked_segments: list[int] | None = None) -> ChannelPlan:
    blocked = set(blocked_segments or [])
    by_id = {p["segment_id"]: p for p in profiles}
    members_all = rfm[["customer_id", "segment_id"]].merge(customers, on="customer_id")
    candidates = []
    for strat in plan.strategies:
        if strat.segment_id not in by_id or strat.segment_id in blocked:
            continue
        profile = by_id[strat.segment_id]
        members = members_all[members_all["segment_id"] == strat.segment_id]
        options = {}
        for ch in CHANNEL_COST_INR:
            audience, no_consent, capped = eligible_audience(members, ch, settings.frequency_cap_days)
            coverage = len(audience) / max(len(members), 1)
            ev = expected_value(profile, ch, strat.offer, strat.discount_pct, settings.gross_margin)
            options[ch] = (ev, coverage, audience, no_consent, capped)

        usable = {ch: o for ch, o in options.items() if o[1] >= settings.min_channel_coverage}
        best = max(usable or options, key=lambda ch: (usable or options)[ch][0])
        pref = strat.preferred_channel
        if pref in usable and options[pref][0] * SWITCH_THRESHOLD >= options[best][0]:
            channel, reason = pref, f"kept strategist's choice ({pref})"
        else:
            why = "low reach" if pref not in usable else "lower expected value"
            channel, reason = best, f"switched {pref} -> {best} ({why})"
        ev, coverage, audience, no_consent, capped = options[channel]
        audience = audience.sort_values(f"{channel}_engagement", ascending=False)
        candidates.append((ev / CHANNEL_COST_INR[channel], strat, profile, channel, ev, audience,
                           no_consent, capped, reason))

    remaining = settings.budget
    allocations = []
    for ratio, strat, profile, channel, ev, audience, no_consent, capped, reason in sorted(
            candidates, key=lambda c: c[0], reverse=True):
        unit = CHANNEL_COST_INR[channel]
        if ev <= 0:
            n, note = 0, "; not funded (expected value <= 0)"
        else:
            n = min(len(audience), int(remaining // unit + 1e-9))
            note = "" if n == len(audience) else f"; budget-limited to {n}/{len(audience)}"
        cost = round(n * unit, 2)
        remaining = round(remaining - cost, 2)
        allocations.append(SegmentAllocation(
            segment_id=strat.segment_id, segment_name=profile["name"], channel=channel,
            eligible=len(audience), targeted=n, cost=cost, expected_value_per_contact=round(ev, 2),
            customer_ids=audience["customer_id"].head(n).tolist(), excluded_no_consent=no_consent,
            excluded_frequency_cap=capped, reason=reason + note,
        ))
    allocations.sort(key=lambda a: a.segment_id)
    return ChannelPlan(allocations=allocations, total_cost=round(sum(a.cost for a in allocations), 2),
                       budget=settings.budget)
