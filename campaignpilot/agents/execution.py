"""Execution Agent: dispatches the approved plan and records outcomes.

Sending is simulated (no real messages leave the system); outcomes come from
``simulator.simulate`` so campaigns can be compared offline.
"""

from __future__ import annotations

import pandas as pd

from ..config import Settings
from ..schemas import ChannelPlan, ExecutionResult, StrategyPlan
from ..simulator import persona_for_segment, simulate


def build_sends(plan: StrategyPlan, channel_plan: ChannelPlan, blocked: list[int] | None = None) -> list[dict]:
    strategies = {s.segment_id: s for s in plan.strategies}
    blocked_set = set(blocked or [])
    sends = []
    for a in channel_plan.allocations:
        if a.segment_id in blocked_set or a.segment_id not in strategies or not a.customer_ids:
            continue
        s = strategies[a.segment_id]
        sends.append({"segment_id": a.segment_id, "segment_name": a.segment_name, "customer_ids": a.customer_ids,
                      "channel": a.channel, "offer": s.offer, "discount_pct": s.discount_pct})
    return sends


def personas_for(rfm: pd.DataFrame, profiles: list[dict], ground_truth: pd.DataFrame | None) -> pd.Series:
    if ground_truth is not None:
        return ground_truth.set_index("customer_id")["persona"]
    names = {p["segment_id"]: persona_for_segment(p["name"]) for p in profiles}
    return rfm.set_index("customer_id")["segment_id"].map(names)


def run_execution_agent(plan: StrategyPlan, channel_plan: ChannelPlan, blocked: list[int],
                        rfm: pd.DataFrame, customers: pd.DataFrame, profiles: list[dict],
                        ground_truth: pd.DataFrame | None, settings: Settings, seed: int = 0) -> ExecutionResult:
    sends = build_sends(plan, channel_plan, blocked)
    result = simulate(sends, customers, rfm, personas_for(rfm, profiles, ground_truth),
                      settings.gross_margin, seed=seed, frequency_cap_days=settings.frequency_cap_days)
    return ExecutionResult.model_validate(result)
