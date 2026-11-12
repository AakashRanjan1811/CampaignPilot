"""Typed contracts between agents. LLM output is validated against these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Objective = Literal["retain", "upsell", "nurture", "win_back", "reactivate"]
Offer = Literal["none", "percent_discount", "free_shipping", "loyalty_points", "early_access", "bundle"]
Channel = Literal["email", "sms", "push", "whatsapp"]


class SegmentStrategy(BaseModel):
    segment_id: int = Field(description="segment_id copied exactly from the segment profile")
    objective: Objective = Field(description="Business objective for this segment")
    offer: Offer = Field(description="Incentive type from the allowed catalog")
    discount_pct: float = Field(ge=0, le=100,
                                description="Discount percentage; must be 0 unless offer is percent_discount")
    preferred_channel: Channel = Field(description="Channel the strategist recommends")
    send_hour: int = Field(ge=0, le=23, description="Local hour (0-23) to send the message")
    message: str = Field(min_length=10, max_length=400, description="Customer-facing message text")
    rationale: str = Field(min_length=5, max_length=400,
                           description="One or two sentences explaining the choice, citing segment data")


class StrategyPlan(BaseModel):
    strategies: list[SegmentStrategy] = Field(description="Exactly one strategy per segment")


class SegmentAllocation(BaseModel):
    segment_id: int
    segment_name: str
    channel: Channel
    eligible: int  # customers reachable after consent + frequency-cap filtering
    targeted: int  # customers actually funded within budget
    cost: float
    expected_value_per_contact: float
    customer_ids: list[str]
    excluded_no_consent: int
    excluded_frequency_cap: int
    reason: str


class ChannelPlan(BaseModel):
    allocations: list[SegmentAllocation]
    total_cost: float
    budget: float


class Violation(BaseModel):
    segment_id: int | None
    rule: str
    severity: Literal["block", "warn"]
    detail: str


class GovernanceReport(BaseModel):
    status: Literal["approved", "revise", "final"]
    violations: list[Violation]
    blocked_segments: list[int] = Field(default_factory=list)
    checks_run: list[str]


class SegmentOutcome(BaseModel):
    segment_id: int
    segment_name: str
    channel: Channel
    offer: str
    sent: int
    conversions: int
    revenue: float
    incentive_cost: float
    channel_cost: float
    profit: float


class ExecutionResult(BaseModel):
    outcomes: list[SegmentOutcome]
    sent: int
    conversions: int
    conversion_rate: float
    revenue: float
    channel_cost: float
    incentive_cost: float
    profit: float
    roi: float  # profit / (channel_cost + incentive_cost)
