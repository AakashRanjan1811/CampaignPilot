"""Governance Agent (deterministic rules engine).

LLMs propose; rules decide. Every check here is plain Python so decisions are
reproducible, explainable and auditable. Consent and frequency caps are
re-verified customer by customer (defence in depth: the channel agent already
filters, governance does not trust it).
"""

from __future__ import annotations

import re

import pandas as pd

from ..catalog import BANNED_PHRASES, PUSH_MAX_CHARS, SMS_MAX_CHARS, VIP_SEGMENT_PREFIXES
from ..config import Settings
from ..schemas import ChannelPlan, GovernanceReport, StrategyPlan, Violation

CHECKS = [
    "coverage: exactly one strategy per segment, no unknown segments",
    "offer consistency: discount only with percent_discount",
    "max discount and VIP margin protection",
    "message states the same discount as the plan",
    "banned / misleading phrases",
    "SMS and push length limits",
    "quiet hours",
    "consent for every targeted customer",
    "frequency cap for every targeted customer",
    "total spend within budget",
    "channel reach (warning)",
]


def _is_vip(name: str) -> bool:
    return name.startswith(VIP_SEGMENT_PREFIXES)


def check_plan(plan: StrategyPlan, channel_plan: ChannelPlan, profiles: list[dict],
               customers: pd.DataFrame, settings: Settings) -> list[Violation]:
    v: list[Violation] = []
    names = {p["segment_id"]: p["name"] for p in profiles}
    sizes = {p["segment_id"]: p["size"] for p in profiles}

    seen: dict[int, int] = {}
    for s in plan.strategies:
        seen[s.segment_id] = seen.get(s.segment_id, 0) + 1
    for sid in names:
        if sid not in seen:
            v.append(Violation(segment_id=sid, rule="missing_strategy", severity="block",
                               detail=f"No strategy for segment {sid} ({names[sid]})."))
    for sid, count in seen.items():
        if sid not in names:
            v.append(Violation(segment_id=None, rule="unknown_segment", severity="block",
                               detail=f"Strategy references unknown segment_id {sid}."))
        elif count > 1:
            v.append(Violation(segment_id=sid, rule="duplicate_strategy", severity="block",
                               detail=f"{count} strategies for segment {sid}."))

    allocations = {a.segment_id: a for a in channel_plan.allocations}
    for s in plan.strategies:
        if s.segment_id not in names:
            continue
        name = names[s.segment_id]
        if s.offer != "percent_discount" and s.discount_pct > 0:
            v.append(Violation(segment_id=s.segment_id, rule="offer_mismatch", severity="block",
                               detail=f"discount_pct={s.discount_pct:g} but offer is {s.offer}."))
        if s.discount_pct > settings.max_discount_pct:
            v.append(Violation(segment_id=s.segment_id, rule="max_discount", severity="block",
                               detail=f"{s.discount_pct:g}% exceeds the {settings.max_discount_pct:g}% limit."))
        elif _is_vip(name) and s.discount_pct > settings.vip_max_discount_pct:
            v.append(Violation(segment_id=s.segment_id, rule="vip_max_discount", severity="block",
                               detail=f"{name} is VIP: {s.discount_pct:g}% exceeds the "
                                      f"{settings.vip_max_discount_pct:g}% margin-protection limit."))
        stated = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", s.message)}
        if stated and stated != {float(s.discount_pct)}:
            v.append(Violation(segment_id=s.segment_id, rule="message_offer_mismatch", severity="block",
                               detail=f"Message mentions {sorted(stated)}% but plan discount is {s.discount_pct:g}%."))
        lowered = s.message.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                v.append(Violation(segment_id=s.segment_id, rule="banned_phrase", severity="block",
                                   detail=f"Message contains banned phrase '{phrase}'."))
        if s.send_hour >= settings.quiet_hours_start or s.send_hour < settings.quiet_hours_end:
            v.append(Violation(segment_id=s.segment_id, rule="quiet_hours", severity="block",
                               detail=f"send_hour {s.send_hour}:00 falls in quiet hours."))
        alloc = allocations.get(s.segment_id)
        if alloc:
            limit = {"sms": SMS_MAX_CHARS, "push": PUSH_MAX_CHARS}.get(alloc.channel)
            if limit and len(s.message) > limit:
                v.append(Violation(segment_id=s.segment_id, rule="message_length", severity="block",
                                   detail=f"{alloc.channel} message is {len(s.message)} chars (limit {limit})."))
            if alloc.eligible < settings.min_channel_coverage * sizes[s.segment_id]:
                v.append(Violation(segment_id=s.segment_id, rule="low_channel_coverage", severity="warn",
                                   detail=f"{alloc.channel} reaches only {alloc.eligible}/{sizes[s.segment_id]} customers."))

    lookup = customers.set_index("customer_id")
    for alloc in channel_plan.allocations:
        if not alloc.customer_ids:
            continue
        rows = lookup.loc[alloc.customer_ids]
        no_consent = int((~rows[f"{alloc.channel}_opt_in"]).sum())
        if no_consent:
            v.append(Violation(segment_id=alloc.segment_id, rule="consent", severity="block",
                               detail=f"{no_consent} targeted customers have not opted in to {alloc.channel}."))
        too_soon = int((rows["days_since_last_contact"] < settings.frequency_cap_days).sum())
        if too_soon:
            v.append(Violation(segment_id=alloc.segment_id, rule="frequency_cap", severity="block",
                               detail=f"{too_soon} customers were contacted in the last "
                                      f"{settings.frequency_cap_days} days."))
    if channel_plan.total_cost > settings.budget + 1e-6:
        v.append(Violation(segment_id=None, rule="budget", severity="block",
                           detail=f"Spend {channel_plan.total_cost:.2f} exceeds budget {settings.budget:.2f}."))
    return v


def run_governance_agent(plan: StrategyPlan, channel_plan: ChannelPlan, profiles: list[dict],
                         customers: pd.DataFrame, settings: Settings, revisions: int) -> GovernanceReport:
    violations = check_plan(plan, channel_plan, profiles, customers, settings)
    blocking = [x for x in violations if x.severity == "block"]
    if not blocking:
        return GovernanceReport(status="approved", violations=violations, checks_run=CHECKS)
    if revisions < settings.max_revisions:
        return GovernanceReport(status="revise", violations=violations, checks_run=CHECKS)
    # Out of revisions: fail safe by blocking every segment that still violates a rule.
    blocked = sorted({x.segment_id for x in blocking if x.segment_id is not None})
    return GovernanceReport(status="final", violations=violations, blocked_segments=blocked, checks_run=CHECKS)
