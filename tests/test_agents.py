from dataclasses import replace

import pytest

from campaignpilot.agents.channel import run_channel_agent
from campaignpilot.agents.governance import check_plan, run_governance_agent
from campaignpilot.agents.strategy import run_strategy_agent
from campaignpilot.catalog import CHANNEL_COST_INR
from campaignpilot.schemas import StrategyPlan
from campaignpilot.simulator import response_probability


def clean_plan(seg, data, llm, settings):
    """Mock plan after it has fixed its deliberate first-draft violation."""
    draft = run_strategy_agent(llm, seg.profiles, settings)
    cp = run_channel_agent(draft, seg.profiles, seg.rfm, data.customers, settings)
    feedback = [v.model_dump() for v in check_plan(draft, cp, seg.profiles, data.customers, settings)]
    return run_strategy_agent(llm, seg.profiles, settings, governance_feedback=feedback)


def rules(violations):
    return {v.rule for v in violations}


# ------------------------------------------------------------------ strategy
def test_mock_strategy_covers_every_segment(seg, llm, settings):
    plan = run_strategy_agent(llm, seg.profiles, settings)
    assert sorted(s.segment_id for s in plan.strategies) == sorted(p["segment_id"] for p in seg.profiles)


def test_first_draft_violates_and_revision_fixes(seg, data, llm, settings):
    draft = run_strategy_agent(llm, seg.profiles, settings)
    cp = run_channel_agent(draft, seg.profiles, seg.rfm, data.customers, settings)
    assert "max_discount" in rules(check_plan(draft, cp, seg.profiles, data.customers, settings))
    fixed = clean_plan(seg, data, llm, settings)
    cp = run_channel_agent(fixed, seg.profiles, seg.rfm, data.customers, settings)
    assert [v for v in check_plan(fixed, cp, seg.profiles, data.customers, settings) if v.severity == "block"] == []


# ------------------------------------------------------------------ channel / budget
@pytest.mark.parametrize("budget", [20.0, 100.0, 250.0, 5000.0])
def test_budget_is_never_exceeded(seg, data, llm, settings, budget):
    s = replace(settings, budget=budget)
    cp = run_channel_agent(clean_plan(seg, data, llm, s), seg.profiles, seg.rfm, data.customers, s)
    assert cp.total_cost <= budget + 1e-9
    for a in cp.allocations:
        assert a.cost == pytest.approx(a.targeted * CHANNEL_COST_INR[a.channel])
        assert len(a.customer_ids) == a.targeted <= a.eligible


def test_only_consented_and_rested_customers_are_targeted(seg, data, llm, settings):
    cp = run_channel_agent(clean_plan(seg, data, llm, settings), seg.profiles, seg.rfm, data.customers, settings)
    lookup = data.customers.set_index("customer_id")
    for a in cp.allocations:
        rows = lookup.loc[a.customer_ids]
        assert rows[f"{a.channel}_opt_in"].all()
        assert (rows["days_since_last_contact"] >= settings.frequency_cap_days).all()


def test_budget_goes_to_highest_value_segments_first(seg, data, llm, settings):
    s = replace(settings, budget=30.0)
    cp = run_channel_agent(clean_plan(seg, data, llm, s), seg.profiles, seg.rfm, data.customers, s)
    funded = [a for a in cp.allocations if a.targeted]
    unfunded = [a for a in cp.allocations if not a.targeted]
    assert funded and unfunded
    ratio = lambda a: a.expected_value_per_contact / CHANNEL_COST_INR[a.channel]  # noqa: E731
    assert min(map(ratio, funded)) >= max(map(ratio, unfunded))


# ------------------------------------------------------------------ governance
def _mutate(plan: StrategyPlan, segment_name: str, profiles, **changes) -> StrategyPlan:
    sid = next(p["segment_id"] for p in profiles if p["name"].startswith(segment_name))
    strategies = [s.model_copy(update=changes) if s.segment_id == sid else s for s in plan.strategies]
    return StrategyPlan(strategies=strategies)


@pytest.mark.parametrize("segment,changes,rule", [
    ("Hibernating", dict(discount_pct=45, message="Take 45% off today!"), "max_discount"),
    ("Champions", dict(offer="percent_discount", discount_pct=20, message="Enjoy 20% off, friend."),
     "vip_max_discount"),
    ("Champions", dict(discount_pct=15), "offer_mismatch"),
    ("Hibernating", dict(message="Come back for 50% off everything we sell!"), "message_offer_mismatch"),
    ("Loyal", dict(message="A guaranteed reward is waiting for you, {{first_name}}."), "banned_phrase"),
    ("Loyal", dict(send_hour=23), "quiet_hours"),
    ("Loyal", dict(send_hour=6), "quiet_hours"),
])
def test_each_governance_rule_fires(seg, data, llm, settings, segment, changes, rule):
    plan = _mutate(clean_plan(seg, data, llm, settings), segment, seg.profiles, **changes)
    cp = run_channel_agent(plan, seg.profiles, seg.rfm, data.customers, settings)
    assert rule in rules(check_plan(plan, cp, seg.profiles, data.customers, settings))


def test_missing_and_unknown_segments_are_blocked(seg, data, llm, settings):
    plan = clean_plan(seg, data, llm, settings)
    bogus = plan.strategies[0].model_copy(update={"segment_id": 999})
    plan = StrategyPlan(strategies=plan.strategies[1:] + [bogus])
    cp = run_channel_agent(plan, seg.profiles, seg.rfm, data.customers, settings)
    found = rules(check_plan(plan, cp, seg.profiles, data.customers, settings))
    assert {"missing_strategy", "unknown_segment"} <= found


def test_sms_length_limit(seg, data, llm, settings):
    plan = clean_plan(seg, data, llm, settings)
    cp = run_channel_agent(plan, seg.profiles, seg.rfm, data.customers, settings)
    alloc = cp.allocations[0]
    alloc.channel = "sms"
    plan = StrategyPlan(strategies=[s.model_copy(update={"message": "x" * 200}) if s.segment_id == alloc.segment_id
                                    else s for s in plan.strategies])
    assert "message_length" in rules(check_plan(plan, cp, seg.profiles, data.customers, settings))


def test_governance_independently_catches_tampered_audience(seg, data, llm, settings):
    plan = clean_plan(seg, data, llm, settings)
    cp = run_channel_agent(plan, seg.profiles, seg.rfm, data.customers, settings)
    alloc = next(a for a in cp.allocations if a.targeted)
    no_consent = data.customers[~data.customers[f"{alloc.channel}_opt_in"]]["customer_id"].iloc[0]
    tired = data.customers[(data.customers["days_since_last_contact"] < settings.frequency_cap_days)
                           & data.customers[f"{alloc.channel}_opt_in"]]["customer_id"].iloc[0]
    alloc.customer_ids += [no_consent, tired]
    found = rules(check_plan(plan, cp, seg.profiles, data.customers, settings))
    assert {"consent", "frequency_cap"} <= found


def test_status_revise_then_final_with_blocking(seg, data, llm, settings):
    draft = run_strategy_agent(llm, seg.profiles, settings)
    cp = run_channel_agent(draft, seg.profiles, seg.rfm, data.customers, settings)
    assert run_governance_agent(draft, cp, seg.profiles, data.customers, settings, revisions=0).status == "revise"
    final = run_governance_agent(draft, cp, seg.profiles, data.customers, settings, revisions=2)
    assert final.status == "final" and final.blocked_segments


# ------------------------------------------------------------------ simulator
def test_response_model_behaves_sensibly():
    assert response_probability("at_risk", "percent_discount", 20, 0.3, False) > \
        response_probability("at_risk", "percent_discount", 0, 0.3, False)
    assert response_probability("loyal", "none", 0, 0.3, True) == pytest.approx(
        0.5 * response_probability("loyal", "none", 0, 0.3, False))
    assert response_probability("champion", "percent_discount", 100, 1.0, False) <= 0.95
