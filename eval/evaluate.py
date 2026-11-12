"""Compare CampaignPilot against simple baselines on simulated campaigns.

Every method gets the same customers, the same budget and the same outcome
simulator. Results are averaged over several random datasets.

    python -m eval.evaluate --seeds 10            # offline, mock strategist
    LLM_PROVIDER=groq python -m eval.evaluate     # with a real LLM strategist

Methods
- blanket_20pct_email : 20% off by email to everyone who opted in to email,
                        random order until the budget runs out (no frequency cap).
- static_rfm_rules    : same segments; 10% off for recent segments, 20% for
                        lapsed ones, email, consent + frequency cap respected.
- agents_no_governance: CampaignPilot's first strategy draft executed directly,
                        without the governance agent or frequency cap (ablation).
- campaignpilot       : full multi-agent pipeline (auto-approved).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from campaignpilot.agents.channel import run_channel_agent
from campaignpilot.agents.execution import build_sends, personas_for
from campaignpilot.agents.strategy import run_strategy_agent
from campaignpilot.catalog import CHANNEL_COST_INR, VIP_SEGMENT_PREFIXES
from campaignpilot.config import Settings
from campaignpilot.data import generate_synthetic
from campaignpilot.graph import CampaignContext, CampaignRunner
from campaignpilot.llm import make_llm
from campaignpilot.schemas import ChannelPlan, StrategyPlan
from campaignpilot.segmentation import segment_customers
from campaignpilot.simulator import simulate

RESULTS_DIR = Path(__file__).parent / "results"


def policy_breaches(sends: list[dict], customers: pd.DataFrame, settings: Settings) -> int:
    """Customers contacted in breach of policy (no consent, frequency cap, over-limit discount)."""
    cust = customers.set_index("customer_id")
    breaches = 0
    for s in sends:
        if not s["customer_ids"]:
            continue
        rows = cust.loc[s["customer_ids"]]
        bad = ~rows[f"{s['channel']}_opt_in"] | (rows["days_since_last_contact"] < settings.frequency_cap_days)
        vip = s["segment_name"].startswith(VIP_SEGMENT_PREFIXES)
        limit = settings.vip_max_discount_pct if vip else settings.max_discount_pct
        if s["discount_pct"] > limit:
            bad[:] = True
        breaches += int(bad.sum())
    return breaches


def fill_budget(ids: list[str], channel: str, budget: float) -> list[str]:
    return ids[: int(budget // CHANNEL_COST_INR[channel])]


def run_seed(seed: int, settings: Settings, llm) -> dict[str, dict]:
    data = generate_synthetic(seed=seed)
    seg = segment_customers(data, settings.k_min, settings.k_max, settings.random_state)
    rfm, profiles, customers = seg.rfm, seg.profiles, data.customers
    personas = personas_for(rfm, profiles, data.ground_truth)
    names = {p["segment_id"]: p["name"] for p in profiles}
    rng = np.random.default_rng(seed)
    sim = lambda sends: simulate(sends, customers, rfm, personas, settings.gross_margin, seed=seed,  # noqa: E731
                                 frequency_cap_days=settings.frequency_cap_days)
    results = {}

    # 1. Blanket discount
    pool = customers[customers["email_opt_in"]].merge(rfm[["customer_id", "segment_id"]], on="customer_id")
    pool = pool.sample(frac=1, random_state=seed)
    chosen = set(fill_budget(pool["customer_id"].tolist(), "email", settings.budget))
    sends = [{"segment_id": sid, "segment_name": names[sid], "channel": "email", "offer": "percent_discount",
              "discount_pct": 20.0, "customer_ids": [c for c in g["customer_id"] if c in chosen]}
             for sid, g in pool.groupby("segment_id")]
    results["blanket_20pct_email"] = {**sim(sends), "breaches": policy_breaches(sends, customers, settings)}

    # 2. Static RFM rules
    median_recency = rfm["recency"].median()
    members = rfm.merge(customers, on="customer_id")
    eligible = members[members["email_opt_in"] & (members["days_since_last_contact"] >= settings.frequency_cap_days)]
    eligible = eligible.sample(frac=1, random_state=int(rng.integers(1e9)))
    chosen = set(fill_budget(eligible["customer_id"].tolist(), "email", settings.budget))
    sends = []
    for p in profiles:
        ids = [c for c in eligible[eligible["segment_id"] == p["segment_id"]]["customer_id"] if c in chosen]
        disc = 10.0 if p["median_recency_days"] <= median_recency else 20.0
        sends.append({"segment_id": p["segment_id"], "segment_name": p["name"], "channel": "email",
                      "offer": "percent_discount", "discount_pct": disc, "customer_ids": ids})
    results["static_rfm_rules"] = {**sim(sends), "breaches": policy_breaches(sends, customers, settings)}

    # 3. Ablation: first LLM draft, no governance, no frequency cap
    draft = run_strategy_agent(llm, profiles, settings)
    cp = run_channel_agent(draft, profiles, rfm, customers, replace(settings, frequency_cap_days=0))
    sends = build_sends(draft, cp)
    results["agents_no_governance"] = {**sim(sends), "breaches": policy_breaches(sends, customers, settings)}

    # 4. Full pipeline
    ctx = CampaignContext(data, settings, llm, auto_approve=True, seed=seed)
    state = CampaignRunner(ctx).start()
    final_sends = build_sends(StrategyPlan.model_validate(state["strategy"]),
                              ChannelPlan.model_validate(state["channel_plan"]),
                              state["governance"]["blocked_segments"])
    results["campaignpilot"] = {**state["execution"], "breaches": policy_breaches(final_sends, customers, settings),
                                "revisions": state["revisions"]}
    return results


def summarise(per_seed: list[dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for method in per_seed[0]:
        vals = pd.DataFrame([r[method] for r in per_seed])
        rows.append({
            "method": method,
            "conversions": f"{vals['conversions'].mean():.1f}",
            "conversion_rate": f"{100 * vals['conversion_rate'].mean():.1f}% ± {100 * vals['conversion_rate'].std():.1f}",
            "profit_inr": f"{vals['profit'].mean():,.0f} ± {vals['profit'].std():,.0f}",
            "roi_on_spend": f"{vals['roi'].mean():.2f}x",
            "policy_breaching_contacts": f"{vals['breaches'].mean():.1f}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    settings = Settings()
    llm = make_llm(settings)
    per_seed = [run_seed(seed, settings, llm) for seed in range(args.seeds)]
    table = summarise(per_seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    header = (f"Strategist: {settings.describe_llm()} | seeds: {args.seeds} | budget: INR {settings.budget:g} "
              f"| 2,000 customers per seed\n\n")
    (RESULTS_DIR / "results.md").write_text(header + table.to_markdown(index=False) + "\n")
    (RESULTS_DIR / "per_seed.json").write_text(json.dumps(per_seed, indent=1, default=str))
    print(header + table.to_markdown(index=False))


if __name__ == "__main__":
    main()
