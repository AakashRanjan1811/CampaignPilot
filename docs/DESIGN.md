# Design decisions & trade-offs

This document explains *why* CampaignPilot is built the way it is. Each section is a question an interviewer is likely to ask.

## 1. Why is segmentation not done by the LLM?

Segmentation is a numerical problem over thousands of rows. K-Means on log-scaled, standardised RFM features is fast, reproducible and measurable (silhouette score), and on the synthetic data it recovers the hidden personas with >90% purity (tested). An LLM would be slower, costlier, non-deterministic and can't see all rows anyway. The LLM is used where it adds value: reasoning over **segment profiles** to choose offers and write copy.

**Why log1p + StandardScaler?** Frequency and spend are heavily right-skewed; without the log, a few big spenders dominate the distance metric. Scaling puts R, F and M on equal footing.

**Why silhouette for k?** It balances cohesion and separation without needing labels. k is searched from 3 to 7 to keep segments actionable for marketers.

## 2. Why a deterministic supervisor instead of an LLM router?

The workflow has a known order and known loop conditions. A pure-Python `route(state)` function is predictable, free, and unit-testable (see `tests/test_graph.py`). An LLM router adds latency and a failure mode without adding capability here. LangGraph still gives us the important parts: shared state, conditional edges, cycles, checkpointing and interrupts.

## 3. Why is governance a rules engine and not an "LLM-as-judge"?

Compliance (consent, frequency caps, discount limits, budget) must be **exact, explainable and auditable**. A rules engine gives the same answer every time and says exactly which rule failed. The governance agent also **re-verifies every targeted customer** instead of trusting the channel agent (defence in depth) — a test tampers with the audience to prove it.

## 4. What happens when the LLM keeps breaking the rules?

Violations are sent back to the Strategy Agent as structured feedback (`rule`, `segment_id`, `detail`). After `MAX_REVISIONS` (default 2), any segment that still violates a blocking rule is **blocked** and excluded from execution. The system fails safe: a misbehaving model can reduce reach, never break policy. Tested with a "stubborn" LLM that ignores feedback.

## 5. How is LLM output kept reliable?

- Output is bound to Pydantic schemas (`Literal` types for objective/offer/channel, numeric bounds for discount and send hour).
- Invalid output triggers a retry that includes the validation error; Groq's `tool_use_failed` errors are retried; auth/network errors fail fast.
- The strategist must use the `segment_id`s it was given; unknown or missing IDs are caught by governance.
- If the message text mentions a discount, it must equal `discount_pct` (catches "35% off" copy on a 30% plan).

## 6. Prompt injection?

Segment profiles are aggregates (no raw customer text reaches the LLM). The system prompt still marks input as untrusted, and — more importantly — nothing the LLM says can bypass governance, because every decision that matters is re-checked deterministically.

## 7. How is the budget allocated?

For each segment and channel the Channel Agent estimates expected value per contact:

`EV = p̂ × AOV × (margin − incentive cost) − message cost`, where `p̂` is a transparent propensity heuristic from recency, frequency and channel engagement.

It keeps the strategist's channel unless another channel is >25% better or the preferred channel reaches <15% of the segment, then funds segments in order of EV per rupee, most-engaged customers first. This is a greedy (value-per-cost) knapsack heuristic: simple, fast and easy to explain, though not guaranteed optimal because customers within a segment differ. A test asserts every funded segment has a higher EV/₹ than every unfunded one.

## 8. Why simulate outcomes?

Real campaign outcomes need real customers. The simulator stands in for the market: response depends on hidden persona, offer, channel engagement and contact fatigue, and profit subtracts incentive and message costs. Agents never see the hidden personas, so the evaluation can't be "cheated". The limitation is stated in the README: in production you would measure uplift with randomised holdout groups.

## 9. What would change in production?

- Persistent checkpointer (Postgres/SQLite) instead of in-memory, so approvals survive restarts.
- Real channel integrations (SES, Twilio, FCM) behind the Execution Agent, with idempotent sends.
- Uplift modelling (two-model or causal forest) to replace the propensity heuristic.
- A/B holdouts for every campaign and drift monitoring on segment composition.
- Auth + role-based approval (marketer drafts, manager approves).
- LangSmith or OpenTelemetry tracing for every LLM call.

## 10. Key numbers to remember

- 6 agents/steps, 1 supervisor, 11 governance checks, 53 tests.
- K-Means k chosen by silhouette; >90% persona purity.
- ~2.1× profit vs static RFM rules, ~2.8× vs blanket discount, 0 policy breaches (10 seeds, mock strategist).
- Governance costs ~4% profit vs the no-governance ablation, which sends ~271 policy-breaching messages.
