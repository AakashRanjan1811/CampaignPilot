# CampaignPilot

[![CI](https://github.com/AakashRanjan1811/CampaignPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/AakashRanjan1811/CampaignPilot/actions/workflows/ci.yml)

A multi-agent system for planning marketing campaigns. It segments customers from their purchase history, uses an LLM to suggest an offer and message for each segment, allocates a fixed budget, checks the plan against business rules, and waits for a human to approve it before "sending" (the sending and the results are simulated).

I built this to try out a supervisor-style agent setup in LangGraph where the LLM only makes suggestions and the important checks are done in normal code.

## How it works

```
customer data
  -> segmentation   (RFM features + K-Means, k picked with silhouette score)
  -> strategy       (LLM: objective, offer, discount, channel, message per segment)
  -> channel/budget (picks channel and spends the budget on the best segments first)
  -> governance     (rules: consent, frequency cap, discount limits, budget, banned words)
       if something fails -> back to strategy with the error list (max 2 times)
  -> human approval (approve / reject / reject with feedback)
  -> execution      (simulated send + results + audit log)
```

A supervisor node decides which step runs next based on the current state. If the LLM still breaks a rule after 2 revisions, that segment is just skipped instead of being sent.

The LLM can be Groq (Llama 3), a local Ollama model, or a simple offline mock (default), so the project runs without any API key.

## Results

I compared it with two simple baselines on 10 randomly generated datasets (2,000 customers each, same ₹250 budget). Run `python -m eval.evaluate` to reproduce.

| Method | Conversion rate | Profit (₹) | Rule-breaking messages |
|---|---|---|---|
| 20% discount to everyone by email | 8.5% | 33,651 | 454 |
| Fixed RFM rules (10% / 20% off) | 8.8% | 45,499 | 0 |
| Agents without the governance step | 13.8% | 97,868 | 271 |
| CampaignPilot | 14.2% | 93,873 | 0 |

Removing the governance step gives about 4% more profit but sends 271 messages that break consent, frequency or discount rules.

Some things to keep in mind:
- The data is synthetic. Customers are generated from 6 hidden types, and only the simulator knows which type each customer is.
- The simulator is a simple model of how customers respond, not real campaign data.
- These numbers use the offline mock planner. Results with a real LLM will be different.

## Setup

```bash
git clone https://github.com/AakashRanjan1811/CampaignPilot.git
cd CampaignPilot
python -m venv .venv
.venv\Scripts\activate        # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # on Mac/Linux: cp .env.example .env
streamlit run app.py
```

To use a real model, get a free key from https://console.groq.com/keys and set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env` (or select groq in the sidebar and paste the key). For Ollama, run `ollama pull llama3.2:3b` and set `LLM_PROVIDER=ollama`.

There is also a small REST API:

```bash
uvicorn api.main:app --reload     # docs at http://localhost:8000/docs
```

You can also upload the UCI Online Retail dataset from the sidebar. It has no consent or engagement columns, so those are randomly generated.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

53 tests, all offline. They cover the RFM calculations, each governance rule, the budget and consent checks, the routing, the revision loop, human approval, the API and the Streamlit app.

## Project structure

```
campaignpilot/   segmentation, agents, LLM wrappers, simulator, LangGraph graph
api/             FastAPI app
app.py           Streamlit app
eval/            baseline comparison
tests/           pytest tests
docs/DESIGN.md   notes on the design choices
```

## License

MIT
