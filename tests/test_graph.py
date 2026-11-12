from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import END
from pydantic import BaseModel

from campaignpilot.graph import CampaignContext, CampaignRunner, route
from campaignpilot.llm import GroqLLM, LLMError, MockLLM, _LangChainLLM, make_llm
from campaignpilot.schemas import StrategyPlan


def runner(data, settings, llm, auto=False):
    return CampaignRunner(CampaignContext(data, settings, llm, auto_approve=auto))


# ------------------------------------------------------------------ routing
@pytest.mark.parametrize("state,expected", [
    ({}, "segmentation"),
    ({"segments": [1]}, "strategy"),
    ({"segments": [1], "strategy": {}}, "strategy"),  # empty plan counts as missing
    ({"segments": [1], "strategy": {"x": 1}}, "channel"),
    ({"segments": [1], "strategy": {"x": 1}, "channel_plan": {"x": 1}}, "governance"),
    ({"segments": [1], "strategy": {"x": 1}, "channel_plan": {"x": 1}, "governance": {"status": "revise"}},
     "strategy"),
    ({"segments": [1], "strategy": {"x": 1}, "channel_plan": {"x": 1}, "governance": {"status": "approved"}},
     "approval"),
    ({"segments": [1], "strategy": {"x": 1}, "channel_plan": {"x": 1}, "governance": {"status": "final"},
      "approval": {"approved": True}}, "execution"),
    ({"segments": [1], "strategy": {"x": 1}, "channel_plan": {"x": 1}, "governance": {"status": "approved"},
      "approval": {"approved": False}}, END),
    ({"error": "boom"}, END),
])
def test_supervisor_routing(state, expected):
    assert route(state) == expected


# ------------------------------------------------------------------ full runs
def test_run_pauses_for_human_then_executes(data, settings, llm):
    r = runner(data, settings, llm)
    state = r.start()
    assert state["awaiting_approval"] and state.get("execution") is None
    assert state["revisions"] == 1  # governance caught the mock's 35% draft once
    assert state["governance"]["status"] == "approved"
    state = r.resume(True)
    assert state["status"] == "executed" and not state["awaiting_approval"]
    assert state["execution"]["sent"] == sum(a["targeted"] for a in state["channel_plan"]["allocations"])
    agents = [a["agent"] for a in state["audit"]]
    for agent in ["segmentation", "strategy", "channel", "governance", "human", "execution", "supervisor"]:
        assert agent in agents


def test_reject_without_feedback_sends_nothing(data, settings, llm):
    r = runner(data, settings, llm)
    r.start()
    state = r.resume(False)
    assert state["status"] == "rejected" and state.get("execution") is None


def test_reject_with_feedback_replans_and_passes_feedback(data, settings):
    seen = []

    class Recording(MockLLM):
        def structured(self, system, user, schema):
            seen.append(user)
            return super().structured(system, user, schema)

    r = runner(data, settings, Recording())
    r.start()
    state = r.resume(False, "No discounts for Hibernating this month")
    assert state["awaiting_approval"] and state["human_revisions"] == 1
    assert "No discounts for Hibernating this month" in seen[-1]
    assert r.resume(True)["status"] == "executed"


def test_stubborn_llm_is_blocked_after_max_revisions(data, settings):
    class Stubborn(MockLLM):
        def structured(self, system, user, schema):  # ignores all feedback
            plan = super().structured(system, user.replace('"governance_feedback": [{', '"ignored": [{'), schema)
            return plan

    r = runner(data, replace(settings, max_revisions=2), Stubborn(), auto=True)
    state = r.start()
    gov = state["governance"]
    assert state["revisions"] == 2 and gov["status"] == "final" and gov["blocked_segments"]
    sent_segments = {o["segment_id"] for o in state["execution"]["outcomes"]}
    assert not sent_segments & set(gov["blocked_segments"])


def test_llm_failure_ends_run_cleanly(data, settings):
    class Broken:
        name = "broken"

        def structured(self, *a):
            raise LLMError("rate limited")

    state = runner(data, settings, Broken()).start()
    assert state["status"] == "failed" and "rate limited" in state["error"]
    assert not state["awaiting_approval"]


# ------------------------------------------------------------------ LLM layer
def test_make_llm_validation(settings):
    assert isinstance(make_llm(settings), MockLLM)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqLLM(replace(settings, llm_provider="groq", groq_api_key=""))
    with pytest.raises(LLMError, match="Unknown"):
        make_llm(replace(settings, llm_provider="gpt-9"))


class Out(BaseModel):
    value: int


class FakeRunnable:
    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    def invoke(self, messages):
        self.prompts.append(messages[-1][1])
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM(_LangChainLLM):
    name = "fake"

    def __init__(self, runnable):
        self.runnable = runnable

    def _bind(self, schema):
        return self.runnable


def test_structured_output_retries_with_error_feedback():
    fake = FakeRunnable([{"value": "not-an-int"}, {"value": 3}])
    assert FakeLLM(fake).structured("sys", "user", Out).value == 3
    assert "previous answer was invalid" in fake.prompts[1]


def test_groq_tool_use_failure_is_retried_but_auth_errors_are_not():
    fake = FakeRunnable([RuntimeError("Error 400: tool_use_failed"), Out(value=1)])
    assert FakeLLM(fake).structured("s", "u", Out).value == 1
    with pytest.raises(LLMError, match="request failed"):
        FakeLLM(FakeRunnable([RuntimeError("401 invalid api key")])).structured("s", "u", Out)


def test_mock_plan_matches_schema(seg, settings):
    from campaignpilot.agents.strategy import run_strategy_agent
    assert isinstance(run_strategy_agent(MockLLM(), seg.profiles, settings), StrategyPlan)


# ------------------------------------------------------------------ API
def test_api_lifecycle(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from api.main import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    run = client.post("/campaigns", json={"n_customers": 600, "seed": 2}).json()
    assert run["status"] == "awaiting_approval"
    assert all("customer_ids" not in a for a in run["channel_plan"]["allocations"])
    done = client.post(f"/campaigns/{run['run_id']}/decision", json={"approved": True}).json()
    assert done["status"] == "executed" and done["execution"]["sent"] > 0
    assert client.post(f"/campaigns/{run['run_id']}/decision", json={"approved": True}).status_code == 409
    assert client.get("/campaigns/unknown").status_code == 404
    assert client.post("/campaigns", json={"n_customers": 5}).status_code == 422
