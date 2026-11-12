"""CampaignPilot REST API.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from campaignpilot.config import Settings
from campaignpilot.data import generate_synthetic
from campaignpilot.graph import CampaignContext, CampaignRunner
from campaignpilot.llm import LLMError, make_llm

app = FastAPI(title="CampaignPilot API", version="1.0.0",
              description="Multi-agent campaign orchestration with governance and human approval.")
RUNS: dict[str, CampaignRunner] = {}  # in-memory; swap for a persistent checkpointer in production


class CampaignRequest(BaseModel):
    n_customers: int = Field(2000, ge=200, le=20000)
    seed: int = 7
    budget: float | None = Field(None, gt=0)


class Decision(BaseModel):
    approved: bool
    feedback: str = ""


def public_view(run_id: str, state: dict) -> dict:
    """Drop bulky per-customer id lists from the response."""
    cp = state.get("channel_plan")
    if cp:
        cp = {**cp, "allocations": [{k: v for k, v in a.items() if k != "customer_ids"} for a in cp["allocations"]]}
    return {
        "run_id": run_id,
        "status": "awaiting_approval" if state.get("awaiting_approval") else state.get("status"),
        "error": state.get("error"),
        "segments": state.get("segments"),
        "strategy": state.get("strategy"),
        "channel_plan": cp,
        "governance": state.get("governance"),
        "revisions": state.get("revisions", 0),
        "execution": state.get("execution"),
        "audit": state.get("audit", []),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": Settings().describe_llm()}


@app.post("/campaigns")
def create_campaign(req: CampaignRequest) -> dict:
    settings = Settings()
    if req.budget is not None:
        settings = replace(settings, budget=req.budget)
    try:
        llm = make_llm(settings)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runner = CampaignRunner(CampaignContext(generate_synthetic(req.n_customers, req.seed), settings, llm,
                                            seed=req.seed))
    state = runner.start()
    RUNS[runner.thread_id] = runner
    return public_view(runner.thread_id, state)


@app.get("/campaigns/{run_id}")
def get_campaign(run_id: str) -> dict:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return public_view(run_id, RUNS[run_id].state())


@app.post("/campaigns/{run_id}/decision")
def decide(run_id: str, decision: Decision) -> dict:
    runner = RUNS.get(run_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if not runner.state().get("awaiting_approval"):
        raise HTTPException(status_code=409, detail="This campaign is not awaiting approval")
    return public_view(run_id, runner.resume(decision.approved, decision.feedback))
