"""LangGraph orchestration: a supervisor routes work between specialist agents.

    START -> supervisor -> segmentation -> supervisor -> strategy -> supervisor
          -> channel -> supervisor -> governance -> supervisor
               (governance says "revise" -> strategy again, at most N times)
          -> approval (human-in-the-loop interrupt) -> supervisor
               (rejected with feedback -> strategy again; rejected -> END)
          -> execution -> supervisor -> END

The supervisor is deterministic: it reads the shared state and picks the next
agent. Large tables (customers, transactions) live in a ``CampaignContext``
outside the checkpointed state; the state holds JSON-serialisable results.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agents.channel import run_channel_agent
from .agents.execution import run_execution_agent
from .agents.governance import run_governance_agent
from .agents.strategy import run_strategy_agent
from .config import Settings
from .data import Dataset
from .llm import LLMClient, LLMError
from .schemas import ChannelPlan, StrategyPlan
from .segmentation import segment_customers


class CampaignState(TypedDict, total=False):
    segments: list[dict]
    k: int
    silhouette_by_k: dict
    strategy: dict | None
    channel_plan: dict | None
    governance: dict | None
    governance_feedback: list[dict]
    human_feedback: str
    revisions: int
    human_revisions: int
    approval: dict | None
    execution: dict | None
    error: str | None
    status: str
    next: str
    audit: Annotated[list[dict], operator.add]


@dataclass
class CampaignContext:
    data: Dataset
    settings: Settings
    llm: LLMClient
    auto_approve: bool = False
    seed: int = 0
    rfm: pd.DataFrame | None = field(default=None, repr=False)


def _audit(agent: str, event: str, detail: str = "") -> list[dict]:
    return [{"at": datetime.now(UTC).isoformat(timespec="seconds"), "agent": agent,
             "event": event, "detail": detail}]


def route(state: CampaignState) -> str:
    """Pure routing function used by the supervisor (unit-tested on its own)."""
    if state.get("error"):
        return END
    if not state.get("segments"):
        return "segmentation"
    if not state.get("strategy"):
        return "strategy"
    if not state.get("channel_plan"):
        return "channel"
    gov = state.get("governance")
    if not gov:
        return "governance"
    if gov["status"] == "revise":
        return "strategy"
    approval = state.get("approval")
    if approval is None:
        return "approval"
    if not approval.get("approved"):
        return END
    if not state.get("execution"):
        return "execution"
    return END


def build_graph(ctx: CampaignContext, checkpointer=None):
    s = ctx.settings

    def supervisor(state: CampaignState) -> dict:
        nxt = route(state)
        update: dict[str, Any] = {"next": nxt}
        if nxt == END:
            if state.get("error"):
                update["status"] = "failed"
            elif state.get("execution"):
                update["status"] = "executed"
            elif state.get("approval") and not state["approval"].get("approved"):
                update["status"] = "rejected"
            update["audit"] = _audit("supervisor", "finished", update.get("status", ""))
        return update

    def segmentation(state: CampaignState) -> dict:
        result = segment_customers(ctx.data, s.k_min, s.k_max, s.random_state)
        ctx.rfm = result.rfm
        names = ", ".join(f"{p['name']} ({p['size']})" for p in result.profiles)
        return {"segments": result.profiles, "k": result.k, "silhouette_by_k": result.silhouette_by_k,
                "audit": _audit("segmentation", f"k={result.k} chosen by silhouette", names)}

    def strategy(state: CampaignState) -> dict:
        gov = state.get("governance")
        is_revision = bool(gov and gov["status"] == "revise")
        feedback = state.get("governance_feedback") if is_revision else None
        try:
            plan = run_strategy_agent(ctx.llm, state["segments"], s, feedback, state.get("human_feedback"))
        except LLMError as exc:
            return {"error": str(exc), "audit": _audit("strategy", "LLM error", str(exc))}
        revisions = state.get("revisions", 0) + (1 if is_revision else 0)
        event = f"revision {revisions} drafted" if is_revision else "initial plan drafted"
        return {"strategy": plan.model_dump(), "channel_plan": None, "governance": None, "approval": None,
                "revisions": revisions,
                "audit": _audit("strategy", event, f"{len(plan.strategies)} strategies via {ctx.llm.name}")}

    def channel(state: CampaignState) -> dict:
        plan = StrategyPlan.model_validate(state["strategy"])
        cp = run_channel_agent(plan, state["segments"], ctx.rfm, ctx.data.customers, s)
        detail = "; ".join(f"{a.segment_name}: {a.channel} x{a.targeted} ({a.reason})" for a in cp.allocations)
        return {"channel_plan": cp.model_dump(),
                "audit": _audit("channel", f"allocated INR {cp.total_cost:.2f} of {cp.budget:.2f}", detail)}

    def governance(state: CampaignState) -> dict:
        plan = StrategyPlan.model_validate(state["strategy"])
        cp = ChannelPlan.model_validate(state["channel_plan"])
        report = run_governance_agent(plan, cp, state["segments"], ctx.data.customers, s,
                                      state.get("revisions", 0))
        blocking = [v.model_dump() for v in report.violations if v.severity == "block"]
        detail = "; ".join(f"[{v['rule']}] {v['detail']}" for v in blocking) or "all checks passed"
        return {"governance": report.model_dump(), "governance_feedback": blocking,
                "audit": _audit("governance", report.status, detail)}

    def approval(state: CampaignState) -> dict:
        if ctx.auto_approve:
            return {"approval": {"approved": True, "by": "auto"},
                    "audit": _audit("human", "auto-approved", "auto_approve=True")}
        decision = interrupt({
            "question": "Approve this campaign for execution?",
            "total_cost": state["channel_plan"]["total_cost"],
            "blocked_segments": state["governance"]["blocked_segments"],
        })
        approved = bool(decision.get("approved"))
        feedback = (decision.get("feedback") or "").strip()
        if approved:
            return {"approval": {"approved": True, "by": decision.get("by", "human")},
                    "audit": _audit("human", "approved", feedback)}
        if feedback and state.get("human_revisions", 0) < s.max_revisions:
            # Send the plan back to the strategist with the reviewer's comments.
            return {"approval": None, "strategy": None, "channel_plan": None, "governance": None,
                    "human_feedback": feedback, "revisions": 0,
                    "human_revisions": state.get("human_revisions", 0) + 1,
                    "audit": _audit("human", "rejected with feedback -> re-plan", feedback)}
        return {"approval": {"approved": False, "by": decision.get("by", "human")},
                "audit": _audit("human", "rejected", feedback)}

    def execution(state: CampaignState) -> dict:
        plan = StrategyPlan.model_validate(state["strategy"])
        cp = ChannelPlan.model_validate(state["channel_plan"])
        result = run_execution_agent(plan, cp, state["governance"]["blocked_segments"], ctx.rfm,
                                     ctx.data.customers, state["segments"], ctx.data.ground_truth, s, ctx.seed)
        return {"execution": result.model_dump(),
                "audit": _audit("execution", "campaign dispatched (simulated)",
                                f"sent={result.sent}, conversions={result.conversions}, "
                                f"profit=INR {result.profit:,.0f}, ROI={result.roi}x")}

    g = StateGraph(CampaignState)
    for name, fn in [("supervisor", supervisor), ("segmentation", segmentation), ("strategy", strategy),
                     ("channel", channel), ("governance", governance), ("approval", approval),
                     ("execution", execution)]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda st: st["next"],
                            ["segmentation", "strategy", "channel", "governance", "approval", "execution", END])
    for worker in ["segmentation", "strategy", "channel", "governance", "approval", "execution"]:
        g.add_edge(worker, "supervisor")
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class CampaignRunner:
    """Convenience wrapper: start a run, then resume it after human review."""

    def __init__(self, ctx: CampaignContext):
        self.ctx = ctx
        self.graph = build_graph(ctx)
        self.thread_id = str(uuid.uuid4())

    @property
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 60}

    def start(self) -> dict:
        initial: CampaignState = {"revisions": 0, "human_revisions": 0, "audit": [], "status": "running"}
        self.graph.invoke(initial, self._config)
        return self.state()

    def resume(self, approved: bool, feedback: str = "", by: str = "human") -> dict:
        self.graph.invoke(Command(resume={"approved": approved, "feedback": feedback, "by": by}), self._config)
        return self.state()

    def state(self) -> dict:
        snapshot = self.graph.get_state(self._config)
        values = dict(snapshot.values)
        values["awaiting_approval"] = bool(snapshot.next) and any(t.interrupts for t in snapshot.tasks)
        return values
