"""CampaignPilot - Streamlit dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from campaignpilot.config import Settings
from campaignpilot.data import generate_synthetic, load_online_retail
from campaignpilot.graph import CampaignContext, CampaignRunner
from campaignpilot.llm import LLMError, make_llm

st.set_page_config(page_title="CampaignPilot", page_icon="🎯", layout="wide")


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:  # no secrets.toml
        return os.getenv(name, "")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Configuration")
defaults = Settings()
providers = ["mock", "groq", "ollama"]
provider = st.sidebar.selectbox("LLM strategist", providers, index=providers.index(defaults.llm_provider)
                                if defaults.llm_provider in providers else 0,
                                help="mock = offline deterministic planner (no API key needed)")
groq_key = secret("GROQ_API_KEY")
if provider == "groq":
    groq_key = st.sidebar.text_input("Groq API key", value=groq_key, type="password")
    groq_model = st.sidebar.text_input("Groq model", value=defaults.groq_model)
else:
    groq_model = defaults.groq_model
ollama_model = st.sidebar.text_input("Ollama model", value=defaults.ollama_model) if provider == "ollama" \
    else defaults.ollama_model

st.sidebar.subheader("Policy")
budget = st.sidebar.slider("Campaign budget (INR)", 50, 1000, int(defaults.budget), step=25)
max_disc = st.sidebar.slider("Max discount %", 5, 50, int(defaults.max_discount_pct))
vip_disc = st.sidebar.slider("VIP max discount %", 0, 30, int(defaults.vip_max_discount_pct))
cap = st.sidebar.slider("Frequency cap (days)", 0, 14, defaults.frequency_cap_days)

st.sidebar.subheader("Data")
source = st.sidebar.radio("Customer data", ["Synthetic (seeded)", "Upload UCI Online Retail CSV"])
if source.startswith("Synthetic"):
    n_customers = st.sidebar.slider("Customers", 500, 5000, 2000, step=500)
    seed = st.sidebar.number_input("Data seed", 0, 9999, 7)
    upload = None
else:
    upload = st.sidebar.file_uploader("Online Retail CSV/XLSX", type=["csv", "xlsx"])
    n_customers, seed = 0, 7

settings = replace(defaults, llm_provider=provider, groq_api_key=groq_key, groq_model=groq_model,
                   ollama_model=ollama_model, budget=float(budget), max_discount_pct=float(max_disc),
                   vip_max_discount_pct=float(vip_disc), frequency_cap_days=int(cap))

# ---------------------------------------------------------------- header
st.title("🎯 CampaignPilot")
st.caption("Multi-agent marketing campaign orchestration: data-driven segmentation → LLM strategy → "
           "budget optimisation → rule-based governance → human approval → execution.")


@st.cache_data(show_spinner=False)
def synthetic(n: int, s: int):
    return generate_synthetic(n_customers=n, seed=s)


if st.button("🚀 Plan campaign", type="primary"):
    try:
        if upload is not None:
            path = f"/tmp/{upload.name}"
            with open(path, "wb") as fh:
                fh.write(upload.getbuffer())
            data = load_online_retail(path)
        elif source.startswith("Synthetic"):
            data = synthetic(n_customers, int(seed))
        else:
            st.warning("Upload a file first.")
            st.stop()
        ctx = CampaignContext(data, settings, make_llm(settings), seed=int(seed))
        runner = CampaignRunner(ctx)
        with st.spinner("Agents at work..."):
            runner.start()
        st.session_state["runner"] = runner
    except LLMError as exc:
        st.error(str(exc))

runner: CampaignRunner | None = st.session_state.get("runner")
if runner is None:
    st.info("Choose settings in the sidebar and click **Plan campaign**. "
            "The mock strategist works offline; pick *groq* and add a free key for a real LLM.")
    st.stop()

state = runner.state()
if state.get("error"):
    st.error(f"Run failed: {state['error']}")

seg_tab, strat_tab, chan_tab, gov_tab, approve_tab, audit_tab = st.tabs(
    ["1 · Segments", "2 · Strategy", "3 · Channel & Budget", "4 · Governance", "5 · Approve & Results", "Audit trail"])

segments = state.get("segments") or []
names = {p["segment_id"]: p["name"] for p in segments}

with seg_tab:
    st.subheader(f"{len(segments)} segments (k={state.get('k')} chosen by silhouette score)")
    st.caption(f"Data: {runner.ctx.data.source}")
    if segments:
        table = pd.DataFrame([{k: v for k, v in p.items() if k != "channels"} for p in segments])
        st.dataframe(table, hide_index=True, width="stretch")
        left, right = st.columns(2)
        rfm = runner.ctx.rfm.assign(segment=lambda d: d["segment_id"].map(names))
        left.plotly_chart(px.scatter(rfm, x="recency", y="monetary", color="segment", log_y=True, opacity=0.6,
                                     title="Recency vs spend", labels={"recency": "days since last order",
                                                                       "monetary": "total spend (INR)"}),
                          width="stretch")
        sil = pd.DataFrame({"k": list(state["silhouette_by_k"]), "silhouette": list(state["silhouette_by_k"].values())})
        right.plotly_chart(px.bar(sil, x="k", y="silhouette", title="Model selection: silhouette by k"),
                           width="stretch")

with strat_tab:
    if state.get("strategy"):
        st.subheader(f"Strategy by {runner.ctx.llm.name} · revisions: {state.get('revisions', 0)}")
        df = pd.DataFrame(state["strategy"]["strategies"])
        df.insert(1, "segment", df["segment_id"].map(names))
        st.dataframe(df, hide_index=True, width="stretch")

with chan_tab:
    cp = state.get("channel_plan")
    if cp:
        st.metric("Allocated budget", f"₹{cp['total_cost']:.2f}", f"of ₹{cp['budget']:.0f}", delta_color="off")
        df = pd.DataFrame(cp["allocations"]).drop(columns="customer_ids")
        st.dataframe(df, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(df, x="segment_name", y=["targeted", "excluded_no_consent", "excluded_frequency_cap"],
                               title="Audience after consent and frequency-cap filtering", barmode="stack"),
                        width="stretch")

with gov_tab:
    gov = state.get("governance")
    if gov:
        icon = {"approved": "✅", "final": "⚠️", "revise": "🔁"}[gov["status"]]
        st.subheader(f"{icon} Governance status: {gov['status']}")
        if gov["blocked_segments"]:
            st.warning("Blocked (still non-compliant after max revisions): "
                       + ", ".join(names[s] for s in gov["blocked_segments"]))
        history = [a for a in state["audit"] if a["agent"] == "governance"]
        st.write("**Review history**")
        st.dataframe(pd.DataFrame(history)[["event", "detail"]], hide_index=True, width="stretch")
        if gov["violations"]:
            v = pd.DataFrame(gov["violations"])
            v["segment"] = v["segment_id"].map(names)
            st.dataframe(v, hide_index=True, width="stretch")
        with st.expander("Checks run"):
            st.write("\n".join(f"- {c}" for c in gov["checks_run"]))

with approve_tab:
    if state.get("awaiting_approval"):
        st.subheader("🧑‍⚖️ Human approval required")
        st.write(f"Total spend: **₹{state['channel_plan']['total_cost']:.2f}** · "
                 f"customers: **{sum(a['targeted'] for a in state['channel_plan']['allocations'])}**")
        feedback = st.text_area("Feedback for the strategist (used if you reject)",
                                placeholder="e.g. No discounts for Bargain Hunters this month.")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve & execute", type="primary"):
            runner.resume(True, feedback)
            st.rerun()
        if c2.button("↩️ Reject"):
            runner.resume(False, feedback)
            st.rerun()
    elif state.get("execution"):
        ex = state["execution"]
        st.subheader("📈 Campaign results (simulated)")
        k = st.columns(5)
        k[0].metric("Messages sent", f"{ex['sent']:,}")
        k[1].metric("Conversions", ex["conversions"], f"{100 * ex['conversion_rate']:.1f}%", delta_color="off")
        k[2].metric("Revenue", f"₹{ex['revenue']:,.0f}")
        k[3].metric("Profit", f"₹{ex['profit']:,.0f}")
        k[4].metric("ROI on spend", f"{ex['roi']:.2f}x")
        out = pd.DataFrame(ex["outcomes"])
        st.dataframe(out, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(out, x="segment_name", y="profit", color="offer", title="Profit by segment"),
                        width="stretch")
    elif state.get("status") == "rejected":
        st.info("Campaign rejected by reviewer. Nothing was sent.")

with audit_tab:
    st.dataframe(pd.DataFrame(state.get("audit", [])), hide_index=True, width="stretch")
    st.download_button("Download audit log (JSON)", pd.DataFrame(state.get("audit", [])).to_json(orient="records"),
                       "audit_log.json", "application/json")
