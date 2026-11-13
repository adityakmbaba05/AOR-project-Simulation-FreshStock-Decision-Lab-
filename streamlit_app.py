"""Streamlit front-end for a transparent perishable inventory decision-support demo.
Deploy entrypoint: streamlit_app.py
"""
from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inventory_engine import ModelConfig, run_experiment

st.set_page_config(page_title="FreshStock | Inventory Decision Support", page_icon="🥬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {--ink: #153b39;}
.block-container{padding-top:1.6rem;max-width:1340px}
.hero{background:linear-gradient(112deg,#123f3a,#276e61);padding:26px 30px;border-radius:18px;color:white;margin-bottom:12px}
.hero h1{color:white;font-size:2.25rem;margin:0 0 .35rem 0}
.hero p{color:#e8f5ed;margin:0;font-size:1.02rem}
.small-note{padding:12px 15px;border:1px solid #dde6df;border-radius:10px;background:#f8fbf9;color:#29453a}
[data-testid="stMetric"]{background:#f5faf7;border:1px solid #e2ede6;border-radius:12px;padding:13px 15px}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🥬 FreshStock Decision Lab</h1>
<p>Cost-aware replenishment for perishable goods | FIFO • Poisson demand • adaptive orders</p></div>""",
unsafe_allow_html=True)
st.caption("Educational prototype inspired by Gulecyuz, O'Sullivan & Tarim (Omega, 2025). "
           "NOT an exact reproduction of the paper's Algorithm 1 or its reported 1.87% optimality gap.")

profiles = {
    "Seasonal grocery (12 periods)": (8, 9, 11, 16, 20, 18, 14, 12, 9, 8, 13, 17),
    "Paper's illustrative demand rates (4 periods)": (6, 9, 3, 8),
    "Increasing demand (8 periods)": (4, 5, 7, 9, 12, 15, 18, 21),
    "Decreasing demand (8 periods)": (21, 18, 15, 12, 9, 7, 5, 4),
    "Custom CSV upload": (),
}

with st.sidebar:
    st.header("Input settings")
    profile = st.selectbox("Demand pattern", list(profiles))
    uploaded = st.file_uploader("Upload CSV with Period, Mean_Demand", type="csv") if profile == "Custom CSV upload" else None
    rates = profiles[profile]
    upload_error = None
    if profile == "Custom CSV upload":
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
                if not {"Period", "Mean_Demand"}.issubset(set(data.columns)):
                    raise ValueError("Required columns: Period and Mean_Demand.")
                data = data.sort_values("Period")
                period_values = data["Period"].astype(int).tolist()
                if period_values != list(range(1, len(period_values) + 1)):
                    raise ValueError("Period must be consecutive integers starting at 1.")
                rates = tuple(float(v) for v in data["Mean_Demand"])
            except Exception as exc:
                upload_error = str(exc)
        else:
            upload_error = "Upload the sample_demand.csv file to continue."
    if upload_error:
        st.warning(upload_error)
    st.caption(f"Loaded forecast: {len(rates)} periods" if rates else "No valid demand forecast yet")
    with st.form("parameter_form"):
        m = st.number_input("Shelf life, M (periods)", min_value=2, max_value=12, value=3, step=1)
        setup = st.number_input("Fixed cost per order", min_value=0.0, value=25.0, step=5.0)
        purchase = st.number_input("Unit purchasing cost", min_value=0.0, value=1.0, step=0.5)
        holding = st.number_input("Unit holding cost / period", min_value=0.0, value=1.0, step=0.5)
        shortage = st.number_input("Unit backlog penalty / period", min_value=0.0, value=10.0, step=1.0)
        waste = st.number_input("Extra unit expiry cost (paper sets this to 0)", min_value=0.0, value=0.0, step=1.0)
        interval = st.number_input("Fixed-interval baseline (periods)", min_value=1, max_value=36, value=3, step=1)
        reps = st.slider("Demand simulations", min_value=5, max_value=150, value=30, step=5)
        seed = st.number_input("Random seed (reproducible)", min_value=0, max_value=1000000, value=42, step=1)
        with st.expander("Advanced: planning precision"):
            planning_scenarios = st.slider("Scenarios per planning arc", 20, 300, 90, 10)
            decision_scenarios = st.slider("Scenarios per adaptive order", 8, 120, 32, 8)
        submitted = st.form_submit_button("▶ Run simulation", type="primary", use_container_width=True)

if not rates:
    st.info("Load a valid demand CSV to run your custom case. Otherwise choose an included example.")
    st.stop()

try:
    cfg = ModelConfig(rates=tuple(rates), shelf_life=int(m), setup_cost=float(setup),
                      unit_cost=float(purchase), holding_cost=float(holding),
                      shortage_cost=float(shortage), waste_cost=float(waste),
                      fixed_interval=int(interval), planning_scenarios=int(planning_scenarios),
                      decision_scenarios=int(decision_scenarios), replications=int(reps), seed=int(seed))
except ValueError as exc:
    st.error(f"Please correct the inputs: {exc}")
    st.stop()

@st.cache_data(show_spinner=False)
def cached_simulation(config: ModelConfig):
    return run_experiment(config)

if submitted or "experiment" not in st.session_state:
    with st.spinner("Building the order schedule and simulating demand ..."):
        st.session_state["experiment"] = cached_simulation(cfg)
        st.session_state["run_config"] = cfg

result = st.session_state["experiment"]
used_cfg: ModelConfig = st.session_state["run_config"]
if cfg != used_cfg:
    st.warning("The controls have changed. Click 'Run simulation' to update the displayed results.")

plan = result["plan"]
means: pd.DataFrame = result["means"]
runs: pd.DataFrame = result["runs"]
summary = means.set_index("Policy")
dp = summary.loc["Paper-inspired DP"]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("DP average cost", f"{dp['Total cost']:,.2f}")
col_b.metric("DP expired units", f"{dp['Waste units']:,.1f}")
col_c.metric("DP orders placed", f"{dp['Orders placed']:,.1f}")
col_d.metric("DP horizon fulfillment", f"{dp['Horizon fulfilled (%)']:,.1f}%")

intro, dashboard, trace_tab, methods, testing = st.tabs([
    "Project overview", "Compare 3 policies", "Single-run explanation", "How the model works", "Tests & downloads"
])

with intro:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("The business decision")
        st.write("A store orders a product that expires after **M periods**. Demand changes from "
                 "period to period and cannot be known exactly. Ordering too often raises setup "
                 "cost; ordering too much raises holding and waste; ordering too little creates backlogs.")
        st.markdown("**Two decisions:** (1) on which periods can we order? "
                    "(2) how many units should we actually order after observing inventory ages?")
        st.markdown("**Stage 1** chooses replenishment dates using a shortest-path "
                    "dynamic program on independent cycle-cost approximations. "
                    "**Stage 2** reoptimizes order sizes using the real FIFO stock-age vector.")
        st.info("The algorithm is a documented educational surrogate, not the published "
                "state-aware, average-cost lookahead heuristic or an optimal SDP solver.")
    with right:
        st.subheader("Demand forecast")
        fig = px.bar(x=list(range(1, len(used_cfg.rates) + 1)), y=used_cfg.rates,
                     labels={"x": "Period", "y": "Mean demand"},
                     title="Poisson demand mean varies by period")
        fig.update_traces(marker_color="#26806c")
        fig.update_layout(margin=dict(t=48, b=10), height=330)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("When the DP allows replenishment")
    arcs = pd.DataFrame(plan["arcs"])
    st.dataframe(arcs.rename(columns={"start": "Start period", "end": "End period", "days": "Cycle length",
                                    "nominal_order": "Stage-1 reference Q",
                                    "approximate_arc_cost": "Stage-1 estimated cost"}),
                 hide_index=True, use_container_width=True)
    st.caption("Reference Q and stage-1 cost assume empty starting inventory and are NOT the "
               "actual simulated quantities/costs. Stage 2 decides actual Q.")

with dashboard:
    st.subheader("Policy comparison across identical demand paths")
    st.dataframe(means[["Policy", "Total cost", "Waste units", "Ending backlog", "Orders placed",
                        "Horizon fulfilled (%)", "Total ordered"]], use_container_width=True, hide_index=True)
    st.caption("Values are Monte Carlo averages, not optimality guarantees. The lower-cost policy can "
               "change with inputs and seed; no policy is universally best.")
    fig = px.bar(means, x="Policy", y="Total cost", color="Policy", text="Total cost",
                 title="Average total inventory cost by policy")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(showlegend=False, height=380, margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)
    components = ["Setup cost", "Purchasing cost", "Holding cost", "Shortage cost", "Expiration cost"]
    stacked = means.melt(id_vars="Policy", value_vars=components, var_name="Cost type", value_name="Average cost")
    fig2 = px.bar(stacked, x="Policy", y="Average cost", color="Cost type", barmode="stack",
                  title="Where the money goes", height=375)
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.box(runs, x="Policy", y="Total cost", color="Policy", points="all",
                  title="Cost variation between random demand scenarios", height=375)
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with trace_tab:
    st.subheader("See every decision in one simulated demand path")
    policy = st.selectbox("Choose the policy", ["Paper-inspired DP", "Every period", "Fixed interval"])
    trace = result["first_rep_details"][policy].copy()
    if policy == "Paper-inspired DP":
        st.caption("Eligible order dates fixed at the start: " + ", ".join(str(x + 1) for x in plan["starts"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trace["Period"], y=trace["Realized demand"], name="Realized demand", opacity=.45))
    fig.add_trace(go.Scatter(x=trace["Period"], y=trace["Order quantity"],
                             name="Actual ordered", mode="lines+markers", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=trace["Period"], y=trace["Ending inventory"],
                             name="Inventory remaining", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=trace["Period"], y=trace["Ending backlog"],
                             name="Unfulfilled demand", mode="lines+markers"))
    fig.update_layout(title="Demand, order quantities and inventory", xaxis_title="Period",
                      yaxis_title="Units", barmode="overlay", height=410)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(trace, use_container_width=True, hide_index=True)
    st.download_button("Download selected run as CSV", trace.to_csv(index=False).encode("utf-8"),
                       "single_run_trace.csv", mime="text/csv")
    st.info("Read row by row: expected demand -> order (if scheduled) -> actual demand -> "
            "FIFO fulfillment -> expiration -> end-of-period holding and backlog costs.")

with methods:
    st.subheader("Model rules")
    st.markdown(r"""
1. **Demand:** each period has its own Poisson mean, and demand is sampled independently.
2. **Shelf life:** a unit can be used during exactly `M` periods (including arrival period).
3. **FIFO:** sell/use the oldest items first. Expire unused items at the end of their final period.
4. **Backlog:** unmet demand carries forward and is penalized each period.
5. **Orders:** arrive instantly; fixed setup cost only if actual quantity is positive.
6. **Period cost:** `setup + unit purchase × Q + holding × surviving stock + shortage × backlog + expiry × expired units`.
7. **Stage 1:** choose a sequence of nonoverlapping cycles of length no more than `M` with independent-cycle Monte Carlo costs.
8. **Stage 2:** for each allowed order date, optimize Q against Monte Carlo demand across the time until the next planned order.
""")
    st.subheader("Transparent pseudocode")
    st.code("""FOR each end period j:
    FOR each feasible start i in [max(1,j-M+1), ..., j]:
        Estimate best order Q for cycle [i..j] assuming empty opening stock
        ApproxCost[i,j] = MonteCarloCycleCost(Q)
        DP[j] = min(DP[i-1] + ApproxCost[i,j])
Backtrack best DP arcs -> eligible replenishment periods
FOR each Monte Carlo replicate:
    Sample one entire demand path
    FOR each period t:
        IF t is eligible to order:
            Observe opening stock by age and outstanding backlog
            Search Q using forecast scenarios until next order date
        Age remaining inventory; receive order; serve backlog + demand FIFO
        Expire old units; record cost, shortage, waste and remaining stock
Compare policies using the SAME sampled demand paths""", language="text")
    st.warning("Paper differences: The published heuristic uses inventory compositions "
               "and a lookahead average-cost decision rule (Algorithm 1), a special "
               "outdating approximation (EWA), and a modified newsvendor formulation. "
               "This teaching app simplifies that stage-1 recursion, uses sampled cycle "
               "costs and grid/local search, and does not implement SDP.")
    st.markdown("**Paper:** Gulecyuz et al., *Omega* 133 (2025) 103267. "
                "[Research DOI](https://doi.org/10.1016/j.omega.2024.103267)")

with testing:
    st.subheader("Model tests")
    st.dataframe(pd.DataFrame([
        ["T01", "FIFO", "Old inventory consumed before fresh units"],
        ["T02", "Expiry", "Items expire only after their final usable period"],
        ["T03", "Backlogging", "Unmet demand carries to the next period"],
        ["T04", "Fixed cost", "Zero order quantity means zero setup cost"],
        ["T05", "Validation", "Invalid negative demand/orders rejected"],
        ["T06", "Planning", "Cycles cover horizon, each length <= shelf life"],
        ["T07", "Reproducibility", "Same seed produces same outputs"],
        ["T08", "Zero-demand edge", "No costs with zero demand and zero orders"],
    ], columns=["Test", "Purpose", "Expected behavior"]), hide_index=True, use_container_width=True)
    st.code("python -m pytest -q", language="bash")
    st.caption("Tests are included in tests/test_inventory_engine.py. Eight initial tests passed "
               "before packaging; rerun them after any change.")
    left, right, third = st.columns(3)
    left.download_button("Download policy averages", means.to_csv(index=False).encode(),
                         "policy_comparison.csv", "text/csv")
    right.download_button("Download all simulation runs", runs.to_csv(index=False).encode(),
                          "all_simulation_runs.csv", "text/csv")
    third.download_button("Download demand CSV", pd.DataFrame({
        "Period": range(1, len(used_cfg.rates) + 1), "Mean_Demand": used_cfg.rates
    }).to_csv(index=False).encode(), "demand_forecast.csv", "text/csv")
    st.markdown("**Decision-support disclaimer:** Costs are in arbitrary currency units. "
                "All forecast inputs and demand paths are synthetic unless replaced by user data. "
                "Real deployments require real demand estimates, business rules, and further validation.")

st.divider()
st.caption("Academic DSS prototype • Source paper: Omega (2025) • Model code: inventory_engine.py • "
           "Prepared for Notion report + Streamlit demonstration")
