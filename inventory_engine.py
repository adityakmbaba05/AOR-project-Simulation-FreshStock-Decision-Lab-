"""Transparent educational implementation of a paper-inspired perishable inventory DSS.

Research basis: Gulecyuz, O'Sullivan & Tarim (2025), Omega 133, 103267.
IMPORTANT: This is NOT the authors' Algorithm 1, EWA, or SDP implementation.
Its two-stage design uses an independent-cycle shortest-path surrogate,
then adaptive Monte Carlo order quantities with FIFO, expiration and backlog.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    rates: tuple[float, ...] = (8, 9, 11, 16, 20, 18, 14, 12, 9, 8, 13, 17)
    shelf_life: int = 3
    setup_cost: float = 25.0
    unit_cost: float = 1.0
    holding_cost: float = 1.0
    shortage_cost: float = 10.0
    waste_cost: float = 0.0
    fixed_interval: int = 3
    planning_scenarios: int = 90
    decision_scenarios: int = 32
    replications: int = 30
    seed: int = 42

    def __post_init__(self) -> None:
        if not 1 <= len(self.rates) <= 36:
            raise ValueError("Enter between 1 and 36 periods.")
        if any(not np.isfinite(x) or x < 0 or x > 250 for x in self.rates):
            raise ValueError("Demand means must be finite and between 0 and 250.")
        if not 2 <= self.shelf_life <= 12:
            raise ValueError("Shelf life must be an integer from 2 through 12.")
        for key in ("setup_cost", "unit_cost", "holding_cost", "shortage_cost", "waste_cost"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite.")
        if not 1 <= self.fixed_interval <= 36:
            raise ValueError("Fixed review interval must be between 1 and 36.")
        if not 10 <= self.planning_scenarios <= 5000 or not 8 <= self.decision_scenarios <= 2000:
            raise ValueError("Scenario counts outside the supported range.")
        if not 1 <= self.replications <= 1000:
            raise ValueError("Replications must be between 1 and 1000.")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative.")


def simulate_period(
    opening_stock: np.ndarray,
    backlog: int,
    order: int,
    demand: int,
    cfg: ModelConfig,
) -> dict[str, Any]:
    """One complete period. Stock is post-expiry stock aged 0..M-2 at PREVIOUS end.

    At this period: age existing stock; receive new order age 0;
    fill old backlog and new demand FIFO (oldest items first); expire age M-1;
    charge holding only to surviving stock and shortage to remaining backlog.
    """
    if order < 0 or demand < 0 or backlog < 0:
        raise ValueError("Demand, backlog, and order cannot be negative.")
    stock = np.asarray(opening_stock, dtype=np.int64)
    if stock.shape != (cfg.shelf_life - 1,) or (stock < 0).any():
        raise ValueError("Opening stock vector must have M-1 nonnegative ages.")
    ages = np.zeros(cfg.shelf_life, dtype=np.int64)
    ages[0] = int(order)
    ages[1:] = stock
    request = int(backlog + demand)
    unmet = request
    for age in range(cfg.shelf_life - 1, -1, -1):
        taken = min(int(ages[age]), unmet)
        ages[age] -= taken
        unmet -= taken
    expired = int(ages[-1])
    ending_stock = ages[:-1].copy()
    held = int(ending_stock.sum())
    served = request - unmet
    costs = {
        "setup": float(cfg.setup_cost if order > 0 else 0),
        "purchase": float(cfg.unit_cost * order),
        "holding": float(cfg.holding_cost * held),
        "shortage": float(cfg.shortage_cost * unmet),
        "expiration": float(cfg.waste_cost * expired),
    }
    return {
        "ending_stock": ending_stock,
        "ending_backlog": unmet,
        "expired": expired,
        "served": served,
        "held": held,
        "costs": costs,
        "total_cost": sum(costs.values()),
    }


def _project_cost(
    cfg: ModelConfig,
    opening_stock: np.ndarray,
    backlog: int,
    order: int,
    demand_paths: np.ndarray,
) -> float:
    """Vectorized expected cycle cost, with no intermediate replenishment.

    IMPORTANT: No terminal salvage/continuation value; planning arcs assume
    no stock from earlier arcs, an intentional surrogate simplification.
    """
    n, length = demand_paths.shape
    ages = np.zeros((n, cfg.shelf_life - 1), dtype=np.int64)
    ages[:] = opening_stock
    needs = np.full(n, int(backlog), dtype=np.int64)
    total = np.full(n, (cfg.setup_cost if order > 0 else 0) + cfg.unit_cost * order, dtype=float)
    for period in range(length):
        this_stock = np.zeros((n, cfg.shelf_life), dtype=np.int64)
        this_stock[:, 0] = int(order) if period == 0 else 0
        this_stock[:, 1:] = ages
        unmet = needs + demand_paths[:, period]
        for age in range(cfg.shelf_life - 1, -1, -1):
            serve = np.minimum(this_stock[:, age], unmet)
            this_stock[:, age] -= serve
            unmet -= serve
        total += cfg.holding_cost * this_stock[:, :-1].sum(axis=1)
        total += cfg.shortage_cost * unmet
        total += cfg.waste_cost * this_stock[:, -1]
        ages = this_stock[:, :-1]
        needs = unmet
    return float(np.mean(total))


def optimize_order(
    cfg: ModelConfig,
    stock: np.ndarray,
    backlog: int,
    future_rates: tuple[float, ...],
    rng: np.random.Generator,
    scenarios: int,
) -> tuple[int, float]:
    """Look ahead to next allowed order; choose integer Q by sampled cost search."""
    if not future_rates:
        raise ValueError("At least one future period is needed.")
    rates = np.asarray(future_rates, dtype=float)
    demand_paths = rng.poisson(rates, size=(scenarios, len(rates)))
    estimate = float(np.sum(rates)) + backlog - float(np.sum(stock))
    cap = max(8, int(math.ceil(estimate + 4 * math.sqrt(max(1.0, float(np.sum(rates)))) + 8)))
    cap = min(cap, 1500)
    stride = max(1, math.ceil(cap / 32))
    candidates = list(range(0, cap + 1, stride))
    if candidates[-1] != cap:
        candidates.append(cap)
    seen = {}

    def score(q: int) -> float:
        if q not in seen:
            seen[q] = _project_cost(cfg, stock, backlog, q, demand_paths)
        return seen[q]

    winner = min(candidates, key=lambda q: (score(q), q))
    for q in range(max(0, winner - stride + 1), min(cap, winner + stride - 1) + 1):
        if score(q) < score(winner) - 1e-10:
            winner = q
    return int(winner), float(score(winner))


def plan_shortest_path(cfg: ModelConfig) -> dict[str, Any]:
    """Stage 1: min-total-cost DAG over cycle lengths <= shelf life.

    Paper inspiration: finite-horizon network representation and arcs <= M.
    Difference: independent empty-start cycle costs and total-cost DP rather
    than the published average-cost, inventory-state-aware recursive heuristic.
    """
    t = len(cfg.rates)
    dp = [float("inf")] * (t + 1)
    parent: list[tuple[int, int, float] | None] = [None] * (t + 1)
    dp[0] = 0.0
    for end in range(1, t + 1):
        for start in range(max(0, end - cfg.shelf_life), end):
            local_rng = np.random.default_rng(cfg.seed + 7919 * start + 104729 * end)
            order, arc_cost = optimize_order(
                cfg, np.zeros(cfg.shelf_life - 1, dtype=np.int64), 0,
                cfg.rates[start:end], local_rng, cfg.planning_scenarios,
            )
            candidate = dp[start] + arc_cost
            if candidate < dp[end] - 1e-9:
                dp[end] = candidate
                parent[end] = (start, order, arc_cost)
    arcs = []
    end = t
    while end > 0:
        item = parent[end]
        assert item is not None
        start, nominal_order, arc_cost = item
        arcs.append({
            "start": start + 1,
            "end": end,
            "days": end - start,
            "nominal_order": nominal_order,
            "approximate_arc_cost": round(arc_cost, 2),
        })
        end = start
    arcs.reverse()
    return {
        "starts": tuple(int(arc["start"]) - 1 for arc in arcs),
        "arcs": arcs,
        "approximated_dp_cost": round(dp[t], 2),
    }


def _schedule(cfg: ModelConfig, policy: str, dp_plan: dict[str, Any]) -> tuple[int, ...]:
    if policy == "Paper-inspired DP":
        return dp_plan["starts"]
    if policy == "Every period":
        return tuple(range(len(cfg.rates)))
    if policy == "Fixed interval":
        return tuple(range(0, len(cfg.rates), cfg.fixed_interval))
    raise ValueError(f"Unknown policy: {policy}")


def simulate_policy(
    cfg: ModelConfig,
    policy: str,
    schedule: tuple[int, ...],
    realized_demand: np.ndarray,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Stage 2: order only in eligible periods; reoptimize Q using real age mix."""
    t = len(cfg.rates)
    if realized_demand.shape != (t,):
        raise ValueError("Realized demand path length must match forecast horizon.")
    if not schedule or schedule[0] != 0 or sorted(set(schedule)) != list(schedule):
        raise ValueError("Schedule must begin at zero with unique ordered indices.")
    if any(s < 0 or s >= t for s in schedule):
        raise ValueError("Schedule entries must fall inside the horizon.")
    next_end = {start: (schedule[i + 1] if i + 1 < len(schedule) else t)
                for i, start in enumerate(schedule)}
    stock = np.zeros(cfg.shelf_life - 1, dtype=np.int64)
    backlog = 0
    rng = np.random.default_rng(random_seed)
    records = []
    total_served = 0
    for p in range(t):
        opened = stock.copy()
        previous_backlog = backlog
        if p in next_end:
            end = next_end[p]
            order, predicted_cycle_cost = optimize_order(
                cfg, stock, backlog, cfg.rates[p:end], rng, cfg.decision_scenarios,
            )
        else:
            order = 0
            predicted_cycle_cost = float("nan")
        result = simulate_period(stock, backlog, order, int(realized_demand[p]), cfg)
        stock = result["ending_stock"]
        backlog = int(result["ending_backlog"])
        total_served += int(result["served"])
        row = {
            "Period": p + 1,
            "Expected demand": cfg.rates[p],
            "Realized demand": int(realized_demand[p]),
            "Opening inventory": int(opened.sum()),
            "Opening backlog": int(previous_backlog),
            "Eligible order date": int(p in next_end),
            "Order quantity": int(order),
            "Units fulfilled": int(result["served"]),
            "Ending inventory": int(result["held"]),
            "Ending backlog": backlog,
            "Expired units": int(result["expired"]),
            "Setup cost": result["costs"]["setup"],
            "Purchasing cost": result["costs"]["purchase"],
            "Holding cost": result["costs"]["holding"],
            "Shortage cost": result["costs"]["shortage"],
            "Expiration cost": result["costs"]["expiration"],
            "Total cost": result["total_cost"],
            "Forecast cycle cost (MC)": predicted_cycle_cost,
        }
        for age in range(cfg.shelf_life - 1):
            row[f"Ending stock age {age}"] = int(stock[age])
        records.append(row)
    df = pd.DataFrame(records)
    demand_total = int(np.sum(realized_demand))
    total = float(df["Total cost"].sum())
    summary: dict[str, float | int | str] = {
        "Policy": policy,
        "Total cost": round(total, 2),
        "Total demand": demand_total,
        "Total ordered": int(df["Order quantity"].sum()),
        "Total fulfilled": total_served,
        "Ending backlog": backlog,
        "Waste units": int(df["Expired units"].sum()),
        "Orders placed": int((df["Order quantity"] > 0).sum()),
        "Horizon fulfilled (%)": round(100 * total_served / demand_total, 2) if demand_total else 100.0,
        "Setup cost": round(float(df["Setup cost"].sum()), 2),
        "Purchasing cost": round(float(df["Purchasing cost"].sum()), 2),
        "Holding cost": round(float(df["Holding cost"].sum()), 2),
        "Shortage cost": round(float(df["Shortage cost"].sum()), 2),
        "Expiration cost": round(float(df["Expiration cost"].sum()), 2),
    }
    return df, summary


def run_experiment(cfg: ModelConfig) -> dict[str, Any]:
    """Compare three policies on IDENTICAL Poisson demand paths (common random numbers)."""
    plan = plan_shortest_path(cfg)
    policies = ["Paper-inspired DP", "Every period", "Fixed interval"]
    demand_rng = np.random.default_rng(cfg.seed + 99_991)
    paths = demand_rng.poisson(np.asarray(cfg.rates), size=(cfg.replications, len(cfg.rates)))
    all_summaries = []
    first_rep_details = {}
    for run in range(cfg.replications):
        for i, policy in enumerate(policies):
            schedule = _schedule(cfg, policy, plan)
            trace, totals = simulate_policy(
                cfg, policy, schedule, paths[run], cfg.seed + run * 1009 + i * 10_007 + 444,
            )
            all_summaries.append({"Run": run + 1, **totals})
            if run == 0:
                first_rep_details[policy] = trace
    runs = pd.DataFrame(all_summaries)
    cols = ["Total cost", "Total ordered", "Total fulfilled", "Ending backlog", "Waste units",
            "Orders placed", "Horizon fulfilled (%)", "Setup cost", "Purchasing cost",
            "Holding cost", "Shortage cost", "Expiration cost"]
    means = runs.groupby("Policy", sort=False)[cols].mean().round(2).reset_index()
    return {"plan": plan, "means": means, "runs": runs,
            "first_rep_details": first_rep_details, "first_path": paths[0]}


if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    cfg = ModelConfig(replications=12, planning_scenarios=60, decision_scenarios=24)
    result = run_experiment(cfg)
    print("Planned order periods:", [x + 1 for x in result["plan"]["starts"]])
    print(result["means"].to_string(index=False))
    print("Seconds:", round(perf_counter() - start, 2))
