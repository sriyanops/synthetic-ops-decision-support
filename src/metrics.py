# src/metrics.py
from __future__ import annotations

import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute recent operational performance metrics using the most recent 14 days of data.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    recent = df.sort_values("date").tail(14)

    metrics = {
        "avg_package_volume": float(recent["package_volume"].mean()),
        "capacity_utilization": float((recent["package_volume"] / recent["network_capacity"]).mean()),
        "on_time_rate_avg": float(recent["on_time_rate"].mean()),
        "avg_exceptions": float(recent["exceptions"].mean()),
        "labor_efficiency": float((recent["package_volume"] / recent["labor_hours"]).mean()),
        "avg_cost_per_package": float(recent["cost_per_package"].mean()),
    }
    return metrics


def build_data_lineage() -> list[dict]:
    """
    Static metric lineage map for the CSV schema. (This is what you show the reader.)
    """
    return [
        {
            "metric": "Avg Package Volume",
            "formula": "mean(package_volume)",
            "source_columns": "package_volume",
        },
        {
            "metric": "Capacity Utilization",
            "formula": "mean(package_volume / network_capacity)",
            "source_columns": "package_volume, network_capacity",
        },
        {
            "metric": "On-Time Delivery Rate",
            "formula": "mean(on_time_rate)",
            "source_columns": "on_time_rate",
        },
        {
            "metric": "Avg Daily Exceptions",
            "formula": "mean(exceptions)",
            "source_columns": "exceptions",
        },
        {
            "metric": "Labor Efficiency (pkgs/hr)",
            "formula": "mean(package_volume / labor_hours)",
            "source_columns": "package_volume, labor_hours",
        },
        {
            "metric": "Avg Cost per Package",
            "formula": "mean(cost_per_package)",
            "source_columns": "cost_per_package",
        },
    ]


def scenario_summaries(df: pd.DataFrame) -> list[dict]:
    """
    Counts and shares by scenario for the analyzed window.
    Expected column: scenario
    """
    if "scenario" not in df.columns:
        return []

    counts = df["scenario"].value_counts(dropna=False)
    total = float(counts.sum()) if float(counts.sum()) > 0 else 1.0

    rows = []
    for scenario, c in counts.items():
        rows.append(
            {
                "scenario": str(scenario),
                "days": int(c),
                "share": float(c) / total,
            }
        )
    return rows


def report_context(df: pd.DataFrame, data_source_name: str, days_analyzed: int) -> dict:
    """
    Minimal context block for the report header.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    date_min = df["date"].min()
    date_max = df["date"].max()

    # Keep same output keys, but ensure consistent string formatting
    return {
        "data_source": data_source_name,
        "days_analyzed": int(days_analyzed),
        "date_range": f"{date_min.date()} → {date_max.date()}",
    }


# ----------------------------
# Added for Option B refactor:
# (moves logic out of export_pdf.py)
# ----------------------------

def scenario_mix_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario mix table used in the PDF.
    Returns a dataframe with:
      scenario, Days, Avg_Volume, Avg_Util, Avg_OnTime, Avg_Exceptions, Avg_LaborEff, Avg_Cost
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    grp = df.groupby("scenario").agg(
        Days=("date", "count"),
        Avg_Volume=("package_volume", "mean"),
        Avg_OnTime=("on_time_rate", "mean"),
        Avg_Exceptions=("exceptions", "mean"),
        Avg_Cost=("cost_per_package", "mean"),
    ).reset_index()

    util = df.assign(util=df["package_volume"] / df["network_capacity"]).groupby("scenario")["util"].mean().reset_index()
    leff = df.assign(leff=df["package_volume"] / df["labor_hours"]).groupby("scenario")["leff"].mean().reset_index()

    grp = grp.merge(util, on="scenario").merge(leff, on="scenario")
    grp = grp.rename(columns={"util": "Avg_Util", "leff": "Avg_LaborEff"})

    # Keep a consistent, readable scenario ordering if present
    order = ["NORMAL", "PEAK", "DISRUPTION", "LABOR_SHORTAGE"]
    if "scenario" in grp.columns:
        grp["scenario"] = pd.Categorical(grp["scenario"], categories=order, ordered=True)
        grp = grp.sort_values("scenario")

    return grp


def top_drivers_tables(df: pd.DataFrame):
    """
    Returns three small dataframes for the report:
      - worst_on_time: bottom 5 on-time rate days
      - high_ex: top 5 exceptions days
      - high_util: top 5 capacity utilization days
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["cap_util"] = df["package_volume"] / df["network_capacity"]

    worst_on_time = df.nsmallest(5, "on_time_rate")[["date", "scenario", "on_time_rate", "package_volume", "exceptions"]]
    high_ex = df.nlargest(5, "exceptions")[["date", "scenario", "exceptions", "on_time_rate", "package_volume"]]
    high_util = df.nlargest(5, "cap_util")[["date", "scenario", "cap_util", "network_capacity", "package_volume"]]

    return worst_on_time, high_ex, high_util
