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


def build_data_lineage() -> list:
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


def scenario_summaries(df: pd.DataFrame) -> list:
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

    return {
        "data_source": data_source_name,
        "days_analyzed": int(days_analyzed),
        "date_range": f"{date_min} → {date_max}",
    }
