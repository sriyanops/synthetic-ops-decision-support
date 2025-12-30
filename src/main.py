import pandas as pd
from rich.console import Console
from rich.table import Table

from metrics import compute_metrics
from rules import generate_recommendations


def _make_kv_table(title: str, rows: list[tuple[str, str]]) -> Table:
    t = Table(title=title, show_lines=True)
    t.add_column("Field")
    t.add_column("Value", overflow="fold")
    for k, v in rows:
        t.add_row(k, v)
    return t


def _top_drivers_tables(df: pd.DataFrame) -> list[Table]:
    """
    Returns small tables that point an operator to the *dates* driving issues.
    """
    tables = []

    df2 = df.copy()
    df2["capacity_util"] = df2["package_volume"] / df2["network_capacity"]

    # Worst on-time days
    worst_on_time = df2.sort_values("on_time_rate", ascending=True).head(5)
    t1 = Table(title="Top Drivers – Worst On-Time Days (bottom 5)", show_lines=True)
    t1.add_column("Date", no_wrap=True)
    t1.add_column("Scenario", no_wrap=True)
    t1.add_column("On-Time", justify="right", no_wrap=True)
    t1.add_column("Volume", justify="right", no_wrap=True)
    t1.add_column("Exceptions", justify="right", no_wrap=True)
    for _, r in worst_on_time.iterrows():
        t1.add_row(
            str(r["date"]),
            str(r.get("scenario", "")),
            f"{r['on_time_rate']*100:.1f}%",
            f"{int(r['package_volume']):,}",
            f"{int(r['exceptions']):,}",
        )
    tables.append(t1)

    # Highest exception days
    worst_ex = df2.sort_values("exceptions", ascending=False).head(5)
    t2 = Table(title="Top Drivers – Highest Exception Days (top 5)", show_lines=True)
    t2.add_column("Date", no_wrap=True)
    t2.add_column("Scenario", no_wrap=True)
    t2.add_column("Exceptions", justify="right", no_wrap=True)
    t2.add_column("On-Time", justify="right", no_wrap=True)
    t2.add_column("Volume", justify="right", no_wrap=True)
    for _, r in worst_ex.iterrows():
        t2.add_row(
            str(r["date"]),
            str(r.get("scenario", "")),
            f"{int(r['exceptions']):,}",
            f"{r['on_time_rate']*100:.1f}%",
            f"{int(r['package_volume']):,}",
        )
    tables.append(t2)

    # Highest capacity utilization days
    worst_cap = df2.sort_values("capacity_util", ascending=False).head(5)
    t3 = Table(title="Top Drivers – Highest Capacity Utilization Days (top 5)", show_lines=True)
    t3.add_column("Date", no_wrap=True)
    t3.add_column("Scenario", no_wrap=True)
    t3.add_column("Cap Util", justify="right", no_wrap=True)
    t3.add_column("Capacity", justify="right", no_wrap=True)
    t3.add_column("Volume", justify="right", no_wrap=True)
    for _, r in worst_cap.iterrows():
        t3.add_row(
            str(r["date"]),
            str(r.get("scenario", "")),
            f"{r['capacity_util']*100:.1f}%",
            f"{int(r['network_capacity']):,}",
            f"{int(r['package_volume']):,}",
        )
    tables.append(t3)

    # Highest cost days
    worst_cost = df2.sort_values("cost_per_package", ascending=False).head(5)
    t4 = Table(title="Top Drivers – Highest Cost per Package Days (top 5)", show_lines=True)
    t4.add_column("Date", no_wrap=True)
    t4.add_column("Scenario", no_wrap=True)
    t4.add_column("Cost/Package", justify="right", no_wrap=True)
    t4.add_column("On-Time", justify="right", no_wrap=True)
    t4.add_column("Volume", justify="right", no_wrap=True)
    for _, r in worst_cost.iterrows():
        t4.add_row(
            str(r["date"]),
            str(r.get("scenario", "")),
            f"${r['cost_per_package']:.2f}",
            f"{r['on_time_rate']*100:.1f}%",
            f"{int(r['package_volume']):,}",
        )
    tables.append(t4)

    return tables


def main():
    console = Console()

    # Load data
    df = pd.read_csv("data/fedex_product_ops.csv")

    # -------------------------
    # Report Header (audit + context)
    # -------------------------
    date_min = str(df["date"].min()) if "date" in df.columns else "N/A"
    date_max = str(df["date"].max()) if "date" in df.columns else "N/A"
    n_rows = len(df)

    scenario_counts = ""
    if "scenario" in df.columns:
        scenario_counts = "\n".join(
            [f"{k}: {v}" for k, v in df["scenario"].value_counts().items()]
        )
    else:
        scenario_counts = "N/A"

    header = _make_kv_table(
        "FedEx Product Ops – Report Header",
        [
            ("Data file", "data/fedex_product_ops.csv"),
            ("Rows analyzed (days)", str(n_rows)),
            ("Date range", f"{date_min} → {date_max}"),
            ("Scenario mix", scenario_counts),
        ],
    )
    console.print(header)
    console.print()

    # Compute metrics
    metrics = compute_metrics(df)

    # -------------------------
    # Metrics Summary
    # -------------------------
    metrics_table = Table(title="FedEx Product Ops – Metrics Summary", show_lines=True)
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right", no_wrap=True)

    metrics_table.add_row("Avg Package Volume", f"{metrics['avg_package_volume']:.0f}")
    metrics_table.add_row("Capacity Utilization", f"{metrics['capacity_utilization']*100:.1f}%")
    metrics_table.add_row("On-Time Delivery Rate", f"{metrics['on_time_rate_avg']*100:.1f}%")
    metrics_table.add_row("Avg Daily Exceptions", f"{metrics['avg_exceptions']:.1f}")
    metrics_table.add_row("Labor Efficiency (pkgs / hr)", f"{metrics['labor_efficiency']:.1f}")
    metrics_table.add_row("Avg Cost per Package", f"${metrics['avg_cost_per_package']:.2f}")

    console.print(metrics_table)
    console.print()

    # -------------------------
    # Data Lineage (Metric -> Formula -> Source Columns)
    # -------------------------
    lineage = Table(title="Data Lineage – How Metrics Are Calculated", show_lines=True, expand=True)
    lineage.add_column("Metric", no_wrap=True, width=26)
    lineage.add_column("Formula", overflow="fold", ratio=3)
    lineage.add_column("Source Columns (fedex_product_ops.csv)", overflow="fold", ratio=3)

    lineage.add_row("Avg Package Volume", "mean(package_volume)", "package_volume")
    lineage.add_row("Capacity Utilization", "mean(package_volume / network_capacity)", "package_volume, network_capacity")
    lineage.add_row("On-Time Delivery Rate", "mean(on_time_rate)", "on_time_rate")
    lineage.add_row("Avg Daily Exceptions", "mean(exceptions)", "exceptions")
    lineage.add_row("Labor Efficiency (pkgs/hr)", "mean(package_volume / labor_hours)", "package_volume, labor_hours")
    lineage.add_row("Avg Cost per Package", "mean(cost_per_package)", "cost_per_package")

    console.print(lineage)
    console.print()

    # -------------------------
    # Top Drivers (dates to investigate)
    # -------------------------
    for t in _top_drivers_tables(df):
        console.print(t)
        console.print()

    # -------------------------
    # Decision Support
    # -------------------------
    recommendations = generate_recommendations(metrics)

    rec_table = Table(
        title="FedEx Product Ops – Decision Support",
        show_lines=True,
        expand=True
    )
    rec_table.add_column("Cat", no_wrap=True, width=8)
    rec_table.add_column("Sev", no_wrap=True, width=6)
    rec_table.add_column("Issue", overflow="fold", ratio=3)
    rec_table.add_column("Rationale", overflow="fold", ratio=3)
    rec_table.add_column("Action", overflow="fold", ratio=4)

    for r in recommendations:
        rec_table.add_row(
            r["category"],
            r["severity"],
            r["issue"],
            r["rationale"],
            r["action"],
        )

    console.print(rec_table)


if __name__ == "__main__":
    main()



