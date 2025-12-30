# src/export_pdf.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

import matplotlib.pyplot as plt

# Pillow (optional) for chart aspect preservation in PDF
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


# Optional: your hand-written insights live here
try:
    from insights import EXEC_SUMMARY, ASSUMPTIONS_LIMITATIONS, ROADMAP
except Exception:
    EXEC_SUMMARY = []
    ASSUMPTIONS_LIMITATIONS = []
    ROADMAP = []


# Repo-relative paths (portfolio-safe, no local machine paths)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample" / "shipments_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "charts"
REPORT_PATH = PROJECT_ROOT / "reports" / "FedEx_Project_Ops_Report.pdf"


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt_pct(x) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def fmt_num(x, nd=1) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _wrap_table_cells(data, font_size=8.5, leading=10.5, wrap_cells=True):
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "CellWrap",
        parent=styles["BodyText"],
        fontSize=font_size,
        leading=leading,
        wordWrap="LTR",
        splitLongWords=False,
    )

    processed = []
    for r_i, row in enumerate(data):
        out_row = []
        for val in row:
            if val is None:
                val = ""
            s = str(val)

            # Header row stays plain strings
            if wrap_cells and r_i != 0 and len(s) > 18:
                out_row.append(Paragraph(s.replace("\n", "<br/>"), cell_style))
            else:
                out_row.append(s)
        processed.append(out_row)
    return processed


def make_table(data, doc_width, col_fracs, repeat_header=True, wrap_cells=True):
    total = sum(col_fracs) if col_fracs else 1.0
    col_fracs = [c / total for c in col_fracs]
    col_widths = [doc_width * c for c in col_fracs]

    processed = _wrap_table_cells(data, wrap_cells=wrap_cells)

    t = Table(
        processed,
        colWidths=col_widths,
        hAlign="LEFT",
        repeatRows=1 if repeat_header else 0,
        splitByRow=1,
    )

    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.Color(0.98, 0.98, 0.98)],
                ),
            ]
        )
    )
    return t


def build_data_lineage_rows():
    return [
        ["Avg Package Volume", "mean(package_volume)", "package_volume"],
        ["Capacity Utilization", "mean(package_volume / network_capacity)", "package_volume, network_capacity"],
        ["On-Time Delivery Rate", "mean(on_time_rate)", "on_time_rate"],
        ["Avg Daily Exceptions", "mean(exceptions)", "exceptions"],
        ["Labor Efficiency (pkgs/hr)", "mean(package_volume / labor_hours)", "package_volume, labor_hours"],
        ["Avg Cost per Package", "mean(cost_per_package)", "cost_per_package"],
    ]


def compute_kpis_recent14(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    recent = df.sort_values("date").tail(14)

    return {
        "avg_package_volume": recent["package_volume"].mean(),
        "capacity_utilization": (recent["package_volume"] / recent["network_capacity"]).mean(),
        "on_time_rate_avg": recent["on_time_rate"].mean(),
        "avg_exceptions": recent["exceptions"].mean(),
        "labor_efficiency": (recent["package_volume"] / recent["labor_hours"]).mean(),
        "avg_cost_per_package": recent["cost_per_package"].mean(),
    }


def _display_scenario(s: str) -> str:
    """LABOR_SHORTAGE -> LABOR SHORTAGE, etc."""
    if s is None:
        return ""
    s = str(s)
    return s.replace("_", " ")


def scenario_mix_rows(df: pd.DataFrame):
    grp = (
        df.groupby("scenario")
        .agg(
            Days=("date", "count"),
            Avg_Volume=("package_volume", "mean"),
            Avg_OnTime=("on_time_rate", "mean"),
            Avg_Exceptions=("exceptions", "mean"),
            Avg_Cost=("cost_per_package", "mean"),
        )
        .reset_index()
    )

    util = (
        df.assign(util=df["package_volume"] / df["network_capacity"])
        .groupby("scenario")["util"]
        .mean()
        .reset_index()
    )
    leff = (
        df.assign(leff=df["package_volume"] / df["labor_hours"])
        .groupby("scenario")["leff"]
        .mean()
        .reset_index()
    )

    grp = grp.merge(util, on="scenario").merge(leff, on="scenario")
    grp = grp.rename(columns={"util": "Avg_Util", "leff": "Avg_LaborEff"})

    order = ["NORMAL", "PEAK", "DISRUPTION", "LABOR_SHORTAGE"]
    grp["scenario"] = pd.Categorical(grp["scenario"], categories=order, ordered=True)
    grp = grp.sort_values("scenario")

    out = []
    for _, r in grp.iterrows():
        out.append(
            [
                _display_scenario(r["scenario"]),
                fmt_int(r["Days"]),
                fmt_int(r["Avg_Volume"]),
                fmt_pct(r["Avg_Util"]),
                fmt_pct(r["Avg_OnTime"]),
                fmt_num(r["Avg_Exceptions"], 0),
                fmt_num(r["Avg_LaborEff"], 1),
                f"${float(r['Avg_Cost']):.2f}",
            ]
        )
    return out


def top_drivers_tables(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["cap_util"] = df["package_volume"] / df["network_capacity"]

    worst_on_time = df.nsmallest(5, "on_time_rate")[
        ["date", "scenario", "on_time_rate", "package_volume", "exceptions"]
    ]
    high_ex = df.nlargest(5, "exceptions")[
        ["date", "scenario", "exceptions", "on_time_rate", "package_volume"]
    ]
    high_util = df.nlargest(5, "cap_util")[
        ["date", "scenario", "cap_util", "network_capacity", "package_volume"]
    ]
    return worst_on_time, high_ex, high_util


def deterministic_signals(metrics: dict) -> list[dict]:
    UTIL_WARN_LOW = 0.90
    UTIL_WARN_HIGH = 0.95
    UTIL_HIGH = 0.98
    ON_TIME_TARGET = 0.96
    EXC_HIGH = 250
    LAB_EFF_LOW = 75
    COST_HIGH = 5.25

    recs = []
    cu = metrics["capacity_utilization"]
    ot = metrics["on_time_rate_avg"]
    ex = metrics["avg_exceptions"]
    le = metrics["labor_efficiency"]
    cp = metrics["avg_cost_per_package"]

    if UTIL_WARN_LOW <= cu < UTIL_WARN_HIGH:
        recs.append(
            {
                "category": "SERVICE",
                "severity": "MEDIUM",
                "issue": "Capacity utilization trending toward saturation.",
                "rationale": f"Capacity utilization = {fmt_pct(cu)} in warning band ({fmt_pct(UTIL_WARN_LOW)}–{fmt_pct(UTIL_WARN_HIGH)}).",
                "action": "Monitor volume growth and prepare contingency capacity plans.",
            }
        )

    if cu >= UTIL_HIGH:
        recs.append(
            {
                "category": "CAPACITY",
                "severity": "HIGH",
                "issue": "Network operating near or above capacity.",
                "rationale": f"Capacity utilization = {fmt_pct(cu)} ≥ {fmt_pct(UTIL_HIGH)}.",
                "action": "Add temporary capacity or reroute volume to underutilized hubs.",
            }
        )

    if ot < ON_TIME_TARGET:
        recs.append(
            {
                "category": "SERVICE",
                "severity": "HIGH",
                "issue": "On-time delivery performance below target.",
                "rationale": f"On-time rate = {fmt_pct(ot)} < {fmt_pct(ON_TIME_TARGET)} target.",
                "action": "Investigate hub bottlenecks and last-mile staffing coverage.",
            }
        )

    if ex > EXC_HIGH:
        recs.append(
            {
                "category": "RISK",
                "severity": "MEDIUM",
                "issue": "Exception volume elevated.",
                "rationale": f"Avg exceptions = {fmt_num(ex, 0)} > {EXC_HIGH}.",
                "action": "Identify top exception drivers and deploy short-term process controls.",
            }
        )

    if le < LAB_EFF_LOW:
        recs.append(
            {
                "category": "LABOR",
                "severity": "LOW",
                "issue": "Labor productivity below expected baseline.",
                "rationale": f"Labor efficiency = {fmt_num(le, 1)} < {LAB_EFF_LOW} pkgs/hr.",
                "action": "Rebalance shifts to better align staffing with demand peaks.",
            }
        )

    if cp > COST_HIGH:
        recs.append(
            {
                "category": "COST",
                "severity": "MEDIUM",
                "issue": "Cost per package above expected operating range.",
                "rationale": f"Cost per package = ${float(cp):.2f} > ${COST_HIGH:.2f}.",
                "action": "Review linehaul, labor mix, and routing efficiency for cost reduction opportunities.",
            }
        )

    if not recs:
        recs.append(
            {
                "category": "STATUS",
                "severity": "LOW",
                "issue": "Operations within expected performance ranges.",
                "rationale": "No KPI thresholds breached in this window.",
                "action": "Maintain current plan and continue monitoring.",
            }
        )

    return recs


def _save_line_chart(df, ycol, title, out_path: Path):
    plt.figure(figsize=(10, 3.2))
    plt.plot(df["date"], df[ycol])
    plt.title(title, fontsize=11)
    plt.xlabel("Date")
    plt.ylabel(ycol)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_charts(df: pd.DataFrame) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["cap_util"] = df["package_volume"] / df["network_capacity"]
    df["labor_eff"] = df["package_volume"] / df["labor_hours"]

    d = df.tail(60)

    paths = {
        "volume": OUTPUT_DIR / "chart_volume.png",
        "util": OUTPUT_DIR / "chart_util.png",
        "ontime": OUTPUT_DIR / "chart_ontime.png",
        "exceptions": OUTPUT_DIR / "chart_exceptions.png",
        "labor_eff": OUTPUT_DIR / "chart_labor_eff.png",
        "cost": OUTPUT_DIR / "chart_cost.png",
    }

    _save_line_chart(d, "package_volume", "Daily Package Volume (last 60 days)", paths["volume"])
    _save_line_chart(d, "cap_util", "Capacity Utilization (last 60 days)", paths["util"])
    _save_line_chart(d, "on_time_rate", "On-Time Rate (last 60 days)", paths["ontime"])
    _save_line_chart(d, "exceptions", "Daily Exceptions (last 60 days)", paths["exceptions"])
    _save_line_chart(d, "labor_eff", "Labor Efficiency pkgs/hr (last 60 days)", paths["labor_eff"])
    _save_line_chart(d, "cost_per_package", "Cost per Package (last 60 days)", paths["cost"])

    return paths


def add_chart(story, img_path: Path, width):
    if not img_path.exists():
        return

    if PILImage is not None:
        im = PILImage.open(img_path)
        w, h = im.size
        aspect = h / float(w)
        story.append(Image(str(img_path), width=width, height=width * aspect))
    else:
        story.append(Image(str(img_path), width=width, height=2.4 * inch))


def make_kpi_table(kpi_rows, doc_width):
    """KPI table with traffic-light coloring on the Value column."""
    t = make_table(kpi_rows, doc_width, col_fracs=[0.65, 0.35], wrap_cells=False)
    return t


def apply_kpi_colors(table_obj, metrics: dict):
    """Applies background colors to KPI Value cells based on thresholds."""

    # Threshold policy (aligned to deterministic signals)
    UTIL_WARN_LOW = 0.90
    UTIL_WARN_HIGH = 0.95
    ON_TIME_TARGET = 0.96
    EXC_HIGH = 250
    LAB_EFF_LOW = 75
    COST_HIGH = 5.25

    GREEN = colors.Color(0.80, 0.93, 0.80)
    YELLOW = colors.Color(1.00, 0.96, 0.70)
    RED = colors.Color(1.00, 0.80, 0.80)

    styles = []

    # Capacity Utilization (row 2) - lower is better
    cu = float(metrics["capacity_utilization"])
    if cu < UTIL_WARN_LOW:
        styles.append(("BACKGROUND", (1, 2), (1, 2), GREEN))
    elif cu < UTIL_WARN_HIGH:
        styles.append(("BACKGROUND", (1, 2), (1, 2), YELLOW))
    else:
        styles.append(("BACKGROUND", (1, 2), (1, 2), RED))

    # On-Time (row 3) - higher is better
    ot = float(metrics["on_time_rate_avg"])
    if ot >= ON_TIME_TARGET:
        styles.append(("BACKGROUND", (1, 3), (1, 3), GREEN))
    elif ot >= (ON_TIME_TARGET - 0.01):
        styles.append(("BACKGROUND", (1, 3), (1, 3), YELLOW))
    else:
        styles.append(("BACKGROUND", (1, 3), (1, 3), RED))

    # Exceptions (row 4) - lower is better
    ex = float(metrics["avg_exceptions"])
    if ex <= EXC_HIGH:
        styles.append(("BACKGROUND", (1, 4), (1, 4), GREEN))
    elif ex <= EXC_HIGH * 1.15:
        styles.append(("BACKGROUND", (1, 4), (1, 4), YELLOW))
    else:
        styles.append(("BACKGROUND", (1, 4), (1, 4), RED))

    # Labor Efficiency (row 5) - higher is better
    le = float(metrics["labor_efficiency"])
    if le >= LAB_EFF_LOW:
        styles.append(("BACKGROUND", (1, 5), (1, 5), GREEN))
    elif le >= LAB_EFF_LOW * 0.95:
        styles.append(("BACKGROUND", (1, 5), (1, 5), YELLOW))
    else:
        styles.append(("BACKGROUND", (1, 5), (1, 5), RED))

    # Cost per package (row 6) - lower is better
    cp = float(metrics["avg_cost_per_package"])
    if cp <= COST_HIGH:
        styles.append(("BACKGROUND", (1, 6), (1, 6), GREEN))
    elif cp <= COST_HIGH * 1.05:
        styles.append(("BACKGROUND", (1, 6), (1, 6), YELLOW))
    else:
        styles.append(("BACKGROUND", (1, 6), (1, 6), RED))

    if styles:
        table_obj.setStyle(TableStyle(styles))


def export():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file: {DATA_PATH}\n"
            f"Expected a synthetic sample CSV at: data/sample/shipments_sample.csv"
        )

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    days = len(df)
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    metrics = compute_kpis_recent14(df)
    signals = deterministic_signals(metrics)
    worst_on_time, high_ex, high_util = top_drivers_tables(df)

    chart_paths = build_charts(df)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    h_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12)

    flow_style = ParagraphStyle(
        "FlowMono",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
        spaceBefore=6,
        spaceAfter=6,
    )
    italic_h3 = ParagraphStyle("H3Italic", parent=styles["Heading3"], fontSize=10, italic=True)

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="FedEx Product Ops — Decision Support Report",
    )

    story = []
    W = doc.width

    # Page 1 — Executive Brief
    story.append(Paragraph("FedEx Product Ops — Decision Support Report", title_style))
    story.append(Paragraph("Executive Brief :", h_style))

    if EXEC_SUMMARY:
        for b in EXEC_SUMMARY:
            story.append(Paragraph(b, body))
    else:
        story.append(
            Paragraph(
                "Purpose: Convert daily network ops data into an auditable KPI summary + decision signals with recommended actions.",
                body,
            )
        )
        story.append(
            Paragraph(
                "Data: Portfolio-safe synthetic dataset shaped like an ops feed. Metrics are computed directly from source columns (see Data Lineage).",
                body,
            )
        )
        story.append(
            Paragraph(
                f"Current performance snapshot: Capacity utilization = {fmt_pct(metrics['capacity_utilization'])}; "
                f"on-time rate = {fmt_pct(metrics['on_time_rate_avg'])}; "
                f"exceptions = {fmt_num(metrics['avg_exceptions'], 1)}/day; "
                f"labor efficiency = {fmt_num(metrics['labor_efficiency'], 1)} pkgs/hr; "
                f"cost per package = ${float(metrics['avg_cost_per_package']):.2f}.",
                body,
            )
        )
        story.append(
            Paragraph(
                "What’s driving risk: Disruption scenarios are typically the main driver of service degradation; confirm using the Top Drivers tables and KPI trend charts.",
                body,
            )
        )
        story.append(
            Paragraph(
                "Immediate actions (7 days): Execute capacity relief + investigate the worst on-time days (hub staffing, sort constraints, last-mile coverage).",
                body,
            )
        )
        story.append(
            Paragraph(
                "Next actions (30 days): Add hub/region drilldowns and alerting so leadership can see where issues originate, not just that they exist.",
                body,
            )
        )

    story.append(Spacer(1, 18))
    story.append(Paragraph("Sriyan S", body))

    # Page 2 — Architecture + Flow + Context + Scenario Mix
    story.append(PageBreak())
    story.append(Paragraph("System Architecture — How This Tool Works", h_style))
    story.append(
        Paragraph(
            "This report is generated by a small decision-support pipeline designed to be auditable and easy to extend.",
            body,
        )
    )

    arch = [
        ["Layer / Module", "What it does"],
        ["1) Data Layer", f"Input dataset: {DATA_PATH.as_posix()} (Synthetic dataset structured like an ops feed)"],
        ["2) Metrics Engine", "Aggregates KPIs (volume, utilization, on-time, exceptions, labor efficiency, cost)"],
        ["3) Rules Engine (deterministic_signals)", "Evaluates KPIs vs thresholds and produces decision signals (category, severity, issue, rationale, action)"],
        ["4) Report Generator (export_pdf.py)", "Builds the PDF: context + scenario mix + KPI summary + data lineage + trends + drivers + signals + actions"],
        ["5) Outputs", f"{REPORT_PATH.name} + charts in {OUTPUT_DIR.as_posix()}/"],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(arch, W, col_fracs=[0.30, 0.70], wrap_cells=True))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Flow (high-level)", italic_h3))

    flow_text = (
        f"{DATA_PATH.as_posix()}<br/>"
        "↓<br/>"
        "export_pdf.py (compute_kpis_recent14 + deterministic_signals)<br/>"
        "↓<br/>"
        "export_pdf.py (assemble report + charts)<br/>"
        "↓<br/>"
        f"{REPORT_PATH.name}"
    )
    story.append(Paragraph(flow_text, flow_style))

    story.append(
        Paragraph(
            "Design choice: rules stay deterministic and auditable (threshold policy), while the system can later add an optional AI summary layer without changing the underlying KPI calculations.",
            body,
        )
    )

    story.append(Spacer(1, 12))
    story.append(Paragraph("Report Context", h_style))
    ctx = [
        ["Field", "Value"],
        ["Data Source", DATA_PATH.as_posix()],
        ["Days Analyzed", str(days)],
        ["Date Range", f"{date_min} → {date_max}"],
    ]
    story.append(make_table(ctx, W, col_fracs=[0.28, 0.72], wrap_cells=True))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Scenario Mix (Counts + Avg Metrics)", h_style))

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Scenario Definitions",
            ParagraphStyle("ScenarioDefTitle", parent=styles["Heading3"], fontSize=10),
        )
    )

    scenario_defs = [
        ["Scenario", "Definition"],
        ["NORMAL", "Baseline operating conditions with expected demand and staffing."],
        ["PEAK", "Elevated demand periods such as seasonal or promotional volume spikes."],
        ["DISRUPTION", "Network disturbances (e.g., weather, facility outages, transport delays)."],
        ["LABOR SHORTAGE", "Reduced labor availability impacting throughput or service levels."],
    ]
    story.append(make_table(scenario_defs, W, col_fracs=[0.25, 0.75], wrap_cells=True))
    story.append(Spacer(1, 12))

    mix = scenario_mix_rows(df)
    mix1 = [["Scenario", "Days", "Avg Volume", "Avg Util"]] + [[r[0], r[1], r[2], r[3]] for r in mix]
    mix2 = [["Scenario", "Avg On-Time", "Avg Exceptions", "Avg Labor Eff (pkgs/hr)", "Avg Cost"]] + [
        [r[0], r[4], r[5], r[6], r[7]] for r in mix
    ]

    story.append(make_table(mix1, W, col_fracs=[0.34, 0.14, 0.27, 0.25], wrap_cells=False))
    story.append(Spacer(1, 10))
    story.append(make_table(mix2, W, col_fracs=[0.24, 0.18, 0.18, 0.26, 0.14], wrap_cells=False))

    # Page 3 — KPI Summary + Data Lineage + Signals
    story.append(PageBreak())
    story.append(Paragraph("KPI Summary (recent 14 days)", h_style))
    kpi_rows = [
        ["Metric", "Value"],
        ["Avg Package Volume", fmt_int(metrics["avg_package_volume"])],
        ["Capacity Utilization", fmt_pct(metrics["capacity_utilization"])],
        ["On-Time Delivery Rate", fmt_pct(metrics["on_time_rate_avg"])],
        ["Avg Daily Exceptions", fmt_num(metrics["avg_exceptions"], 0)],
        ["Labor Efficiency (pkgs/hr)", fmt_num(metrics["labor_efficiency"], 1)],
        ["Avg Cost per Package", f"${float(metrics['avg_cost_per_package']):.2f}"],
    ]

    kpi_table = make_kpi_table(kpi_rows, W)
    apply_kpi_colors(kpi_table, metrics)
    story.append(kpi_table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Data Lineage — How Metrics Are Calculated", h_style))
    lineage = [["Metric", "Formula", f"Source Columns ({DATA_PATH.name})"]] + build_data_lineage_rows()
    story.append(make_table(lineage, W, col_fracs=[0.28, 0.34, 0.38], wrap_cells=True))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Decision Signals", h_style))

    sig1 = [["Category", "Severity", "Issue"]]
    sig2 = [["Rationale", "Action"]]
    for s in signals:
        sig1.append([s["category"], s["severity"], s["issue"]])
        sig2.append([s["rationale"], s["action"]])

    story.append(make_table(sig1, W, col_fracs=[0.20, 0.18, 0.62], wrap_cells=True))
    story.append(Spacer(1, 10))
    story.append(make_table(sig2, W, col_fracs=[0.52, 0.48], wrap_cells=True))

    # Page 4 — Charts
    story.append(PageBreak())
    story.append(Paragraph("KPI Trends (last 60 days)", h_style))
    story.append(Paragraph("These charts visualize the same source columns used in the KPI calculations.", body))
    story.append(Spacer(1, 8))

    for key, label in [
        ("volume", "Daily Package Volume"),
        ("util", "Capacity Utilization"),
        ("ontime", "On-Time Rate"),
        ("exceptions", "Daily Exceptions"),
        ("labor_eff", "Labor Efficiency"),
        ("cost", "Cost per Package"),
    ]:
        p = chart_paths.get(key)
        if p and p.exists():
            story.append(Paragraph(label, ParagraphStyle(f"Chart_{key}", parent=styles["Heading3"], fontSize=10)))
            add_chart(story, p, W)
            story.append(Spacer(1, 10))

    # Page 5 — Top Drivers
    story.append(PageBreak())
    story.append(Paragraph("Top Drivers — Dates to Investigate", h_style))
    story.append(Paragraph("Specific dates most responsible for poor performance or cost.", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Worst On-Time Days (bottom 5)", ParagraphStyle("Sub", parent=styles["Heading3"], fontSize=10)))
    w = [["Date", "Scenario", "On-Time", "Volume", "Exceptions"]]
    for _, r in worst_on_time.iterrows():
        w.append(
            [
                str(pd.to_datetime(r["date"]).date()),
                _display_scenario(r["scenario"]),
                fmt_pct(r["on_time_rate"]),
                fmt_int(r["package_volume"]),
                fmt_int(r["exceptions"]),
            ]
        )
    story.append(make_table(w, W, col_fracs=[0.24, 0.20, 0.16, 0.20, 0.20], wrap_cells=False))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Highest Exception Days (top 5)", ParagraphStyle("Sub2", parent=styles["Heading3"], fontSize=10)))
    e = [["Date", "Scenario", "Exceptions", "On-Time", "Volume"]]
    for _, r in high_ex.iterrows():
        e.append(
            [
                str(pd.to_datetime(r["date"]).date()),
                _display_scenario(r["scenario"]),
                fmt_int(r["exceptions"]),
                fmt_pct(r["on_time_rate"]),
                fmt_int(r["package_volume"]),
            ]
        )
    story.append(make_table(e, W, col_fracs=[0.24, 0.20, 0.19, 0.16, 0.21], wrap_cells=False))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph("Highest Capacity Utilization Days (top 5)", ParagraphStyle("Sub3", parent=styles["Heading3"], fontSize=10))
    )
    u = [["Date", "Scenario", "Cap Util", "Capacity", "Volume"]]
    for _, r in high_util.iterrows():
        u.append(
            [
                str(pd.to_datetime(r["date"]).date()),
                _display_scenario(r["scenario"]),
                fmt_pct(r["cap_util"]),
                fmt_int(r["network_capacity"]),
                fmt_int(r["package_volume"]),
            ]
        )
    story.append(make_table(u, W, col_fracs=[0.24, 0.20, 0.16, 0.20, 0.20], wrap_cells=False))

    # Page 6 — Next Actions + Roadmap
    story.append(PageBreak())
    story.append(Paragraph("Next Actions (7 Days / 30 Days)", h_style))

    next7 = [
        "Validate the bottom-5 on-time days with lane/hub notes and confirm root-cause category (hub constraint vs last-mile staffing vs weather).",
        "Run a 1-week capacity relief plan: reroute volume to underutilized hubs or add temporary linehaul capacity during peak windows.",
        "Create an exception driver Pareto (top reason codes) and deploy 1–2 process controls to reduce the highest-frequency exception types.",
        "Confirm if cost/pack spikes correlate with disruption days (overtime, re-handling, re-routes).",
    ]
    next30 = [
        "Add feature-level breakdowns (by region/hub/lane) to pinpoint where service degradation is concentrated.",
        "Introduce an alerting threshold policy (targets per KPI) and log signal history to prove the system catches issues early.",
        "Build a simple scenario impact estimator: predicted on-time/cost deltas under PEAK vs DISRUPTION conditions.",
        "Optionally add an AI summary layer: generate a short exec brief from metrics + drivers + recommendations (keep deterministic logic intact).",
    ]

    story.append(Paragraph("<b>7-Day</b>", body))
    for b in next7:
        story.append(Paragraph(f"• {b}", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>30-Day</b>", body))
    for b in next30:
        story.append(Paragraph(f"• {b}", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Roadmap — What This Tool Would Do Next (Product Ops Enhancements)", h_style))
    if ROADMAP:
        for b in ROADMAP:
            story.append(Paragraph(f"• {b}", body))
    else:
        story.append(Paragraph("• Drilldowns (hub/region/lane) to localize root cause.", body))
        story.append(Paragraph("• Driver attribution for exceptions and service misses.", body))
        story.append(Paragraph("• Early-warning alerts (trend-to-threshold) instead of threshold-only.", body))
        story.append(Paragraph("• Policy tuning via config profiles (PEAK vs NORMAL).", body))
        story.append(Paragraph("• Optional AI narrative layer that does not alter KPI calculations.", body))

    doc.build(story)
    print(f"PDF generated: {REPORT_PATH}")


if __name__ == "__main__":
    export()

