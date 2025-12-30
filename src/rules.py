# src/rules.py
from __future__ import annotations


def generate_recommendations(metrics: dict) -> list[dict]:
    """
    Deterministic, auditable rules engine.
    Input: metrics dict (from metrics.compute_metrics)
    Output: list of recommendations:
      {category, severity, issue, action, rationale}
    """
    recommendations: list[dict] = []

    # -------------------------
    # Thresholds (tune as needed)
    # -------------------------
    CAPACITY_HIGH = 0.95
    UTILIZATION_WARNING = 0.90
    ON_TIME_LOW = 0.96
    EXCEPTIONS_HIGH = 250
    LABOR_EFFICIENCY_LOW = 75
    COST_HIGH = 5.25

    # Defensive casts (keeps behavior stable even if numbers arrive as strings)
    cu = float(metrics.get("capacity_utilization", 0.0))
    ot = float(metrics.get("on_time_rate_avg", 0.0))
    ex = float(metrics.get("avg_exceptions", 0.0))
    le = float(metrics.get("labor_efficiency", 0.0))
    cp = float(metrics.get("avg_cost_per_package", 0.0))

    # -------------------------
    # SERVICE – early warning (near saturation but not critical)
    # -------------------------
    if cu >= UTILIZATION_WARNING and cu < CAPACITY_HIGH:
        recommendations.append({
            "category": "SERVICE",
            "severity": "MEDIUM",
            "issue": "Capacity utilization trending toward saturation.",
            "action": "Monitor volume growth and prepare contingency capacity plans.",
            "rationale": f"Capacity utilization = {cu*100:.1f}% in warning band ({UTILIZATION_WARNING*100:.1f}%–{CAPACITY_HIGH*100:.1f}%)",
        })

    # -------------------------
    # CAPACITY – hard limit
    # -------------------------
    if cu >= CAPACITY_HIGH:
        recommendations.append({
            "category": "CAPACITY",
            "severity": "HIGH",
            "issue": "Network operating near or above capacity.",
            "action": "Add temporary capacity or reroute volume to underutilized hubs.",
            "rationale": f"Capacity utilization = {cu*100:.1f}% ≥ {CAPACITY_HIGH*100:.1f}% threshold",
        })

    # -------------------------
    # SERVICE – SLA risk (on-time)
    # -------------------------
    if ot < ON_TIME_LOW:
        recommendations.append({
            "category": "SERVICE",
            "severity": "HIGH",
            "issue": "On-time delivery performance below target.",
            "action": "Investigate hub bottlenecks and last-mile staffing coverage.",
            "rationale": f"On-time rate = {ot*100:.1f}% < {ON_TIME_LOW*100:.1f}% target",
        })

    # -------------------------
    # COST
    # -------------------------
    if cp > COST_HIGH:
        recommendations.append({
            "category": "COST",
            "severity": "MEDIUM",
            "issue": "Cost per package above expected operating range.",
            "action": "Review linehaul, labor mix, and routing efficiency for cost reduction opportunities.",
            "rationale": f"Avg cost per package = ${cp:.2f} > ${COST_HIGH:.2f} threshold",
        })

    # -------------------------
    # RISK / EXCEPTIONS
    # -------------------------
    if ex > EXCEPTIONS_HIGH:
        recommendations.append({
            "category": "RISK",
            "severity": "MEDIUM",
            "issue": "Exception volume elevated.",
            "action": "Identify top exception drivers and deploy short-term process controls.",
            "rationale": f"Avg daily exceptions = {ex:.1f} > {EXCEPTIONS_HIGH} threshold",
        })

    # -------------------------
    # LABOR
    # -------------------------
    if le < LABOR_EFFICIENCY_LOW:
        recommendations.append({
            "category": "LABOR",
            "severity": "LOW",
            "issue": "Labor productivity below expected baseline.",
            "action": "Rebalance shifts to better align staffing with demand peaks.",
            "rationale": f"Labor efficiency = {le:.1f} pkgs/hr < {LABOR_EFFICIENCY_LOW} baseline",
        })

    # -------------------------
    # STATUS fallback
    # -------------------------
    if not recommendations:
        recommendations.append({
            "category": "STATUS",
            "severity": "LOW",
            "issue": "Operations within expected performance ranges.",
            "action": "Maintain current operating plan and continue monitoring.",
            "rationale": "No thresholds were breached in the selected window.",
        })

    return recommendations
