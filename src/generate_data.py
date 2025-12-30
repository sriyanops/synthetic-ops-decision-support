import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _apply_scenario(base: dict, scenario: str) -> dict:
    """
    Adjusts the base daily values to simulate different operating conditions.
    Scenarios are designed to trigger different rule categories realistically.
    """
    d = dict(base)

    if scenario == "NORMAL":
        # Small noise around baseline
        d["package_volume"] *= random.uniform(0.97, 1.03)
        d["on_time_rate"] *= random.uniform(0.995, 1.005)
        d["exceptions"] *= random.uniform(0.95, 1.05)
        d["labor_hours"] *= random.uniform(0.98, 1.02)
        d["cost_per_package"] *= random.uniform(0.98, 1.02)

    elif scenario == "PEAK":
        # Volume surges, capacity tighter, labor stretched, cost slightly up
        d["package_volume"] *= random.uniform(1.10, 1.25)
        d["on_time_rate"] *= random.uniform(0.985, 0.998)
        d["exceptions"] *= random.uniform(1.05, 1.20)
        d["labor_hours"] *= random.uniform(1.05, 1.15)
        d["cost_per_package"] *= random.uniform(1.02, 1.08)

    elif scenario == "DISRUPTION":
        # Weather/ops disruption: on-time drops, exceptions spike, cost spikes
        d["package_volume"] *= random.uniform(0.95, 1.10)
        d["on_time_rate"] *= random.uniform(0.92, 0.965)
        d["exceptions"] *= random.uniform(1.40, 2.20)
        d["labor_hours"] *= random.uniform(1.05, 1.20)
        d["cost_per_package"] *= random.uniform(1.10, 1.35)

    elif scenario == "LABOR_SHORTAGE":
        # Understaffed: labor hours may drop vs need, efficiency drops, exceptions rise, on-time slightly down
        d["package_volume"] *= random.uniform(0.98, 1.08)
        d["on_time_rate"] *= random.uniform(0.955, 0.985)
        d["exceptions"] *= random.uniform(1.15, 1.60)
        d["labor_hours"] *= random.uniform(0.88, 0.97)  # fewer hours available
        d["cost_per_package"] *= random.uniform(1.05, 1.15)

    else:
        # Unknown scenario -> treat as normal
        return _apply_scenario(base, "NORMAL")

    return d


def generate_synthetic_fedex_ops_data(
    days: int = 60,
    seed: int = 42,
    scenario_plan: str = "MIXED",
) -> pd.DataFrame:
    """
    Generates a realistic-ish synthetic dataset for product ops decision support.

    scenario_plan:
      - "MIXED": mixes NORMAL / PEAK / DISRUPTION / LABOR_SHORTAGE across the period
      - "NORMAL_ONLY": all days are normal
      - "PEAK_ONLY": all days peak
      - "DISRUPTION_ONLY": all days disruption
      - "LABOR_ONLY": all days labor shortage
    """
    random.seed(seed)
    np.random.seed(seed)

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days - 1)

    # Baselines (tuneable)
    baseline = {
        "network_capacity": 140_000,     # capacity per day
        "package_volume": 125_000,       # average daily volume
        "on_time_rate": 0.965,           # baseline on-time
        "exceptions": 160,               # baseline exceptions/day
        "labor_hours": 1500,             # baseline labor hours/day
        "cost_per_package": 4.65,        # baseline cost
    }

    # Build scenario schedule
    scenario_choices = ["NORMAL", "PEAK", "DISRUPTION", "LABOR_SHORTAGE"]

    rows = []
    for i in range(days):
        day = start_date + timedelta(days=i)

        # Determine scenario
        if scenario_plan == "NORMAL_ONLY":
            scenario = "NORMAL"
        elif scenario_plan == "PEAK_ONLY":
            scenario = "PEAK"
        elif scenario_plan == "DISRUPTION_ONLY":
            scenario = "DISRUPTION"
        elif scenario_plan == "LABOR_ONLY":
            scenario = "LABOR_SHORTAGE"
        else:
            # MIXED: bias toward normal but insert blocks of stress
            # Every ~3 weeks: a short disruption block; weekends slightly more "peak"
            dow = day.weekday()  # 0=Mon ... 6=Sun
            roll = random.random()

            # Short disruption block: ~3-4 days every ~21 days
            if (i % 21) in (14, 15, 16) and roll < 0.85:
                scenario = "DISRUPTION"
            # Peak tendency on Mon + Fri
            elif dow in (0, 4) and roll < 0.55:
                scenario = "PEAK"
            # Labor shortage pops up occasionally
            elif roll < 0.12:
                scenario = "LABOR_SHORTAGE"
            else:
                scenario = "NORMAL"

        # Base with noise
        base_day = dict(baseline)

        # Slight seasonality/weekly drift
        weekly_factor = 1.0 + 0.03 * np.sin(2 * np.pi * i / 7.0)
        base_day["package_volume"] *= weekly_factor

        # Apply scenario adjustments
        d = _apply_scenario(base_day, scenario)

        # Derived
        capacity = baseline["network_capacity"] * random.uniform(0.98, 1.02)  # small daily capacity noise
        volume = max(0, int(d["package_volume"]))
        on_time = float(np.clip(d["on_time_rate"], 0.80, 0.995))
        exceptions = max(0, int(d["exceptions"]))
        labor_hours = max(1.0, float(d["labor_hours"]))  # avoid divide-by-zero
        cost = max(0.50, float(d["cost_per_package"]))

        rows.append({
            "date": str(day),
            "scenario": scenario,
            "network_capacity": int(capacity),
            "package_volume": volume,
            "on_time_rate": on_time,
            "exceptions": exceptions,
            "labor_hours": labor_hours,
            "cost_per_package": cost,
        })

    return pd.DataFrame(rows)


def main():
    df = generate_synthetic_fedex_ops_data(days=60, seed=42, scenario_plan="MIXED")

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "fedex_product_ops.csv")
    df.to_csv(out_path, index=False)

    print("Synthetic FedEx product ops data generated:")
    print(out_path)
    print("\nScenario counts:")
    print(df["scenario"].value_counts().to_string())


if __name__ == "__main__":
    main()

