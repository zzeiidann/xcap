"""Forward financial-deterioration proxy construction."""

from __future__ import annotations

import pandas as pd

from .config import TARGET_THRESHOLDS


def build_deterioration_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach an auditable t-to-t+1 multi-indicator deterioration target."""
    result = panel.sort_values(["Kode_Bank", "Tahun"]).copy()
    grouped = result.groupby("Kode_Bank", sort=False)
    result["target_year"] = grouped["Tahun"].shift(-1)
    consecutive = result["target_year"].eq(result["Tahun"] + 1)
    component_columns: list[str] = []
    future_complete = pd.Series(True, index=result.index)

    for ratio, threshold in TARGET_THRESHOLDS.items():
        future = grouped[ratio].shift(-1)
        change = future - result[ratio]
        future_complete &= future.notna()
        result[f"target_change__{ratio}"] = change
        component = f"target_component__{ratio}"
        result[component] = change.ge(threshold) if threshold > 0 else change.le(threshold)
        component_columns.append(component)

    current_complete = result[list(TARGET_THRESHOLDS)].notna().all(axis=1)
    eligible = consecutive & current_complete & future_complete
    result["target_eligible"] = eligible
    result["target_adverse_component_count"] = result[component_columns].sum(axis=1).where(eligible)
    result["next_period_deterioration"] = (
        result["target_adverse_component_count"].ge(2).astype("Int64").where(eligible)
    )
    return result
