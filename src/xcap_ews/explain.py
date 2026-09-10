"""Model-agnostic explanations tied to actual fitted predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "NPL_Neto": "high NPL net",
    "ROA": "profitability level",
    "BOPO": "operating efficiency pressure",
    "LDR": "loan-to-deposit position",
    "KPMM": "capital adequacy level",
    "Cash_Ratio": "cash liquidity position",
    "log_assets": "asset scale",
    "log_loans": "loan scale",
    "equity_to_assets": "equity-to-assets position",
    "provision_to_loans": "provision coverage",
    "loans_to_deposits_calc": "funding pressure",
    "deposit_share": "deposit funding mix",
    "delta_npl_net_1y": "one-year NPL net movement",
    "delta_roa_1y": "one-year ROA movement",
    "delta_bopo_1y": "one-year BOPO movement",
    "delta_ldr_1y": "one-year LDR movement",
    "delta_kpmm_1y": "one-year KPMM movement",
    "asset_growth_1y": "asset growth",
    "loan_growth_1y": "loan growth",
    "deposit_growth_1y": "deposit growth",
    "loan_minus_deposit_growth": "loan growth versus deposit growth",
    "npl_net_slope_3y": "three-year NPL net trend",
    "roa_slope_3y": "three-year ROA trend",
    "bopo_slope_3y": "three-year BOPO trend",
    "kpmm_slope_3y": "three-year capital trend",
    "npl_net_vs_history": "NPL net versus bank history",
    "roa_vs_history": "ROA versus bank history",
}


def local_median_replacement_contributions(
    model: object,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate local risk contributions by replacing one feature at a time.

    A positive contribution means the observed feature raises predicted risk
    relative to replacing it with the training median. This is an auditable,
    model-agnostic perturbation explanation, not a causal effect.
    """
    baseline = model.predict_proba(frame)[:, 1]
    medians = reference.median(numeric_only=True)
    contributions = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for column in frame.columns:
        counterfactual = frame.copy()
        counterfactual[column] = medians[column]
        contributions[column] = baseline - model.predict_proba(counterfactual)[:, 1]

    def describe(row: pd.Series) -> str:
        positive = row[row.gt(0)].sort_values(ascending=False).head(top_n)
        if positive.empty:
            positive = row.abs().sort_values(ascending=False).head(top_n)
        return "; ".join(
            f"{FEATURE_LABELS.get(feature, feature)} ({value:+.3f})"
            for feature, value in positive.items()
        )

    explanations = contributions.apply(describe, axis=1).rename("top_risk_drivers").to_frame()
    global_importance = contributions.abs().mean().sort_values(ascending=False).rename("mean_absolute_probability_contribution").to_frame()
    return explanations, global_importance
