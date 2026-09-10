"""Loading and conservative repairs for retained OJK extracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import MONETARY_COLUMNS


def load_bpr_panel(repo_root: str | Path) -> pd.DataFrame:
    """Load the retained conventional-BPR panel using repository-relative paths."""
    path = Path(repo_root) / "Client-BPRSK" / "BPRSK" / "FinalBPRK.csv"
    panel = pd.read_csv(path)
    panel["Kode_Bank"] = panel["Kode_Bank"].astype(str)
    panel["Tahun"] = panel["Tahun"].astype(int)
    return panel.sort_values(["Kode_Bank", "Tahun"]).reset_index(drop=True)


def harmonize_2024_monetary_units(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert 2024 monetary fields to the pre-2024 scale.

    The audit finds a common approximately 1,000-fold discontinuity for both BPR
    and BPRS monetary stocks. This explicit provisional repair is isolated here
    and flagged. It must ultimately be confirmed against OJK report unit labels.
    """
    result = panel.copy()
    mask = result["Tahun"].eq(2024)
    present = [column for column in MONETARY_COLUMNS if column in result]
    result[present] = result[present].astype(float)
    result.loc[mask, present] = result.loc[mask, present].div(1_000.0)
    result["unit_adjustment_flag"] = mask.astype("int8")
    return result


def validate_panel(panel: pd.DataFrame) -> None:
    """Raise on identity/grain violations that invalidate panel modeling."""
    required = {"Kode_Bank", "Tahun", "Nama_BPR", "Provinsi", "Kabupaten_Kota"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if panel.duplicated(["Kode_Bank", "Tahun"]).any():
        raise ValueError("Duplicate conventional BPR bank-year keys found")
    if not panel["Tahun"].between(2010, 2024).all():
        raise ValueError("Unexpected year outside audited 2010-2024 range")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while treating zero denominators and infinities as unavailable."""
    result = numerator.div(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def add_ratio_quality_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Flag severe ratio anomalies for review without silently deleting them."""
    result = panel.copy()
    checks = {
        "NPL_Neto": (0.0, 100.0),
        "ROA": (-50.0, 50.0),
        "BOPO": (0.0, 500.0),
        "LDR": (0.0, 500.0),
        "KPMM": (0.0, 500.0),
        "Cash_Ratio": (0.0, 500.0),
    }
    reasons = pd.Series("", index=result.index, dtype="object")
    for column, (lower, upper) in checks.items():
        invalid = result[column].notna() & ~result[column].between(lower, upper)
        reasons.loc[invalid] = reasons.loc[invalid].map(
            lambda existing: f"{existing}; {column} outside audit range".lstrip("; ")
        )
    result["data_quality_flag"] = reasons.ne("")
    result["data_quality_reason"] = reasons.replace("", pd.NA)
    return result
