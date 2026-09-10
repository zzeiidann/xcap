"""Out-of-time validation and risk-oriented evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TemporalFold:
    validation_target_year: int
    minimum_train_target_year: int = 2016


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
    return float(np.max(true_positive_rate - false_positive_rate))


def calibration_summary(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs")
    calibration_model.fit(logit.reshape(-1, 1), y_true)
    return float(calibration_model.intercept_[0]), float(calibration_model.coef_[0, 0])


def score_metrics(y_true: np.ndarray, probability: np.ndarray, review_fraction: float = 0.20) -> dict[str, float]:
    cutoff = np.quantile(probability, 1 - review_fraction)
    selected = probability >= cutoff
    event_rate = float(np.mean(y_true))
    top_rate = float(np.mean(y_true[selected])) if selected.any() else np.nan
    calibration_intercept, calibration_slope = calibration_summary(y_true, probability)
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "ks": ks_statistic(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "recall_at_20pct": recall_score(y_true, selected),
        "precision_at_20pct": precision_score(y_true, selected, zero_division=0),
        "lift_at_20pct": top_rate / event_rate if event_rate else np.nan,
        "event_rate": event_rate,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def logistic_pipeline(estimator: object) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def walk_forward_validate(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, object],
    validation_years: list[int],
    minimum_train_year: int = 2016,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model/ablation on expanding target-year folds."""
    eligible = frame.dropna(subset=["next_period_deterioration", "target_year"]).copy()
    eligible["next_period_deterioration"] = eligible["next_period_deterioration"].astype(int)
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for validation_year in validation_years:
        train_mask = eligible["target_year"].between(minimum_train_year, validation_year - 1)
        valid_mask = eligible["target_year"].eq(validation_year)
        train = eligible.loc[train_mask]
        valid = eligible.loc[valid_mask]
        if train.empty or valid.empty:
            continue
        for feature_set, columns in feature_sets.items():
            for model_name, estimator in models.items():
                fitted = clone(estimator)
                fitted.fit(train[columns], train["next_period_deterioration"])
                probability = fitted.predict_proba(valid[columns])[:, 1]
                row = score_metrics(valid["next_period_deterioration"].to_numpy(), probability)
                row.update({
                    "validation_target_year": validation_year,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_train": len(train),
                    "n_validation": len(valid),
                })
                metrics.append(row)
                prediction = valid[["Kode_Bank", "Nama_BPR", "Tahun", "target_year", "next_period_deterioration"]].copy()
                prediction["probability"] = probability
                prediction["feature_set"] = feature_set
                prediction["model"] = model_name
                predictions.append(prediction)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def decile_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build high-to-low risk deciles and lift within each validation year."""
    result = predictions.copy()
    result["risk_decile"] = result.groupby("target_year")["probability"].transform(
        lambda values: 10 - pd.qcut(values.rank(method="first"), 10, labels=False)
    )
    overall = result["next_period_deterioration"].mean()
    table = result.groupby("risk_decile", as_index=False).agg(
        observations=("Kode_Bank", "size"),
        average_probability=("probability", "mean"),
        deterioration_rate=("next_period_deterioration", "mean"),
        captured_events=("next_period_deterioration", "sum"),
    )
    table["lift"] = table["deterioration_rate"] / overall
    return table.sort_values("risk_decile")
