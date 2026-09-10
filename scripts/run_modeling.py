"""Run the complete conventional-BPR early-warning experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from xcap_ews.config import (
    CURRENT_FEATURES,
    PEER_FEATURES,
    RANDOM_SEED,
    SPATIAL_FEATURES,
    TREND_FEATURES,
)
from xcap_ews.data import add_ratio_quality_flags, harmonize_2024_monetary_units, load_bpr_panel, validate_panel
from xcap_ews.evaluation import (
    decile_table,
    ks_statistic,
    logistic_pipeline,
    walk_forward_validate,
)
from xcap_ews.explain import local_median_replacement_contributions
from xcap_ews.features import build_feature_panel
from xcap_ews.spatial import knn_weights, morans_i_permutation_test
from xcap_ews.target import build_deterioration_target
from xcap_ews.visualization import (
    plot_ablation,
    plot_calibration,
    plot_cumulative_capture,
    plot_deciles,
    plot_feature_importance,
    plot_historical_trends,
    plot_metric_stability,
    plot_migration,
    plot_peer_comparison,
    plot_province_risk,
    plot_risk_bands,
    plot_spatial_peer_context,
    plot_spatial_risk,
    plot_target_stability,
    set_tiket_style,
)

LOGGER = logging.getLogger("xcap_ews")


def assign_capacity_risk_bands(probability: pd.Series) -> pd.Series:
    """Assign bands matching an explicit 20% review and 5% escalation capacity."""
    percentile = probability.rank(method="first", pct=True)
    return pd.cut(
        percentile,
        bins=[0, 0.50, 0.80, 0.95, 1.0],
        labels=["Low Risk", "Moderate Risk", "Watchlist", "High Risk"],
        include_lowest=True,
    )


def model_definitions() -> dict[str, object]:
    return {
        "logistic": logistic_pipeline(
            LogisticRegression(C=0.3, max_iter=2_000, random_state=RANDOM_SEED)
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=2.0,
            random_state=RANDOM_SEED,
        ),
    }


def feature_definitions() -> dict[str, list[str]]:
    return {
        "financial": list(CURRENT_FEATURES),
        "financial_trend": list(CURRENT_FEATURES + TREND_FEATURES),
        "financial_trend_peer": list(CURRENT_FEATURES + TREND_FEATURES + PEER_FEATURES),
        "financial_trend_peer_spatial": list(
            CURRENT_FEATURES + TREND_FEATURES + PEER_FEATURES + SPATIAL_FEATURES
        ),
    }


def risk_migration(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.sort_values(["Kode_Bank", "target_year"]).copy()
    scored["risk_band"] = scored.groupby("target_year", group_keys=False)["probability"].apply(assign_capacity_risk_bands)
    scored["previous_band"] = scored.groupby("Kode_Bank")["risk_band"].shift(1)
    return pd.crosstab(scored["previous_band"], scored["risk_band"], normalize="index").reset_index()


def run(repo_root: Path) -> None:
    set_tiket_style()
    output_dir = repo_root / "outputs"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading and validating conventional BPR panel")
    panel = load_bpr_panel(repo_root)
    validate_panel(panel)
    panel = harmonize_2024_monetary_units(panel)
    panel = add_ratio_quality_flags(panel)
    panel = build_deterioration_target(panel)
    panel = build_feature_panel(panel, spatial_k=8)
    target_profile = (
        panel.dropna(subset=["next_period_deterioration"])
        .groupby("target_year", as_index=False)
        .agg(
            eligible=("Kode_Bank", "size"),
            events=("next_period_deterioration", "sum"),
            event_rate=("next_period_deterioration", "mean"),
        )
    )
    plot_target_stability(target_profile, figure_dir / "target_stability.png")

    feature_sets = feature_definitions()
    models = model_definitions()
    metrics, predictions = walk_forward_validate(
        panel, feature_sets, models, validation_years=list(range(2019, 2025)), minimum_train_year=2016
    )
    metrics.to_csv(output_dir / "model_metrics_by_year.csv", index=False)
    predictions.to_csv(output_dir / "out_of_time_predictions.csv", index=False)

    development = metrics.loc[metrics["validation_target_year"].lt(2024)]
    summary = (
        development.groupby(["feature_set", "model"], as_index=False)
        [["roc_auc", "pr_auc", "ks", "brier", "recall_at_20pct", "precision_at_20pct", "lift_at_20pct"]]
        .mean()
        .sort_values("ks", ascending=False)
    )
    summary.to_csv(output_dir / "ablation_summary_development.csv", index=False)
    plot_ablation(summary, figure_dir / "ablation_ks.png")

    # Champion is chosen on 2019-2023 mean KS only; 2024 is the locked final assessment.
    champion_row = summary.iloc[0]
    champion_feature_set = str(champion_row["feature_set"])
    champion_model_name = str(champion_row["model"])
    champion_features = feature_sets[champion_feature_set]
    plot_metric_stability(
        metrics,
        champion_feature_set,
        champion_model_name,
        figure_dir / "champion_metric_stability.png",
    )
    champion_predictions = predictions.loc[
        predictions["feature_set"].eq(champion_feature_set)
        & predictions["model"].eq(champion_model_name)
    ].copy()
    deciles = decile_table(champion_predictions)
    deciles.to_csv(output_dir / "champion_risk_deciles.csv", index=False)
    plot_deciles(deciles, figure_dir / "champion_risk_deciles.png")
    migration = risk_migration(champion_predictions)
    migration.to_csv(output_dir / "champion_risk_migration.csv", index=False)
    plot_migration(migration, figure_dir / "champion_risk_migration.png")
    final_oot = champion_predictions.loc[champion_predictions["target_year"].eq(2024)]
    plot_calibration(
        final_oot["next_period_deterioration"],
        final_oot["probability"],
        figure_dir / "final_holdout_calibration.png",
    )
    plot_cumulative_capture(
        final_oot["next_period_deterioration"],
        final_oot["probability"],
        figure_dir / "final_holdout_cumulative_capture.png",
    )

    labeled = panel.loc[
        panel["target_year"].between(2016, 2024) & panel["next_period_deterioration"].notna()
    ].copy()
    labeled["next_period_deterioration"] = labeled["next_period_deterioration"].astype(int)
    latest = panel.loc[panel["Tahun"].eq(2024) & panel["Total_Aset"].notna()].copy()
    champion = model_definitions()[champion_model_name]
    champion.fit(labeled[champion_features], labeled["next_period_deterioration"])
    latest["predicted_deterioration_probability"] = champion.predict_proba(latest[champion_features])[:, 1]
    latest["risk_band"] = assign_capacity_risk_bands(latest["predicted_deterioration_probability"])

    explanations, global_contribution = local_median_replacement_contributions(
        champion, latest[champion_features], labeled[champion_features]
    )
    latest = latest.join(explanations)
    global_contribution.to_csv(output_dir / "global_model_contributions.csv")

    risk_columns = [
        "Kode_Bank", "Nama_BPR", "Provinsi", "Kabupaten_Kota", "Latitude", "Longitude",
        "NPL_Neto", "ROA", "BOPO", "LDR", "KPMM", "Cash_Ratio",
        "delta_npl_net_1y", "delta_roa_1y", "delta_bopo_1y", "delta_kpmm_1y",
        "peer_city_npl_net_mean", "peer_city_roa_mean", "peer_city_bopo_mean",
        "spatial_npl_net_mean", "spatial_roa_mean", "spatial_bopo_mean",
        "predicted_deterioration_probability", "risk_band", "top_risk_drivers",
        "data_quality_flag", "data_quality_reason", "shared_coordinate_flag",
    ]
    risk_table = latest[risk_columns].sort_values(
        "predicted_deterioration_probability", ascending=False
    )
    risk_table.to_csv(output_dir / "latest_bank_risk_table.csv", index=False)
    plot_risk_bands(risk_table, figure_dir / "latest_risk_band_distribution.png")
    plot_peer_comparison(risk_table, figure_dir / "latest_peer_comparison.png")
    boundary_path = repo_root / "data" / "external" / "indonesia_adm0.geojson"
    plot_spatial_risk(
        risk_table, figure_dir / "latest_spatial_risk_map.png", boundary_path
    )
    plot_spatial_peer_context(
        risk_table, figure_dir / "latest_spatial_peer_context.png", boundary_path
    )
    plot_historical_trends(panel, risk_table, figure_dir / "risk_band_financial_trends.png")
    plot_province_risk(risk_table, figure_dir / "latest_province_risk.png")

    final_rows = labeled.loc[labeled["target_year"].eq(2024)]
    if not final_rows.empty:
        final_model = model_definitions()[champion_model_name]
        pre_final = labeled.loc[labeled["target_year"].lt(2024)]
        final_model.fit(pre_final[champion_features], pre_final["next_period_deterioration"])
        def ks_scorer(estimator: object, features: pd.DataFrame, target: pd.Series) -> float:
            probability = estimator.predict_proba(features)[:, 1]
            return ks_statistic(target.to_numpy(), probability)

        importance = permutation_importance(
            final_model,
            final_rows[champion_features],
            final_rows["next_period_deterioration"],
            scoring=ks_scorer,
            n_repeats=15,
            random_state=RANDOM_SEED,
        )
        importance_frame = pd.DataFrame({
            "feature": champion_features,
            "ks_importance_mean": importance.importances_mean,
            "ks_importance_std": importance.importances_std,
        }).sort_values("ks_importance_mean", ascending=False)
        importance_frame.to_csv(
            output_dir / "final_holdout_permutation_importance.csv", index=False
        )
        plot_feature_importance(
            importance_frame, figure_dir / "final_holdout_feature_importance.png"
        )

    spatial_sample = panel.loc[
        panel["Tahun"].eq(2023), ["Kode_Bank", "NPL_Neto", "Latitude", "Longitude"]
    ].dropna()
    weights = knn_weights(
        spatial_sample["Latitude"].to_numpy(), spatial_sample["Longitude"].to_numpy(), k=8
    )
    spatial_result = morans_i_permutation_test(
        spatial_sample["NPL_Neto"].to_numpy(), weights, permutations=999, seed=RANDOM_SEED
    )
    pd.DataFrame([{"year": 2023, "indicator": "NPL_Neto", "k": 8, **spatial_result}]).to_csv(
        output_dir / "spatial_diagnostics.csv", index=False
    )

    final_metric = metrics.loc[
        metrics["validation_target_year"].eq(2024)
        & metrics["feature_set"].eq(champion_feature_set)
        & metrics["model"].eq(champion_model_name)
    ].iloc[0]
    metadata = {
        "random_seed": RANDOM_SEED,
        "champion_selection_period": "target years 2019-2023",
        "champion_selection_metric": "mean KS statistic",
        "champion_model": champion_model_name,
        "champion_feature_set": champion_feature_set,
        "final_holdout_target_year": 2024,
        "final_holdout_ks": float(final_metric["ks"]),
        "final_holdout_pr_auc": float(final_metric["pr_auc"]),
        "final_holdout_roc_auc": float(final_metric["roc_auc"]),
        "final_holdout_brier": float(final_metric["brier"]),
        "risk_band_capacity_assumption": {
            "High Risk": "top 5%",
            "Watchlist": "next 15%",
            "Moderate Risk": "next 30%",
            "Low Risk": "bottom 50%",
        },
        "regional_features": "not available; intentionally omitted",
        "target_semantics": "next-year multi-indicator deterioration proxy; not default probability",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Completed. Champion: %s / %s", champion_model_name, champion_feature_set)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(Path(__file__).resolve().parents[1])
