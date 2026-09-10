"""Portfolio visualizations using a tiket.com-inspired blue/yellow palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve

TIKET = {
    "blue": "#0064D2",
    "deep_blue": "#004A9F",
    "navy": "#102A43",
    "sky": "#DDF1FF",
    "yellow": "#FFC400",
    "orange": "#FF9F1C",
    "coral": "#FF5A5F",
    "teal": "#00A6A6",
    "muted": "#66788A",
    "grid": "#D7E8F7",
    "panel": "#F7FBFF",
}

RISK_COLORS = {
    "Low Risk": TIKET["sky"],
    "Moderate Risk": TIKET["blue"],
    "Watchlist": TIKET["yellow"],
    "High Risk": TIKET["coral"],
}

RISK_ORDER = ["Low Risk", "Moderate Risk", "Watchlist", "High Risk"]


def set_tiket_style() -> None:
    """Apply a consistent friendly, high-contrast visual theme."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": TIKET["panel"],
        "axes.edgecolor": TIKET["grid"],
        "axes.labelcolor": TIKET["navy"],
        "axes.titlecolor": TIKET["navy"],
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": TIKET["grid"],
        "grid.alpha": 0.65,
        "grid.linewidth": 0.8,
        "xtick.color": TIKET["muted"],
        "ytick.color": TIKET["muted"],
        "text.color": TIKET["navy"],
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_stability(target_profile: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(target_profile["target_year"], target_profile["eligible"], color=TIKET["sky"], label="Eligible bank-years")
    ax1.set_ylabel("Eligible observations")
    ax2 = ax1.twinx()
    ax2.plot(target_profile["target_year"], target_profile["event_rate"], color=TIKET["blue"], marker="o", linewidth=2.5, label="Deterioration rate")
    ax2.set_ylabel("Deterioration rate")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax1.set(title="Target coverage and deterioration stability", xlabel="Target year")
    lines = [ax1.patches[0], ax2.lines[0]]
    ax1.legend(lines, ["Eligible bank-years", "Deterioration rate"], loc="upper left")
    _save(fig, path)


def plot_ablation(summary: pd.DataFrame, path: Path) -> None:
    chart = summary.pivot(index="feature_set", columns="model", values="pr_auc")
    chart = chart.reindex([name for name in ["financial", "financial_trend", "financial_trend_peer", "financial_trend_peer_spatial"] if name in chart.index])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    chart.plot(kind="bar", ax=ax, color=[TIKET["blue"], TIKET["yellow"]], width=0.72)
    ax.set(title="Do peer and spatial layers improve early warning?", xlabel="Feature layer", ylabel="Mean out-of-time PR-AUC (2019–2023)")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Model")
    _save(fig, path)


def plot_metric_stability(metrics: pd.DataFrame, feature_set: str, model: str, path: Path) -> None:
    selected = metrics.loc[metrics["feature_set"].eq(feature_set) & metrics["model"].eq(model)].sort_values("validation_target_year")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, label, color in zip(
        axes,
        ["roc_auc", "pr_auc", "brier"],
        ["ROC-AUC", "PR-AUC", "Brier score (lower is better)"],
        [TIKET["blue"], TIKET["yellow"], TIKET["coral"]],
    ):
        ax.plot(selected["validation_target_year"], selected[metric], marker="o", linewidth=2.5, color=color)
        ax.set(title=label, xlabel="Target year")
        ax.axvline(2024, color=TIKET["navy"], linestyle="--", alpha=0.5)
    fig.suptitle("Champion stability across economic regimes", fontsize=15, fontweight="bold")
    _save(fig, path)


def plot_deciles(table: pd.DataFrame, path: Path) -> None:
    ordered = table.sort_values("risk_decile")
    colors = [TIKET["coral"]] + [TIKET["orange"]] * 2 + [TIKET["yellow"]] * 2 + [TIKET["blue"]] * 5
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["risk_decile"].astype(str), ordered["deterioration_rate"], color=colors[: len(ordered)])
    ax.set(title="Realized deterioration concentrates in high-risk deciles", xlabel="Risk decile (1 = highest)", ylabel="Deterioration rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _save(fig, path)


def plot_calibration(y_true: pd.Series, probability: pd.Series, path: Path) -> None:
    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=TIKET["muted"], label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", linewidth=2.5, color=TIKET["blue"], label="2024 holdout")
    ax.fill_between(predicted, predicted, observed, color=TIKET["yellow"], alpha=0.25)
    ax.set(title="Probability calibration on the 2024 holdout", xlabel="Mean predicted probability", ylabel="Observed deterioration rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    _save(fig, path)


def plot_cumulative_capture(y_true: pd.Series, probability: pd.Series, path: Path) -> None:
    ordered = pd.DataFrame({"event": y_true.to_numpy(), "probability": probability.to_numpy()}).sort_values("probability", ascending=False)
    population = np.arange(1, len(ordered) + 1) / len(ordered)
    captured = ordered["event"].cumsum() / ordered["event"].sum()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(population, captured, color=TIKET["blue"], linewidth=3, label="Champion")
    ax.plot([0, 1], [0, 1], linestyle="--", color=TIKET["muted"], label="Random review")
    ax.axvspan(0, 0.20, color=TIKET["yellow"], alpha=0.22, label="20% review capacity")
    ax.set(title="Cumulative capture of deteriorating banks", xlabel="Share of banks reviewed", ylabel="Share of deterioration events captured", xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_risk_bands(risk_table: pd.DataFrame, path: Path) -> None:
    counts = risk_table["risk_band"].value_counts().reindex(RISK_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index, counts.values, color=[RISK_COLORS[label] for label in counts.index])
    ax.bar_label(bars, padding=4, color=TIKET["navy"], fontweight="bold")
    ax.set(title="2024 portfolio allocation by monitoring band", xlabel="", ylabel="Banks")
    _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    selected = importance.nlargest(top_n, "pr_auc_importance_mean").sort_values("pr_auc_importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(selected["feature"], selected["pr_auc_importance_mean"], xerr=selected["pr_auc_importance_std"], color=TIKET["blue"], ecolor=TIKET["yellow"], capsize=3)
    ax.set(title="What drives out-of-time discrimination?", xlabel="Decrease in holdout PR-AUC after permutation", ylabel="")
    _save(fig, path)


def plot_migration(migration: pd.DataFrame, path: Path) -> None:
    matrix = migration.set_index("previous_band").reindex(index=RISK_ORDER, columns=RISK_ORDER).fillna(0)
    cmap = LinearSegmentedColormap.from_list("tiket_blue", ["#FFFFFF", TIKET["sky"], TIKET["blue"], TIKET["navy"]])
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=max(0.01, matrix.to_numpy().max()))
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value > matrix.to_numpy().max() * 0.55 else TIKET["navy"])
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(title="Out-of-time risk-band migration", xlabel="Current band", ylabel="Previous band")
    fig.colorbar(image, ax=ax, label="Transition share")
    _save(fig, path)


def plot_peer_comparison(risk_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for band in RISK_ORDER:
        subset = risk_table.loc[risk_table["risk_band"].eq(band)]
        ax.scatter(subset["peer_city_npl_net_mean"], subset["NPL_Neto"], s=25, alpha=0.65, color=RISK_COLORS[band], label=band, edgecolors="none")
    limit = np.nanpercentile(pd.concat([risk_table["peer_city_npl_net_mean"], risk_table["NPL_Neto"]]), 98)
    ax.plot([0, limit], [0, limit], linestyle="--", color=TIKET["muted"], label="Equal to peers")
    ax.set(title="Is elevated NPL isolated or shared with local peers?", xlabel="City/regency peer mean NPL net", ylabel="Bank NPL net", xlim=(0, limit), ylim=(0, limit))
    ax.legend(ncol=2)
    _save(fig, path)


def plot_spatial_risk(risk_table: pd.DataFrame, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("tiket_risk", [TIKET["sky"], TIKET["blue"], TIKET["yellow"], TIKET["coral"]])
    fig, ax = plt.subplots(figsize=(12, 5.8))
    points = ax.scatter(risk_table["Longitude"], risk_table["Latitude"], c=risk_table["predicted_deterioration_probability"], cmap=cmap, s=20, alpha=0.82, linewidths=0)
    ax.set(title="Geographic distribution of predicted deterioration risk", xlabel="Longitude", ylabel="Latitude")
    ax.grid(alpha=0.3)
    fig.colorbar(points, ax=ax, label="Predicted deterioration probability")
    _save(fig, path)


def plot_historical_trends(panel: pd.DataFrame, risk_table: pd.DataFrame, path: Path) -> None:
    band_lookup = risk_table.set_index("Kode_Bank")["risk_band"]
    history = panel.loc[panel["Tahun"].between(2019, 2024)].copy()
    history["risk_band"] = history["Kode_Bank"].map(band_lookup)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, metric in zip(axes, ["NPL_Neto", "ROA", "BOPO"]):
        medians = history.groupby(["Tahun", "risk_band"], observed=True)[metric].median().unstack()
        for band in RISK_ORDER:
            if band in medians:
                ax.plot(medians.index, medians[band], marker="o", linewidth=2.2, color=RISK_COLORS[band], label=band)
        ax.set(title=f"Median {metric}", xlabel="Year")
    axes[0].set_ylabel("Reported ratio")
    axes[-1].legend(loc="best")
    fig.suptitle("Financial paths of banks by their latest monitoring band", fontsize=15, fontweight="bold")
    _save(fig, path)


def plot_province_risk(risk_table: pd.DataFrame, path: Path) -> None:
    province = risk_table.groupby("Provinsi", as_index=False).agg(
        banks=("Kode_Bank", "size"), mean_risk=("predicted_deterioration_probability", "mean")
    )
    province = province.loc[province["banks"].ge(10)].nlargest(12, "mean_risk").sort_values("mean_risk")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(province["Provinsi"].str.replace("Provinsi ", "", regex=False), province["mean_risk"], color=TIKET["blue"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in province["mean_risk"]], padding=4, color=TIKET["navy"])
    ax.set(title="Highest average predicted risk by province", xlabel="Mean predicted deterioration probability", ylabel="")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _save(fig, path)
