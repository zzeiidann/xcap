"""Leakage-safe bank, trend, administrative-peer, and spatial features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import safe_divide


def _rolling_slope(values: np.ndarray) -> float:
    if len(values) < 3 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.astype(float), 1)[0])


def add_bank_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create financially interpretable current-condition and historical features."""
    result = panel.sort_values(["Kode_Bank", "Tahun"]).copy()
    deposits = result["Tabungan"] + result["Deposito"]
    result["log_assets"] = np.log1p(result["Total_Aset"].clip(lower=0))
    result["log_loans"] = np.log1p(result["Jumlah_Kredit"].clip(lower=0))
    result["equity_to_assets"] = safe_divide(result["Total_Ekuitas"], result["Total_Aset"])
    result["provision_to_loans"] = safe_divide(
        result["Cadangan_Kerugian_Penurunan_Nilai"], result["Jumlah_Kredit"]
    )
    result["loans_to_deposits_calc"] = safe_divide(result["Jumlah_Kredit"], deposits)
    result["deposit_share"] = safe_divide(result["Deposito"], deposits)

    grouped = result.groupby("Kode_Bank", sort=False)
    for source, output in {
        "NPL_Neto": "delta_npl_net_1y",
        "ROA": "delta_roa_1y",
        "BOPO": "delta_bopo_1y",
        "LDR": "delta_ldr_1y",
        "KPMM": "delta_kpmm_1y",
    }.items():
        result[output] = grouped[source].diff()

    for source, output in {
        "Total_Aset": "asset_growth_1y",
        "Jumlah_Kredit": "loan_growth_1y",
    }.items():
        previous = grouped[source].shift(1)
        result[output] = safe_divide(result[source], previous) - 1.0

    result["total_deposits"] = deposits
    previous_deposits = result.groupby("Kode_Bank", sort=False)["total_deposits"].shift(1)
    result["deposit_growth_1y"] = safe_divide(result["total_deposits"], previous_deposits) - 1.0
    result["loan_minus_deposit_growth"] = result["loan_growth_1y"] - result["deposit_growth_1y"]

    for source, output in {
        "NPL_Neto": "npl_net_slope_3y",
        "ROA": "roa_slope_3y",
        "BOPO": "bopo_slope_3y",
        "KPMM": "kpmm_slope_3y",
    }.items():
        result[output] = grouped[source].transform(
            lambda series: series.rolling(3, min_periods=3).apply(_rolling_slope, raw=True)
        )

    result["npl_net_vs_history"] = result["NPL_Neto"] - grouped["NPL_Neto"].transform(
        lambda series: series.expanding(min_periods=2).median()
    )
    result["roa_vs_history"] = result["ROA"] - grouped["ROA"].transform(
        lambda series: series.expanding(min_periods=2).median()
    )
    result["current_deteriorating"] = (
        result["delta_npl_net_1y"].gt(0) | result["delta_roa_1y"].lt(0)
    ).astype(float)
    return result


def _leave_one_out_mean(values: pd.Series, groups: pd.core.groupby.SeriesGroupBy) -> pd.Series:
    totals = groups.transform("sum")
    counts = groups.transform("count")
    return (totals - values).div((counts - values.notna().astype(int)).replace(0, np.nan))


def add_administrative_peer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create contemporaneous, leave-one-bank-out city/regency peer features."""
    result = panel.copy()
    keys = [result["Tahun"], result["Kabupaten_Kota"]]
    for source, output in {
        "NPL_Neto": "peer_city_npl_net_mean",
        "ROA": "peer_city_roa_mean",
        "BOPO": "peer_city_bopo_mean",
        "current_deteriorating": "peer_city_deteriorating_share",
    }.items():
        grouped = result[source].groupby(keys, dropna=False)
        result[output] = _leave_one_out_mean(result[source], grouped)

    result["city_peer_count"] = result.groupby(["Tahun", "Kabupaten_Kota"])["Kode_Bank"].transform("size") - 1
    city_npl_median = result.groupby(["Tahun", "Kabupaten_Kota"])["NPL_Neto"].transform("median")
    city_roa_median = result.groupby(["Tahun", "Kabupaten_Kota"])["ROA"].transform("median")
    result["npl_net_vs_city_median"] = result["NPL_Neto"] - city_npl_median
    result["roa_vs_city_median"] = result["ROA"] - city_roa_median
    result["npl_net_city_percentile"] = result.groupby(["Tahun", "Kabupaten_Kota"])["NPL_Neto"].rank(pct=True)
    return result


def haversine_distance_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Return pairwise great-circle distance in kilometres."""
    lat = np.radians(latitude.astype(float))
    lon = np.radians(longitude.astype(float))
    delta_lat = lat[:, None] - lat[None, :]
    delta_lon = lon[:, None] - lon[None, :]
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2) ** 2
    return 6_371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_spatial_features(panel: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Add same-year k-nearest-bank averages using static audited coordinates."""
    result = panel.copy()
    banks = (result.sort_values("Tahun").drop_duplicates("Kode_Bank", keep="last")
             [["Kode_Bank", "Latitude", "Longitude"]].dropna())
    distances = haversine_distance_matrix(banks["Latitude"].to_numpy(), banks["Longitude"].to_numpy())
    np.fill_diagonal(distances, np.inf)
    neighbor_positions = np.argpartition(distances, kth=min(k, len(banks) - 1) - 1, axis=1)[:, :k]
    bank_codes = banks["Kode_Bank"].to_numpy()
    neighbor_map = {bank_codes[i]: bank_codes[neighbor_positions[i]] for i in range(len(bank_codes))}

    coordinate_counts = banks.groupby(["Latitude", "Longitude"])["Kode_Bank"].transform("size")
    shared = dict(zip(banks["Kode_Bank"], coordinate_counts.gt(1).astype(int)))
    result["shared_coordinate_flag"] = result["Kode_Bank"].map(shared).fillna(0).astype(int)

    outputs = {
        "NPL_Neto": "spatial_npl_net_mean",
        "ROA": "spatial_roa_mean",
        "BOPO": "spatial_bopo_mean",
        "current_deteriorating": "spatial_deteriorating_share",
    }
    for output in outputs.values():
        result[output] = np.nan
    result["spatial_valid_neighbor_count"] = 0

    for year, year_index in result.groupby("Tahun").groups.items():
        year_frame = result.loc[year_index].set_index("Kode_Bank")
        for row_index in year_index:
            code = result.at[row_index, "Kode_Bank"]
            neighbors = neighbor_map.get(code, np.array([], dtype=str))
            available = year_frame.reindex(neighbors)
            result.at[row_index, "spatial_valid_neighbor_count"] = available["NPL_Neto"].notna().sum()
            for source, output in outputs.items():
                result.at[row_index, output] = available[source].mean()
    return result


def build_feature_panel(panel: pd.DataFrame, spatial_k: int = 8) -> pd.DataFrame:
    """Apply feature layers in leakage-safe order."""
    result = add_bank_features(panel)
    result = add_administrative_peer_features(result)
    return add_spatial_features(result, k=spatial_k)
