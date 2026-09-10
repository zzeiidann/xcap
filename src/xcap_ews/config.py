"""Central, reviewable analytical configuration."""

from __future__ import annotations

RANDOM_SEED = 42

# Percentage-point changes from observation year t to target year t+1.
TARGET_THRESHOLDS: dict[str, float] = {
    "NPL_Neto": 2.0,
    "ROA": -1.0,
    "BOPO": 5.0,
    "KPMM": -5.0,
}

MONETARY_COLUMNS: tuple[str, ...] = (
    "Total_Aset",
    "a. Kepada BPR",
    "b. Kepada Bank Umum",
    "c. Kepada non bank – pihak terkait",
    "d. Kepada non bank – pihak tidak terkait",
    "Cadangan_Kerugian_Penurunan_Nilai",
    "Jumlah_Kredit",
    "Tabungan",
    "Deposito",
    "Total_Liabilitas",
    "Laba_tahun_berjalan",
    "Total_Ekuitas",
    "Agunan_yang_Diambil_Alih",
    "Jumlah_Pendapatan_Bunga",
    "Jumlah_Pendapatan_Operasional",
    "Jumlah_Beban_Operasional",
    "Laba_Rugi_Operasional",
)
CURRENT_FEATURES: tuple[str, ...] = (
    "NPL_Neto",
    "ROA",
    "BOPO",
    "LDR",
    "KPMM",
    "Cash_Ratio",
    "log_assets",
    "log_loans",
    "equity_to_assets",
    "provision_to_loans",
    "loans_to_deposits_calc",
    "deposit_share",
)

TREND_FEATURES: tuple[str, ...] = (
    "delta_npl_net_1y",
    "delta_roa_1y",
    "delta_bopo_1y",
    "delta_ldr_1y",
    "delta_kpmm_1y",
    "asset_growth_1y",
    "loan_growth_1y",
    "deposit_growth_1y",
    "loan_minus_deposit_growth",
    "npl_net_slope_3y",
    "roa_slope_3y",
    "bopo_slope_3y",
    "kpmm_slope_3y",
    "npl_net_vs_history",
    "roa_vs_history",
)

PEER_FEATURES: tuple[str, ...] = (
    "peer_city_npl_net_mean",
    "peer_city_roa_mean",
    "peer_city_bopo_mean",
    "peer_city_deteriorating_share",
    "npl_net_vs_city_median",
    "roa_vs_city_median",
    "npl_net_city_percentile",
    "city_peer_count",
)

SPATIAL_FEATURES: tuple[str, ...] = (
    "spatial_npl_net_mean",
    "spatial_roa_mean",
    "spatial_bopo_mean",
    "spatial_deteriorating_share",
    "spatial_valid_neighbor_count",
    "shared_coordinate_flag",
)
