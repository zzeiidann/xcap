import numpy as np
import pandas as pd

from xcap_ews.data import harmonize_2024_monetary_units
from xcap_ews.features import add_administrative_peer_features, add_bank_features
from xcap_ews.target import build_deterioration_target


def _panel() -> pd.DataFrame:
    rows = []
    for bank, npl in [("A", [1.0, 4.0, 5.0]), ("B", [2.0, 2.5, 3.0])]:
        for year, value in zip([2022, 2023, 2024], npl):
            rows.append({
                "Kode_Bank": bank, "Tahun": year, "Nama_BPR": bank,
                "Provinsi": "P", "Kabupaten_Kota": "C", "Latitude": -6.0,
                "Longitude": 107.0 + (bank == "B"), "NPL_Neto": value,
                "ROA": 3 - (year - 2022), "BOPO": 80 + 6 * (year - 2022),
                "LDR": 80, "KPMM": 30 - 6 * (year - 2022), "Cash_Ratio": 20,
                "Total_Aset": 1_000, "Jumlah_Kredit": 700, "Tabungan": 200,
                "Deposito": 500, "Total_Ekuitas": 100,
                "Cadangan_Kerugian_Penurunan_Nilai": 20,
            })
    return pd.DataFrame(rows)


def test_target_is_forward_and_consecutive() -> None:
    result = build_deterioration_target(_panel())
    first_a = result.loc[(result.Kode_Bank == "A") & (result.Tahun == 2022)].iloc[0]
    assert first_a.target_year == 2023
    assert first_a.next_period_deterioration == 1
    assert pd.isna(result.loc[result.Tahun.eq(2024), "next_period_deterioration"]).all()


def test_2024_unit_adjustment_is_explicit() -> None:
    result = harmonize_2024_monetary_units(_panel())
    assert result.loc[result.Tahun.eq(2024), "Total_Aset"].eq(1).all()
    assert result.loc[result.Tahun.ne(2024), "Total_Aset"].eq(1_000).all()


def test_peer_mean_excludes_self() -> None:
    featured = add_administrative_peer_features(add_bank_features(_panel()))
    row = featured.loc[(featured.Kode_Bank == "A") & (featured.Tahun == 2022)].iloc[0]
    assert np.isclose(row.peer_city_npl_net_mean, 2.0)
