"""
Summarize meteorological trend results with Figure8-style count heatmaps.

For each variable (precipitation, temperature, ET, snow, rain), builds rows
Annual + DJF/MAM/JJA/SON and columns Positive / Negative / significant counts,
annotated with glaciated-basin counts like ``Figure8_visualize_trend_summary_with_heatmap.py``.

Outputs one PNG+PDF per variable under the chosen output folder.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import geopandas as gpd
from matplotlib import rcParams

from config import CATCHMENT_ATTRIBUTES_FILE, OUTPUT_DIR
from Figure8_visualize_trend_summary_with_heatmap import plot_trend_heatmaps

MET_COLUMNS = [
    ("annual_trend", "pval", "Annual"),
    ("trend_DJF", "pval_DJF", "DJF"),
    ("trend_MAM", "pval_MAM", "MAM"),
    ("trend_JJA", "pval_JJA", "JJA"),
    ("trend_SON", "pval_SON", "SON"),
]

VARIABLE_LABELS = {
    "prec": "Precipitation",
    "2m_temp_mean": "2 m temperature",
    "total_et": "Evapotranspiration",
    "snowfall": "Snowfall",
    "rainfall": "Rainfall",
}

# Manuscript subfigure order (a–j) and short names for panel titles: "Trends in …"
MET_HEATMAP_PANELS: list[tuple[str, str, tuple[str, str]]] = [
    ("2m_temp_mean", "temperature", ("a", "b")),
    ("prec", "precipitation", ("c", "d")),
    ("rainfall", "rainfall", ("e", "f")),
    ("snowfall", "snowfall", ("g", "h")),
    ("total_et", "evapotranspiration", ("i", "j")),
]


def _attach_g_frac(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    c = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)
    c = c.set_index(c["id"].astype(int))
    if "g_frac" not in c.columns:
        raise ValueError(f"{CATCHMENT_ATTRIBUTES_FILE} has no g_frac column")
    out = gdf.copy()
    ids = out.index.astype(int)
    out["g_frac"] = c["g_frac"].reindex(ids).values
    return out


def main(
    pickle_dir: Path | None = None,
    pkl_suffix: str = "",
    output_dir: Path | None = None,
) -> None:
    pickle_dir = pickle_dir or OUTPUT_DIR
    out = Path(output_dir or (OUTPUT_DIR / "meteorological_trends_figures" / "met_trend_summary_heatmaps"))
    out = out.resolve()
    os.makedirs(out, exist_ok=True)

    p73 = pickle_dir / f"merged_results_dict_1973-2023{pkl_suffix}.pkl"
    p93 = pickle_dir / f"merged_results_dict_1993-2023{pkl_suffix}.pkl"
    if not p73.is_file():
        raise FileNotFoundError(p73)
    if not p93.is_file():
        raise FileNotFoundError(p93)

    with open(p73, "rb") as f:
        merged_1973: dict = pickle.load(f)
    with open(p93, "rb") as f:
        merged_1993: dict = pickle.load(f)

    rcParams["font.family"] = "Arial"
    rcParams["font.size"] = 20

    for var, title_entity, letters in MET_HEATMAP_PANELS:
        label = VARIABLE_LABELS[var]
        k73 = f"{var}_1973-2023"
        k93 = f"{var}_1993-2023"
        if k73 not in merged_1973 or k93 not in merged_1993:
            print(f"Skip {var}: missing keys {k73!r} / {k93!r}")
            continue
        df73 = _attach_g_frac(merged_1973[k73])
        df93 = _attach_g_frac(merged_1993[k93])
        print(f"Heatmap: {label} ...")
        plot_trend_heatmaps(
            df73,
            df93,
            MET_COLUMNS,
            label,
            str(out),
            panel_letters=letters,
            title_entity=title_entity,
        )

    print(f"Done. Figures in: {out}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Met trend summary heatmaps (counts by direction).")
    ap.add_argument("--pickle-dir", type=Path, default=None)
    ap.add_argument("--pkl-suffix", type=str, default="")
    ap.add_argument("--output-dir", type=Path, default=None)
    ns = ap.parse_args()
    main(pickle_dir=ns.pickle_dir, pkl_suffix=ns.pkl_suffix, output_dir=ns.output_dir)


if __name__ == "__main__":
    _cli()
