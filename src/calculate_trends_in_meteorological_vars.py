"""
Compute met trends using:

- **LamaH-Ice** daily met CSVs: ``prec``, ``2m_temp_mean``, ``total_et``
- **Snowfall pickle** ``data/daily_dfs_snowfall_runoff.p`` (local, may include ``snowmelt_sum``)
  or ``data/daily_dfs_snowfall_runoff.p.gz`` (repo: gzip, ``snowfall_sum`` only); **rainfall** =
  LamaH ``prec`` minus ``snowfall_sum`` (``runoff_sum`` is not used by this script).

Outputs ``merged_results_dict_<start>-<end>.pkl`` in the chosen output directory and,
if ``--compare-archived`` is set, prints / writes a numeric comparison to archived pickles
(e.g. ``merged_results_dict_1973-2023.pkl`` from the prior ERA5/Caravan pipeline).

Example::

    cd src
    conda activate py312_clean
    python calculate_trends_in_meteorological_vars.py --compare-archived
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    LAMAH_ICE_BASE_PATH,
    OUTPUT_DIR,
    load_snowfall_runoff_pickle,
    resolve_snowfall_runoff_pickle_path,
)
from met_trends_core import calc_trends

# --- defaults (override with CLI) ---
# LamaH met CSV folder (derived from config)
DEFAULT_LAMAH_MET_DIR = (
    LAMAH_ICE_BASE_PATH / r"A_basins_total_upstrm\2_timeseries\daily\meteorological_data"
)

# Gauges shapefile is LamaH base + D_gauges/...
DEFAULT_GAUGES_SHP = LAMAH_ICE_BASE_PATH / r"D_gauges\3_shapefiles\gauges.shp"

# Default outputs go to configured OUTPUT_DIR
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_ARCHIVED_1973 = DEFAULT_OUTPUT_DIR / "merged_results_dict_1973-2023.pkl"
DEFAULT_ARCHIVED_1993 = DEFAULT_OUTPUT_DIR / "merged_results_dict_1993-2023.pkl"

RESAMPLE_METHODS = {
    "prec": "sum",
    "rainfall": "sum",
    "snowfall": "sum",
    "total_et": "sum",
    "2m_temp_mean": "mean",
}
VARIABLES = ["prec", "2m_temp_mean", "total_et", "snowfall", "rainfall"]
PERIODS = [
    {"start": "1973-10-01", "end": "2023-09-30"},
    {"start": "1993-10-01", "end": "2023-09-30"},
]


def _load_lamah_basin_met(basin_id: int, lamah_dir: Path) -> pd.DataFrame:
    path = lamah_dir / f"ID_{basin_id}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";")
    for col in ("YYYY", "MM", "DD"):
        if col not in df.columns:
            raise ValueError(f"{path.name}: missing {col}")
    ts = pd.to_datetime(
        df["YYYY"].astype(int).astype(str)
        + "-"
        + df["MM"].astype(int).astype(str).str.zfill(2)
        + "-"
        + df["DD"].astype(int).astype(str).str.zfill(2),
        errors="coerce",
    )
    df = df.assign(_date=ts).dropna(subset=["_date"]).set_index("_date").sort_index()
    need = ["prec", "2m_temp_mean", "total_et"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path.name}: missing {miss}")
    return df[need].apply(pd.to_numeric, errors="coerce")


def _basin_runoff_df(runoff: dict, basin_id: int) -> pd.DataFrame:
    for k in (f"lamahice_{basin_id}", f"lamahice_{int(basin_id)}"):
        if k in runoff:
            return runoff[k]
    for k in runoff.keys():
        if isinstance(k, str) and k.startswith("lamahice_"):
            try:
                if int(k.split("_", 1)[1]) == int(basin_id):
                    return runoff[k]
            except (IndexError, ValueError):
                continue
    raise KeyError(basin_id)


def _discover_basin_ids(lamah_dir: Path, runoff: dict) -> list[int]:
    ids: list[int] = []
    for p in sorted(lamah_dir.glob("ID_*.csv")):
        try:
            bid = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        try:
            _basin_runoff_df(runoff, bid)
        except KeyError:
            continue
        ids.append(bid)
    return sorted(ids)


def build_combined_dfs(lamah_dir: Path, runoff_pickle: Path) -> dict[str, pd.DataFrame]:
    runoff = load_snowfall_runoff_pickle(runoff_pickle)
    basin_ids = _discover_basin_ids(lamah_dir, runoff)
    if not basin_ids:
        raise RuntimeError("No basins with both LamaH met CSV and runoff pickle entry.")

    lamah_blocks: dict[int, pd.DataFrame] = {}
    snow_blocks: dict[int, pd.Series] = {}
    for bid in basin_ids:
        lamah_blocks[bid] = _load_lamah_basin_met(bid, lamah_dir)
        ro = _basin_runoff_df(runoff, bid)
        if "snowfall_sum" not in ro.columns:
            raise ValueError(f"basin {bid}: runoff DF missing snowfall_sum")
        s = pd.to_numeric(ro["snowfall_sum"], errors="coerce")
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].sort_index()
        snow_blocks[bid] = s

    def _wide_from_lamah(col: str) -> pd.DataFrame:
        series_list = []
        for bid in basin_ids:
            series_list.append(lamah_blocks[bid][col].rename(bid))
        return pd.concat(series_list, axis=1).sort_index()

    combined: dict[str, pd.DataFrame] = {
        "prec": _wide_from_lamah("prec"),
        "2m_temp_mean": _wide_from_lamah("2m_temp_mean"),
        "total_et": _wide_from_lamah("total_et"),
    }
    combined["snowfall"] = pd.concat([snow_blocks[b].rename(b) for b in basin_ids], axis=1).sort_index()

    rain_cols = {}
    for bid in basin_ids:
        p = combined["prec"][bid]
        sn = snow_blocks[bid]
        j = pd.concat([p.rename("p"), sn.rename("s")], axis=1).dropna()
        rain_cols[bid] = j["p"] - j["s"]
    combined["rainfall"] = pd.DataFrame(rain_cols).sort_index()

    # align all to common inner date range across variables (optional trim)
    return combined


def _load_gauges(shp: Path) -> gpd.GeoDataFrame:
    gauges = gpd.read_file(shp)
    gauges.index = gauges["id"].astype(int)
    gauges.index.name = "id"
    return gauges.sort_index()


def _is_precip_variable(variable: str) -> bool:
    return variable in ("prec", "total_et", "snowfall", "rainfall")


def _compare_one_merged(old: pd.DataFrame, new: pd.DataFrame, key: str) -> dict:
    o = old.drop(columns=["geometry"], errors="ignore")
    n = new.drop(columns=["geometry"], errors="ignore")
    num_cols = [c for c in o.columns if c in n.columns and np.issubdtype(o[c].dtype, np.number)]
    if not num_cols:
        return {"key": key, "note": "no overlapping numeric columns"}
    diffs = []
    for c in num_cols:
        a, b = o[c].align(n[c], join="inner")
        d = (a - b).abs()
        diffs.append((c, float(d.max()), float(d.mean())))
    worst = max(diffs, key=lambda x: x[1])
    return {
        "key": key,
        "n_gauges_old": len(old),
        "n_gauges_new": len(new),
        "n_numeric_cols_compared": len(num_cols),
        "worst_col": worst[0],
        "worst_max_abs": worst[1],
        "worst_mean_abs": worst[2],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lamah-dir", type=Path, default=DEFAULT_LAMAH_MET_DIR)
    p.add_argument(
        "--runoff-pickle",
        type=Path,
        default=None,
        help="Default: data/daily_dfs_snowfall_runoff.p if present, else daily_dfs_snowfall_runoff.p.gz",
    )
    p.add_argument("--gauges-shp", type=Path, default=DEFAULT_GAUGES_SHP)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Filename suffix before .pkl (merged_results_dict_<y>-<y><suffix>.pkl)",
    )
    p.add_argument(
        "--compare-archived",
        action="store_true",
        help="Compare new merged dicts to archived merged_results_dict_*.pkl in same folder",
    )
    p.add_argument("--archived-1973", type=Path, default=DEFAULT_ARCHIVED_1973)
    p.add_argument("--archived-1993", type=Path, default=DEFAULT_ARCHIVED_1993)
    p.add_argument(
        "--comparison-report",
        type=Path,
        default=None,
        help="Write comparison summary lines to this path (default: output_dir / met_trend_compare_lamah_snowpickle.txt)",
    )
    p.add_argument(
        "--snow-rain-end",
        type=str,
        default=None,
        help=(
            "Last date (inclusive) for snowfall/rainfall trend input only, e.g. 2021-09-30 "
            "to match the archived script. Default: same as period end (e.g. 2023-09-30)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runoff_pickle is None:
        args.runoff_pickle = resolve_snowfall_runoff_pickle_path()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.comparison_report
        if args.comparison_report is not None
        else args.output_dir / "met_trend_compare_lamah_snowpickle.txt"
    )

    print("LamaH met dir:", args.lamah_dir)
    print("Runoff pickle:", args.runoff_pickle)
    print("Gauges:", args.gauges_shp)
    print("Output dir:", args.output_dir)

    combined_dfs = build_combined_dfs(args.lamah_dir, args.runoff_pickle)
    print("Built combined_dfs:", {k: v.shape for k, v in combined_dfs.items()})

    gauges = _load_gauges(args.gauges_shp)

    archived: dict[str, dict] = {}
    if args.compare_archived:
        for label, path in ("1973", args.archived_1973), ("1993", args.archived_1993):
            if path.is_file():
                with open(path, "rb") as f:
                    archived[label] = pickle.load(f)
                print(f"Loaded archived {label}:", path)
            else:
                print("WARN: archived file missing:", path, file=sys.stderr)

    lines_out: list[str] = []

    for period in PERIODS:
        start = period["start"]
        end = period["end"]
        start_year = start[:4]
        end_year = end[:4]
        merged_results_dict: dict[str, pd.DataFrame] = {}

        print(f"\n=== Period {start_year}-{end_year} ===")
        snow_rain_end = args.snow_rain_end if args.snow_rain_end is not None else end
        for variable in VARIABLES:
            print("Variable:", variable)
            is_precip = _is_precip_variable(variable)
            if variable in ("snowfall", "rainfall"):
                variable_data = combined_dfs[variable].loc[start:snow_rain_end]
            else:
                variable_data = combined_dfs[variable].loc[start:end]
            results = calc_trends(
                variable_data,
                start_year,
                end_year,
                resample_method=RESAMPLE_METHODS[variable],
                is_precip=is_precip,
            )
            merged_results = gauges.merge(
                results.sort_index()[:107],
                left_index=True,
                right_index=True,
            )
            key = f"{variable}_{start_year}-{end_year}"
            merged_results_dict[key] = merged_results
            csv_name = f"lamah_snowpickle_results_{variable}_{start_year}_{end_year}.csv"
            results.sort_index().to_csv(args.output_dir / csv_name, sep=";")
            print("  wrote", args.output_dir / csv_name)

        out_pkl = args.output_dir / f"merged_results_dict_{start_year}-{end_year}{args.suffix}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(merged_results_dict, f)
        print("Wrote", out_pkl)

        arch = archived.get(start_year)
        if arch is not None:
            lines_out.append(f"\n## Compare new vs archived (period start {start_year})\n")
            for key in merged_results_dict:
                if key not in arch:
                    lines_out.append(f"{key}: MISSING in archived\n")
                    continue
                info = _compare_one_merged(arch[key], merged_results_dict[key], key)
                lines_out.append(str(info) + "\n")
                # annual_trend correlation
                o = arch[key]["annual_trend"]
                n = merged_results_dict[key]["annual_trend"]
                j = pd.concat([o.rename("o"), n.rename("n")], axis=1).dropna()
                if len(j) > 2:
                    r = float(j["o"].corr(j["n"]))
                    lines_out.append(f"  annual_trend Pearson r (old vs new): {r:.6f}\n")

    if lines_out:
        hdr = (
            "NOTE: Archived calculate_met_trends.py used snowfall/rainfall daily data only through "
            "2021-09-30. This run uses each period's end for snow/rain unless you pass "
            "--snow-rain-end 2021-09-30 to match that archive window.\n\n"
        )
        report_path.write_text(hdr + "".join(lines_out), encoding="utf-8")
        print("\nComparison report:", report_path.resolve())
        print("".join(lines_out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
