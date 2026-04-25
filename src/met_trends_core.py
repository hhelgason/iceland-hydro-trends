"""Theil–Sen + Hamed–Rao MK trend helpers (same logic as ``calculate_met_trends.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymannkendall import hamed_rao_modification_test
from scipy.stats import theilslopes

tmul = 10


def calc_trend_and_pval(data):
    trend, intercept, _, _ = theilslopes(data, range(len(data)))
    _, _, pval, _, _, _, _, _, _ = hamed_rao_modification_test(data)
    return trend * tmul, pval


def calc_trends(df, start_year, end_year, resample_method="mean", is_precip=False):
    df_resampled_annual = df.resample("A-SEP").agg(resample_method)
    df_resampled_monthly = df.resample("M").agg(resample_method)
    df_filtered = df.loc["%s-12-01" % start_year : "%s-08-31" % end_year]
    df_resampled_seasonal = df_filtered.resample("QS-DEC").agg(resample_method)

    results = pd.DataFrame(columns=["annual_trend", "pval"])

    for col in df.columns:
        annual_trend, annual_pval = calc_trend_and_pval(df_resampled_annual[col].values)
        if is_precip:
            results.loc[col, "annual_trend_mm"] = annual_trend
            annual_trend = (annual_trend / df_resampled_annual[col].mean()) * 100
        results.loc[col, "annual_trend"] = annual_trend
        results.loc[col, "pval"] = annual_pval

        for i, month in enumerate(range(1, 13)):
            monthly_trend, monthly_pval = calc_trend_and_pval(
                df_resampled_monthly[df_resampled_monthly.index.month == month][col].values
            )
            if is_precip:
                monthly_trend = (
                    monthly_trend
                    / df_resampled_monthly[df_resampled_monthly.index.month == month][col].mean()
                    * 100
                )
            results.loc[col, f"trend_month_{i+1}"] = monthly_trend
            results.loc[col, f"pval_month_{i+1}"] = monthly_pval

        for month, season in zip([12, 3, 6, 9], ["DJF", "MAM", "JJA", "SON"]):
            seasonal_trend, seasonal_pval = calc_trend_and_pval(
                df_resampled_seasonal[df_resampled_seasonal.index.month == month][col].values
            )
            if is_precip:
                results.loc[col, f"trend_{season}_mm"] = seasonal_trend
                seasonal_trend = (
                    seasonal_trend
                    / df_resampled_seasonal[df_resampled_seasonal.index.month == month][col].mean()
                    * 100
                )
            results.loc[col, f"trend_{season}"] = seasonal_trend
            results.loc[col, f"pval_{season}"] = seasonal_pval
    results.index = results.index.astype(int)
    results = results.astype(float).round(3)
    return results
