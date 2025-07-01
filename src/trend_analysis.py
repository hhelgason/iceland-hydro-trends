"""
Trend Analysis for Streamflow Data in Iceland

This script analyzes trends in various streamflow metrics for Icelandic rivers. It is part of the 
supplementary material for the paper "Understanding Changes in Iceland's Streamflow Dynamics in Response to Climate Change",
submitted to HESS, 2024.

The script calculates trends in:
- Annual and seasonal mean flows
- Flow variability (standard deviation, coefficient of variation)
- Flashiness indices
- Rising and falling sequences
- Baseflow indices
- Low and high flow metrics

Methods:
- Trend calculation using Theil-Sen slope estimator
- Significance testing using modified Mann-Kendall test (Hamed and Rao modification)
- Digital filtering for baseflow separation following Ladson et al., 2013

Dependencies:
- pandas
- numpy
- scipy
- pymannkendall
- datetime
- pathlib
- collections

Author: Hordur Bragi Helgason
"""

import pandas as pd
import numpy as np
import datetime as dt
from collections import defaultdict
from pymannkendall import hamed_rao_modification_test
from scipy.stats import theilslopes
import os
from pathlib import Path
from config import LAMAH_ICE_BASE_PATH

def return_df(df, start_year, end_year, missing_data_threshold=0.8):
    """
    Filter data based on coverage requirements.
    
    Args:
        df (pd.DataFrame): Input streamflow data
        start_year (int): Start year for analysis
        end_year (int): End year for analysis
        missing_data_threshold (float): Minimum fraction of valid data required (default: 0.8)
    
    Returns:
        pd.DataFrame: Filtered data containing only gauges meeting coverage requirements
    """
    year_data = df['%s-10-01' % (int(start_year)):'%s-09-30'% (int(end_year))].copy()
    gauge_ids_to_export = []
    for gauge_id in year_data.columns:
        to_plot = year_data[gauge_id]
        if len(to_plot.dropna()) / len(pd.date_range('%s-10-01' % start_year,'%s-09-30' % end_year)) > missing_data_threshold:
            gauge_ids_to_export.append(gauge_id)
        else:
            print('Gauge omitted: Overall raw coverage between %s and %s is less than %s for gauge id %s' % (start_year, end_year, missing_data_threshold, gauge_id))
    return year_data[gauge_ids_to_export]

def calc_trend_and_pval(data):
    """
    Calculate trend and p-value using Theil-Sen slope estimator and modified Mann-Kendall test.
    
    Args:
        data (np.array): Time series data
        
    Returns:
        tuple: (trend per decade in %, p-value, trend, intercept, x values)
    """
    tmul = 10  # Multiplier to convert to per-decade values
    data = np.asarray(data, dtype=np.float64)
    trend_ts, intercept_ts, _, _ = theilslopes(data, range(len(data)))
    _, _, mod_pval, _, _, _, _, _, _ = hamed_rao_modification_test(data)
    trend_percent_increase_per_decade_ts = (trend_ts / np.mean(data)) * 100 * tmul
    x_values = np.arange(len(data))
    
    return trend_percent_increase_per_decade_ts, mod_pval, trend_ts, intercept_ts, x_values

def get_water_year_index(df):
    """
    Create water year index for a DataFrame (October 1 - September 30).
    
    Args:
        df (pd.DataFrame): Input data with datetime index
        
    Returns:
        pd.Series: Water year for each date
    """
    return pd.Series([(d - dt.timedelta(days=273)).year for d in df.index], index=df.index)

def compute_annual_stats(df):
    """
    Compute annual statistics (mean, standard deviation, coefficient of variation).
    
    Args:
        df (pd.DataFrame): Input streamflow data
        
    Returns:
        tuple: (annual_mean, annual_std, annual_cv) DataFrames
    """
    water_years = get_water_year_index(df)
    annual_avg = df.groupby(water_years).mean()
    annual_std = df.groupby(water_years).std()
    annual_cv = annual_std / annual_avg
    index = pd.to_datetime([f"{y}-12-31" for y in annual_avg.index])
    for d in [annual_avg, annual_std, annual_cv]:
        d.index = index
    return annual_avg, annual_std, annual_cv

def compute_low_high_flows(df):
    water_years = get_water_year_index(df)
    low = df.groupby(water_years).apply(lambda x: x.quantile(0.1))
    high = df.groupby(water_years).apply(lambda x: x.quantile(0.9))
    index = pd.to_datetime([f"{y}-12-31" for y in low.index])
    low.index = high.index = index
    return low, high

def compute_rising_falling_sequences(df):
    water_years = get_water_year_index(df)
    rise_counts = pd.DataFrame(index=np.unique(water_years), columns=df.columns)
    fall_counts = pd.DataFrame(index=np.unique(water_years), columns=df.columns)
    for col in df.columns:
        # Handle NaN values by doing calculations on clean data only
        data = df[col].copy()
        diff = data.diff()
        # Only consider non-NaN differences for sequence detection
        valid_diff = diff.dropna()
        sign = valid_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        change = sign.ne(sign.shift()).cumsum()
        temp = pd.DataFrame({"sign": sign, "segment": change, "water_year": water_years[diff.notna()]})
        
        # Group sequences by water year and count
        yearly = temp.groupby(["water_year", "segment"])['sign'].first()
        yearly = yearly.reset_index().groupby(["water_year", "sign"]).count().unstack(fill_value=0)
        
        # Fill in the counts, using 0 for years with no valid sequences
        rise_counts[col] = yearly.get(1, pd.Series(index=rise_counts.index, dtype=int))
        fall_counts[col] = yearly.get(-1, pd.Series(index=fall_counts.index, dtype=int))
    
    rise_counts.index = fall_counts.index = pd.to_datetime([f"{y}-12-31" for y in rise_counts.index])
    return rise_counts, fall_counts

def compute_flashiness(df):
    water_years = get_water_year_index(df)
    years = np.unique(water_years)
    flashiness = pd.DataFrame(index=years, columns=df.columns)
    for year in years:
        subset = df[water_years == year]
        for col in df.columns:
            q = subset[col].dropna()
            abs_diff_sum = q.diff().abs().sum()
            total_flow = q.sum()
            flashiness.loc[year, col] = abs_diff_sum / total_flow if total_flow != 0 else np.nan
    flashiness.index = pd.to_datetime([f"{y}-12-31" for y in flashiness.index])
    return flashiness

def compute_seasonal_stats(df):
    seasonal_groups = df.resample('QS-DEC')
    seasonal_cv = seasonal_groups.std() / seasonal_groups.mean()
    flashiness = seasonal_groups.apply(lambda x: x.diff().abs().sum() / x.sum())
    rising_counts = seasonal_groups.apply(lambda x: (x.diff() > 0).astype(int).diff().fillna(0).eq(1).sum())
    falling_counts = seasonal_groups.apply(lambda x: (x.diff() < 0).astype(int).diff().fillna(0).eq(1).sum())
    jas_groups = df[df.index.month.isin([7, 8, 9])].resample('YE-SEP')
    jas_cv = jas_groups.std() / jas_groups.mean()
    jas_flashiness = jas_groups.apply(lambda x: x.diff().abs().sum() / x.sum())
    jas_rising_counts = jas_groups.apply(lambda x: (x.diff() > 0).astype(int).diff().fillna(0).eq(1).sum())
    jas_falling_counts = jas_groups.apply(lambda x: (x.diff() < 0).astype(int).diff().fillna(0).eq(1).sum())
    return seasonal_cv, flashiness, rising_counts, falling_counts, jas_cv, jas_flashiness, jas_rising_counts, jas_falling_counts

def forward_pass(q, alpha):
    qf = [q[0]]
    for i in range(1, len(q)):
        qf.append(alpha * qf[i-1] + 0.5 * (1 + alpha) * (q[i] - q[i-1]))
    qb = [qt-fl if fl > 0 else qt for (qt, fl) in zip(q, qf)]
    return (qf, qb)

def backward_pass(qb, alpha):
    qb_flipped = np.flip(qb)
    qf_new_flipped, qb_new_flipped = forward_pass(qb_flipped, alpha)
    qf_new = np.flip(qf_new_flipped)
    qb_new = np.flip(qb_new_flipped)
    return(qf_new, qb_new)

def get_bf(streamflow, alpha, num_filters, num_reflect):
    """Calculate baseflow using digital filter method.
    
    Args:
        streamflow: Array of streamflow values
        alpha: Filter parameter (default 0.925)
        num_filters: Number of filter passes (default 3)
        num_reflect: Number of days to reflect at start/end (default 30)
        
    Returns:
        Tuple of (quickflow, baseflow) arrays
    """
    # Add reflected values
    q_reflect = np.zeros(2*num_reflect + len(streamflow))
    q_reflect[:num_reflect] = np.flip(streamflow[:num_reflect])
    q_reflect[num_reflect:len(q_reflect)-num_reflect] = streamflow
    q_reflect[len(q_reflect)-num_reflect:] = np.flip(streamflow[len(streamflow)-num_reflect:])

    # Run the filters
    for i in range(num_filters):
        if i % 2 == 0:  # Forward filter
            if i == 0:  # The input is q_reflect
                qf, qb = forward_pass(q_reflect, alpha)
            else:  # The input is qb
                qf, qb = forward_pass(qb, alpha)
        else:  # Backward filter
            qf, qb = backward_pass(qb, alpha)

    # Remove the reflected values
    qf = qf[num_reflect:len(q_reflect)-num_reflect]
    qb = qb[num_reflect:len(q_reflect)-num_reflect]
    return (qf, qb)

def calculate_baseflow_index(streamflow, alpha=0.925, num_filters=3, num_reflect=30):
    """
    Calculate baseflow index following Ladson et al., 2013.
    Simplified version that assumes no quality flags (all data is good quality).
    """
    # Calculate baseflow
    qf, qb = get_bf(streamflow, alpha, num_filters, num_reflect)
    # Calculate BFI
    bfi = np.sum(qb)/np.sum(streamflow)
    return float(bfi)

def compute_baseflow_index(df, output_dir=None):
    """
    Calculate annual baseflow index for each gauge using digital filtering method.
    
    First calculates baseflow for the entire time series to avoid edge effects,
    then aggregates to annual values. Uses the Lyne and Hollick digital filter
    following Ladson et al., 2013.
    
    Args:
        df (pd.DataFrame): Streamflow data
        output_dir (str, optional): Directory to save baseflow series
        
    Returns:
        pd.DataFrame: Annual baseflow index values for each gauge
    """
    water_years = get_water_year_index(df)
    years = np.unique(water_years)
    bfi = pd.DataFrame(index=years, columns=df.columns)
    
    # Create dictionary to store baseflow series
    baseflow_series = {}
    
    for col in df.columns:
        # Calculate baseflow for entire time series
        q = df[col].dropna()
        if len(q) > 0:
            try:
                # Get baseflow values for entire series
                _, qb = get_bf(q.values, alpha=0.925, num_filters=3, num_reflect=30)
                # Create series with same index as original data
                baseflow = pd.Series(qb, index=q.index)
                # Store baseflow series
                baseflow_series[col] = pd.DataFrame({
                    'streamflow': q,
                    'baseflow': baseflow
                })
                # Group by water year and calculate BFI
                yearly_total = df[col].groupby(water_years).sum()
                yearly_baseflow = baseflow.groupby(water_years).sum()
                # Calculate BFI for each year
                yearly_bfi = yearly_baseflow / yearly_total
                # Assign to output DataFrame
                bfi[col] = yearly_bfi
            except:
                bfi[col] = np.nan
        else:
            bfi[col] = np.nan
    
    # Save baseflow series if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for col in baseflow_series:
            baseflow_series[col].to_csv(os.path.join(output_dir, f'baseflow_series_{col}.csv'))
    
    bfi.index = pd.to_datetime([f"{y}-12-31" for y in bfi.index])
    return bfi

def calc_all_trends(df, baseflow_series_path=None):
    """
    Calculate comprehensive set of streamflow trends.
    
    Calculates trends in:
    - Annual and seasonal mean flows
    - Flow variability (standard deviation, coefficient of variation)
    - Flashiness indices
    - Rising and falling sequences
    - Baseflow indices
    - Low and high flow metrics
    
    Args:
        df (pd.DataFrame): Input streamflow data
        baseflow_series_path (str, optional): Path to save baseflow series
        
    Returns:
        tuple: (results DataFrame, valid data dictionary, invalid data dictionary)
    """
    results = defaultdict(dict)
    valid_data_dict = {}
    invalid_data_dict = {}

    # Define thresholds
    annual_threshold = 0.8
    seasonal_threshold = 0.8
    within_year_and_season_coverage_threshold = 0.9

    # Load catchment attributes to get degimpact
    degimpact_path = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"
    degimpact_df = pd.read_csv(degimpact_path, sep=';').set_index('id')
    influenced_gauges = set(degimpact_df[degimpact_df['degimpact'] == 's'].index.astype(str))

    # Calculate annual (water years) statistics
    water_years = get_water_year_index(df)
    annual_avg = df.groupby(water_years).mean()
    annual_std = df.groupby(water_years).std()
    annual_cv = annual_std / annual_avg
    annual_count = df.groupby(water_years).count()
    annual_bfi = compute_baseflow_index(df, output_dir=baseflow_series_path)  # Add baseflow index calculation

    # Set datetime index
    for data in [annual_avg, annual_std, annual_cv, annual_count, annual_bfi]:
        # Convert year numbers to datetime, checking if conversion is needed
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime([f"{int(y)}-12-31" for y in data.index])

    # Calculate monthly and seasonal averages
    monthly_avg = df.resample('ME').mean()
    monthly_count = df.resample('ME').count()
    seasonal_avg = df.resample('QS-DEC').mean()
    seasonal_count = df.resample('QS-DEC').count()

    # Calculate flashiness indices
    flashiness = pd.DataFrame(index=np.unique(water_years), columns=df.columns)
    rising_counts = pd.DataFrame(index=np.unique(water_years), columns=df.columns)
    falling_counts = pd.DataFrame(index=np.unique(water_years), columns=df.columns)

    for year in np.unique(water_years):
        year_data = df[water_years == year]
        for col in df.columns:
            # Flashiness index
            data = year_data[col].dropna()
            if len(data) > 0:
                flashiness.loc[year, col] = data.diff().abs().sum() / data.sum()

            # Rising and falling sequences
            diff = data.diff()
            sign = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            change = sign.ne(sign.shift()).cumsum()
            sequences = pd.DataFrame({"sign": sign, "segment": change})
            rising_counts.loc[year, col] = (sequences.groupby(["segment"])['sign'].first() == 1).sum()
            falling_counts.loc[year, col] = (sequences.groupby(["segment"])['sign'].first() == -1).sum()

    flashiness.index = pd.to_datetime([f"{y}-12-31" for y in flashiness.index])
    rising_counts.index = flashiness.index
    falling_counts.index = flashiness.index

    # Calculate trends for different metrics
    for col in df.columns:
        col_str = str(col)
        is_influenced = col_str in influenced_gauges
        # Annual mean flow: include all gauges
        # All other metrics: skip if influenced
        # Annual mean flow
        valid_annual = annual_avg[col][(annual_count[col].dropna() / 365) > within_year_and_season_coverage_threshold]
        valid_data_percent = len(valid_annual) / len(annual_avg[col])
        annual_data = valid_annual.dropna()
        if valid_data_percent >= annual_threshold:
            trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(annual_data.values)
            results[col]['annual_avg_flow_trend_per_decade'] = trend_per_dec
            results[col]['annual_avg_flow_trend'] = trend
            results[col]['annual_intercept'] = intercept
            results[col]['pval'] = pval
            valid_data_dict[(str(col), 'annual')] = annual_data
            
            # All other metrics: skip if influenced
            if not is_influenced:
                # Annual variability trends - only calculate if we have enough valid years
                for metric_name, metric_data in [
                    ('annual_std', annual_std),
                    ('annual_cv', annual_cv),
                    ('flashiness', flashiness),
                    ('rising_seq', rising_counts),
                    ('falling_seq', falling_counts),
                    ('baseflow_index', annual_bfi)
                ]:
                    valid_data = metric_data[col][(annual_count[col].dropna() / 365) > within_year_and_season_coverage_threshold].dropna()                
                    if len(valid_data) > 0:
                        trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(valid_data.values)
                        results[col][f'trend_{metric_name}_per_decade'] = trend_per_dec
                        results[col][f'trend_{metric_name}'] = trend
                        results[col][f'intercept_{metric_name}'] = intercept
                        results[col][f'pval_{metric_name}'] = pval
                        valid_data_dict[(str(col), metric_name)] = valid_data
        else:
            # Set NaN values for annual mean flow metrics
            results[col]['annual_avg_flow_trend_per_decade'] = np.nan
            results[col]['annual_avg_flow_trend'] = np.nan
            results[col]['annual_intercept'] = np.nan
            results[col]['pval'] = np.nan
            invalid_data_dict[(str(col), 'annual')] = annual_data
            
            # Set NaN values for all variability metrics
            for metric_name in ['annual_std', 'annual_cv', 'flashiness', 'rising_seq', 'falling_seq', 'baseflow_index']:
                results[col][f'trend_{metric_name}_per_decade'] = np.nan
                results[col][f'trend_{metric_name}'] = np.nan
                results[col][f'intercept_{metric_name}'] = np.nan
                results[col][f'pval_{metric_name}'] = np.nan

        # Low/High flow trends
        for flow_type, percentile in [('low_flow', 0.1), ('high_flow', 0.9)]:
            # Calculate annual quantiles
            flow_data = df[col].groupby(water_years).quantile(percentile)
            flow_data.index = pd.to_datetime([f"{y}-12-31" for y in flow_data.index])
            
            # Filter to years with sufficient coverage
            valid_years = annual_count[col][(annual_count[col].dropna() / 365) > within_year_and_season_coverage_threshold].index
            valid_flow_data = flow_data[flow_data.index.isin(valid_years)].dropna()
            
            if len(valid_flow_data) > 0:
                trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(valid_flow_data.values)
                results[col][f'{flow_type}_trend_per_decade'] = trend_per_dec
                results[col][f'{flow_type}_trend'] = trend
                results[col][f'{flow_type}_intercept'] = intercept
                results[col][f'{flow_type}_pval'] = pval
                valid_data_dict[(str(col), flow_type)] = valid_flow_data

        # Seasonal trends
        if not is_influenced:  # Skip seasonal trends for influenced gauges
            # JAS (July-August-Sptember) trends
            jas_data = df[col][df.index.month.isin([7, 8, 9])].resample('YE-SEP').mean()
            jas_count = df[col][df.index.month.isin([7, 8, 9])].resample('YE-SEP').count()
            valid_jas = jas_data[(jas_count / 90) > seasonal_threshold]
            valid_jas_percent = len(valid_jas) / len(jas_data)
            valid_jas_data = valid_jas.dropna()        
            if valid_jas_percent >= seasonal_threshold:
                trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(valid_jas_data.values)
                results[col]['trend_JAS_per_decade'] = trend_per_dec
                results[col]['trend_JAS'] = trend
                results[col]['intercept_JAS'] = intercept
                results[col]['pval_JAS'] = pval
                valid_data_dict[(str(col), 'JAS')] = valid_jas_data
            else:
                invalid_data_dict[(str(col), 'JAS')] = valid_jas

            # Seasonal trends
            for month, season in zip([12, 3, 6, 9], ['DJF', 'MAM', 'JJA', 'SON']):
                season_data = seasonal_avg[col][seasonal_avg.index.month == month]
                season_count = seasonal_count[col][seasonal_count.index.month == month]
                valid_season = season_data[(season_count / 90) > seasonal_threshold]
                valid_season_percent = len(valid_season) / len(season_data)
                valid_season_data = valid_season.dropna()

                if valid_season_percent >= seasonal_threshold:
                    print(f"    Season {season} passed threshold ({valid_season_percent:.2%} >= {seasonal_threshold})")
                    trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(valid_season_data.values)
                    results[col][f'trend_{season}_per_decade'] = trend_per_dec
                    results[col][f'trend_{season}'] = trend
                    results[col][f'intercept_{season}'] = intercept
                    results[col][f'pval_{season}'] = pval
                    valid_data_dict[(str(col), season)] = valid_season_data

                    # Calculate seasonal flashiness
                    if season == 'DJF':
                        season_months = [12, 1, 2]
                    elif season == 'MAM':
                        season_months = [3, 4, 5]
                    elif season == 'JJA':
                        season_months = [6, 7, 8]
                    else:  # 'SON'
                        season_months = [9, 10, 11]

                    # Calculate seasonal flashiness for each year
                    seasonal_flashiness = pd.Series(index=valid_season.index)
                    for year in valid_season.index.year:
                        # Get data for this season
                        if season == 'DJF':
                            # For DJF, we need to handle the year transition
                            winter_data = df[col][
                                ((df.index.year == year-1) & (df.index.month == 12)) |
                                ((df.index.year == year) & (df.index.month.isin([1, 2])))
                            ].dropna()
                        else:
                            # For other seasons, all months are in the same year
                            winter_data = df[col][
                                (df.index.year == year) & 
                                (df.index.month.isin(season_months))
                            ].dropna()
                        
                        if len(winter_data) > 0:
                            seasonal_flashiness[f"{year}-{season_months[0]:02d}-01"] = winter_data.diff().abs().sum() / winter_data.sum()

                    # Calculate trend for seasonal flashiness
                    valid_flashiness = seasonal_flashiness.dropna()
                    if len(valid_flashiness) >= len(valid_season) * seasonal_threshold:
                        trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(valid_flashiness.values)
                        results[col][f'flashiness_{season}_trend_per_decade'] = trend_per_dec
                        results[col][f'flashiness_{season}_pval'] = pval
                        valid_data_dict[(str(col), f'flashiness_{season}')] = valid_flashiness
                    else:
                        results[col][f'flashiness_{season}_trend_per_decade'] = np.nan
                        results[col][f'flashiness_{season}_pval'] = np.nan
                        invalid_data_dict[(str(col), f'flashiness_{season}')] = seasonal_flashiness
                    
                    # Get all data for this gauge first
                    full_data = df[col].dropna()
                    if len(full_data) > 0:
                        # Calculate and save seasonal std and cv
                        seasonal_daily_data = df[col][df.index.month.isin(season_months)]
                        seasonal_water_years = get_water_year_index(seasonal_daily_data)
                        seasonal_std = seasonal_daily_data.groupby(seasonal_water_years).std()
                        seasonal_mean = seasonal_daily_data.groupby(seasonal_water_years).mean()
                        seasonal_cv = seasonal_std / seasonal_mean
                        if len(seasonal_std) > 0:
                            # Calculate std trends
                            trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(seasonal_std.values)
                            results[col][f'std_{season}_trend_per_decade'] = trend_per_dec
                            results[col][f'std_{season}_trend'] = trend
                            results[col][f'std_{season}_intercept'] = intercept
                            results[col][f'std_{season}_pval'] = pval
                            valid_data_dict[(str(col), f'std_{season}')] = seasonal_std
                            
                            # Calculate CV trends
                            trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(seasonal_cv.values)
                            results[col][f'cv_{season}_trend_per_decade'] = trend_per_dec
                            results[col][f'cv_{season}_trend'] = trend
                            results[col][f'cv_{season}_intercept'] = intercept
                            results[col][f'cv_{season}_pval'] = pval
                            valid_data_dict[(str(col), f'cv_{season}')] = seasonal_cv

                        # Now calculate baseflow index trends
                        try:
                            # Calculate baseflow for entire series first
                            _, qb = get_bf(full_data.values, alpha=0.925, num_filters=3, num_reflect=30)
                            # Create series with same index as original data
                            baseflow = pd.Series(qb, index=full_data.index)
                            
                            # Now filter to seasonal data and calculate seasonal BFI
                            seasonal_baseflow = baseflow[baseflow.index.month.isin(season_months)]
                            
                            # Create index using datetime for the first month of each season
                            valid_years = seasonal_water_years.unique()
                            seasonal_index = pd.to_datetime([f"{y}-{season_months[0]:02d}-01" for y in valid_years])
                            seasonal_bfi = pd.Series(index=seasonal_index)
                            
                            # Calculate seasonal BFI using the pre-calculated baseflow
                            for year in valid_years:
                                year_mask = seasonal_water_years == year
                                year_flow = seasonal_daily_data[year_mask]
                                year_baseflow = seasonal_baseflow[year_mask]
                                if len(year_flow) > 0:
                                    seasonal_bfi.loc[pd.to_datetime(f"{year}-{season_months[0]:02d}-01")] = year_baseflow.sum() / year_flow.sum()
                                else:
                                    seasonal_bfi.loc[pd.to_datetime(f"{year}-{season_months[0]:02d}-01")] = np.nan
                            
                            # Calculate trend for seasonal BFI
                            valid_bfi = seasonal_bfi.dropna()
                            if len(valid_bfi) >= len(valid_season) * seasonal_threshold:
                                trend_per_dec, pval, trend, intercept, x_values = calc_trend_and_pval(valid_bfi.values)
                                results[col][f'baseflow_index_{season}_trend_per_decade'] = trend_per_dec
                                results[col][f'baseflow_index_{season}_trend'] = trend
                                results[col][f'baseflow_index_{season}_intercept'] = intercept
                                results[col][f'baseflow_index_{season}_pval'] = pval
                                valid_data_dict[(str(col), f'baseflow_index_{season}')] = valid_bfi
                            else:
                                results[col][f'baseflow_index_{season}_trend_per_decade'] = np.nan
                                results[col][f'baseflow_index_{season}_trend'] = np.nan
                                results[col][f'baseflow_index_{season}_intercept'] = np.nan
                                results[col][f'baseflow_index_{season}_pval'] = np.nan
                                invalid_data_dict[(str(col), f'baseflow_index_{season}')] = seasonal_bfi
                        except Exception as e:
                            print(f"Error calculating seasonal BFI for gauge {col}, season {season}: {e}")
                            results[col][f'baseflow_index_{season}_trend_per_decade'] = np.nan
                            results[col][f'baseflow_index_{season}_trend'] = np.nan
                            results[col][f'baseflow_index_{season}_intercept'] = np.nan
                            results[col][f'baseflow_index_{season}_pval'] = np.nan
                else:
                    print(f"    Season {season} failed threshold ({valid_season_percent:.2%} < {seasonal_threshold})")
                    invalid_data_dict[(str(col), season)] = valid_season
                    results[col][f'trend_{season}_per_decade'] = np.nan
                    results[col][f'trend_{season}'] = np.nan
                    results[col][f'intercept_{season}'] = np.nan
                    results[col][f'pval_{season}'] = np.nan
                    results[col][f'flashiness_{season}_trend_per_decade'] = np.nan
                    results[col][f'flashiness_{season}_pval'] = np.nan
                    results[col][f'baseflow_index_{season}_trend_per_decade'] = np.nan
                    results[col][f'baseflow_index_{season}_trend'] = np.nan
                    results[col][f'baseflow_index_{season}_intercept'] = np.nan
                    results[col][f'baseflow_index_{season}_pval'] = np.nan
                    results[col][f'std_{season}_trend_per_decade'] = np.nan
                    results[col][f'std_{season}_trend'] = np.nan
                    results[col][f'std_{season}_intercept'] = np.nan
                    results[col][f'std_{season}_pval'] = np.nan
                    results[col][f'cv_{season}_trend_per_decade'] = np.nan
                    results[col][f'cv_{season}_trend'] = np.nan
                    results[col][f'cv_{season}_intercept'] = np.nan
                    results[col][f'cv_{season}_pval'] = np.nan
        else:
            # Set NaN values for all seasonal metrics if gauge is influenced
            for season in ['JAS', 'DJF', 'MAM', 'JJA', 'SON']:
                results[col][f'trend_{season}_per_decade'] = np.nan
                results[col][f'trend_{season}'] = np.nan
                results[col][f'intercept_{season}'] = np.nan
                results[col][f'pval_{season}'] = np.nan
                results[col][f'flashiness_{season}_trend_per_decade'] = np.nan
                results[col][f'flashiness_{season}_pval'] = np.nan
                results[col][f'baseflow_index_{season}_trend_per_decade'] = np.nan
                results[col][f'baseflow_index_{season}_trend'] = np.nan
                results[col][f'baseflow_index_{season}_intercept'] = np.nan
                results[col][f'baseflow_index_{season}_pval'] = np.nan
                results[col][f'std_{season}_trend_per_decade'] = np.nan
                results[col][f'std_{season}_trend'] = np.nan
                results[col][f'std_{season}_intercept'] = np.nan
                results[col][f'std_{season}_pval'] = np.nan
                results[col][f'cv_{season}_trend_per_decade'] = np.nan
                results[col][f'cv_{season}_trend'] = np.nan
                results[col][f'cv_{season}_intercept'] = np.nan
                results[col][f'cv_{season}_pval'] = np.nan

    return pd.DataFrame.from_dict(results, orient='index'), valid_data_dict, invalid_data_dict