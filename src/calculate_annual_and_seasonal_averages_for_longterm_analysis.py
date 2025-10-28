"""
Calculate annual (water year) and seasonal averages of streamflow for long-term analysis.

This script:
1. Reads the cleaned daily streamflow data from pre_process_streamflow_measurements_from_LamaH_Ice.py
2. Reads a csv with long-term series for Jökulsá á Dal river: combines gauge data with the calculated reservoir inflow into Hálslón Reservoir. This replaces gauge 43 from LamaH.
3. Filters to include only gauges that started on or before 1980
4. Calculates water year (Oct 1 - Sep 30) means
5. Calculates seasonal means (DJF, MAM, JJA, SON)
6. Applies a 90% daily data availability threshold for each water year/season
7. Requires at least 30 years of valid water year data
8. Creates visualizations for each gauge
9. Saves the annual and seasonal averages to CSV format

The annual and seasonal data are used for climate indices correlation analysis.

This follows the same filtering logic as trend_analysis.py to ensure consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import geopandas as gpd
import datetime as dt
from config import STREAMFLOW_DATA_PATH, OUTPUT_DIR

# Set global font properties
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10

# Thresholds (same as trend_analysis.py)
WITHIN_YEAR_COVERAGE_THRESHOLD = 0.9  # 90% of daily data required per water year
MIN_YEARS = 30  # Minimum number of valid water years required
START_YEAR_THRESHOLD = 1980  # Gauges must start on or before this year

def get_water_year_index(df):
    """
    Create water year index for a DataFrame (October 1 - September 30).
    
    The water year is labeled by the year in which it ends (Sep 30).
    For example, the water year Oct 1, 2022 - Sep 30, 2023 is labeled as 2023.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with datetime index
        
    Returns
    -------
    pd.Series
        Water year for each date
    """
    def get_wy(date):
        # If month is Oct, Nov, or Dec, the water year is next year
        if date.month >= 10:
            return date.year + 1
        else:
            return date.year
    
    return pd.Series([get_wy(d) for d in df.index], index=df.index)

def calculate_annual_means_with_threshold(df):
    """
    Calculate water year means with data availability threshold.
    
    If a water year has less than 90% of daily data available for a gauge,
    the mean for that water year is set to NaN for that gauge.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with daily streamflow data and datetime index
        
    Returns
    -------
    pd.DataFrame
        DataFrame with annual means, with NaN for years with insufficient data
    """
    # Get water year for each date
    water_years = get_water_year_index(df)
    
    # Calculate annual means
    annual_avg = df.groupby(water_years).mean()
    
    # Count valid (non-NaN) data points per water year
    annual_count = df.groupby(water_years).count()
    
    # Apply threshold: keep only years with >90% daily data
    # For each gauge (column), check if the year has enough data
    for col in df.columns:
        # Calculate the fraction of valid data for each year
        valid_fraction = annual_count[col] / 365.0
        # Set annual mean to NaN where data coverage is insufficient
        insufficient_data_mask = valid_fraction <= WITHIN_YEAR_COVERAGE_THRESHOLD
        annual_avg.loc[insufficient_data_mask, col] = np.nan
    
    # Convert water year index to datetime (December 31 of the water year)
    annual_avg.index = pd.to_datetime([f"{y}-12-31" for y in annual_avg.index])
    
    return annual_avg

def get_season(month):
    """
    Assign season based on month.
    
    Seasons:
    - DJF (December, January, February): Winter
    - MAM (March, April, May): Spring
    - JJA (June, July, August): Summer
    - SON (September, October, November): Fall
    
    Parameters
    ----------
    month : int
        Month number (1-12)
        
    Returns
    -------
    str
        Season code ('DJF', 'MAM', 'JJA', 'SON')
    """
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    else:  # month in [9, 10, 11]
        return 'SON'

def get_season_year(date):
    """
    Get the year for a given season.
    For DJF, the year is the year of January (not December).
    
    Parameters
    ----------
    date : datetime
        Date
        
    Returns
    -------
    int
        Year for the season
    """
    if date.month == 12:
        return date.year + 1
    else:
        return date.year

def calculate_seasonal_means_with_threshold(df):
    """
    Calculate seasonal means with data availability threshold.
    
    If a season has less than 90% of days available for a gauge,
    the mean for that season is set to NaN for that gauge.
    
    Seasons are defined as:
    - DJF: December, January, February (90 days)
    - MAM: March, April, May (92 days)
    - JJA: June, July, August (92 days)
    - SON: September, October, November (91 days)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with daily streamflow data and datetime index
        
    Returns
    -------
    dict
        Dictionary with season codes as keys and DataFrames as values
    """
    # Add season and season-year columns
    df_season = df.copy()
    df_season['season'] = df_season.index.month.map(get_season)
    df_season['season_year'] = df_season.index.map(get_season_year)
    
    # Expected number of days per season (accounting for leap years, use average)
    season_days = {
        'DJF': 90,  # Dec (31) + Jan (31) + Feb (28-29), average ~90
        'MAM': 92,  # Mar (31) + Apr (30) + May (31)
        'JJA': 92,  # Jun (30) + Jul (31) + Aug (31)
        'SON': 91   # Sep (30) + Oct (31) + Nov (30)
    }
    
    seasonal_data = {}
    
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        # Filter data for this season
        season_df = df_season[df_season['season'] == season].drop(columns=['season', 'season_year'])
        season_years = df_season[df_season['season'] == season]['season_year']
        
        # Calculate seasonal means
        seasonal_avg = season_df.groupby(season_years).mean()
        
        # Count valid data points per season
        seasonal_count = season_df.groupby(season_years).count()
        
        # Apply threshold: keep only seasons with >90% daily data
        for col in df.columns:
            # Calculate the fraction of valid data for each season
            valid_fraction = seasonal_count[col] / season_days[season]
            # Set seasonal mean to NaN where data coverage is insufficient
            insufficient_data_mask = valid_fraction <= WITHIN_YEAR_COVERAGE_THRESHOLD
            seasonal_avg.loc[insufficient_data_mask, col] = np.nan
        
        # Convert index to datetime (end of season)
        # DJF: Feb 28, MAM: May 31, JJA: Aug 31, SON: Nov 30
        season_end_month = {'DJF': 2, 'MAM': 5, 'JJA': 8, 'SON': 11}
        season_end_day = {'DJF': 28, 'MAM': 31, 'JJA': 31, 'SON': 30}
        
        seasonal_avg.index = pd.to_datetime([
            f"{y}-{season_end_month[season]:02d}-{season_end_day[season]:02d}" 
            for y in seasonal_avg.index
        ])
        
        seasonal_data[season] = seasonal_avg
    
    return seasonal_data

def plot_annual_timeseries(df_annual, output_dir, gauges_gdf):
    """
    Plot annual streamflow time series for all gauges.
    
    Parameters
    ----------
    df_annual : pd.DataFrame
        DataFrame with annual streamflow data (gauges as columns)
    output_dir : Path
        Directory to save the plots
    gauges_gdf : gpd.GeoDataFrame
        GeoDataFrame with gauge information (names, rivers)
    """
    print("\n" + "="*80)
    print("CREATING ANNUAL TIMESERIES PLOTS")
    print("="*80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each gauge individually
    for i, col in enumerate(df_annual.columns, 1):
        gauge_id = int(col)
        print(f"  Plotting gauge {col} ({i}/{len(df_annual.columns)})")
        
        # Get gauge name
        if gauge_id in gauges_gdf.index:
            river = gauges_gdf.loc[gauge_id, 'river'] if 'river' in gauges_gdf.columns else ''
            name = gauges_gdf.loc[gauge_id, 'name'] if 'name' in gauges_gdf.columns else ''
            gauge_title = f"{river}, {name} (ID {gauge_id})"
        else:
            gauge_title = f"Gauge {col}"
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get valid data
        valid_data = df_annual[col].dropna()
        
        if len(valid_data) > 0:
            # Plot annual time series
            ax.plot(valid_data.index, valid_data.values, linewidth=1.5, 
                   marker='o', markersize=4, color='steelblue', alpha=0.7)
            
            # Add 5-year rolling mean
            rolling_mean = valid_data.rolling(window=5, center=True).mean()
            ax.plot(rolling_mean.index, rolling_mean.values, linewidth=2.5, 
                   color='darkred', label='5-year rolling mean', alpha=0.9)
            
            # Formatting
            ax.set_xlabel('Water Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Annual Mean Streamflow (m³/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'Annual Streamflow - {gauge_title}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Add statistics text
            stats_text = (f"Valid years: {len(valid_data)}\n"
                         f"Period: {valid_data.index.min().year}-{valid_data.index.max().year}\n"
                         f"Mean: {valid_data.mean():.2f} m³/s\n"
                         f"Std: {valid_data.std():.2f} m³/s")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure
            output_file = output_dir / f'gauge_{col}_annual_timeseries.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print(f"    Warning: No valid data for gauge {col}")
    
    print(f"\n  Saved {len(df_annual.columns)} plots to: {output_dir}")
    
    # Create a summary plot with all gauges (small multiples)
    print("\n  Creating summary plot with all gauges...")
    n_gauges = len(df_annual.columns)
    n_cols = 4
    n_rows = int(np.ceil(n_gauges / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_gauges > 1 else [axes]
    
    for idx, col in enumerate(df_annual.columns):
        ax = axes[idx]
        gauge_id = int(col)
        
        # Get gauge name
        if gauge_id in gauges_gdf.index:
            river = gauges_gdf.loc[gauge_id, 'river'] if 'river' in gauges_gdf.columns else ''
            name = gauges_gdf.loc[gauge_id, 'name'] if 'name' in gauges_gdf.columns else ''
            gauge_title = f"{river}, {name}\n(ID {gauge_id})"
        else:
            gauge_title = f"Gauge {col}"
        
        valid_data = df_annual[col].dropna()
        
        if len(valid_data) > 0:
            ax.plot(valid_data.index, valid_data.values, linewidth=1, 
                   marker='o', markersize=2, color='steelblue', alpha=0.7)
            
            # Add 5-year rolling mean
            rolling_mean = valid_data.rolling(window=5, center=True).mean()
            ax.plot(rolling_mean.index, rolling_mean.values, linewidth=2, 
                   color='darkred', alpha=0.8)
            
            ax.set_title(gauge_title, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gauge_title, fontsize=9, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_gauges, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    summary_file = output_dir / 'all_gauges_annual_summary.png'
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary plot: {summary_file}")

def main():
    """
    Main function to calculate annual streamflow averages.
    """
    print("="*80)
    print("CALCULATE ANNUAL (WATER YEAR) STREAMFLOW AVERAGES")
    print("="*80)
    
    # Check if input file exists
    if not STREAMFLOW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Streamflow data file not found: {STREAMFLOW_DATA_PATH}\n"
            f"Please run pre_process_streamflow_measurements_from_LamaH_Ice.py first."
        )
    
    # Read the cleaned daily streamflow data
    print(f"\nReading daily streamflow data from: {STREAMFLOW_DATA_PATH}")
    df_daily = pd.read_csv(STREAMFLOW_DATA_PATH, index_col=0, parse_dates=True)
    
    print(f"  Loaded data: {df_daily.shape[0]} days, {df_daily.shape[1]} gauges")
    print(f"  Date range: {df_daily.index.min()} to {df_daily.index.max()}")
    
    # Read the long-term series for Jökulsá á Dal river (replaces gauge 43 from LamaH)
    kiwis_file = Path('../data/Jokulsa_a_dal_river_longterm_series.csv')
    if kiwis_file.exists():
        print(f"\nReading KIWIS data from: {kiwis_file}")
        print("  This will replace gauge 43 data from LamaH with KIWIS data")
        df_kiwis = pd.read_csv(kiwis_file, index_col=0, parse_dates=True)
        df_kiwis.columns = ['43']  # Gauge ID 43
        
        # Convert to timezone-naive if needed
        if df_kiwis.index.tz is not None:
            df_kiwis.index = df_kiwis.index.tz_localize(None)
        
        print(f"  Loaded KIWIS data: {df_kiwis.shape[0]} days")
        print(f"  Date range: {df_kiwis.index.min()} to {df_kiwis.index.max()}")
        print(f"  Valid values: {df_kiwis['43'].notna().sum()}")
        
        # Replace gauge 43 if it exists, otherwise add it
        if '43' in df_daily.columns:
            print(f"  Replacing existing gauge 43 data from LamaH")
            df_daily = df_daily.drop(columns=['43'])
        
        # Merge with main dataframe
        df_daily = df_daily.join(df_kiwis, how='outer')
        print(f"\n  After merging KIWIS data: {df_daily.shape[0]} days, {df_daily.shape[1]} gauges")
    else:
        print(f"\nWarning: KIWIS data file not found: {kiwis_file}")
        print("  Skipping KIWIS data. Run read_kiwis_timeseries.py first to include it.")
    
    # Filter gauges: keep only those that start on or before START_YEAR_THRESHOLD
    print(f"\nFiltering gauges (must start on or before {START_YEAR_THRESHOLD})...")
    gauges_to_keep = []
    gauges_removed = []
    
    for col in df_daily.columns:
        # Get first non-null date for this gauge
        first_valid_date = df_daily[col].first_valid_index()
        if first_valid_date is not None and first_valid_date.year <= START_YEAR_THRESHOLD:
            gauges_to_keep.append(col)
        else:
            gauges_removed.append((col, first_valid_date.year if first_valid_date else 'No data'))
    
    print(f"  Kept {len(gauges_to_keep)} gauges (started ≤ {START_YEAR_THRESHOLD})")
    print(f"  Removed {len(gauges_removed)} gauges (started > {START_YEAR_THRESHOLD} or no data)")
    
    if gauges_removed:
        print(f"\n  Removed gauges:")
        for gauge_id, start_year in sorted(gauges_removed):
            print(f"    Gauge {gauge_id}: starts in {start_year}")
    
    # Filter the dataframe
    df_daily_filtered = df_daily[gauges_to_keep]
    print(f"\n  Filtered data: {df_daily_filtered.shape[0]} days, {df_daily_filtered.shape[1]} gauges")
    
    # Calculate annual means with threshold
    print(f"\nCalculating annual means (water years with ≥{WITHIN_YEAR_COVERAGE_THRESHOLD*100:.0f}% daily data)...")
    df_annual = calculate_annual_means_with_threshold(df_daily_filtered)
    
    print(f"  Annual data: {df_annual.shape[0]} water years, {df_annual.shape[1]} gauges")
    print(f"  Water year range: {df_annual.index.min().year} to {df_annual.index.max().year}")
    
    # Print data availability statistics
    total_cells = df_annual.shape[0] * df_annual.shape[1]
    valid_cells = df_annual.notna().sum().sum()
    print(f"  Data availability: {valid_cells}/{total_cells} ({100*valid_cells/total_cells:.1f}%)")
    
    # Filter gauges: require at least MIN_YEARS years of valid data
    print(f"\nFiltering gauges (must have at least {MIN_YEARS} valid water years)...")
    
    gauges_to_keep_final = []
    gauges_removed_short = []
    
    for col in df_annual.columns:
        valid_count = df_annual[col].notna().sum()
        
        if valid_count >= MIN_YEARS:
            gauges_to_keep_final.append(col)
        else:
            gauges_removed_short.append((col, valid_count))
    
    print(f"  Kept {len(gauges_to_keep_final)} gauges (≥{MIN_YEARS} valid water years)")
    print(f"  Removed {len(gauges_removed_short)} gauges (<{MIN_YEARS} valid water years)")
    
    if gauges_removed_short:
        print("\n  Removed gauges (insufficient valid water years):")
        for gauge_id, years in sorted(gauges_removed_short):
            print(f"    Gauge {gauge_id}: {years} valid years")
    
    # Filter the dataframe
    df_annual = df_annual[gauges_to_keep_final]
    print(f"\n  Final data: {df_annual.shape[0]} water years, {df_annual.shape[1]} gauges")
    
    # Calculate seasonal means for the same filtered daily data
    print("\n" + "="*80)
    print("CALCULATING SEASONAL MEANS")
    print("="*80)
    print(f"\nCalculating seasonal means (seasons with ≥{WITHIN_YEAR_COVERAGE_THRESHOLD*100:.0f}% daily data)...")
    print("  Seasons: DJF (Dec-Jan-Feb), MAM (Mar-Apr-May), JJA (Jun-Jul-Aug), SON (Sep-Oct-Nov)")
    
    # Use the same filtered daily data (by start year) before the final 30-year filter
    seasonal_data_all = calculate_seasonal_means_with_threshold(df_daily_filtered)
    
    # Apply the same 30-year filter to seasonal data
    seasonal_data_filtered = {}
    for season, df_season in seasonal_data_all.items():
        # Only keep gauges that passed the annual filter
        df_season_filtered = df_season[gauges_to_keep_final]
        seasonal_data_filtered[season] = df_season_filtered
        
        valid_cells = df_season_filtered.notna().sum().sum()
        total_cells = df_season_filtered.shape[0] * df_season_filtered.shape[1]
        print(f"  {season}: {df_season_filtered.shape[0]} seasons, {df_season_filtered.shape[1]} gauges, "
              f"{100*valid_cells/total_cells:.1f}% data availability")
    
    # Save annual data to CSV
    output_file_annual = OUTPUT_DIR / "annual_streamflow_averages_longterm.csv"
    output_file_annual.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    print(f"\nSaving annual averages to: {output_file_annual}")
    df_annual.to_csv(output_file_annual)
    
    # Save seasonal data to CSV
    for season, df_season in seasonal_data_filtered.items():
        output_file_season = OUTPUT_DIR / f"seasonal_streamflow_averages_longterm_{season}.csv"
        print(f"Saving {season} averages to: {output_file_season}")
        df_season.to_csv(output_file_season)
    
    # Print summary for each gauge
    print("\n" + "="*80)
    print("SUMMARY BY GAUGE")
    print("="*80)
    print(f"{'Gauge ID':<10} {'Valid Years':<15} {'Coverage %':<12} {'Water Year Range'}")
    print("-"*80)
    
    for col in df_annual.columns:
        valid_count = df_annual[col].notna().sum()
        total_count = len(df_annual)
        coverage = 100 * valid_count / total_count
        
        # Get date range for this gauge
        valid_data = df_annual[col].dropna()
        if len(valid_data) > 0:
            date_range = f"{valid_data.index.min().year} to {valid_data.index.max().year}"
        else:
            date_range = "No data"
        
        print(f"{col:<10} {valid_count:<15} {coverage:<12.1f} {date_range}")
    
    # Load gauge names from shapefile
    print("\n" + "="*80)
    print("Loading gauge names for visualization...")
    gauges_shp_path = Path(r"C:\Users\hordurbhe\OneDrive - Landsvirkjun\Documents\Vinna\lamah\lamah_ice\lamah_ice\D_gauges\3_shapefiles\gauges.shp")
    gauges_gdf = gpd.read_file(gauges_shp_path)
    gauges_gdf['id'] = gauges_gdf['id'].astype(int)
    gauges_gdf = gauges_gdf.set_index('id')
    
    # Create visualizations
    viz_output_dir = OUTPUT_DIR / "annual_means_viz_longterm"
    plot_annual_timeseries(df_annual, viz_output_dir, gauges_gdf)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Annual: {output_file_annual}")
    print(f"  Seasonal (DJF): {OUTPUT_DIR / 'seasonal_streamflow_averages_longterm_DJF.csv'}")
    print(f"  Seasonal (MAM): {OUTPUT_DIR / 'seasonal_streamflow_averages_longterm_MAM.csv'}")
    print(f"  Seasonal (JJA): {OUTPUT_DIR / 'seasonal_streamflow_averages_longterm_JJA.csv'}")
    print(f"  Seasonal (SON): {OUTPUT_DIR / 'seasonal_streamflow_averages_longterm_SON.csv'}")
    print(f"\nVisualization folder: {viz_output_dir}")
    print(f"\nThresholds used:")
    print(f"  - Within-year/season coverage threshold: {WITHIN_YEAR_COVERAGE_THRESHOLD*100:.0f}%")
    print(f"  - Minimum valid water years: {MIN_YEARS}")
    print(f"  - Start year threshold: ≤{START_YEAR_THRESHOLD}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

