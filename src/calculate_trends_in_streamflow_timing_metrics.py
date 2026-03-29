import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

import trend_analysis
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from scipy.stats import kendalltau, theilslopes
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from pymannkendall import hamed_rao_modification_test, original_test
import scipy.stats as stats
from config import (
    LAMAH_ICE_BASE_PATH,
    OUTPUT_DIR,
    ICELAND_SHAPEFILE,
    GLACIER_SHAPEFILE,
    MISSING_DATA_THRESHOLD,
    STREAMFLOW_DATA_PATH
)

facecolor = 'white' #'lightgrey'
plt.rcParams['font.family'] = 'Arial' 
streamflow_markersize = 200
streamflow_sign_size = 22
maps_fontsize = 20
map_fontsize_sea = 35
shift_value = 100

time_periods = [(1973, 2023), (1993, 2023)]
missing_data_threshold = MISSING_DATA_THRESHOLD

# Define output base folder
base_folder = OUTPUT_DIR / 'timing_analysis'

# Create output directory if it doesn't exist
os.makedirs(base_folder, exist_ok=True)

# Create figures directory
maps_path = OUTPUT_DIR / 'timing_trends_figures'
maps_path.mkdir(parents=True, exist_ok=True)

# Read streamflow data
print("Reading streamflow data from:", STREAMFLOW_DATA_PATH)
df_with_data = pd.read_csv(STREAMFLOW_DATA_PATH, index_col=0, parse_dates=True)

# Read the gauges shapefile that contains the indices and V numbers
gauges_shapefile = LAMAH_ICE_BASE_PATH / "D_gauges/3_shapefiles/gauges.shp"
gauges_gdf = gpd.read_file(gauges_shapefile)
gauges_gdf = gauges_gdf.set_index('id')
gauges_gdf = gauges_gdf.set_crs('epsg:3057')
gauges = gauges_gdf.copy()

# Read the catchment characteristics - Extract area_calc and human influence
catchments_chara = pd.read_csv(LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv", sep=';')
catchments_chara = catchments_chara.set_index('id')
mask = catchments_chara['degimpact'] != 's'

# Read the catchment characteristics - Extract area_calc and human influence
hydro_sign = pd.read_csv(LAMAH_ICE_BASE_PATH / "D_gauges/1_attributes/hydro_indices_1981_2018_unfiltered.csv", sep=';')
hydro_sign = hydro_sign.set_index('id')

# Read catchments
catchments = gpd.read_file(LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/3_shapefiles/Basins_A.shp")
catchments = catchments.set_index('id')
catchments = catchments.set_crs('epsg:3057')

bmap = gpd.read_file(ICELAND_SHAPEFILE)
glaciers = gpd.read_file(GLACIER_SHAPEFILE)
    
# Specify some plot attributes
iceland_shapefile_color = 'gray'
glaciers_color = 'white'
# Define plot specifications
colormap = 'RdBu'

shift_value = 100
plot_gap_filling_plots = False
plot_freshet_hydrographs = True  # Plot hydrographs with Mode 0 and Mode 1 markers
plot_mean_seasonal = True  # Plot mean seasonal hydrographs with variability bands

# Gauges to exclude from freshet analysis due to high boundary-hitting rate (>45%)
# These rivers lack a distinct seasonal freshet pulse, causing the detection algorithm
# to frequently return dates at the search window boundaries (April 1 or June 30)
FRESHET_EXCLUDE_PERIOD1 = [8, 37, 39, 58, 64, 67, 79, 82, 95, 98]  # 1973-2023
FRESHET_EXCLUDE_PERIOD2 = [8, 26, 37, 39, 58, 64, 67, 79, 82, 91, 98]  # 1993-2023

def reikna_Q_g_optimized(df, idx):
    idx_start = idx - 15
    idx_end = idx + 15

    Q_o = df.iloc[idx_start:idx_end+1]['Value'].dropna().to_numpy()
    
    if len(Q_o) < 5:
        K = 1
    else:
        Q_m = df[df['DOY'].isin(df.iloc[idx_start:idx_end+1]['DOY'])]['Value'].mean()
        K = np.median(Q_o) / Q_m

    Q_m_j = df[df['DOY'] == df.iloc[idx]['DOY']]['Value'].mean()

    Q_g = Q_m_j * K

    return Q_g

def fill_missing_data(streamflow_data, max_gap_days=60):
    filled_df = streamflow_data.copy()
    
    for column in streamflow_data.columns:
        print(f"Processing column: {column}")
        
        # Calculate the percentage of missing data
        total_values = len(streamflow_data[column])
        missing_values = streamflow_data[column].isna().sum()
        missing_percentage = (missing_values / total_values) * 100
        
        df = pd.DataFrame(filled_df[column])
        df.columns = ['Value']
        df['DOY'] = df.index.dayofyear

        # Identify the range with valid data
        first_valid_idx = df['Value'].first_valid_index()
        last_valid_idx = df['Value'].last_valid_index()
        
        # Calculate the percentage of missing data within the valid range
        valid_data_range = df.loc[first_valid_idx:last_valid_idx]
        valid_missing_values = valid_data_range['Value'].isna().sum()
        valid_total_values = len(valid_data_range)
        valid_missing_percentage = (valid_missing_values / valid_total_values) * 100

        # Check if there is any missing data within the valid range
        if valid_missing_values == 0:
            print("No data is missing within the valid range.")
            filled_df[column] = df['Value']
        else:
            print(f'{valid_missing_percentage:.2f}% of data is missing between {first_valid_idx} and {last_valid_idx}')

            # Create an array to store the filled values, initialized with the original data
            A = df['Value'].to_numpy()

            # Process only gaps that are <= max_gap_days
            mask = df['Value'].isna()
            gap_sizes = mask.groupby((mask != mask.shift()).cumsum()).transform('size')
            
            # Only fill gaps of size less than or equal to max_gap_days
            for i in range(max(15, df.index.get_loc(first_valid_idx)), 
                           min(len(df) - 15, df.index.get_loc(last_valid_idx) + 1)):
                if np.isnan(df.iloc[i]['Value']) and gap_sizes.iloc[i] <= max_gap_days:
                    A[i] = reikna_Q_g_optimized(df, i)

            # Create a new filled column with the adjusted values
            df['Q_g'] = A
            df['brúuð röð'] = df['Value'].fillna(df['Q_g'])
            filled_df[column] = df['brúuð röð']
    
    return filled_df

def plot_figs(basemap,glaciers,ax,iceland_shapefile_color,glaciers_color):
    # This function plots the basemap, the glaciers and sets x,ylimits and x,yticks
    # Set figure limits
    minx, miny = 222375, 307671
    maxx, maxy = 765246, 697520
    
    basemap.plot(ax=ax, color=iceland_shapefile_color,edgecolor='darkgray')
    glaciers.plot(ax=ax,facecolor=glaciers_color,edgecolor='none')
    ax.set_xlim(minx,maxx)
    ax.set_ylim(miny,maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def day_of_year_to_date_fixed(year, day_of_year):
    start_of_year = datetime(year, 1, 1)
    return start_of_year + timedelta(days=int(day_of_year) - 1)

def get_water_year(date):
    """Get water year for a given date. Water year starts Oct 1."""
    if date.month >= 10:
        return date.year + 1
    return date.year

def get_day_of_water_year(date):
    """Get day of water year (1 = Oct 1, 365/366 = Sept 30)."""
    water_year = get_water_year(date)
    water_year_start = pd.Timestamp(year=water_year - 1, month=10, day=1)
    return (date - water_year_start).days + 1

def water_year_day_to_date(water_year, day_of_water_year):
    """Convert water year and day of water year to a date."""
    water_year_start = pd.Timestamp(year=water_year - 1, month=10, day=1)
    return water_year_start + pd.Timedelta(days=day_of_water_year - 1)

# Freshet detection window in water year days:
# Oct: days 1-31, Nov: 32-61, Dec: 62-92, Jan: 93-123, Feb: 124-151/152
# Mar: 152-182/183, Apr: 183-212/213, May: 213-243/244, Jun: 244-273/274
# April 1 in water year ≈ day 183, June 30 ≈ day 273
FRESHET_START_DAY_WY = 183  # April 1 in water year
FRESHET_END_DAY_WY = 273    # June 30 in water year

def pulse_method_strict_fix(yearly_flow_array, freshet_start_day=FRESHET_START_DAY_WY, freshet_end_day=FRESHET_END_DAY_WY):
    """Detect freshet timing using cumulative departure method.
    
    Args:
        yearly_flow_array: 2D array of shape (n_years, 365/366) with daily flows
        freshet_start_day: Start of freshet search window (day of water year)
        freshet_end_day: End of freshet search window (day of water year)
    
    Returns:
        day_mode_0: Day of minimum cumulative departure (water year day)
        day_mode_1: Refined start by walking backwards to find where flow starts rising
    """
    n_years = yearly_flow_array.shape[0]
    n_days = yearly_flow_array.shape[1]
    
    # Adjust end day if array is shorter (non-leap year)
    freshet_end_day = min(freshet_end_day, n_days)
    freshet_range = range(freshet_start_day, freshet_end_day)
    
    hydro_dept = np.cumsum(yearly_flow_array - np.reshape(yearly_flow_array.mean(1), (-1, 1)), 1)
    day_mode_0 = np.argmin(hydro_dept[:, freshet_range], 1) + freshet_start_day
    
    day_mode_1 = np.zeros(n_years, dtype=int)
    for i in range(n_years):
        day = day_mode_0[i]
        while day > freshet_start_day and yearly_flow_array[i, day] >= yearly_flow_array[i, day - 1]:
            day -= 1
        day_mode_1[i] = max(day, freshet_start_day)
    
    return day_mode_0, day_mode_1

def calculate_day_of_water_year(timestamp):
    if timestamp.month >= 10:  # If the month is October or later, the water year started in the same year
        start_of_water_year = pd.Timestamp(year=timestamp.year, month=10, day=1)
    else:  # If the month is before October, the water year started in the previous year
        start_of_water_year = pd.Timestamp(year=timestamp.year - 1, month=10, day=1)
    return (timestamp - start_of_water_year).days + 1

def plot_freshet_series_with_subplots(filled_df, df_orig, timing_results_all, save_path, years_per_plot=8, freshet=True):
    """Plot original and gap-filled data for visual comparison.
    
    Args:
        filled_df: DataFrame with gap-filled values
        df_orig: Original DataFrame with missing values
        timing_results_all: Dictionary with freshet/centroid/peak timing results (can be None for gap-filling plots)
        save_path: Path to save the plots
        years_per_plot: Number of years to show in each subplot
        freshet: If True, plot freshet timing, if False plot centroid/peak timing
    """
    os.makedirs(save_path, exist_ok=True)
    # Ensure that the index of filled_df is in datetime format
    filled_df.index = pd.to_datetime(filled_df.index)
    
    # Define the number of years per subplot and number of subplots
    total_years = filled_df.index.year.max() - filled_df.index.year.min() + 1
    num_subplots = (total_years + years_per_plot - 1) // years_per_plot  # Calculate number of subplots needed
    
    # Iterate over each column in the filled_df (each streamflow series)
    for column in filled_df.columns:
        print(f"Plotting series: {column}")
        
        # Create a figure with subplots for the specified number of subplots
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots), sharex=False)
        if num_subplots == 1:
            axes = [axes]  # Make axes iterable when there's only one subplot
        
        # Add data to each subplot
        for i in range(num_subplots):
            start_year = filled_df.index.year.min() + i * years_per_plot
            end_year = start_year + years_per_plot - 1
            ax = axes[i]
            
            # Filter the original (un-filled) data for the current year range
            data_range_orig = df_orig.loc[
                (df_orig.index.year >= start_year) & (df_orig.index.year <= end_year), column
            ]
            
            # Filter the filled data for the current year range
            data_range = filled_df.loc[
                (filled_df.index.year >= start_year) & (filled_df.index.year <= end_year), column
            ]
            
            # Plot the streamflow data for the current range
            ax.plot(data_range.index, data_range, label='Streamflow (filled)', color='red', alpha=0.3)
            ax.plot(data_range_orig.index, data_range_orig, label='Streamflow', color='blue', alpha=0.6)
            
            # Set x-axis limits to match the data range
            ax.set_xlim([data_range.index.min(), data_range.index.max()])
            
            # Add timing markers if timing_results_all is provided
            if timing_results_all is not None:
                timing_data = timing_results_all.get(column)
                if timing_data is not None:
                    # Check if this is freshet data (has both Mode 0 and Mode 1)
                    if 'Mode_0_Freshet_Date' in timing_data.columns and 'Mode_1_Freshet_Date' in timing_data.columns:
                        # Plot both Mode 0 and Mode 1 with different markers
                        for idx, row in timing_data.iterrows():
                            mode_0_date = pd.to_datetime(row['Mode_0_Freshet_Date'])
                            mode_1_date = pd.to_datetime(row['Mode_1_Freshet_Date'])
                            
                            if start_year <= mode_0_date.year <= end_year:
                                # Mode 0 - green triangle (min cumulative departure)
                                closest_date_0 = data_range.index.get_indexer([mode_0_date], method='nearest')[0]
                                closest_date_0 = data_range.index[closest_date_0]
                                ax.axvline(x=closest_date_0, color='green', linestyle='--', alpha=0.7, 
                                          label='Mode 0 (min cum. dep.)' if idx == 0 and i == 0 else "")
                                
                                # Mode 1 - red triangle (refined start)
                                closest_date_1 = data_range.index.get_indexer([mode_1_date], method='nearest')[0]
                                closest_date_1 = data_range.index[closest_date_1]
                                ax.axvline(x=closest_date_1, color='red', linestyle='-', alpha=0.7,
                                          label='Mode 1 (refined start)' if idx == 0 and i == 0 else "")
                    elif 'CentroidDate' in timing_data.columns:
                        date_col = 'CentroidDate'
                        marker_label = 'Centroid of Timing'
                        for idx, row in timing_data.iterrows():
                            timing_date = pd.to_datetime(row[date_col])
                            if start_year <= timing_date.year <= end_year:
                                closest_date = data_range.index.get_indexer([timing_date], method='nearest')[0]
                                closest_date = data_range.index[closest_date]
                                ax.plot(closest_date, data_range.loc[closest_date], 'ro',
                                       label=marker_label if idx == 0 else "")
                    elif 'PeakFlowDate' in timing_data.columns:
                        date_col = 'PeakFlowDate'
                        marker_label = 'Peak Flow Timing'
                        for idx, row in timing_data.iterrows():
                            timing_date = pd.to_datetime(row[date_col])
                            if start_year <= timing_date.year <= end_year:
                                closest_date = data_range.index.get_indexer([timing_date], method='nearest')[0]
                                closest_date = data_range.index[closest_date]
                                ax.plot(closest_date, data_range.loc[closest_date], 'ro',
                                       label=marker_label if idx == 0 else "")
                    else:
                        print(f"No recognized date column found for gauge {column}")
            
            # Add titles and labels
            if i == 0:  # Only add title to first subplot
                ax.set_title(f"{gauges.loc[int(column)]['river']}, {gauges.loc[int(column)]['name']}, {gauges.loc[int(column)]['V_no']}, ID {column}")
            ax.set_ylabel("Daily Streamflow [m³/s]")
            ax.grid(True, which='both', alpha=0.3)
            
            # Format x-axis with month ticks
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.tick_params(axis='x', which='major', labelsize=8)
            ax.tick_params(axis='x', which='minor', length=2)
            
            # Add legend to first subplot only
            if i == 0:
                ax.legend(loc='upper right')
        
        # Set a common x-axis label for the bottom subplot
        axes[-1].set_xlabel("Date")
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(save_path / f"{column}.png", dpi=300)
        plt.close()

def plot_mean_seasonal_hydrograph(df, save_path, use_water_year=True):
    """Plot mean seasonal hydrograph for each gauge showing average daily flow and variability.
    
    Args:
        df: DataFrame with daily streamflow data (columns are gauge IDs, index is datetime)
        save_path: Path to save the plots
        use_water_year: If True, use water year (Oct 1 - Sept 30), otherwise calendar year
    """
    os.makedirs(save_path, exist_ok=True)
    df.index = pd.to_datetime(df.index)
    
    for column in df.columns:
        print(f"Plotting mean seasonal hydrograph: {column}")
        
        dff = df[column].dropna()
        if dff.empty:
            continue
        
        # Create DataFrame with day of year
        streamflow_data = pd.DataFrame(dff)
        streamflow_data['Timestamp'] = pd.to_datetime(dff.index)
        
        if use_water_year:
            # Use day of water year (Oct 1 = day 1)
            streamflow_data['DayOfYear'] = streamflow_data['Timestamp'].apply(get_day_of_water_year)
            x_label = "Day of Water Year (Oct 1 = Day 1)"
            # Create month labels for water year
            month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
            month_starts = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 336]
        else:
            streamflow_data['DayOfYear'] = streamflow_data['Timestamp'].dt.dayofyear
            x_label = "Day of Year"
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        
        # Calculate statistics for each day of year
        daily_stats = streamflow_data.groupby('DayOfYear')[column].agg(['mean', 'std', 'min', 'max', 'median'])
        daily_stats['q25'] = streamflow_data.groupby('DayOfYear')[column].quantile(0.25)
        daily_stats['q75'] = streamflow_data.groupby('DayOfYear')[column].quantile(0.75)
        daily_stats['q10'] = streamflow_data.groupby('DayOfYear')[column].quantile(0.10)
        daily_stats['q90'] = streamflow_data.groupby('DayOfYear')[column].quantile(0.90)
        
        # Sort by day of year
        daily_stats = daily_stats.sort_index()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        days = daily_stats.index.values
        
        # Plot 10th-90th percentile range (lighter shade)
        ax.fill_between(days, daily_stats['q10'], daily_stats['q90'], 
                       alpha=0.2, color='blue', label='10th-90th percentile')
        
        # Plot 25th-75th percentile range (darker shade)
        ax.fill_between(days, daily_stats['q25'], daily_stats['q75'], 
                       alpha=0.3, color='blue', label='25th-75th percentile')
        
        # Plot the median (main line)
        ax.plot(days, daily_stats['median'], color='blue', linewidth=2, label='Median')
        
        # Plot the mean (red dashed, thinner)
        ax.plot(days, daily_stats['mean'], color='red', linewidth=1, linestyle='--', label='Mean')
        
        # Add month labels on x-axis
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
        ax.set_xlim([1, 366])
        
        # Add vertical lines at month boundaries
        for month_start in month_starts[1:]:
            ax.axvline(x=month_start, color='gray', linestyle='--', alpha=0.3)
        
        # Labels and title
        try:
            gauge_info = gauges.loc[int(column)]
            title = f"{gauge_info['river']}, {gauge_info['name']}, {gauge_info['V_no']}, ID {column}"
        except:
            title = f"Gauge {column}"
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Daily Streamflow [m³/s]", fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(save_path) / f"{column}_mean_seasonal.png", dpi=300)
        plt.close()

def plot_timing_series(gauge_id, timing_data, trend_results, title, ylabel, save_path, water_year_offset=274):
    """Plot timing series for a single gauge.
    
    Args:
        gauge_id: ID of the gauge
        timing_data: DataFrame containing timing values
        trend_results: DataFrame containing trend data
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the plot
        water_year_offset: Days to add to convert to water year days (default 274 for Oct 1)
    """
    # Get the timing column name based on what's available
    if 'Mode_1_Freshet_Day' in timing_data.columns:
        timing_col = 'Mode_1_Freshet_Day'
        year_col = 'WaterYear'
    elif 'CentroidOfTiming' in timing_data.columns:
        timing_col = 'CentroidOfTiming'
        year_col = 'WaterYear'
    elif 'PeakFlowDay' in timing_data.columns:
        timing_col = 'PeakFlowDay'
        year_col = 'WaterYear'
    else:
        print(f"No timing column found for gauge {gauge_id}")
        return
    
    # Filter for 1973-2023 period
    timing_data = timing_data[
        (timing_data[year_col] >= 1973) & 
        (timing_data[year_col] <= 2023)
    ]
    
    if timing_data.empty:
        print(f"No data in 1973-2023 period for gauge {gauge_id}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot timing data points
    plt.scatter(timing_data[year_col], timing_data[timing_col], 
               color='blue', s=50, alpha=0.6, label='Annual timing')
    
    # Plot trend line if exists
    trend = trend_results.iloc[0]  # Get the trend data from the first row
    
    # Calculate trend line using the same approach as plotting.py
    trend_years = pd.date_range(str(timing_data[year_col].min()), 
                              str(timing_data[year_col].max()), 
                              freq='YE').year
    trend_line = trend['trend_days_per_year'] * np.arange(len(trend_years)) + trend['intercept']
    
    if trend['pval'] < 0.05:
        line_style = '-'
        significance = 'significant'
    else:
        line_style = '--'
        significance = 'not significant'
        
    trend_label = (f"Trend: {trend['trend_days_per_decade']:.1f} days/decade\n"
                  f"(p={trend['pval']:.3f}, {significance})")
    
    plt.plot(trend_years, trend_line, 'r'+line_style,
            label=trend_label, linewidth=2)
    
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.8)
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel('Water Year', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits to focus on the relevant range
    y_min = max(1, timing_data[timing_col].min() - 10)
    y_max = min(366, timing_data[timing_col].max() + 10)
    plt.ylim(y_min, y_max)
    
    # Format x-axis to show years clearly
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    save_name = f'{gauge_id}_timing_ts_mk.png'
    plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
    plt.close() 

def run_freshet_analysis_for_all_gauges(df):
    """Run freshet analysis using water years (Oct 1 - Sept 30)."""
    freshet_results_all = {}
    trends_results = []
    
    # Calculate minimum required water years (80% of period length)
    # Estimate water year range from data
    min_date = df.index.min()
    max_date = df.index.max()
    first_water_year = get_water_year(min_date)
    last_water_year = get_water_year(max_date)
    period_length = last_water_year - first_water_year + 1
    min_years_required = int(0.8 * period_length)
    print(f"\nWater year range: {first_water_year} to {last_water_year}")
    print(f"Period length: {period_length} water years")
    print(f"Minimum years required (80%): {min_years_required} water years")
    
    for gauge in df.columns:
        dff = df[gauge].dropna()  # Drop missing data
        if dff.empty:
            continue
        
        streamflow_data = pd.DataFrame(dff)
        streamflow_data['Timestamp'] = pd.to_datetime(dff.index)
        
        # Calculate water year and day of water year
        streamflow_data['WaterYear'] = streamflow_data['Timestamp'].apply(get_water_year)
        streamflow_data['DayOfWaterYear'] = streamflow_data['Timestamp'].apply(get_day_of_water_year)

        # Check data completeness for each water year
        yearly_data_counts = streamflow_data.groupby('WaterYear').size()
        complete_years = yearly_data_counts[yearly_data_counts >= 0.9 * 365].index  # Keep water years with at least 90% of data
        incomplete_years = set(streamflow_data['WaterYear']) - set(complete_years)

        if incomplete_years:
            print(f"Gauge: {gauge} - Excluded water years due to insufficient data (>10% missing): {sorted(incomplete_years)}")
        
        # Filter out incomplete water years
        streamflow_data = streamflow_data[streamflow_data['WaterYear'].isin(complete_years)]
        
        # Skip if insufficient complete water years are available
        if len(complete_years) < min_years_required:
            print(f"Gauge: {gauge} - Insufficient complete water years ({len(complete_years)} years, need {min_years_required}), skipping")
            continue

        # Pivot by water year and day of water year
        yearly_flow = streamflow_data.pivot(index='WaterYear', columns='DayOfWaterYear', values=gauge).fillna(0)
        yearly_flow_array = yearly_flow.to_numpy()

        day_mode_0_final, day_mode_1_final = pulse_method_strict_fix(yearly_flow_array)
        
        # Create a dataframe to hold the freshet results for the current gauge
        freshet_results_pulse = pd.DataFrame({
            'WaterYear': yearly_flow.index,
            'Mode_0_Freshet_Day': day_mode_0_final,
            'Mode_1_Freshet_Day': day_mode_1_final
        })

        # Convert water year day to actual date
        freshet_results_pulse['Mode_0_Freshet_Date'] = freshet_results_pulse.apply(
            lambda row: water_year_day_to_date(row['WaterYear'], row['Mode_0_Freshet_Day']), axis=1
        )

        freshet_results_pulse['Mode_1_Freshet_Date'] = freshet_results_pulse.apply(
            lambda row: water_year_day_to_date(row['WaterYear'], row['Mode_1_Freshet_Day']), axis=1
        )
        
        # Add Year column for compatibility (calendar year of freshet date)
        freshet_results_pulse['Year'] = freshet_results_pulse['Mode_0_Freshet_Date'].dt.year

        freshet_results_all[gauge] = freshet_results_pulse
        
        # Calculate the trend in Mode_0 Freshet Day using Theil-Sen and modified Mann-Kendall
        trend_per_decade, pval, trend, intercept, _ = trend_analysis.calc_trend_and_pval(freshet_results_pulse['Mode_0_Freshet_Day'].values)
        trend_per_decade = trend * 10  # Convert trend to days/decade
        
        trends_results.append({
            'Gauge': gauge,
            'Trend (days/decade)': trend_per_decade,
            'Mann-Kendall p-value': pval,
            'Mann-Kendall Trend': trend,
            'Number of years': len(complete_years)
        })
    
    # Convert trends results to a DataFrame
    trends_df = pd.DataFrame(trends_results)
    
    return freshet_results_all, trends_df

def run_centroid_timing_analysis_for_all_gauges(df):
    centroid_results_all = {}
    trends_results = []
    
    # Calculate minimum required years (80% of period length)
    period_length = df.index.year.max() - df.index.year.min() + 1
    min_years_required = int(0.8 * period_length)
    print(f"\nPeriod length: {period_length} years")
    print(f"Minimum years required (80%): {min_years_required} years")
    
    for gauge in df.columns:
        print('processing %s' % gauge)
        dff = df[gauge].dropna()  # Drop missing data
        if dff.empty:
            continue
        
        # Convert to DataFrame and set up datetime info
        streamflow_data = pd.DataFrame(dff)
        streamflow_data['Timestamp'] = pd.to_datetime(dff.index)
        streamflow_data['Year'] = streamflow_data['Timestamp'].dt.year
        streamflow_data['Month'] = streamflow_data['Timestamp'].dt.month
        
        # Adjust for water year: Set the year to the previous year for dates between October and December
        streamflow_data.loc[streamflow_data['Month'] >= 10, 'Year'] += 1
        streamflow_data['WaterYear'] = streamflow_data['Year']
        
        # Check data completeness for each water year
        yearly_data_counts = streamflow_data.groupby('WaterYear').size()
        complete_years = yearly_data_counts[yearly_data_counts >= 0.9 * 365].index  # Keep years with at least 90% of data
        incomplete_years = set(streamflow_data['WaterYear']) - set(complete_years)

        if incomplete_years:
            print(f"Gauge: {gauge} - Excluded water years due to insufficient data (>10% missing): {sorted(incomplete_years)}")
        
        # Filter out incomplete years
        streamflow_data = streamflow_data[streamflow_data['WaterYear'].isin(complete_years)]
        
        # Skip if insufficient complete years are available
        if len(complete_years) < min_years_required:
            print(f"Gauge: {gauge} - Insufficient complete years ({len(complete_years)} years, need {min_years_required}), skipping")
            continue

        # Correctly calculate the DayOfWaterYear
        streamflow_data['DayOfWaterYear'] = streamflow_data['Timestamp'].apply(calculate_day_of_water_year)

        # Calculate the 50% centroid of timing for each water year
        def calculate_centroid_of_timing(df):
            df = df.sort_values('DayOfWaterYear')
            df['CumulativeFlow'] = df[gauge].cumsum()
            total_flow = df[gauge].sum()
            # Find the day when 50% of the total flow is reached
            target_flow = total_flow * 0.5
            centroid_day_row = df.loc[df['CumulativeFlow'] >= target_flow].iloc[0]
            return centroid_day_row['DayOfWaterYear']

        centroid_timing = streamflow_data.groupby('WaterYear').apply(calculate_centroid_of_timing).dropna()

        # Create the datetime object for the centroid of timing for each water year
        def calculate_centroid_datetime(water_year, day_of_water_year):
            start_of_water_year = pd.Timestamp(year=water_year - 1, month=10, day=1)
            return start_of_water_year + pd.Timedelta(days=day_of_water_year - 1)

        centroid_dates = [calculate_centroid_datetime(wy, doy) for wy, doy in zip(centroid_timing.index, centroid_timing.values)]
        
        # Create a dataframe to hold the centroid results for the current gauge
        centroid_results = pd.DataFrame({
            'WaterYear': centroid_timing.index,
            'CentroidOfTiming': centroid_timing.values,
            'CentroidDate': centroid_dates  # Add the datetime column
        })
        
        centroid_results_all[gauge] = centroid_results
        
        # Calculate the trend in centroid timing using Theil-Sen and modified Mann-Kendall
        trend_per_decade, pval, trend, intercept, _ = trend_analysis.calc_trend_and_pval(centroid_results['CentroidOfTiming'].values)
        trend_per_decade = trend * 10  # Convert trend to days/decade
        
        trends_results.append({
            'Gauge': gauge,
            'Trend (days/decade)': trend_per_decade,
            'Mann-Kendall p-value': pval,
            'Mann-Kendall Trend': trend,
            'Number of years': len(complete_years)
        })
    
    # Convert trends results to a DataFrame
    trends_df = pd.DataFrame(trends_results)
    
    return centroid_results_all, trends_df

def run_peak_flow_timing_analysis_for_all_gauges(df):
    """
    Calculate the timing of annual peak flow for each gauge.
    Similar to centroid timing analysis but finds the day of maximum flow in each water year.
    
    Args:
        df: DataFrame with datetime index and gauge columns
        
    Returns:
        peak_results_all: Dictionary with peak flow timing results for each gauge
        trends_df: DataFrame with trend statistics
    """
    peak_results_all = {}
    trends_results = []
    
    # Calculate minimum required years (80% of period length)
    period_length = df.index.year.max() - df.index.year.min() + 1
    min_years_required = int(0.8 * period_length)
    print(f"\nPeriod length: {period_length} years")
    print(f"Minimum years required (80%): {min_years_required} years")
    
    for gauge in df.columns:
        print('processing %s' % gauge)
        dff = df[gauge].dropna()  # Drop missing data
        if dff.empty:
            continue
        
        # Convert to DataFrame and set up datetime info
        streamflow_data = pd.DataFrame(dff)
        streamflow_data['Timestamp'] = pd.to_datetime(dff.index)
        streamflow_data['Year'] = streamflow_data['Timestamp'].dt.year
        streamflow_data['Month'] = streamflow_data['Timestamp'].dt.month
        
        # Adjust for water year
        streamflow_data.loc[streamflow_data['Month'] >= 10, 'Year'] += 1
        streamflow_data['WaterYear'] = streamflow_data['Year']
        
        # Calculate day of water year
        streamflow_data['DayOfWaterYear'] = streamflow_data['Timestamp'].apply(calculate_day_of_water_year)

        # Check data completeness for each water year
        yearly_data_counts = streamflow_data.groupby('WaterYear').size()
        complete_years = yearly_data_counts[yearly_data_counts >= 0.9 * 365].index  # Keep years with at least 90% of data
        incomplete_years = set(streamflow_data['WaterYear']) - set(complete_years)

        if incomplete_years:
            print(f"Gauge: {gauge} - Excluded water years due to insufficient data (>10% missing): {sorted(incomplete_years)}")
        
        # Filter out incomplete years
        streamflow_data = streamflow_data[streamflow_data['WaterYear'].isin(complete_years)]
        
        # Skip if insufficient complete years are available
        if len(complete_years) < min_years_required:
            print(f"Gauge: {gauge} - Insufficient complete years ({len(complete_years)} years, need {min_years_required}), skipping")
            continue

        # Find the day of peak flow for each water year
        def find_peak_flow_timing(df):
            peak_idx = df[gauge].idxmax()
            return df.loc[peak_idx, 'DayOfWaterYear']

        peak_timing = streamflow_data.groupby('WaterYear').apply(find_peak_flow_timing).dropna()

        # Calculate peak flow dates
        def calculate_peak_datetime(water_year, day_of_water_year):
            start_of_water_year = pd.Timestamp(year=water_year - 1, month=10, day=1)
            return start_of_water_year + pd.Timedelta(days=day_of_water_year - 1)

        peak_dates = [calculate_peak_datetime(wy, doy) for wy, doy in zip(peak_timing.index, peak_timing.values)]
        
        # Create results DataFrame
        peak_results = pd.DataFrame({
            'WaterYear': peak_timing.index,
            'PeakFlowDay': peak_timing.values,
            'PeakFlowDate': peak_dates
        })
        
        peak_results_all[gauge] = peak_results
        
        # Calculate trend in peak flow timing using Theil-Sen and modified Mann-Kendall
        trend_per_decade, pval, trend, intercept, _ = trend_analysis.calc_trend_and_pval(peak_results['PeakFlowDay'].values)
        trend_per_decade = trend * 10  # Convert trend to days/decade
        
        trends_results.append({
            'Gauge': gauge,
            'Trend (days/decade)': trend_per_decade,
            'Mann-Kendall p-value': pval,
            'Mann-Kendall Trend': trend,
            'Number of years': len(complete_years)
        })
    
    # Convert trends results to a DataFrame
    trends_df = pd.DataFrame(trends_results)
    
    return peak_results_all, trends_df

def prepare_and_plot(trends_df, ax, title, label, vmin, vmax, subplot_letter):
    trends_df = trends_df.set_index('Gauge').sort_index()
    trends_df.index = trends_df.index.astype('int')
    
    # Double check to ensure gauge 66 is excluded for 1973-2023 period
    if start_year == 1973 and '66' in trends_df.index:
        print(f"Warning: Gauge 66 found in trends data for {start_year}-2023 period when it should have been excluded")
        trends_df = trends_df.drop('66', errors='ignore')
    
    merged_gdf = gauges_gdf.merge(trends_df, left_index=True, right_index=True)

    plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
    merged_gdf.loc[mask].plot(
        column='Trend (days/decade)', ax=ax, cmap=colormap, legend=False, 
        s=200, zorder=2, vmin=vmin, vmax=vmax
    )
    
    significant_points = merged_gdf.loc[mask][merged_gdf.loc[mask]['Mann-Kendall p-value'] < 0.05]
    ax.plot(
        significant_points.geometry.x, significant_points.geometry.y,
        marker='o', markersize=22, markerfacecolor='none', markeredgecolor='k',
        linestyle='none', markeredgewidth=2, zorder=3
    )
    
    # Add subplot letter to the title
    title_with_letter = f"{subplot_letter}) {title}"
    ax.set_title(title_with_letter, size=20, y=0.95) 

def run_analysis_for_period(start_year, end_year):
    """
    Run timing analysis for a specific period.
    
    Args:
        start_year: Start year of analysis period
        end_year: End year of analysis period
    """
    df_with_data_over_thresh = trend_analysis.return_df(df_with_data, start_year, end_year, missing_data_threshold)
    print(f"\nBefore dropping gauges - Available gauges: {sorted(df_with_data_over_thresh.columns.tolist())}")
    
    # Drop gauges with human-influenced data (7, 13, 102) and jökulsá á dal (43 - combined series from multiple gauges) and Álftafitjakvísl (96 - influenced by leakage from Þórisvatn reservoir)
    gauges_to_drop = ['7','13', '102','43','96']
    
    # Drop gauge 66 for 1973-2023 period since its data starts in 1982
    if start_year == 1973:
        gauges_to_drop.append('66')
    
    df_with_data_over_thresh = df_with_data_over_thresh.drop(columns=gauges_to_drop, errors='ignore')
    print(f"After dropping gauges - Available gauges: {sorted(df_with_data_over_thresh.columns.tolist())}")
    
    # Check if filled data exists
    period_folder = base_folder / f"{start_year}_{end_year}"
    period_folder.mkdir(parents=True, exist_ok=True)
    
    filled_file = period_folder / f'filled_{start_year}_{end_year}_missingthresh_{missing_data_threshold}.csv'
    if filled_file.exists():
        filled = pd.read_csv(filled_file, index_col=0)
        filled.index = pd.to_datetime(filled.index)
    else:
        filled = fill_missing_data(df_with_data_over_thresh)
        filled.to_csv(filled_file)
    
    print(f"After filling data - Available gauges: {sorted(filled.columns.tolist())}")
    
    if plot_gap_filling_plots:
        gap_fill_plots_folder = period_folder / 'gap_filling_plots'
        gap_fill_plots_folder.mkdir(parents=True, exist_ok=True)
        plot_freshet_series_with_subplots(filled, df_with_data_over_thresh, None, gap_fill_plots_folder, years_per_plot=8, freshet=False)
    
    # Run freshet timing analysis
    print("\nCalculating freshet timing...\n")
    freshet_results_all, freshet_trends_df = run_freshet_analysis_for_all_gauges(filled)
    
    # Filter out excluded gauges from freshet analysis (high BFI / groundwater-dominated)
    if start_year == 1973:
        freshet_exclude = FRESHET_EXCLUDE_PERIOD1
    else:
        freshet_exclude = FRESHET_EXCLUDE_PERIOD2
    
    freshet_exclude_str = [str(g) for g in freshet_exclude]
    excluded_gauges = [g for g in freshet_results_all.keys() if g in freshet_exclude_str]
    if excluded_gauges:
        print(f"\nExcluding {len(excluded_gauges)} gauges from freshet analysis (high BFI): {excluded_gauges}")
        for g in excluded_gauges:
            del freshet_results_all[g]
        freshet_trends_df = freshet_trends_df[~freshet_trends_df['Gauge'].isin(freshet_exclude_str)]
        print(f"Remaining gauges for freshet analysis: {len(freshet_results_all)}")
    
    # Run centroid timing analysis
    print("\nCalculating centroid timing...\n")
    centroid_results_all, centroid_trends_df = run_centroid_timing_analysis_for_all_gauges(filled)
    
    # Run peak flow timing analysis
    print("\nCalculating peak flow timing...\n")
    peak_results_all, peak_trends_df = run_peak_flow_timing_analysis_for_all_gauges(filled)
    
    # Save all trend results
    freshet_trends_file = period_folder / f'freshet_trends_{start_year}_{end_year}_ts_mk.csv'
    centroid_trends_file = period_folder / f'centroid_trends_{start_year}_{end_year}_ts_mk.csv'
    peak_trends_file = period_folder / f'peak_flow_trends_{start_year}_{end_year}_ts_mk.csv'
    
    freshet_trends_df.to_csv(freshet_trends_file, index=False)
    centroid_trends_df.to_csv(centroid_trends_file, index=False)
    peak_trends_df.to_csv(peak_trends_file, index=False)
    
    print(f"Saved timing trends to {period_folder}")
    
    # Plot hydrographs with Mode 0 and Mode 1 freshet markers
    if plot_freshet_hydrographs:
        print("\nPlotting freshet hydrographs with Mode 0 and Mode 1 markers...")
        freshet_hydrograph_folder = period_folder / 'freshet_hydrographs'
        freshet_hydrograph_folder.mkdir(parents=True, exist_ok=True)
        plot_freshet_series_with_subplots(filled, df_with_data_over_thresh, freshet_results_all, 
                                          freshet_hydrograph_folder, years_per_plot=10, freshet=True)
    
    # Plot mean seasonal hydrographs
    if plot_mean_seasonal:
        print("\nPlotting mean seasonal hydrographs...")
        mean_seasonal_folder = period_folder / 'mean_seasonal_hydrographs'
        mean_seasonal_folder.mkdir(parents=True, exist_ok=True)
        plot_mean_seasonal_hydrograph(filled, mean_seasonal_folder, use_water_year=True)
    
    # Create folders for timing series plots
    freshet_series_folder = period_folder / 'freshet_annual_series'
    centroid_series_folder = period_folder / 'centroid_annual_series'
    peak_series_folder = period_folder / 'peak_flow_annual_series'
    
    for folder in [freshet_series_folder, centroid_series_folder, peak_series_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Process and plot freshet timing results
    print("\nProcessing freshet timing results...")
    for gauge_id, freshet_data in freshet_results_all.items():
        trend_row = freshet_trends_df.loc[freshet_trends_df['Gauge'] == gauge_id].iloc[0]
        years = freshet_data['Year'].values
        days = freshet_data['Mode_1_Freshet_Day'].values
        slope = trend_row['Trend (days/decade)'] / 10
        intercept = np.mean(days) - slope * (np.mean(years) - years[0])
        
        trend_data = pd.DataFrame({
            'trend_days_per_year': [trend_row['Trend (days/decade)'] / 10],
            'trend_days_per_decade': [trend_row['Trend (days/decade)']],
            'pval': [trend_row['Mann-Kendall p-value']],
            'intercept': [intercept]
        })
        
        title = f"{gauges.loc[int(gauge_id)]['river']}, {gauges.loc[int(gauge_id)]['name']}, {gauges.loc[int(gauge_id)]['V_no']}" if int(gauge_id) in gauges.index else f"Gauge {gauge_id}"
        
        plot_timing_series(
            gauge_id=str(gauge_id),
            timing_data=freshet_data,
            trend_results=trend_data,
            title=title,
            ylabel='Day of Year',
            save_path=freshet_series_folder
        )
    
    # Process and plot centroid timing results
    print("\nProcessing centroid timing results...")
    for gauge_id, centroid_data in centroid_results_all.items():
        trend_row = centroid_trends_df.loc[centroid_trends_df['Gauge'] == gauge_id].iloc[0]
        years = centroid_data['WaterYear'].values
        days = centroid_data['CentroidOfTiming'].values
        slope = trend_row['Trend (days/decade)'] / 10
        intercept = np.mean(days) - slope * (np.mean(years) - years[0])
        
        trend_data = pd.DataFrame({
            'trend_days_per_year': [trend_row['Trend (days/decade)'] / 10],
            'trend_days_per_decade': [trend_row['Trend (days/decade)']],
            'pval': [trend_row['Mann-Kendall p-value']],
            'intercept': [intercept]
        })
        
        title = f"{gauges.loc[int(gauge_id)]['river']}, {gauges.loc[int(gauge_id)]['name']}, {gauges.loc[int(gauge_id)]['V_no']}" if int(gauge_id) in gauges.index else f"Gauge {gauge_id}"
        
        plot_timing_series(
            gauge_id=str(gauge_id),
            timing_data=centroid_data,
            trend_results=trend_data,
            title=title,
            ylabel='Day of Water Year',
            save_path=centroid_series_folder
        )
    
    # Process and plot peak flow timing results
    print("\nProcessing peak flow timing results...")
    for gauge_id, peak_data in peak_results_all.items():
        trend_row = peak_trends_df.loc[peak_trends_df['Gauge'] == gauge_id].iloc[0]
        years = peak_data['WaterYear'].values
        days = peak_data['PeakFlowDay'].values
        slope = trend_row['Trend (days/decade)'] / 10
        intercept = np.mean(days) - slope * (np.mean(years) - years[0])
        
        trend_data = pd.DataFrame({
            'trend_days_per_year': [trend_row['Trend (days/decade)'] / 10],
            'trend_days_per_decade': [trend_row['Trend (days/decade)']],
            'pval': [trend_row['Mann-Kendall p-value']],
            'intercept': [intercept]
        })
        
        title = f"{gauges.loc[int(gauge_id)]['river']}, {gauges.loc[int(gauge_id)]['name']}, {gauges.loc[int(gauge_id)]['V_no']}" if int(gauge_id) in gauges.index else f"Gauge {gauge_id}"
        
        plot_timing_series(
            gauge_id=str(gauge_id),
            timing_data=peak_data,
            trend_results=trend_data,
            title=title,
            ylabel='Day of Water Year',
            save_path=peak_series_folder
        )

# Run the analysis for each period and save the results
for start_year, end_year in time_periods:
    run_analysis_for_period(start_year, end_year)

# Parameters for plotting
overall_vmin, overall_vmax = float('inf'), -float('inf')

# Load the data and calculate the overall vmin and vmax
results = {}
for start_year, end_year in time_periods:
    period_folder = base_folder / f"{start_year}_{end_year}"
    
    # Load trend files from the period folder
    trends_df = pd.read_csv(period_folder / f'freshet_trends_{start_year}_{end_year}_ts_mk.csv')
    centroid_trends_df = pd.read_csv(period_folder / f'centroid_trends_{start_year}_{end_year}_ts_mk.csv')
    peak_trends_df = pd.read_csv(period_folder / f'peak_flow_trends_{start_year}_{end_year}_ts_mk.csv')
    
    # Filter out gauge 66 for 1973-2023 period
    if start_year == 1973:
        trends_df = trends_df[trends_df['Gauge'] != '66']
        centroid_trends_df = centroid_trends_df[centroid_trends_df['Gauge'] != '66']
        peak_trends_df = peak_trends_df[peak_trends_df['Gauge'] != '66']
    
    results[(start_year, end_year)] = (trends_df, centroid_trends_df, peak_trends_df)
    
    # Calculate vmin and vmax for all trends
    for df in [trends_df, centroid_trends_df, peak_trends_df]:
        trends_min = df['Trend (days/decade)'].min()
        trends_max = df['Trend (days/decade)'].max()
        overall_vmin = min(overall_vmin, trends_min)
        overall_vmax = max(overall_vmax, trends_max)

print(f"Overall vmin: {overall_vmin}, Overall vmax: {overall_vmax}")

# We manually set the vmax/vmin
overall_vmax = 12
overall_vmin = -12
extend = 'both'

# Create a 2x3 subplot for plotting
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
fig.patch.set_facecolor('white')

# Define subplot letters
subplot_letters = [['a', 'b', 'c'], ['d', 'e', 'f']]

# Loop over the time periods and plot results with the common vmin and vmax
for i, (start_year, end_year) in enumerate(time_periods):
    trends_df, centroid_trends_df, peak_trends_df = results[(start_year, end_year)]
    row = i  # 0 for 1973-2023, 1 for 1993-2023

    # Plot freshet results with a common color scale
    prepare_and_plot(
        trends_df, axes[row, 0], f'Trend in the start of spring freshet ({start_year}-{end_year})',
        f'Trend in start of spring freshet from {start_year}-{end_year} (days/decade)',
        vmin=overall_vmin, vmax=overall_vmax,
        subplot_letter=subplot_letters[row][0]
    )
    catchments.loc[trends_df.dropna()['Gauge'].values].plot(facecolor='none', edgecolor='black', ax=axes[row, 0], zorder=3, lw=0.25)

    # Plot centroid of timing results with a common color scale
    prepare_and_plot(
        centroid_trends_df, axes[row, 1], f'Trend in the centroid of timing ({start_year}-{end_year})',
        f'Trend in centroid of timing from {start_year}-{end_year} (days/decade)',
        vmin=overall_vmin, vmax=overall_vmax,
        subplot_letter=subplot_letters[row][1]
    )
    catchments.loc[centroid_trends_df.dropna()['Gauge'].values].plot(facecolor='none', edgecolor='black', ax=axes[row, 1], zorder=3, lw=0.25)

    # Plot peak flow results with a common color scale
    prepare_and_plot(
        peak_trends_df, axes[row, 2], f'Trend in peak flow timing ({start_year}-{end_year})',
        f'Trend in peak flow timing from {start_year}-{end_year} (days/decade)',
        vmin=overall_vmin, vmax=overall_vmax,
        subplot_letter=subplot_letters[row][2]
    )
    catchments.loc[peak_trends_df.dropna()['Gauge'].values].plot(facecolor='none', edgecolor='black', ax=axes[row, 2], zorder=3, lw=0.25)

# Add a shared colorbar below the plots
cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=overall_vmin, vmax=overall_vmax), orientation='horizontal', extend=extend)
cb.set_label('Trend (days/decade)', size=maps_fontsize)
cb.ax.tick_params(labelsize=maps_fontsize)

# Save and show the figure
plt.tight_layout(rect=[0, 0.1, 1, 1])
save_path = os.path.join(maps_path, 'freshet_centroid_peak_flow_trends_ts_mk.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
save_path = os.path.join(maps_path, 'freshet_centroid_peak_flow_trends_ts_mk.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Generate summary statistics for timing trends
# =============================================================================
print("\n" + "="*60)
print("Generating summary statistics for timing trends...")
print("="*60)

def calculate_trend_summary(df, metric_name, filter_mask, period_start_year, gauges_gdf_index, debug=False):
    """Calculate summary statistics for a trend dataframe.
    
    Mirrors exactly the filtering done in prepare_and_plot:
    1. Set index to 'Gauge' and convert to int
    2. Merge with gauges_gdf
    3. Apply mask (degimpact != 's')
    
    Args:
        df: DataFrame with trend results
        metric_name: Name of the metric for the summary
        filter_mask: Boolean mask for filtering gauges (degimpact != 's')
        period_start_year: Start year of the period (for gauge 66 exclusion)
        gauges_gdf_index: Index from gauges_gdf for proper alignment
        debug: If True, print debug information
    """
    trend_col = 'Trend (days/decade)'
    pval_col = 'Mann-Kendall p-value'
    
    # Mirror prepare_and_plot: set index and convert to int
    df_indexed = df.set_index('Gauge').sort_index()
    df_indexed.index = df_indexed.index.astype('int')
    
    # Exclude gauge 66 for 1973-2023 period (same as prepare_and_plot)
    if period_start_year == 1973 and 66 in df_indexed.index:
        df_indexed = df_indexed.drop(66, errors='ignore')
    
    # Only keep gauges that exist in both df_indexed and the mask
    # This mirrors: merged_gdf = gauges_gdf.merge(trends_df, left_index=True, right_index=True)
    common_indices = df_indexed.index.intersection(filter_mask.index)
    
    # Apply the mask (degimpact != 's')
    df_filtered = df_indexed.loc[common_indices]
    mask_filtered = filter_mask.loc[common_indices]
    df_filtered = df_filtered.loc[mask_filtered]
    
    # Get valid (non-null) trends
    valid = df_filtered[trend_col].notna()
    trends = df_filtered.loc[valid, trend_col]
    pvals = df_filtered.loc[valid, pval_col]
    
    # Use >= 0 for positive to include any near-zero trends (ensures all significant trends are counted)
    positive = (trends >= 0).sum()
    positive_sig = ((trends >= 0) & (pvals < 0.05)).sum()
    negative = (trends < 0).sum()
    negative_sig = ((trends < 0) & (pvals < 0.05)).sum()
    total_sig = (pvals < 0.05).sum()
    
    if debug:
        print(f"\n  DEBUG {metric_name} ({period_start_year}):")
        print(f"    Gauges in df_indexed: {len(df_indexed)}")
        print(f"    Common indices with mask: {len(common_indices)}")
        print(f"    Gauges after mask filter: {len(df_filtered)}")
        print(f"    Total significant (p<0.05): {total_sig}")
        print(f"    Positive (>=0) sig: {positive_sig}, Negative (<0) sig: {negative_sig}")
        print(f"    Sum of pos+neg sig: {positive_sig + negative_sig}")
        sig_mask = pvals < 0.05
        if sig_mask.sum() > 0:
            sig_indices = trends[sig_mask].index
            print(f"    Significant gauges: {sorted(sig_indices.tolist())}")
            for idx in sig_indices:
                t = df_filtered.loc[idx, trend_col]
                p = df_filtered.loc[idx, pval_col]
                direction = "positive" if t >= 0 else "negative"
                print(f"      Gauge {idx}: trend={t:.4f} ({direction}), p={p:.4f}")
    
    return {
        'Metric': metric_name,
        'Positive': positive,
        'Positive significant': positive_sig,
        'Negative': negative,
        'Negative significant': negative_sig,
        'Total': len(trends)
    }

# Create summary for each time period
summary_rows = []
for period_start_year, period_end_year in time_periods:
    trends_df, centroid_trends_df, peak_trends_df = results[(period_start_year, period_end_year)]
    period_label = f"{period_start_year}-{period_end_year}"
    
    # Enable debug for spring freshet period 2 to diagnose discrepancy
    debug_freshet = (period_start_year == 1993)
    
    summary_rows.append({
        'Period': period_label,
        **calculate_trend_summary(trends_df, 'Spring freshet', mask, period_start_year, gauges_gdf.index, debug=debug_freshet)
    })
    summary_rows.append({
        'Period': period_label,
        **calculate_trend_summary(centroid_trends_df, 'Centroid of timing', mask, period_start_year, gauges_gdf.index)
    })
    summary_rows.append({
        'Period': period_label,
        **calculate_trend_summary(peak_trends_df, 'Peak flow timing', mask, period_start_year, gauges_gdf.index)
    })

# Create summary dataframe
summary_df = pd.DataFrame(summary_rows)

# Save to CSV
summary_csv_path = os.path.join(maps_path, 'timing_trends_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary saved to: {summary_csv_path}")

# Print summary to console
print("\nTiming Trends Summary:")
print("-" * 80)
print(summary_df.to_string(index=False))

# Create heatmap visualization
import seaborn as sns
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 14

# Create separate heatmaps for each period
for start_year, end_year in time_periods:
    period_label = f"{start_year}-{end_year}"
    period_data = summary_df[summary_df['Period'] == period_label].copy()
    period_data = period_data.set_index('Metric')[['Positive', 'Negative', 'Positive significant', 'Negative significant']]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create annotation strings
    annotations = period_data.astype(int).astype(str)
    
    sns.heatmap(period_data, annot=annotations, fmt='', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Number of gauges'})
    ax.set_title(f'Timing Trends Summary ({period_label})', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('')
    
    plt.tight_layout()
    heatmap_path = os.path.join(maps_path, f'timing_trends_summary_heatmap_{start_year}_{end_year}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    heatmap_path_pdf = os.path.join(maps_path, f'timing_trends_summary_heatmap_{start_year}_{end_year}.pdf')
    plt.savefig(heatmap_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")

# Create combined heatmap with both periods side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for i, (start_year, end_year) in enumerate(time_periods):
    period_label = f"{start_year}-{end_year}"
    period_data = summary_df[summary_df['Period'] == period_label].copy()
    period_data = period_data.set_index('Metric')[['Positive', 'Negative', 'Positive significant', 'Negative significant']]
    
    annotations = period_data.astype(int).astype(str)
    
    cmap = 'Blues' if i == 0 else 'Greens'
    panel_label = 'a)' if i == 0 else 'b)'
    
    sns.heatmap(period_data, annot=annotations, fmt='', cmap=cmap, ax=axes[i],
                cbar_kws={'label': 'Number of gauges'})
    axes[i].set_title(f'{panel_label} Timing Trends ({period_label})', fontsize=16)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_ylabel('')

plt.tight_layout()
combined_heatmap_path = os.path.join(maps_path, 'timing_trends_summary_heatmap_combined.png')
plt.savefig(combined_heatmap_path, dpi=300, bbox_inches='tight')
combined_heatmap_path_pdf = os.path.join(maps_path, 'timing_trends_summary_heatmap_combined.pdf')
plt.savefig(combined_heatmap_path_pdf, dpi=300, bbox_inches='tight')
plt.close()
print(f"Combined heatmap saved to: {combined_heatmap_path}")

print("\n" + "="*60)
print("Summary generation complete!")
print("="*60) 