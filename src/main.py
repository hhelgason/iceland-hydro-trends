"""
Main Script for Streamflow Trend Analysis in Iceland

This script is the main entry point for analyzing trends in streamflow data across Iceland.
It is part of the supplementary material for the paper "Understanding Changes in Iceland's Streamflow Dynamics in Response to Climate Change".

The script performs the following tasks:
1. Loads streamflow data from LamaH-Ice dataset
2. Reads gauge locations and catchment characteristics
3. Calculates trends in various streamflow metrics
4. Generates visualizations of the results

Dependencies:
- pandas: Data manipulation and analysis
- geopandas: Spatial data operations
- pathlib: Path manipulation
- pickle: Data serialization
- trend_analysis: Custom module for trend calculations
- plotting: Custom module for visualization

Author: Hordur Bragi Helgason
"""

# Standard library imports
import os
import pickle
from pathlib import Path

# Third party imports
import pandas as pd
import geopandas as gpd

# Local imports
import trend_analysis
import plotting
from config import (
    LAMAH_ICE_BASE_PATH,
    OUTPUT_DIR,
    STREAMFLOW_DATA_PATH,
    START_YEAR,
    END_YEAR,
    MISSING_DATA_THRESHOLD
)

# Create output folder for results
output_dir = os.path.join(OUTPUT_DIR, "results_lamah_data")
os.makedirs(output_dir, exist_ok=True)

# Define paths for saved results
results_file = os.path.join(output_dir, f'results_{START_YEAR}_{END_YEAR}.csv')
valid_data_file = os.path.join(output_dir, f"valid_data_dict_{START_YEAR}_{END_YEAR}.p")
invalid_data_file = os.path.join(output_dir, f"invalid_data_dict_{START_YEAR}_{END_YEAR}.p")
merged_gdf_file = os.path.join(output_dir, f'merged_gdf_{START_YEAR}_{END_YEAR}.gpkg')

# Read streamflow data
print("Loading streamflow data...")
df_with_data = pd.read_csv(STREAMFLOW_DATA_PATH, index_col=0, parse_dates=True)

# Read spatial data
print("Loading spatial data...")
# Read the gauges shapefile that contains the indices and V numbers
gauges_shapefile = LAMAH_ICE_BASE_PATH / "D_gauges/3_shapefiles/gauges.shp"
gauges_gdf = gpd.read_file(gauges_shapefile)
gauges_gdf = gauges_gdf.set_index('id')
gauges_gdf = gauges_gdf.set_crs('epsg:3057')

# Read catchment characteristics and extract glacier fraction and degree of impact
catchment_attributes_file = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"
catchments_chara = pd.read_csv(catchment_attributes_file, sep=';')
catchments_chara = catchments_chara.set_index('id')
gauges_gdf['g_frac'] = catchments_chara['g_frac']
gauges_gdf['degimpact'] = catchments_chara['degimpact']  # Add degree of impact for filtering anthropogenic influences

# Read catchment boundaries
catchments_shapefile = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/3_shapefiles/Basins_A.shp"
catchments = gpd.read_file(catchments_shapefile)
catchments = catchments.set_index('id')
catchments = catchments.set_crs('epsg:3057')

# Create folders for different types of plots
print("Creating output directories...")
daily_timeseries_path, annual_autocorrelation_path, maps_path, raster_trends_path, \
seasonal_trends_path_mod_ts, annual_trends_path, monthly_trends_path_mod_ts, \
annual_mean_flow_path, annual_cv_path, annual_std_path, flashiness_path, sequences_path, \
baseflow_index_path, baseflow_series_path = plotting.create_folders(OUTPUT_DIR, START_YEAR, END_YEAR)

# Check if trend results already exist and load them if they do
if (os.path.exists(results_file) and 
    os.path.exists(valid_data_file) and 
    os.path.exists(invalid_data_file) and 
    os.path.exists(merged_gdf_file)):
    print("Loading existing trend results...")
    results = pd.read_csv(results_file, sep=';', index_col=0)
    valid_data_dict = pickle.load(open(valid_data_file, "rb"))
    invalid_data_dict = pickle.load(open(invalid_data_file, "rb"))
    merged_gdf = gpd.read_file(merged_gdf_file)
    # Restore index from column when loading from GPKG
    if 'index' in merged_gdf.columns:
        merged_gdf = merged_gdf.set_index('index')
else:
    print("Calculating trends...")
    # Calculate trends in the streamflow timeseries
    df = trend_analysis.return_df(df_with_data, START_YEAR, END_YEAR, MISSING_DATA_THRESHOLD)

    # Calculate trends and get results
    results, valid_data_dict, invalid_data_dict = trend_analysis.calc_all_trends(df, baseflow_series_path=baseflow_series_path)
    results.index = results.index.astype('int')
    results = results.astype('float').round(3)
    merged_gdf = gauges_gdf.merge(results, left_index=True, right_index=True)
    merged_gdf["index"] = merged_gdf.index

    # Save results
    print("Saving trend results...")
    results.to_csv(results_file, sep=';')
    pickle.dump(valid_data_dict, open(valid_data_file, "wb"))
    pickle.dump(invalid_data_dict, open(invalid_data_file, "wb"))
    merged_gdf.to_file(merged_gdf_file, driver="GPKG")

# Define which plots to create
# True = plot will be created, False = plot will be skipped
which_plots = {
    # Annual maps
    'annual_map': True,              # Annual mean flow trends
    'annual_std_map': True,          # Annual standard deviation trends
    'annual_cv_map': True,           # Annual coefficient of variation trends
    'flashiness_map': True,          # Annual flashiness index trends
    'rising_falling_map': True,      # Annual rising/falling sequences trends
    'baseflow_index_map': True,      # Annual baseflow index trends
    'low_high_flow_map': True,       # Annual low/high flow trends
    
    # Seasonal maps
    'seasonal_map': True,            # Seasonal mean flow trends
    'seasonal_std_map': True,        # Seasonal standard deviation trends
    'seasonal_cv_map': True,         # Seasonal coefficient of variation trends
    'seasonal_flashiness_map': True, # Seasonal flashiness trends
    'seasonal_baseflow_index_map': True,  # Seasonal baseflow index trends
    'seasonal_rising_falling_map': True,  # Seasonal rising/falling sequences
    'seasonal_low_high_flow_map': True,   # Seasonal low/high flow trends
    
    # Monthly maps
    'monthly_map': False,            # Monthly trend maps (disabled)
    
    # Time series plots
    'annual_series': True,          # Annual mean flow time series
    'annual_std_series': True,      # Annual standard deviation time series
    'annual_cv_series': True,       # Annual CV time series
    'flashiness_series': True,      # Flashiness index time series
    'rising_falling_series': True,  # Rising/falling sequences time series
    'baseflow_index_series': True,   # Baseflow index time series
    'low_flow_series': True,        # Low flow time series
    'high_flow_series': True,       # High flow time series
    
    # Seasonal and monthly time series
    'seasonal_series': True,        # Seasonal mean flow time series
    'seasonal_std_series': True,    # Seasonal standard deviation time series
    'seasonal_cv_series': True,     # Seasonal CV time series
    'seasonal_flashiness_series': True,  # Seasonal flashiness time series
    'seasonal_baseflow_index_series': True,  # Seasonal baseflow index time series
    'seasonal_rising_falling_series': True,  # Seasonal rising/falling sequences
    'seasonal_low_high_flow_series': True,   # Seasonal low/high flow time series
    
    # Additional analyses
    'autocorrelation': False,        # Autocorrelation analysis
    'raster_trends': False           # Raster-based trend analysis
}

# Generate all requested plots
print("Generating plots...")
plotting.plot_trendfigs(
    catchments, which_plots, merged_gdf, START_YEAR, END_YEAR, results,
    valid_data_dict, invalid_data_dict, daily_timeseries_path, annual_autocorrelation_path, maps_path,
    raster_trends_path, seasonal_trends_path_mod_ts, annual_trends_path, monthly_trends_path_mod_ts, 
    annual_mean_flow_path, annual_cv_path, annual_std_path, flashiness_path, sequences_path, baseflow_index_path
)

print("Analysis complete!")
