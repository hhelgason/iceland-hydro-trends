"""
Script for creating formatted tables for manuscript publication.

This script is part of the analysis for the paper:
"Understanding Changes in Iceland's Streamflow Dynamics in Response to Climate Change"

The script processes data from multiple sources and creates formatted tables for publication:
1. Table 1: Basic station information
   - River and station names
   - Impact assessment
   - Observation periods
   - Glacier fractions
2. Table 1b: Period means
   - Streamflow means
   - Temperature means
   - Precipitation means
3. Tables 2a-e: Catchment characteristics
   - Geographic attributes
   - Meteorological and hydrological attributes
   - Soil and geology attributes
   - Land cover attributes
   - Glacier attributes
4. Table 3: Trend results
   - 1973-2023 period
   - 1993-2023 period
   - Annual and seasonal trends

All tables are saved to a single Excel file with multiple sheets.

Dependencies:
    pandas
    numpy
    geopandas
    pickle
    pathlib
    sys
    os
    datetime

Author: Hordur Bragi
Created: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
import geopandas as gpd
import os
from datetime import datetime

def check_file_exists(file_path, description):
    """
    Check if a required file exists and exit with helpful message if not.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to check
    description : str
        Description of the file for error messages
        
    Raises
    ------
    SystemExit
        If the file does not exist
    """
    if not file_path.exists():
        print(f"Error: Could not find {description} at path:")
        print(f"  {file_path}")
        print("\nPlease ensure the file exists and the path is correct.")
        sys.exit(1)

# Define paths
base_path = Path(r"C:\Users\hordurbhe\OneDrive - Landsvirkjun\Changes in streamflow in Iceland")
lamah_ice_base_path = Path(r"C:\Users\hordurbhe\OneDrive - Landsvirkjun\Documents\Vinna\lamah\lamah_ice\lamah_ice")
notebooks_path = base_path / "Notebooks" / "June2024"  # Updated path
data_path = base_path / "data"

# Create output directories
tables_path = notebooks_path / "manuscript_tables"
os.makedirs(tables_path, exist_ok=True)

# Path to valid_data_dict and results
results_path = Path(r"C:\Users\hordurbhe\Not_backed_up\Changes in streamflow in Iceland paper\final_testing_prior_to_committing\results_lamah_data")
valid_data_1973_file = results_path / "valid_data_dict_1973_2023.p"
valid_data_1993_file = results_path / "valid_data_dict_1993_2023.p"

# Define trend results paths
df_1973_path = results_path / 'results_1973_2023.csv'
df_1993_path = results_path / 'results_1993_2023.csv'

# Define output Excel file
output_excel = tables_path / "manuscript_tables.xlsx"

# If the file exists and can't be written to, try with a timestamp
if output_excel.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = tables_path / f"manuscript_tables_{timestamp}.xlsx"

print("Checking file paths...")

# Read the gauges shapefile that contains station information
gauges_shapefile = lamah_ice_base_path / "D_gauges/3_shapefiles/gauges.shp"
check_file_exists(gauges_shapefile, "gauges shapefile")

print("Reading station information...")
gauges = gpd.read_file(gauges_shapefile)
print("\nAvailable columns in gauges shapefile:", gauges.columns.tolist())
gauges = gauges.set_index('id')

# Check and read trend results
check_file_exists(df_1973_path, "1973 trend results CSV file")
check_file_exists(df_1993_path, "1993 trend results CSV file")

print("Reading trend results...")
df_1973_output = pd.read_csv(df_1973_path, index_col=0, sep=';')
df_1993_output = pd.read_csv(df_1993_path, index_col=0, sep=';')

# Add station information to trend results
station_cols = ['river', 'name', 'typimpact', 'degimpact', 'obsbeg_day', 'obsend_day']
df_1973_output = df_1973_output.join(gauges[station_cols])
df_1993_output = df_1993_output.join(gauges[station_cols])

# Rename columns to match expected names
df_1973_output = df_1973_output.rename(columns={'typimpact': 'impact_type', 'degimpact': 'impact_degree', 'obsbeg_day': 'obsbeg', 'obsend_day': 'obsend'})
df_1993_output = df_1993_output.rename(columns={'typimpact': 'impact_type', 'degimpact': 'impact_degree', 'obsbeg_day': 'obsbeg', 'obsend_day': 'obsend'})

# Check and read catchment attributes
catchment_attributes_file = lamah_ice_base_path / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"
check_file_exists(catchment_attributes_file, "catchment attributes CSV file")

print("Reading catchment attributes...")
catchments_chara = pd.read_csv(catchment_attributes_file, sep=';')
catchments_chara = catchments_chara.set_index('id')

# Read hydro indices for BFI
hydro_indices_file = lamah_ice_base_path / "D_gauges/1_attributes/hydro_indices_1981_2018_unfiltered.csv"
check_file_exists(hydro_indices_file, "hydro indices file")

print("Reading hydro indices...")
hydro_indices = pd.read_csv(hydro_indices_file, sep=';')
hydro_indices = hydro_indices.set_index('id')

# Add BFI and glacial fraction to trend results
df_1973_output['BFI'] = hydro_indices['baseflow_index_ladson']
df_1973_output['g_frac'] = catchments_chara['g_frac']
df_1993_output['BFI'] = hydro_indices['baseflow_index_ladson']
df_1993_output['g_frac'] = catchments_chara['g_frac']

# Read streamflow data from valid_data_dict
print("Reading streamflow data...")
with open(valid_data_1973_file, 'rb') as f:
    valid_data_1973 = pickle.load(f)
with open(valid_data_1993_file, 'rb') as f:
    valid_data_1993 = pickle.load(f)

def calculate_mean_flow(valid_data_dict, period='annual'):
    """
    Calculate mean flow for stations from valid_data_dict.
    
    Parameters
    ----------
    valid_data_dict : dict
        Dictionary containing flow data for each station
    period : str, optional
        Period to calculate means for, by default 'annual'
        
    Returns
    -------
    pandas.Series
        Series containing mean flow values indexed by station number
    """
    means = {}
    for station_id in valid_data_dict:
        if isinstance(station_id, tuple) and station_id[1] == period:
            # Get the station number from the tuple
            station_num = station_id[0]
            # Calculate mean flow for this station
            means[station_num] = valid_data_dict[station_id].mean()
    return pd.Series(means)

def calculate_meteo_means(station_id, start_date, end_date):
    """
    Calculate meteorological means for a station over a specified period.
    
    Parameters
    ----------
    station_id : int
        Station identifier
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
        
    Returns
    -------
    tuple
        (temperature_mean, precipitation_mean) in °C and mm/year
        Returns (None, None) if data not available or error occurs
    """
    try:
        # Read meteorological data for the station
        meteo_file = lamah_ice_base_path / f"A_basins_total_upstrm/2_timeseries/daily/meteorological_data/ID_{station_id}.csv"
        if not meteo_file.exists():
            return None, None
        
        # Read the CSV file
        df = pd.read_csv(meteo_file, sep=';')
        
        # Create datetime index
        df['date'] = pd.to_datetime({
            'year': df['YYYY'],
            'month': df['MM'],
            'day': df['DD']
        })
        df = df.set_index('date')
        
        # Calculate means for the period
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_data = df[mask]
        
        if len(period_data) == 0:
            return None, None
        
        # Calculate means
        temp_mean = period_data['2m_temp_mean'].mean()
        prec_mean = period_data['prec'].mean() * 365  # Convert to mm/year
        
        return temp_mean, prec_mean
    except Exception as e:
        print(f"Warning: Error processing meteorological data for station {station_id}: {str(e)}")
        return None, None

print("\nCalculating mean flows...")
# Calculate mean flows for both periods
mean_flow_1973 = calculate_mean_flow(valid_data_1973)
mean_flow_1993 = calculate_mean_flow(valid_data_1993)

# Print some debug information
print("\nNumber of stations with mean flow data (1973-2023):", len(mean_flow_1973))
print("Number of stations with mean flow data (1993-2023):", len(mean_flow_1993))
print("\nSample of mean flows (1973-2023):", mean_flow_1973.head())
print("\nSample of mean flows (1993-2023):", mean_flow_1993.head())

def create_tables_for_manuscript(df_1973, df_1993, catchments_chara, mean_flow_1973, mean_flow_1993):
    """
    Create formatted tables for manuscript publication.
    
    Parameters
    ----------
    df_1973 : DataFrame
        Trend results for 1973-2023 period
    df_1993 : DataFrame
        Trend results for 1993-2023 period
    catchments_chara : DataFrame
        Catchment characteristics data
    mean_flow_1973 : Series
        Mean flows for 1973-2023 period
    mean_flow_1993 : Series
        Mean flows for 1993-2023 period
        
    Returns
    -------
    tuple
        (station_info, means_table, attributes_table_2a, attributes_table_2b,
         attributes_table_2c, attributes_table_2d, attributes_table_2e,
         trends_1973, trends_1993)
        
    Notes
    -----
    Creates multiple tables formatted for publication:
    - Table 1: Basic Station Information
    - Table 1b: Period means
    - Tables 2a-e: Various catchment attributes
    - Table 3: Trend results
    """
    print("\nCreating manuscript tables...")
    
    # Table 1: Basic Station Information
    station_info = df_1993[['river', 'name', 'impact_degree', 'g_frac', 'obsbeg', 'obsend']].copy()
    station_info['g_frac'] = 100 * station_info['g_frac']  # Convert to percentage
    
    # Add catchment area from catchments_chara
    station_info['area_calc'] = catchments_chara['area_calc']
    
    # Reorder columns to put area after name
    station_info = station_info[['river', 'name', 'area_calc', 'impact_degree', 'g_frac', 'obsbeg', 'obsend']]
    
    # Table 1b: Period Means
    means_table = pd.DataFrame(index=station_info.index)
    
    # Add streamflow means and convert index to integer
    mean_flow_1973.index = mean_flow_1973.index.astype(int)
    mean_flow_1993.index = mean_flow_1993.index.astype(int)
    means_table.index = means_table.index.astype(int)
    
    # Add streamflow means
    means_table['Q1'] = mean_flow_1973
    means_table['Q2'] = mean_flow_1993
    
    # Verify the data was added correctly
    print("\nNumber of non-null values in Q1:", means_table['Q1'].count())
    print("Number of non-null values in Q2:", means_table['Q2'].count())
    
    print("Calculating meteorological means (this may take a while)...")
    
    # Calculate meteorological means
    temp_means_1973 = {}
    prec_means_1973 = {}
    temp_means_1993 = {}
    prec_means_1993 = {}
    
    for station_id in means_table.index:
        # Calculate means for both periods
        temp_mean_1973, prec_mean_1973 = calculate_meteo_means(
            station_id, '1973-10-01', '2023-09-30'
        )
        temp_mean_1993, prec_mean_1993 = calculate_meteo_means(
            station_id, '1993-10-01', '2023-09-30'
        )
        
        # Store means
        temp_means_1973[station_id] = temp_mean_1973
        prec_means_1973[station_id] = prec_mean_1973
        temp_means_1993[station_id] = temp_mean_1993
        prec_means_1993[station_id] = prec_mean_1993
    
    # Add meteorological means to table
    means_table['T1'] = pd.Series(temp_means_1973)
    means_table['T2'] = pd.Series(temp_means_1993)
    means_table['P1'] = pd.Series(prec_means_1973)
    means_table['P2'] = pd.Series(prec_means_1993)
    
    # 2a: Geographic Attributes
    geographic_vars = [
        'area_calc', 'elev_mean', 'elev_std', 'elev_ran', 'slope_mean', 'asp_mean', 
        'elon_ratio', 'strm_dens'
    ]
    
    # 2b: Meteorological and Hydrological Attributes
    meteo_hydro_vars = [
        'p_mean', 'aridity', 'frac_snow', 'p_season',
        'q_mean', 'runoff_ratio'
    ]
    
    # 2c: Soil and Geology Attributes
    soil_geo_vars = [
        'sand_fra', 'silt_fra', 'clay_fra', 'oc_fra', 'root_dep', 
        'soil_tawc', 'soil_poros', 'bedrk_dep', 'ndvi_max'
    ]
    
    # 2d: Land Cover Attributes
    land_cover_vars = [
        'bare_fra', 'forest_fra', 'lake_fra', 'agr_fra'
    ]
    
    # 2e: Glacier Attributes
    glacier_vars = [
        'g_area', 'g_mean_el', 'g_max_el', 'g_min_el', 'g_slope', 'g_slopel20', 
        'g_frac'
    ]
    
    # Create separate tables
    attributes_table_2a = pd.DataFrame(index=station_info.index)
    attributes_table_2b = pd.DataFrame(index=station_info.index)
    attributes_table_2c = pd.DataFrame(index=station_info.index)
    attributes_table_2d = pd.DataFrame(index=station_info.index)
    attributes_table_2e = pd.DataFrame(index=station_info.index)
    
    # Add category-specific attributes
    for var in geographic_vars:
        if var in catchments_chara.columns:
            attributes_table_2a[var] = catchments_chara[var]
    
    # Add BFI to table 2b
    attributes_table_2b['BFI'] = hydro_indices['baseflow_index_ladson']
    
    for var in meteo_hydro_vars:
        if var in catchments_chara.columns:
            attributes_table_2b[var] = catchments_chara[var]
    
    for var in soil_geo_vars:
        if var in catchments_chara.columns:
            attributes_table_2c[var] = catchments_chara[var]
    
    for var in land_cover_vars:
        if var in catchments_chara.columns:
            attributes_table_2d[var] = catchments_chara[var]
    
    for var in glacier_vars:
        if var in catchments_chara.columns:
            if var == 'g_frac':
                attributes_table_2e['Glacial Fraction'] = station_info['g_frac']
            else:
                attributes_table_2e[var] = catchments_chara[var]
    
    # Column renaming dictionary
    column_renames = {
        # Geographic
        'area_calc': 'Catchment Area',
        'elev_mean': 'Mean Elevation',
        'elev_std': 'Elevation Std',
        'elev_ran': 'Elevation Range',
        'slope_mean': 'Mean Slope',
        'asp_mean': 'Mean Aspect',
        'elon_ratio': 'Elongation Ratio',
        'strm_dens': 'Stream Density',
        
        # Meteorological and Hydrological
        'p_mean': 'Mean Annual Precipitation',
        'aridity': 'Aridity Index',
        'frac_snow': 'Snow Fraction',
        'p_season': 'Precipitation Seasonality',
        'q_mean': 'Mean Discharge',
        'runoff_ratio': 'Runoff Ratio',
        
        # Soil and Geology
        'sand_fra': 'Sand Fraction',
        'silt_fra': 'Silt Fraction',
        'clay_fra': 'Clay Fraction',
        'oc_fra': 'Organic Carbon Fraction',
        'root_dep': 'Root Depth',
        'soil_tawc': 'Soil Water Capacity',
        'soil_poros': 'Soil Porosity',
        'bedrk_dep': 'Bedrock Depth',
        'ndvi_max': 'NDVI Max',
        
        # Land Cover
        'bare_fra': 'Bare Ground Fraction',
        'forest_fra': 'Forest Fraction',
        'lake_fra': 'Lake Fraction',
        'agr_fra': 'Agricultural Fraction',
        
        # Glacier
        'g_area': 'Glacier Area',
        'g_mean_el': 'Glacier Mean Elevation',
        'g_max_el': 'Glacier Max Elevation',
        'g_min_el': 'Glacier Min Elevation',
        'g_slope': 'Glacier Slope',
        'g_slopel20': 'Glacier Slope Lower20'
    }
    
    # Rename columns in each table
    attributes_table_2a = attributes_table_2a.rename(columns=column_renames)
    attributes_table_2b = attributes_table_2b.rename(columns=column_renames)
    attributes_table_2c = attributes_table_2c.rename(columns=column_renames)
    attributes_table_2d = attributes_table_2d.rename(columns=column_renames)
    attributes_table_2e = attributes_table_2e.rename(columns=column_renames)
    
    # Table 3: Trend Results
    trend_columns = [
        'annual_avg_flow_trend_per_decade', 'pval',  # Annual trend and p-value
        'trend_DJF_per_decade', 'pval_DJF',  # Winter
        'trend_MAM_per_decade', 'pval_MAM',  # Spring
        'trend_JJA_per_decade', 'pval_JJA',  # Summer
        'trend_SON_per_decade', 'pval_SON'   # Autumn
    ]
    
    trends_1973 = df_1973[trend_columns].copy()
    trends_1993 = df_1993[trend_columns].copy()
    
    # Rename columns to be more readable
    column_renames = {
        'annual_avg_flow_trend_per_decade': 'Annual Trend (percent per decade)',
        'pval': 'Annual P Value',
        'trend_DJF_per_decade': 'Winter Trend (percent per decade)',
        'pval_DJF': 'Winter P Value',
        'trend_MAM_per_decade': 'Spring Trend (percent per decade)',
        'pval_MAM': 'Spring P Value',
        'trend_JJA_per_decade': 'Summer Trend (percent per decade)',
        'pval_JJA': 'Summer P Value',
        'trend_SON_per_decade': 'Autumn Trend (percent per decade)',
        'pval_SON': 'Autumn P Value'
    }
    
    trends_1973 = trends_1973.rename(columns=column_renames)
    trends_1993 = trends_1993.rename(columns=column_renames)
    
    # Get all unique station IDs
    all_stations = sorted(set(list(station_info.index) + 
                            list(means_table.index) +
                            list(attributes_table_2a.index) + 
                            list(attributes_table_2b.index) + 
                            list(attributes_table_2c.index) + 
                            list(attributes_table_2d.index) + 
                            list(attributes_table_2e.index) + 
                            list(trends_1973.index) + 
                            list(trends_1993.index)))
    print(f"\nTotal unique stations across all tables: {len(all_stations)}")
    print("Stations in each table:")
    print(f"Station info table: {len(station_info)}")
    print(f"Means table: {len(means_table)}")
    print(f"Table 2a: {len(attributes_table_2a)}")
    print(f"Table 2b: {len(attributes_table_2b)}")
    print(f"Table 2c: {len(attributes_table_2c)}")
    print(f"Table 2d: {len(attributes_table_2d)}")
    print(f"Table 2e: {len(attributes_table_2e)}")
    print(f"1973-2023 trends: {len(trends_1973)}")
    print(f"1993-2023 trends: {len(trends_1993)}")
    
    return station_info, means_table, attributes_table_2a, attributes_table_2b, attributes_table_2c, attributes_table_2d, attributes_table_2e, trends_1973, trends_1993

# Create the tables
station_info, means_table, attributes_table_2a, attributes_table_2b, attributes_table_2c, attributes_table_2d, attributes_table_2e, trends_1973, trends_1993 = create_tables_for_manuscript(
    df_1973_output,
    df_1993_output,
    catchments_chara,
    mean_flow_1973,
    mean_flow_1993
)

# Save tables to Excel files
print("\nSaving tables...")

# Sort all tables by index (gauge ID)
station_info = station_info.sort_index()
means_table = means_table.sort_index()
attributes_table_2a = attributes_table_2a.sort_index()
attributes_table_2b = attributes_table_2b.sort_index()
attributes_table_2c = attributes_table_2c.sort_index()
attributes_table_2d = attributes_table_2d.sort_index()
attributes_table_2e = attributes_table_2e.sort_index()
trends_1973 = trends_1973.sort_index()
trends_1993 = trends_1993.sort_index()

# Format all numbers in means_table (Table 1b) to 1 decimal place
for col in means_table.columns:
    means_table[col] = means_table[col].round(1)

# Format all numbers in attributes_table_2a (Table 2a) to 1 decimal place
for col in attributes_table_2a.columns:
    attributes_table_2a[col] = attributes_table_2a[col].round(1)

# Print first few rows of means table to verify data
print("\nFirst few rows of means table:")
print(means_table[['Q1', 'Q2']].head())

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    station_info.to_excel(writer, sheet_name='Table1 Station Info', float_format='%.1f')
    means_table.to_excel(writer, sheet_name='Table1b Period Means', float_format='%.1f')
    attributes_table_2a.to_excel(writer, sheet_name='Table2a Geographic', float_format='%.1f')
    attributes_table_2b.to_excel(writer, sheet_name='Table2b Meteo Hydro', float_format='%.3f')
    attributes_table_2c.to_excel(writer, sheet_name='Table2c Soil Geology', float_format='%.3f')
    attributes_table_2d.to_excel(writer, sheet_name='Table2d Land Cover', float_format='%.3f')
    attributes_table_2e.to_excel(writer, sheet_name='Table2e Glacier', float_format='%.3f')
    trends_1973.to_excel(writer, sheet_name='Table3a Trends 1973-2023', float_format='%.3f')
    trends_1993.to_excel(writer, sheet_name='Table3b Trends 1993-2023', float_format='%.3f')

print("\nTables have been saved to:", output_excel)

def print_summary_statistics(df_1973, df_1993, mean_flow_1973, mean_flow_1993, temp_means_1973, prec_means_1973, temp_means_1993, prec_means_1993):
    """
    Print summary statistics for use in manuscript text.
    
    Parameters
    ----------
    df_1973, df_1993 : DataFrame
        Trend results for both periods
    mean_flow_1973, mean_flow_1993 : Series
        Mean flows for both periods
    temp_means_1973, temp_means_1993 : Series
        Temperature means for both periods
    prec_means_1973, prec_means_1993 : Series
        Precipitation means for both periods
        
    Prints
    ------
    Summary statistics including:
    - Mean streamflow ranges
    - Mean temperature ranges
    - Mean precipitation ranges
    For both study periods
    """
    print("\nSummary statistics for manuscript text:\n")
    
    # Period 1973-2023
    print("Period 1973-2023:")
    if not mean_flow_1973.empty:
        print(f"Mean streamflow range: {mean_flow_1973.min():.1f} - {mean_flow_1973.max():.1f} m³/s")
    else:
        print("Mean streamflow range: No data available")
        
    temp_range_1973 = pd.Series(temp_means_1973).dropna()
    prec_range_1973 = pd.Series(prec_means_1973).dropna()
    if not temp_range_1973.empty:
        print(f"Mean temperature range: {temp_range_1973.min():.1f} - {temp_range_1973.max():.1f} °C")
    if not prec_range_1973.empty:
        print(f"Mean precipitation range: {prec_range_1973.min():.1f} - {prec_range_1973.max():.1f} mm/year")
    print()
    
    # Period 1993-2023
    print("Period 1993-2023:")
    if not mean_flow_1993.empty:
        print(f"Mean streamflow range: {mean_flow_1993.min():.1f} - {mean_flow_1993.max():.1f} m³/s")
    else:
        print("Mean streamflow range: No data available")
        
    temp_range_1993 = pd.Series(temp_means_1993).dropna()
    prec_range_1993 = pd.Series(prec_means_1993).dropna()
    if not temp_range_1993.empty:
        print(f"Mean temperature range: {temp_range_1993.min():.1f} - {temp_range_1993.max():.1f} °C")
    if not prec_range_1993.empty:
        print(f"Mean precipitation range: {prec_range_1993.min():.1f} - {prec_range_1993.max():.1f} mm/year")

# Print summary statistics for the manuscript text
print_summary_statistics(df_1973_output, df_1993_output, mean_flow_1973, mean_flow_1993, means_table['T1'], means_table['P1'], means_table['T2'], means_table['P2']) 