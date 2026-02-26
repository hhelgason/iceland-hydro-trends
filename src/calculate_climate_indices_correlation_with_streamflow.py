"""
Calculate correlations between climate indices and streamflow/glacier mass balance.

This script:
1. Calculates annual correlations between climate indices (AO, NAO, and others) and streamflow
2. Calculates seasonal correlations
3. Calculates correlations with glacier mass balance
4. Exports results to CSV files

Author: Hordur Bragi Helgason
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
from config import OUTPUT_DIR
STREAMFLOW_ANNUAL_FILE = OUTPUT_DIR / "annual_streamflow_averages_longterm.csv"
STREAMFLOW_SEASONAL_FILES = {
    'DJF': OUTPUT_DIR / "seasonal_streamflow_averages_longterm_DJF.csv",
    'MAM': OUTPUT_DIR / "seasonal_streamflow_averages_longterm_MAM.csv",
    'JJA': OUTPUT_DIR / "seasonal_streamflow_averages_longterm_JJA.csv",
    'SON': OUTPUT_DIR / "seasonal_streamflow_averages_longterm_SON.csv"
}
CLIMATE_INDICES_FILE = Path(r"C:\Users\hordurbhe\OneDrive - Landsvirkjun\HOPIG\CEATI-project_monitor\November_2024_Update\Streamflow Toolkit\streamflow-toolkit\Data\Indices_2024.xlsx")
GLACIER_MB_FILE = Path(r"C:\Users\hordurbhe\OneDrive - Landsvirkjun\Changes in streamflow in Iceland\data\glacier_mass_balance\glacier-mass-balance-vatnaj-langj-hofsj-20241105.csv")

# Analysis parameters
START_YEAR = 1950
END_YEAR = 2024
HYDROLOGICAL_YEAR_FIRST_MONTH = 10
MIN_YEARS_OVERLAP = 20  # Minimum years of overlap for correlations (streamflow and glacier)
LAG_YEARS = [1, 2, 3]  # Time lags (in years) for lagged correlation analysis

# AO and NAO data URLs
AO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii"
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"

# Seasons definition (matching calculate_annual_averages_for_longterm_analysis.py)
SEASONS = {
    'DJF': [12, 1, 2],     # Winter (Dec, Jan, Feb)
    'MAM': [3, 4, 5],      # Spring (Mar, Apr, May)
    'JJA': [6, 7, 8],      # Summer (Jun, Jul, Aug)
    'SON': [9, 10, 11]     # Autumn (Sep, Oct, Nov)
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def filter_29_of_february(df):
    """Remove February 29th from the DataFrame."""
    return df[~((df.index.month == 2) & (df.index.day == 29))]

def detrend_normalize_data(values, normalize=True):
    """
    Detrend and optionally normalize a time series.
    
    Parameters
    ----------
    values : array-like
        Time series values
    normalize : bool
        Whether to normalize after detrending
        
    Returns
    -------
    array
        Detrended (and normalized) values
    """
    values = np.array(values)
    values_length = len(values)
    values_range = range(1, values_length + 1)
    
    # Handle NaN values
    finite_index = np.argwhere(~np.isnan(values)).T.tolist()
    if len(finite_index) == 0 or len(finite_index[0]) < 2:
        return values
    
    values_range_poly_fit = [values_range[index] for index in finite_index[0]]
    values_poly_fit = [values[index] for index in finite_index[0]]
    
    # Fit linear trend
    poly_fit = np.polyfit(values_range_poly_fit, values_poly_fit, 1)
    poly_val = np.polyval(poly_fit, values_range)
    
    # Detrend
    if np.isnan(poly_fit[0]):
        x_detrend = values
    else:
        x_detrend = values - poly_val
    
    # Normalize if requested
    if normalize:
        x_detrend_normalized = (x_detrend - np.nanmean(x_detrend)) / np.nanstd(x_detrend)
        return x_detrend_normalized
    else:
        return x_detrend

def get_seasonal_data(df, season_name):
    """
    Filter data for the given season.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    season_name : str
        Season name ('JFM', 'AMJ', 'JAS', 'OND')
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame for the season
    """
    if season_name not in SEASONS:
        raise ValueError(f"Invalid season: {season_name}")
    return df[df.index.month.isin(SEASONS[season_name])]

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def read_ao_nao_index(url):
    """
    Read AO or NAO index from NOAA URL.
    
    Parameters
    ----------
    url : str
        URL to the ASCII file
        
    Returns
    -------
    pd.DataFrame
        Monthly index values with DatetimeIndex
    """
    print(f"Reading index from: {url}")
    df_index = pd.read_csv(url, delim_whitespace=True, header=None, names=['Year', 'Month', 'Value'])
    df_index['date'] = pd.to_datetime(df_index[['Year', 'Month']].assign(DAY=1))
    df_index.set_index('date', inplace=True)
    df_index.drop(columns=['Year', 'Month'], inplace=True)
    return df_index

def read_climate_indices_from_excel(file_path):
    """
    Read all climate indices from the Excel file.
    
    Parameters
    ----------
    file_path : Path
        Path to the Excel file containing climate indices
        
    Returns
    -------
    dict
        Dictionary of {index_name: pd.DataFrame} with monthly data
    """
    print(f"Reading climate indices from: {file_path}")
    
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return {}
    
    # Read the Excel file
    xl_file = pd.ExcelFile(file_path)
    
    indices_dict = {}
    
    for sheet_name in xl_file.sheet_names:
        # Skip info sheet
        if sheet_name.lower() == 'info':
            continue
            
        try:
            df = pd.read_excel(xl_file, sheet_name=sheet_name)
            
            # Check if data is in wide format (year column + 12 month columns)
            if 'year' in df.columns and all(m in df.columns for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
                # Wide format: convert to long format with datetime index
                records = []
                for _, row in df.iterrows():
                    year = int(row['year'])
                    for month in range(1, 13):
                        date = pd.Timestamp(year=year, month=month, day=1)
                        value = row[month]
                        if pd.notna(value):
                            records.append({'date': date, 'Value': value})
                
                if records:
                    df_index = pd.DataFrame(records)
                    df_index.set_index('date', inplace=True)
                    indices_dict[sheet_name] = df_index
                    print(f"  Loaded index: {sheet_name} ({len(df_index)} records)")
            else:
                # Try to identify date/time column
                date_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'month']):
                        date_col = col
                        break
                
                if date_col is not None:
                    # Assume the value column is the first non-date column
                    value_cols = [col for col in df.columns if col != date_col]
                    if value_cols:
                        df_index = pd.DataFrame()
                        df_index['date'] = pd.to_datetime(df[date_col])
                        df_index['Value'] = df[value_cols[0]]
                        df_index.set_index('date', inplace=True)
                        indices_dict[sheet_name] = df_index
                        print(f"  Loaded index: {sheet_name} ({len(df_index)} records)")
        except Exception as e:
            print(f"  Warning: Could not load {sheet_name}: {e}")
            continue
    
    return indices_dict

def load_annual_streamflow_data():
    """
    Load the annual streamflow averages from CSV file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with gauge IDs as columns and annual (water year) data
    """
    print(f"Loading annual streamflow data from: {STREAMFLOW_ANNUAL_FILE}")
    
    if not STREAMFLOW_ANNUAL_FILE.exists():
        raise FileNotFoundError(f"Annual streamflow file not found: {STREAMFLOW_ANNUAL_FILE}")
    
    df = pd.read_csv(STREAMFLOW_ANNUAL_FILE, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df.columns)} gauges with {len(df)} annual records")
    
    return df

def load_seasonal_streamflow_data():
    """
    Load the seasonal streamflow averages from CSV files.
    
    Returns
    -------
    dict
        Dictionary of {season_name: DataFrame} with seasonal data
    """
    print(f"Loading seasonal streamflow data...")
    
    seasonal_data = {}
    for season, filepath in STREAMFLOW_SEASONAL_FILES.items():
        if not filepath.exists():
            print(f"  Warning: Seasonal file not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        seasonal_data[season] = df
        print(f"  Loaded {season}: {len(df.columns)} gauges with {len(df)} records")
    
    return seasonal_data

def read_annual_streamflow_data(gauge_id, df_streamflow_all, start_year, end_year):
    """
    Extract annual streamflow data for a specific gauge from the full dataframe.
    
    Parameters
    ----------
    gauge_id : int
        Gauge ID
    df_streamflow_all : pd.DataFrame
        Full streamflow dataframe with all gauges (annual data)
    start_year : int
        Start year
    end_year : int
        End year
        
    Returns
    -------
    pd.DataFrame or None
        Annual streamflow DataFrame with DatetimeIndex and 'Value' column, or None if gauge not found
    """
    # Check if gauge exists in the dataframe
    if str(gauge_id) not in df_streamflow_all.columns and int(gauge_id) not in df_streamflow_all.columns:
        return None
    
    try:
        # Extract the gauge column
        gauge_col = str(gauge_id) if str(gauge_id) in df_streamflow_all.columns else int(gauge_id)
        df_gauge = df_streamflow_all[[gauge_col]].copy()
        df_gauge.columns = ['Value']
        
        # Filter by date range
        start_date = pd.Timestamp(f'{start_year}-01-01')
        end_date = pd.Timestamp(f'{end_year}-12-31')
        df_gauge = df_gauge.loc[start_date:end_date]
        
        # Remove rows with all NaN
        df_gauge = df_gauge.dropna(how='all')
        
        return df_gauge
        
    except Exception as e:
        print(f"Error extracting gauge {gauge_id}: {e}")
        return None

def read_seasonal_streamflow_data(gauge_id, df_seasonal_dict, start_year, end_year):
    """
    Extract seasonal streamflow data for a specific gauge.
    
    Parameters
    ----------
    gauge_id : int
        Gauge ID
    df_seasonal_dict : dict
        Dictionary of {season_name: DataFrame} with seasonal data
    start_year : int
        Start year
    end_year : int
        End year
        
    Returns
    -------
    dict or None
        Dictionary of {season_name: DataFrame} for this gauge, or None if gauge not found
    """
    result = {}
    
    for season, df_season_all in df_seasonal_dict.items():
        # Check if gauge exists
        if str(gauge_id) not in df_season_all.columns and int(gauge_id) not in df_season_all.columns:
            continue
        
        try:
            # Extract the gauge column
            gauge_col = str(gauge_id) if str(gauge_id) in df_season_all.columns else int(gauge_id)
            df_gauge = df_season_all[[gauge_col]].copy()
            df_gauge.columns = ['Value']
            
            # Filter by date range
            start_date = pd.Timestamp(f'{start_year}-01-01')
            end_date = pd.Timestamp(f'{end_year}-12-31')
            df_gauge = df_gauge.loc[start_date:end_date]
            
            # Remove rows with all NaN
            df_gauge = df_gauge.dropna(how='all')
            
            result[season] = df_gauge
            
        except Exception as e:
            print(f"Error extracting gauge {gauge_id} season {season}: {e}")
            continue
    
    return result if result else None

def get_available_gauges(df_streamflow_all):
    """
    Get list of available gauge IDs from the streamflow dataframe.
    
    Parameters
    ----------
    df_streamflow_all : pd.DataFrame
        Full streamflow dataframe
        
    Returns
    -------
    list
        List of gauge IDs (as integers)
    """
    # Try to convert column names to integers
    gauge_ids = []
    for col in df_streamflow_all.columns:
        try:
            gauge_ids.append(int(col))
        except ValueError:
            continue
    
    return sorted(gauge_ids)

# ============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# ============================================================================

def calculate_annual_correlation(streamflow_annual_df, climate_index_monthly_df, normalize_climate_index=False):
    """
    Calculate annual correlation between streamflow and climate index.
    
    Parameters
    ----------
    streamflow_annual_df : pd.DataFrame
        Annual streamflow data with 'Value' column
    climate_index_monthly_df : pd.DataFrame
        Monthly climate index data with 'Value' column
    normalize_climate_index : bool
        Whether to normalize the climate index (set to False for indices already normalized)
        
    Returns
    -------
    tuple
        (correlation_coefficient, p_value, n_overlap) or (np.nan, np.nan, 0) if insufficient data
    """
    # Resample climate index to annual (water year mean)
    # The streamflow is already annual (water year), so we just need to aggregate the climate index
    climate_annual = climate_index_monthly_df.resample('YE').mean()
    
    # Merge the two dataframes
    merged = pd.merge(streamflow_annual_df, climate_annual, left_index=True, right_index=True, 
                     how='inner', suffixes=('_flow', '_index'))
    
    # Detrend and normalize streamflow
    flow_detrended = detrend_normalize_data(merged['Value_flow'].values, normalize=True)
    
    # Detrend and optionally normalize climate index
    index_detrended = detrend_normalize_data(merged['Value_index'].values, normalize=normalize_climate_index)
    
    # Calculate correlation
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(flow_detrended) | np.isnan(index_detrended))
        n_overlap = valid_mask.sum()
        
        if n_overlap < MIN_YEARS_OVERLAP:
            return np.nan, np.nan, n_overlap
        
        corr, pval = stats.pearsonr(flow_detrended[valid_mask], index_detrended[valid_mask])
        return corr, pval, n_overlap
    except:
        return np.nan, np.nan, 0

def calculate_seasonal_correlation(streamflow_seasonal_df, climate_index_monthly_df, season_name, normalize_climate_index=False):
    """
    Calculate seasonal correlation between streamflow and climate index.
    
    Parameters
    ----------
    streamflow_seasonal_df : pd.DataFrame
        Seasonal streamflow data with 'Value' column (already aggregated)
    climate_index_monthly_df : pd.DataFrame
        Monthly climate index data with 'Value' column
    season_name : str
        Season name ('DJF', 'MAM', 'JJA', 'SON')
    normalize_climate_index : bool
        Whether to normalize the climate index
        
    Returns
    -------
    tuple
        (correlation_coefficient, p_value, n_overlap) or (np.nan, np.nan, 0) if insufficient data
    """
    # Filter climate index for the season months
    climate_seasonal = get_seasonal_data(climate_index_monthly_df, season_name)
    
    # Resample climate index to annual (one value per year for this season)
    # Using YE (year end) to align with water year ending
    climate_seasonal_annual = climate_seasonal.resample('YE').mean()
    
    # Merge the two dataframes
    merged = pd.merge(streamflow_seasonal_df, climate_seasonal_annual, left_index=True, right_index=True,
                     how='inner', suffixes=('_flow', '_index'))
    
    # Detrend and normalize streamflow
    flow_detrended = detrend_normalize_data(merged['Value_flow'].values, normalize=True)
    index_detrended = detrend_normalize_data(merged['Value_index'].values, normalize=normalize_climate_index)
    
    # Calculate correlation
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(flow_detrended) | np.isnan(index_detrended))
        n_overlap = valid_mask.sum()
        
        if n_overlap < MIN_YEARS_OVERLAP:
            return np.nan, np.nan, n_overlap
        
        corr, pval = stats.pearsonr(flow_detrended[valid_mask], index_detrended[valid_mask])
        return corr, pval, n_overlap
    except:
        return np.nan, np.nan, 0

def calculate_lagged_correlation(streamflow_annual_df, climate_index_monthly_df, lag_years, normalize_climate_index=False):
    """
    Calculate lagged correlation between climate index and future streamflow.
    
    For example, with lag_years=1, correlates climate index in year t with streamflow in year t+1.
    
    Parameters
    ----------
    streamflow_annual_df : pd.DataFrame
        Annual streamflow data with 'Value' column
    climate_index_monthly_df : pd.DataFrame
        Monthly climate index data with 'Value' column
    lag_years : int
        Number of years to lag (positive = climate index leads streamflow)
    normalize_climate_index : bool
        Whether to normalize the climate index
        
    Returns
    -------
    tuple
        (correlation_coefficient, p_value, n_overlap) or (np.nan, np.nan, 0) if insufficient data
    """
    # Resample climate index to annual
    climate_annual = climate_index_monthly_df.resample('YE').mean()
    
    # Shift streamflow backward by lag_years (so climate index in year t aligns with streamflow in year t+lag)
    # Example: lag=1 means climate in 2019 predicts streamflow in 2020
    # We relabel streamflow 2020 as 2019, then merge with climate 2019
    # Result: climate[2019] paired with streamflow[originally 2020]
    streamflow_lagged = streamflow_annual_df.copy()
    streamflow_lagged.index = streamflow_lagged.index - pd.DateOffset(years=lag_years)
    
    # Merge the two dataframes
    merged = pd.merge(streamflow_lagged, climate_annual, left_index=True, right_index=True,
                     how='inner', suffixes=('_flow', '_index'))
    
    # Detrend and normalize
    flow_detrended = detrend_normalize_data(merged['Value_flow'].values, normalize=True)
    index_detrended = detrend_normalize_data(merged['Value_index'].values, normalize=normalize_climate_index)
    
    # Calculate correlation
    try:
        # Remove NaN values
        valid_mask = ~(np.isnan(flow_detrended) | np.isnan(index_detrended))
        n_overlap = valid_mask.sum()
        
        if n_overlap < MIN_YEARS_OVERLAP:
            return np.nan, np.nan, n_overlap
        
        corr, pval = stats.pearsonr(flow_detrended[valid_mask], index_detrended[valid_mask])
        return corr, pval, n_overlap
    except:
        return np.nan, np.nan, 0

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_ao_nao_correlations(df_streamflow_annual):
    """
    Analyze correlations between AO/NAO and streamflow for all gauges.
    
    Parameters
    ----------
    df_streamflow_annual : pd.DataFrame
        Annual streamflow dataframe with all gauges
    
    Returns
    -------
    pd.DataFrame
        Results dataframe with correlations for each gauge
    """
    print("\n" + "="*80)
    print("ANALYZING AO AND NAO CORRELATIONS WITH STREAMFLOW")
    print("="*80)
    
    # Load AO and NAO data
    ao_data = read_ao_nao_index(AO_URL)
    nao_data = read_ao_nao_index(NAO_URL)
    
    # Get available gauges
    gauge_ids = get_available_gauges(df_streamflow_annual)
    print(f"\nFound {len(gauge_ids)} gauges")
    
    results = []
    years_list = []  # Track number of years for each gauge
    
    for gauge_id in gauge_ids:
        print(f"\nProcessing gauge {gauge_id}...", end=" ")
        
        # Load streamflow data (annual)
        flow_data = read_annual_streamflow_data(gauge_id, df_streamflow_annual, START_YEAR, END_YEAR)
        
        if flow_data is None or len(flow_data) < MIN_YEARS_OVERLAP:
            print("Insufficient data")
            continue
        
        # Calculate annual correlations (climate indices are already normalized, so set to False)
        ao_corr, ao_pval, ao_overlap = calculate_annual_correlation(flow_data, ao_data, normalize_climate_index=False)
        nao_corr, nao_pval, nao_overlap = calculate_annual_correlation(flow_data, nao_data, normalize_climate_index=False)
        
        # Use the maximum overlap for tracking
        n_overlap = max(ao_overlap, nao_overlap)
        if n_overlap >= MIN_YEARS_OVERLAP:
            years_list.append(n_overlap)
        
        results.append({
            'Streamflow Gauge': gauge_id,
            'N_Years_Overlap': n_overlap,
            'Correlation AO': ao_corr,
            'P-value AO': ao_pval,
            'Correlation NAO': nao_corr,
            'P-value NAO': nao_pval
        })
        
        print(f"AO: r={ao_corr:.3f} (p={ao_pval:.3f}), NAO: r={nao_corr:.3f} (p={nao_pval:.3f}), N_overlap={n_overlap}")
    
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    if years_list:
        print("\n" + "="*80)
        print("SERIES LENGTH STATISTICS")
        print("="*80)
        print(f"Number of gauges analyzed: {len(years_list)}")
        print(f"Average number of years per gauge: {np.mean(years_list):.1f}")
        print(f"Minimum number of years: {np.min(years_list)}")
        print(f"Maximum number of years: {np.max(years_list)}")
        print(f"Median number of years: {np.median(years_list):.1f}")
    
    # Save to CSV
    output_file = 'climate_index_correlation_results_CI_not_normalized.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved AO/NAO results to: {output_file}")
    
    return df_results

def analyze_extended_climate_indices(df_streamflow_annual, df_streamflow_seasonal):
    """
    Analyze correlations between all climate indices and streamflow.
    Includes AO/NAO from NOAA URLs plus other indices from Excel file.
    
    Parameters
    ----------
    df_streamflow_annual : pd.DataFrame
        Annual streamflow dataframe with all gauges
    df_streamflow_seasonal : dict
        Dictionary of {season_name: DataFrame} with seasonal data
    
    Returns
    -------
    pd.DataFrame
        Extended results dataframe
    """
    print("\n" + "="*80)
    print("ANALYZING EXTENDED CLIMATE INDICES CORRELATIONS")
    print("="*80)
    
    # Load AO and NAO from NOAA URLs
    print("\nLoading AO and NAO from NOAA...")
    ao_data = read_ao_nao_index(AO_URL)
    nao_data = read_ao_nao_index(NAO_URL)
    
    # Combine AO/NAO with other indices
    climate_indices = {
        'AO': ao_data,
        'NAO': nao_data
    }
    
    # Load other climate indices from Excel file
    print("\nLoading other climate indices from Excel...")
    excel_indices = read_climate_indices_from_excel(CLIMATE_INDICES_FILE)
    
    if excel_indices:
        # Exclude AO and NAO from Excel since we're using NOAA versions
        excel_indices_filtered = {k: v for k, v in excel_indices.items() if k not in ['AO', 'NAO']}
        climate_indices.update(excel_indices_filtered)
        print(f"  Total indices: {len(climate_indices)} (AO, NAO from NOAA + {len(excel_indices_filtered)} from Excel)")
    else:
        print("  Note: No additional indices loaded from Excel. Using only AO and NAO.")
    
    # Get available gauges
    gauge_ids = get_available_gauges(df_streamflow_annual)
    
    results = []
    years_list = []  # Track number of years for each gauge
    
    for gauge_id in gauge_ids:
        print(f"\nProcessing gauge {gauge_id}...")
        
        # Load annual streamflow data
        flow_annual = read_annual_streamflow_data(gauge_id, df_streamflow_annual, START_YEAR, END_YEAR)
        
        if flow_annual is None or len(flow_annual) < MIN_YEARS_OVERLAP:
            continue
        
        # Load seasonal streamflow data
        flow_seasonal = read_seasonal_streamflow_data(gauge_id, df_streamflow_seasonal, START_YEAR, END_YEAR)
        
        gauge_results = {
            'Gauge_ID': gauge_id
        }
        
        # Track overlap for each index
        max_overlap = 0
        
        # Annual correlations
        for index_name, index_data in climate_indices.items():
            # AO and NAO are already normalized, others need normalization
            normalize = index_name not in ['AO', 'NAO']
            corr, pval, n_overlap = calculate_annual_correlation(flow_annual, index_data, normalize_climate_index=normalize)
            gauge_results[f'{index_name}_annual_corr'] = corr
            gauge_results[f'{index_name}_annual_pval'] = pval
            max_overlap = max(max_overlap, n_overlap)
        
        # Seasonal correlations
        if flow_seasonal:
            for season in SEASONS.keys():
                if season not in flow_seasonal:
                    continue
                for index_name, index_data in climate_indices.items():
                    # AO and NAO are already normalized, others need normalization
                    normalize = index_name not in ['AO', 'NAO']
                    corr, pval, n_overlap = calculate_seasonal_correlation(flow_seasonal[season], index_data, season, normalize_climate_index=normalize)
                    gauge_results[f'{index_name}_{season}_corr'] = corr
                    gauge_results[f'{index_name}_{season}_pval'] = pval
                    max_overlap = max(max_overlap, n_overlap)
        
        gauge_results['N_Years_Overlap'] = max_overlap
        if max_overlap >= MIN_YEARS_OVERLAP:
            years_list.append(max_overlap)
        
        results.append(gauge_results)
        print(f"  Completed: {len(climate_indices)} indices x 5 periods (annual + 4 seasons), N_overlap={max_overlap} years")
    
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    if years_list:
        print("\n" + "="*80)
        print("SERIES LENGTH STATISTICS")
        print("="*80)
        print(f"Number of gauges analyzed: {len(years_list)}")
        print(f"Average number of years per gauge: {np.mean(years_list):.1f}")
        print(f"Minimum number of years: {np.min(years_list)}")
        print(f"Maximum number of years: {np.max(years_list)}")
        print(f"Median number of years: {np.median(years_list):.1f}")
    
    # Save to CSV
    output_file = 'climate_indices_correlation_results_extended_feb_2025.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved extended results to: {output_file}")
    
    return df_results

def analyze_lagged_climate_correlations(df_streamflow_annual):
    """
    Analyze lagged correlations between climate indices and streamflow.
    Tests whether climate indices can predict future streamflow.
    
    Parameters
    ----------
    df_streamflow_annual : pd.DataFrame
        Annual streamflow dataframe with all gauges
    
    Returns
    -------
    pd.DataFrame
        Lagged correlation results
    """
    print("\n" + "="*80)
    print("ANALYZING LAGGED CLIMATE-STREAMFLOW CORRELATIONS")
    print("="*80)
    print(f"Testing lags: {LAG_YEARS} years")
    print("(Climate index in year t vs. streamflow in year t+lag)")
    
    # Load AO and NAO from NOAA URLs
    print("\nLoading AO and NAO from NOAA...")
    ao_data = read_ao_nao_index(AO_URL)
    nao_data = read_ao_nao_index(NAO_URL)
    
    # Combine AO/NAO with other indices
    climate_indices = {
        'AO': ao_data,
        'NAO': nao_data
    }
    
    # Load other climate indices from Excel file
    print("\nLoading other climate indices from Excel...")
    excel_indices = read_climate_indices_from_excel(CLIMATE_INDICES_FILE)
    
    if excel_indices:
        # Exclude AO and NAO from Excel since we're using NOAA versions
        excel_indices_filtered = {k: v for k, v in excel_indices.items() if k not in ['AO', 'NAO']}
        climate_indices.update(excel_indices_filtered)
        print(f"  Total indices: {len(climate_indices)} (AO, NAO from NOAA + {len(excel_indices_filtered)} from Excel)")
    else:
        print("  Note: No additional indices loaded from Excel. Using only AO and NAO.")
    
    # Get available gauges
    gauge_ids = get_available_gauges(df_streamflow_annual)
    
    results = []
    years_list = []  # Track number of years for each gauge
    
    for gauge_id in gauge_ids:
        print(f"\nProcessing gauge {gauge_id}...")
        
        # Load annual streamflow data
        flow_annual = read_annual_streamflow_data(gauge_id, df_streamflow_annual, START_YEAR, END_YEAR)
        
        if flow_annual is None or len(flow_annual) < MIN_YEARS_OVERLAP:
            continue
        
        gauge_results = {
            'Gauge_ID': gauge_id
        }
        
        # Track maximum overlap
        max_overlap = 0
        
        # Calculate lagged correlations for each climate index and lag
        for index_name, index_data in climate_indices.items():
            # AO and NAO are already normalized, others need normalization
            normalize = index_name not in ['AO', 'NAO']
            
            for lag in LAG_YEARS:
                corr, pval, n_overlap = calculate_lagged_correlation(
                    flow_annual, index_data, lag, normalize_climate_index=normalize
                )
                gauge_results[f'{index_name}_lag{lag}_corr'] = corr
                gauge_results[f'{index_name}_lag{lag}_pval'] = pval
                max_overlap = max(max_overlap, n_overlap)
        
        gauge_results['N_Years_Overlap'] = max_overlap
        if max_overlap >= MIN_YEARS_OVERLAP:
            years_list.append(max_overlap)
        
        results.append(gauge_results)
        print(f"  Completed: {len(climate_indices)} indices x {len(LAG_YEARS)} lags, N_overlap={max_overlap} years")
    
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    if years_list:
        print("\n" + "="*80)
        print("SERIES LENGTH STATISTICS")
        print("="*80)
        print(f"Number of gauges analyzed: {len(years_list)}")
        print(f"Average number of years per gauge: {np.mean(years_list):.1f}")
        print(f"Minimum number of years: {np.min(years_list)}")
        print(f"Maximum number of years: {np.max(years_list)}")
        print(f"Median number of years: {np.median(years_list):.1f}")
    
    # Save to CSV
    output_file = 'climate_indices_lagged_correlations.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved lagged correlation results to: {output_file}")
    
    # Print summary of significant lagged correlations
    print("\n" + "="*80)
    print("SIGNIFICANT LAGGED CORRELATIONS SUMMARY (p < 0.05)")
    print("="*80)
    
    for lag in LAG_YEARS:
        print(f"\nLag {lag} year(s):")
        for index_name in climate_indices.keys():
            corr_col = f'{index_name}_lag{lag}_corr'
            pval_col = f'{index_name}_lag{lag}_pval'
            
            if corr_col in df_results.columns and pval_col in df_results.columns:
                significant = df_results[df_results[pval_col] < 0.05]
                if len(significant) > 0:
                    pos = (significant[corr_col] > 0).sum()
                    neg = (significant[corr_col] < 0).sum()
                    print(f"  {index_name}: {len(significant)} significant ({pos} positive, {neg} negative)")
    
    return df_results

def analyze_lagged_glacier_mb_correlations():
    """
    Analyze lagged correlations between climate indices and glacier mass balance.
    Tests whether climate indices can predict future glacier mass balance.
    
    Returns
    -------
    pd.DataFrame
        Lagged glacier correlation results
    """
    print("\n" + "="*80)
    print("ANALYZING LAGGED GLACIER MASS BALANCE CORRELATIONS")
    print("="*80)
    print(f"Testing lags: {LAG_YEARS} years")
    
    # Check if glacier MB file exists
    if not GLACIER_MB_FILE.exists():
        print(f"Warning: Glacier MB file not found: {GLACIER_MB_FILE}")
        print("Skipping lagged glacier analysis.")
        return None
    
    print(f"Loading glacier mass balance data from: {GLACIER_MB_FILE}")
    
    # Load glacier mass balance data
    glacmb = pd.read_csv(GLACIER_MB_FILE)
    
    # Extract individual glaciers
    langjokull = glacmb[glacmb['glims_id'] == 'G339764E64629N'].copy()
    langjokull.set_index('yr', inplace=True)
    
    hofsjokull = glacmb[glacmb['glims_id'] == 'G341164E64838N'].copy()
    hofsjokull.set_index('yr', inplace=True)
    
    vatnajokull = glacmb[glacmb['glims_id'] == 'G343222E64409N'].copy()
    vatnajokull.set_index('yr', inplace=True)
    
    print(f"  Langjökull: {len(langjokull)} years")
    print(f"  Hofsjökull: {len(hofsjokull)} years")
    print(f"  Vatnajökull: {len(vatnajokull)} years")
    
    # Detrend and normalize glacier data
    langjokull_detrended = detrend_normalize_glacier_mb(langjokull)
    hofsjokull_detrended = detrend_normalize_glacier_mb(hofsjokull)
    vatnajokull_detrended = detrend_normalize_glacier_mb(vatnajokull)
    
    glaciers = {
        'Langjokull': langjokull_detrended,
        'Hofsjokull': hofsjokull_detrended,
        'Vatnajokull': vatnajokull_detrended
    }
    
    # Load AO and NAO from NOAA
    print("\nLoading climate indices...")
    ao_data = read_ao_nao_index(AO_URL)
    nao_data = read_ao_nao_index(NAO_URL)
    
    # Load other indices from Excel
    excel_indices = read_climate_indices_from_excel(CLIMATE_INDICES_FILE)
    
    # Combine all indices
    climate_indices = {
        'AO': ao_data,
        'NAO': nao_data
    }
    if excel_indices:
        # Exclude AO and NAO from Excel
        excel_indices_filtered = {k: v for k, v in excel_indices.items() if k not in ['AO', 'NAO']}
        climate_indices.update(excel_indices_filtered)
    
    print(f"  Total climate indices: {len(climate_indices)}")
    
    # Detrend and normalize climate indices
    detrended_normalized_indices = {}
    for index_name, index_data in climate_indices.items():
        if index_name in ['AO', 'NAO']:
            df_detrended = index_data.copy()
            df_detrended.columns = [f'detrended_normalized_{index_name}']
        else:
            detrended = detrend_normalize_data(index_data['Value'].values, normalize=True)
            df_detrended = pd.DataFrame(
                {f'detrended_normalized_{index_name}': detrended},
                index=index_data.index
            )
        
        if not isinstance(df_detrended.index, pd.DatetimeIndex):
            df_detrended.index = pd.to_datetime(df_detrended.index)
        
        detrended_normalized_indices[index_name] = df_detrended
    
    # Calculate lagged correlations
    print(f"\nCalculating lagged correlations (min {MIN_YEARS_OVERLAP} years overlap)...")
    results = []
    
    for glacier_name, df_glacier in glaciers.items():
        print(f"\n  Processing {glacier_name}...")
        
        for climate_index, df_climate in detrended_normalized_indices.items():
            col_name = f'detrended_normalized_{climate_index}'
            
            for lag in LAG_YEARS:
                # Aggregate climate index by different periods
                series_wy = climate_index_by_water_year(df_climate, col_name)
                series_winter = climate_index_by_winter(df_climate, col_name)
                series_summer = climate_index_by_summer(df_climate, col_name)
                
                # Shift climate indices to lag them
                series_wy_lagged = series_wy.copy()
                series_wy_lagged.index = series_wy_lagged.index - lag
                
                series_winter_lagged = series_winter.copy()
                series_winter_lagged.index = series_winter_lagged.index - lag
                
                series_summer_lagged = series_summer.copy()
                series_summer_lagged.index = series_summer_lagged.index - lag
                
                # Join with glacier data
                df_ = df_glacier[['detrended_normalized_bw',
                                  'detrended_normalized_bs',
                                  'detrended_normalized_ba']].copy()
                df_['winter_index'] = series_winter_lagged
                df_['summer_index'] = series_summer_lagged
                df_['annual_index'] = series_wy_lagged
                
                df_.dropna(inplace=True)
                
                # Check the number of overlapping years
                if len(df_) < MIN_YEARS_OVERLAP:
                    continue
                
                # Correlate
                corr_bw, p_bw = stats.pearsonr(df_['detrended_normalized_bw'], df_['winter_index'])
                corr_bs, p_bs = stats.pearsonr(df_['detrended_normalized_bs'], df_['summer_index'])
                corr_ba, p_ba = stats.pearsonr(df_['detrended_normalized_ba'], df_['annual_index'])
                
                # Store results
                results.append({
                    'Glacier': glacier_name,
                    'ClimateIndex': climate_index,
                    'Lag_Years': lag,
                    'corr_bw': corr_bw,
                    'p_bw': p_bw,
                    'corr_bs': corr_bs,
                    'p_bs': p_bs,
                    'corr_ba': corr_ba,
                    'p_ba': p_ba,
                    'n_overlap': len(df_)
                })
        
        completed_count = len([r for r in results if r['Glacier'] == glacier_name])
        print(f"    Completed {completed_count} index-lag combinations with sufficient overlap")
    
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'glacier_mb_climate_lagged_correlations.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved lagged glacier correlation results to: {output_file}")
    print(f"  Total correlations: {len(df_results)}")
    
    return df_results

def detrend_normalize_glacier_mb(df_glacier):
    """
    Detrend and normalize glacier mass balance data (bw, bs, ba).
    
    Parameters
    ----------
    df_glacier : pd.DataFrame
        Glacier mass balance data with columns: bw, bs, ba
        
    Returns
    -------
    pd.DataFrame
        Detrended and normalized glacier data
    """
    result = pd.DataFrame(index=df_glacier.index)
    
    for col in ['bw', 'bs', 'ba']:
        if col in df_glacier.columns:
            detrended = detrend_normalize_data(df_glacier[col].values, normalize=True)
            result[f'detrended_normalized_{col}'] = detrended
    
    return result

def assign_water_year(dt):
    """Assign water year (Oct-Sep)."""
    if dt.month >= 10:
        return dt.year + 1
    else:
        return dt.year

def assign_glacier_winter(dt):
    """Assign glacier winter (Oct-May)."""
    if dt.month >= 10:
        return dt.year + 1
    elif dt.month <= 5:
        return dt.year
    else:
        return None

def assign_glacier_summer(dt):
    """Assign glacier summer (Jun-Sep)."""
    if 6 <= dt.month <= 9:
        return dt.year
    else:
        return None

def climate_index_by_water_year(df_monthly, column_name):
    """Aggregate climate index by water year."""
    df = df_monthly[[column_name]].copy()
    df['WY'] = df.index.to_series().apply(assign_water_year)
    result = df.groupby('WY')[column_name].mean()
    # Ensure index is integer
    result.index = result.index.astype(int)
    return result

def climate_index_by_winter(df_monthly, column_name):
    """Aggregate climate index by glacier winter (Oct-May)."""
    df = df_monthly[[column_name]].copy()
    df['winter_label'] = df.index.to_series().apply(assign_glacier_winter)
    df = df.dropna(subset=['winter_label'])
    result = df.groupby('winter_label')[column_name].mean()
    # Ensure index is integer
    result.index = result.index.astype(int)
    return result

def climate_index_by_summer(df_monthly, column_name):
    """Aggregate climate index by glacier summer (Jun-Sep)."""
    df = df_monthly[[column_name]].copy()
    df['summer_label'] = df.index.to_series().apply(assign_glacier_summer)
    df = df.dropna(subset=['summer_label'])
    result = df.groupby('summer_label')[column_name].mean()
    # Ensure index is integer
    result.index = result.index.astype(int)
    return result

def analyze_glacier_mb_correlations():
    """
    Analyze correlations between climate indices and glacier mass balance.
    
    Returns
    -------
    pd.DataFrame
        Glacier correlation results
    """
    print("\n" + "="*80)
    print("ANALYZING GLACIER MASS BALANCE CORRELATIONS")
    print("="*80)
    
    # Check if glacier MB file exists
    if not GLACIER_MB_FILE.exists():
        print(f"Warning: Glacier MB file not found: {GLACIER_MB_FILE}")
        print("Skipping glacier analysis.")
        return None
    
    print(f"Loading glacier mass balance data from: {GLACIER_MB_FILE}")
    
    # Load glacier mass balance data
    glacmb = pd.read_csv(GLACIER_MB_FILE)
    
    # Extract individual glaciers
    langjokull = glacmb[glacmb['glims_id'] == 'G339764E64629N'].copy()
    langjokull.set_index('yr', inplace=True)
    
    hofsjokull = glacmb[glacmb['glims_id'] == 'G341164E64838N'].copy()
    hofsjokull.set_index('yr', inplace=True)
    
    vatnajokull = glacmb[glacmb['glims_id'] == 'G343222E64409N'].copy()
    vatnajokull.set_index('yr', inplace=True)
    
    print(f"  Langjökull: {len(langjokull)} years")
    print(f"  Hofsjökull: {len(hofsjokull)} years")
    print(f"  Vatnajökull: {len(vatnajokull)} years")
    
    # Detrend and normalize glacier data
    langjokull_detrended = detrend_normalize_glacier_mb(langjokull)
    hofsjokull_detrended = detrend_normalize_glacier_mb(hofsjokull)
    vatnajokull_detrended = detrend_normalize_glacier_mb(vatnajokull)
    
    glaciers = {
        'Langjokull': langjokull_detrended,
        'Hofsjokull': hofsjokull_detrended,
        'Vatnajokull': vatnajokull_detrended
    }
    
    # Load AO and NAO from NOAA
    print("\nLoading climate indices...")
    ao_data = read_ao_nao_index(AO_URL)
    nao_data = read_ao_nao_index(NAO_URL)
    
    # Load other indices from Excel
    excel_indices = read_climate_indices_from_excel(CLIMATE_INDICES_FILE)
    
    # Combine all indices
    climate_indices = {
        'AO': ao_data,
        'NAO': nao_data
    }
    if excel_indices:
        climate_indices.update(excel_indices)
    
    print(f"  Total climate indices: {len(climate_indices)}")
    
    # Detrend and normalize climate indices
    detrended_normalized_indices = {}
    for index_name, index_data in climate_indices.items():
        # For AO/NAO, they're already normalized, so just use them as-is
        # For others, detrend and normalize
        if index_name in ['AO', 'NAO']:
            df_detrended = index_data.copy()
            # AO/NAO have 'Value' column, rename it
            df_detrended.columns = [f'detrended_normalized_{index_name}']
        else:
            detrended = detrend_normalize_data(index_data['Value'].values, normalize=True)
            df_detrended = pd.DataFrame(
                {f'detrended_normalized_{index_name}': detrended},
                index=index_data.index
            )
        
        # Ensure the dataframe has a DatetimeIndex
        if not isinstance(df_detrended.index, pd.DatetimeIndex):
            df_detrended.index = pd.to_datetime(df_detrended.index)
        
        detrended_normalized_indices[index_name] = df_detrended
    
    # Calculate correlations
    print(f"\nCalculating correlations (min {MIN_YEARS_OVERLAP} years overlap)...")
    results = []
    
    for glacier_name, df_glacier in glaciers.items():
        print(f"\n  Processing {glacier_name}...")
        
        for climate_index, df_climate in detrended_normalized_indices.items():
            col_name = f'detrended_normalized_{climate_index}'
            
            # Aggregate climate index by different periods
            series_wy = climate_index_by_water_year(df_climate, col_name)
            series_winter = climate_index_by_winter(df_climate, col_name)
            series_summer = climate_index_by_summer(df_climate, col_name)
            
            # Join with glacier data
            df_ = df_glacier[['detrended_normalized_bw',
                              'detrended_normalized_bs',
                              'detrended_normalized_ba']].copy()
            df_['winter_index'] = series_winter
            df_['summer_index'] = series_summer
            df_['annual_index'] = series_wy
            
            df_.dropna(inplace=True)
            
            # Check the number of overlapping years
            if len(df_) < MIN_YEARS_OVERLAP:
                continue
            
            # Correlate
            corr_bw, p_bw = stats.pearsonr(df_['detrended_normalized_bw'], df_['winter_index'])
            corr_bs, p_bs = stats.pearsonr(df_['detrended_normalized_bs'], df_['summer_index'])
            corr_ba, p_ba = stats.pearsonr(df_['detrended_normalized_ba'], df_['annual_index'])
            
            # Store results
            results.append({
                'Glacier': glacier_name,
                'ClimateIndex': climate_index,
                'corr_bw': corr_bw,
                'p_bw': p_bw,
                'corr_bs': corr_bs,
                'p_bs': p_bs,
                'corr_ba': corr_ba,
                'p_ba': p_ba,
                'n_overlap': len(df_)
            })
        
        print(f"    Completed {len([r for r in results if r['Glacier'] == glacier_name])} indices with sufficient overlap")
    
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'glacier_mb_climate_correlation_filtered.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved glacier correlation results to: {output_file}")
    print(f"  Total correlations: {len(df_results)}")
    
    return df_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("CLIMATE INDICES CORRELATION ANALYSIS")
    print("="*80)
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Hydrological year starts: Month {HYDROLOGICAL_YEAR_FIRST_MONTH}")
    
    # Load streamflow data (annual and seasonal)
    print("\n" + "="*80)
    print("LOADING STREAMFLOW DATA")
    print("="*80)
    df_streamflow_annual = load_annual_streamflow_data()
    df_streamflow_seasonal = load_seasonal_streamflow_data()
    
    # 1. Analyze AO and NAO correlations
    df_ao_nao = analyze_ao_nao_correlations(df_streamflow_annual)
    
    # 2. Analyze extended climate indices (if file exists)
    if CLIMATE_INDICES_FILE.exists():
        df_extended = analyze_extended_climate_indices(df_streamflow_annual, df_streamflow_seasonal)
    else:
        print(f"\nWarning: Climate indices file not found: {CLIMATE_INDICES_FILE}")
        print("Skipping extended analysis.")
    
    # 3. Analyze lagged correlations (predictive potential)
    df_lagged = analyze_lagged_climate_correlations(df_streamflow_annual)
    
    # 4. Analyze glacier mass balance correlations
    df_glacier = analyze_glacier_mb_correlations()
    
    # 5. Analyze lagged glacier mass balance correlations
    df_glacier_lagged = analyze_lagged_glacier_mb_correlations()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - climate_index_correlation_results_CI_not_normalized.csv")
    if CLIMATE_INDICES_FILE.exists():
        print("  - climate_indices_correlation_results_extended_feb_2025.csv")
    print("  - climate_indices_lagged_correlations.csv")
    if df_glacier is not None:
        print("  - glacier_mb_climate_correlation_filtered.csv")
    if df_glacier_lagged is not None:
        print("  - glacier_mb_climate_lagged_correlations.csv")

if __name__ == "__main__":
    main()

