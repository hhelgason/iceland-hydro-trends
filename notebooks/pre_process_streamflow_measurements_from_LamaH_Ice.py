"""
Pre-process streamflow measurements from the LamaH-Ice dataset for trend analysis.

This script performs the following operations:
1. Reads daily streamflow measurements from LamaH-Ice CSV files
2. Combines the series into one dataframe
3. Removes gauges that are strongly influenced by human activities (degimpact='s'),
   except for specific gauges where annual flows are not significantly altered
4. Handles specific data quality issues:
   - Merges Kelduá ofan Grjótár (id 55) data into Kelduá ofan Folavatns (id 56)
   - Removes gauges with known data quality issues
5. Saves the cleaned dataset

Input data format:
- Daily streamflow measurements in CSV files (separator=';')
- File columns: YYYY, MM, DD, qobs (flow), qc_flag (quality flag)
- Catchment attributes in CSV format with 'degimpact' column indicating human influence

Output:
- DataFrame with datetime index and gauge IDs as columns
- Data range: through 2023-09-30
- Saved in both CSV and pickle formats for accessibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import sys
from config import (
    LAMAH_ICE_BASE_PATH,
    OUTPUT_DIR,
    STREAMFLOW_DATA_PATH,
    GAUGES_TO_KEEP,
    GAUGES_TO_REMOVE,
    WITHIN_YEAR_COVERAGE_THRESHOLD
)

def process_streamflow_data(path_gauges_ts):
    """Process streamflow data and return a dictionary with streamflow series.

    Parameters:
    ----------
    path_gauges_ts : Path
        Path to directory containing streamflow measurement files.
        Expected format: ID_*.csv files with columns YYYY, MM, DD, qobs, qc_flag

    Returns:
    -------
    dict
        Dictionary mapping gauge IDs to their streamflow DataFrames
    """
    meas_dict = {}

    # Verify directory exists
    if not path_gauges_ts.exists():
        raise FileNotFoundError(f"Streamflow data directory not found: {path_gauges_ts}")

    # Loop through gauge files
    for gauge_file in path_gauges_ts.glob("ID_*.csv"):
        try:
            catchment_id = gauge_file.stem.split("_")[1]  # Extract ID
            df = pd.read_csv(gauge_file, sep=";")

            # Convert to datetime index
            df['date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].astype(str).agg('-'.join, axis=1))
            df.set_index('date', inplace=True)

            # Rename columns
            df = df.rename(columns={'qobs': 'Value', 'qc_flag': 'Quality'})
            df = df.drop(columns=['YYYY', 'MM', 'DD', 'Quality'])

            # Store in dictionary
            meas_dict[catchment_id] = df
        except Exception as e:
            print(f"Error processing file {gauge_file}: {e}", file=sys.stderr)
            continue

    if not meas_dict:
        raise ValueError("No valid streamflow data files were processed")

    return meas_dict

def main():
    # Create output directory if it doesn't exist
    STREAMFLOW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read catchment characteristics
    catchment_attributes_file = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"
    if not catchment_attributes_file.exists():
        raise FileNotFoundError(f"Catchment attributes file not found: {catchment_attributes_file}")

    catchments_chara = pd.read_csv(catchment_attributes_file, sep=';')
    catchments_chara = catchments_chara.set_index('id')

    # Process streamflow data
    meas_dict = process_streamflow_data(LAMAH_ICE_BASE_PATH / "D_gauges/2_timeseries/daily")

    # Convert to DataFrame format
    gauge_data_nan_cleaned_value_only = {}
    for key in meas_dict.keys():
        series = meas_dict[key]['Value']
        series.index = series.index.date
        gauge_data_nan_cleaned_value_only[key] = series

    df_with_d = pd.DataFrame(gauge_data_nan_cleaned_value_only)
    df_with_d.index = pd.to_datetime(df_with_d.index)
    df_with_d.columns = df_with_d.columns.astype(int)

    # Data quality corrections
    # Merge Kelduá gauges: Insert data from Kelduá ofan Grjótár (id 55) into Kelduá ofan Folavatns (id 56)
    # These gauges are in the same river, located close to one another
    print("Merging Kelduá gauge data (ID 55 into ID 56)")
    df_with_d[56]['1998-10-13':'2006-08-17'] = df_with_d[55]['1998-10-13':'2006-08-17']
    df_with_d[55] = np.nan

    # Remove strongly influenced gauges, except those explicitly kept
    masker = catchments_chara[
        (catchments_chara['degimpact'] == 's') & 
        (~catchments_chara.index.isin(GAUGES_TO_KEEP))
    ].index
    df_cleaned = df_with_d.drop(columns=masker)

    # Remove gauges with known data quality issues
    df_cleaned = df_cleaned.drop(columns=GAUGES_TO_REMOVE, errors='ignore')
    
    # Ensure consistent column format and date range
    df_cleaned.columns = df_cleaned.columns.astype(str)
    df_cleaned = df_cleaned[:'2023-09-30']

    # Save outputs
    print(f"Saving cleaned data to {STREAMFLOW_DATA_PATH}")
    print(f"DataFrame shape: {df_cleaned.shape}")
    print(f"Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
    print(f"Number of gauges: {len(df_cleaned.columns)}")
    
    # Save as CSV
    df_cleaned.to_csv(STREAMFLOW_DATA_PATH)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)