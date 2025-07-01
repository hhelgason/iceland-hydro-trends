"""
Script for analyzing correlations between streamflow trends and catchment characteristics in Iceland.

This script is part of the analysis for the paper:
"Understanding Changes in Iceland's Streamflow Dynamics in Response to Climate Change"

The script performs correlation analysis between:
- Streamflow trends (annual and seasonal)
- Catchment physical characteristics
- Meteorological trends
- Glacier attributes

For each correlation:
1. Calculates Pearson correlation coefficient and p-value
2. Creates scatter plots for significant correlations (p<0.05)
3. Saves results to CSV files for further analysis
4. Generates separate analyses for:
   - All rivers
   - Glacial rivers (glacier fraction >= 0.1)
   - Non-glacial rivers (glacier fraction < 0.1)

Dependencies:
    pandas
    numpy
    matplotlib
    seaborn
    scipy
    geopandas
    pathlib
    os

Author: Hordur Bragi
Created: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import geopandas as gpd
import os
from config import (
    OUTPUT_DIR,
    PERIOD,
    CATCHMENT_ATTRIBUTES_FILE
)

# --- CONFIG ---
results_file = OUTPUT_DIR / 'results_lamah_data' / f'results_{PERIOD}.csv'
output_folder = OUTPUT_DIR / f'{PERIOD}/correlations'
split_output_folder = output_folder / 'split_glac_and_non_glac_rivers'
glac_output_folder = split_output_folder / 'correlations_glac_rivers'
non_glac_output_folder = split_output_folder / 'correlations_non_glac_rivers'

def create_all_directories():
    """
    Create all necessary directories for storing analysis results.
    
    Creates directory structure for:
    - Base output directory
    - Results directory
    - Correlation analysis directories
    - Separate directories for glacial and non-glacial analyses
    - Metric-specific directories
    
    Prints status messages for each directory creation attempt.
    """
    print("\n=== Creating directory structure ===")
    # Create base directories
    directories = [
        OUTPUT_DIR,
        OUTPUT_DIR / 'results_lamah_data',
        output_folder,
        split_output_folder,
        glac_output_folder,
        non_glac_output_folder
    ]
    
    # Add metric directories for each output folder
    for metric_folder in trend_metrics.values():
        directories.extend([
            output_folder / metric_folder,
            glac_output_folder / metric_folder,
            non_glac_output_folder / metric_folder
        ])
    
    # Create all directories
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")

# --- METRICS TO ANALYZE ---
trend_metrics = {
    'annual_avg_flow_trend_per_decade': 'annual_mean_flow',
    'trend_annual_cv_per_decade': 'annual_cv',
    'trend_annual_std_per_decade': 'annual_std',
    'trend_flashiness_per_decade': 'flashiness',
    'trend_rising_seq_per_decade': 'rising_sequences',
    'trend_falling_seq_per_decade': 'falling_sequences',
    'trend_baseflow_index_per_decade': 'baseflow_index',
}

# Add seasonal metrics with flattened directory structure
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    trend_metrics[f'trend_{season}_per_decade'] = f'mean_flow_{season}'
    trend_metrics[f'std_{season}_trend_per_decade'] = f'std_{season}'
    trend_metrics[f'cv_{season}_trend_per_decade'] = f'cv_{season}'
    trend_metrics[f'flashiness_{season}_trend_per_decade'] = f'flashiness_{season}'
    trend_metrics[f'rising_{season}_trend_per_decade'] = f'rising_sequences_{season}'
    trend_metrics[f'falling_{season}_trend_per_decade'] = f'falling_sequences_{season}'
    trend_metrics[f'baseflow_index_{season}_trend_per_decade'] = f'baseflow_index_{season}'

# Create all necessary directories at the start
create_all_directories()

# --- LOAD DATA ---
print("\n=== Loading trend results ===")
print(f"From file: {results_file}")
results = pd.read_csv(results_file, sep=';', index_col=0)
print("\nResults DataFrame:")
print(results.head())
print(f"\nResults index range: {results.index.min()} to {results.index.max()}")
print(f"Results index values: {sorted(results.index.tolist())}")

print("\n=== Loading catchment characteristics ===")
print(f"From file: {CATCHMENT_ATTRIBUTES_FILE}")
catchments_chara = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)
# Ensure id column is integer type before setting as index
catchments_chara['id'] = catchments_chara['id'].astype(int)
catchments_chara = catchments_chara.set_index('id')
print("\nCatchment characteristics DataFrame:")
print(catchments_chara.head())
print(f"\nCatchment characteristics index range: {catchments_chara.index.min()} to {catchments_chara.index.max()}")
print(f"Catchment characteristics index values: {sorted(catchments_chara.index.tolist())}")

# Verify that indices match
print("\n=== Verifying indices ===")
common_indices = set(results.index).intersection(set(catchments_chara.index))
print(f"Number of common indices: {len(common_indices)} out of {len(results.index)} results and {len(catchments_chara.index)} catchments")
print(f"Common indices: {sorted(list(common_indices))}")

# Check for any results indices not in catchment characteristics
missing_indices = set(results.index) - set(catchments_chara.index)
if missing_indices:
    print(f"\nWARNING: Found {len(missing_indices)} indices in results that are not in catchment characteristics:")
    print(sorted(list(missing_indices)))

# Ensure indices match before proceeding
if len(missing_indices) > 0:
    raise ValueError("Some indices in results are not found in catchment characteristics. Please check the data.")

# --- Load and merge meteorological trends for this period (copied from timing_trends_catchment_analysis.py) ---
meteo_vars_list = ['prec', '2m_temp_mean', 'total_et', 'snowfall', 'rainfall']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
start_year, end_year = PERIOD.split('_')
meteo_var_labels = {}

# Define paths for both periods
results_paths = {
    '1973': Path("merged_results_dict_1973-2023_june_2025.pkl"),
    '1993': Path("merged_results_dict_1993-2023_june_2025.pkl")
}

# Load results for the current period
current_results_path = results_paths[start_year]
print(f"\nLoading meteorological trends from: {current_results_path}")

if current_results_path.exists():
    import pickle
    with open(current_results_path, 'rb') as f:
        merged_results_dict = pickle.load(f)
    for var in meteo_vars_list:
        key = f'{var}_{start_year}-{end_year}'
        if key in merged_results_dict:
            df = merged_results_dict[key]
            # Annual trend
            colname = f'trend_{var}_{start_year}_{end_year}'
            catchments_chara[colname] = df['annual_trend']
            meteo_var_labels[colname] = f"{var} annual trend ({start_year}-{end_year})"
            # Seasonal trends
            for season in seasons:
                colname_season = f'trend_{var}_{season}_{start_year}_{end_year}'
                if f'trend_{season}' in df.columns:
                    catchments_chara[colname_season] = df[f'trend_{season}']
                    meteo_var_labels[colname_season] = f"{var} {season} trend ({start_year}-{end_year})"
else:
    print(f"Warning: Meteorological trends file not found: {current_results_path}")

# Update meteo_vars to only include meteorological trend columns for this period
meteo_vars = [k for k in meteo_var_labels.keys() if k in catchments_chara.columns]

# --- CATCHMENT VARIABLES ---
catchment_vars = [
    'area_calc', 'elev_mean', 'elev_med', 'elev_std', 'elev_ran', 'slope_mean', 'asp_mean', 'elon_ratio',
    'strm_dens', 'g_lat', 'g_lon', 'p_mean', 'aridity', 'frac_snow', 'p_season', 'bare_fra', 'forest_fra',
    'glac_fra', 'g_frac', 'lake_fra', 'wetl_fra', 'urban_fra', 'agr_fra', 'sand_fra', 'silt_fra', 'clay_fra',
    'oc_fra', 'root_dep', 'soil_tawc', 'soil_poros', 'bedrk_dep', 'ndvi_max', 'ndvi_min', 'lai_max',
    'lai_diff', 'gvf_max', 'gvf_diff', 'q_mean', 'runoff_ratio', 'baseflow_index_ladson', 'hfd_mean',
    'slope_fdc', 'high_q_dur', 'low_q_dur', 'high_q_freq', 'low_q_freq', 'Q5', 'Q95',
    'g_frac', 'g_lat', 'g_lon', 'g_mean_el', 'g_min_el', 'g_slope', 'g_slopel20', 'glac_fra'
]

# Define glacier-related attributes (excluding g_frac which is used for filtering)
glacier_attributes = ['g_lat', 'g_lon', 'g_mean_el', 'g_min_el', 'g_slope', 'g_slopel20', 'glac_fra']

# --- CORRELATION ANALYSIS ---
def run_correlation_analysis(results, catchments_chara, output_folder, river_type=None, g_frac_threshold=0.1):
    """
    Run correlation analysis between streamflow trends and catchment characteristics.
    
    Parameters
    ----------
    results : DataFrame
        DataFrame containing streamflow trend results
    catchments_chara : DataFrame
        DataFrame containing catchment characteristics
    output_folder : Path
        Directory to save correlation results and plots
    river_type : str, optional
        Type of rivers to analyze ('glacial', 'non_glacial', or None for all)
    g_frac_threshold : float, optional
        Threshold for glacier fraction to classify rivers, by default 0.1
        
    Returns
    -------
    list
        List of dictionaries containing significant correlation results
        
    Notes
    -----
    For each metric and catchment characteristic:
    1. Calculates Pearson correlation
    2. Creates scatter plot if correlation is significant (p<0.05)
    3. Saves results to CSV files
    4. For glacier attributes, only includes catchments with g_frac > 0.05
    """
    global_significant_results = []
    
    # Filter data based on river type if specified
    if river_type == 'glacial':
        catchments_chara = catchments_chara[catchments_chara['g_frac'] >= g_frac_threshold].copy()
        results = results[results.index.isin(catchments_chara.index)].copy()
    elif river_type == 'non_glacial':
        catchments_chara = catchments_chara[catchments_chara['g_frac'] < g_frac_threshold].copy()
        results = results[results.index.isin(catchments_chara.index)].copy()
    
    print(f"\n=== Running correlation analysis for {river_type if river_type else 'all'} rivers ===")
    print(f"Number of catchments: {len(catchments_chara)}")
    
    # Create all necessary directories upfront
    all_metric_dirs = []
    for metric, metric_folder in trend_metrics.items():
        if metric in results.columns:
            metric_dir = output_folder / metric_folder
            all_metric_dirs.append(metric_dir)
    
    # Create all directories at once
    for directory in [output_folder] + all_metric_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
    
    for metric, metric_folder in trend_metrics.items():
        if metric not in results.columns:
            print(f"Skipping {metric} - not found in results")
            continue
        
        metric_out = output_folder / metric_folder
        print(f"\nProcessing {metric} -> {metric_out}")
        
        metric_label = metric.replace('_', ' ').title()
        significant_results = []
        
        # 1. Correlate with catchment attributes
        for var in catchment_vars:
            if var not in catchments_chara.columns:
                continue
            
            try:
                # For glacier attributes, only include catchments with g_frac > 0.05
                if var in glacier_attributes:
                    # Skip g_dom_NI if it's non-numeric
                    if var == 'g_dom_NI' and not pd.api.types.is_numeric_dtype(catchments_chara[var]):
                        continue
                        
                    merged = pd.DataFrame({
                        metric: results[metric],
                        var: catchments_chara[var],
                        'g_frac': catchments_chara['g_frac']
                    }).dropna()
                    # Filter for glacierized catchments
                    merged = merged[merged['g_frac'] > 0.05]
                    if len(merged) < 3:  # Skip if too few glacierized catchments
                        continue
                    # Remove g_frac column before correlation
                    merged = merged.drop('g_frac', axis=1)
                else:
                    merged = pd.DataFrame({
                        metric: results[metric],
                        var: catchments_chara[var]
                    }).dropna()
                    if len(merged) < 3:
                        continue
                
                corr, pval = stats.pearsonr(merged[metric], merged[var])
                
                if pval < 0.05:
                    significant_results.append({
                        'metric': metric,
                        'metric_label': metric_label,
                        'var': var,
                        'corr': corr,
                        'pval': pval,
                        'n': len(merged),
                        'type': 'catchment'
                    })
                    global_significant_results.append({
                        'metric': metric,
                        'metric_label': metric_label,
                        'var': var,
                        'corr': corr,
                        'pval': pval,
                        'n': len(merged),
                        'type': 'catchment'
                    })
                    
                    plt.figure(figsize=(5,4))
                    sns.scatterplot(x=merged[var], y=merged[metric])
                    z = np.polyfit(merged[var], merged[metric], 1)
                    p = np.poly1d(z)
                    plt.plot(merged[var], p(merged[var]), "r--", alpha=0.8)
                    plt.title(f"{metric_label} vs {var}\nr = {corr:.2f}, p = {pval:.3f}")
                    plt.xlabel(var)
                    plt.ylabel(metric_label)
                    
                    # Highlight significant Mann-Kendall trends
                    pval_col = f"pval_{metric}" if f"pval_{metric}" in results.columns else ("pval" if "pval" in results.columns else None)
                    if pval_col and pval_col in results.columns:
                        sig_idx = results.index[results[pval_col] < 0.05]
                        sig_points = merged.loc[sig_idx.intersection(merged.index)]
                        if not sig_points.empty:
                            plt.scatter(sig_points[var], sig_points[metric], facecolors='none', edgecolors='black', s=100, label='Significant MK trend (p < 0.05)')
                            plt.legend()
                    
                    plt.tight_layout()
                    
                    try:
                        plot_path = metric_out / f"{metric}_vs_{var}.png"
                        plt.savefig(plot_path, dpi=200)
                        print(f"Saved plot to: {plot_path}")
                    except Exception as e:
                        print(f"Error saving plot {plot_path}: {e}")
                    finally:
                        plt.close()
                        
            except Exception as e:
                print(f"Error processing {metric} vs {var}: {e}")
                continue
        
        # 2. Correlate with meteorological trends
        for var in meteo_vars:
            try:
                if var in results.columns:
                    merged = pd.DataFrame({metric: results[metric], var: results[var]}).dropna()
                elif var in catchments_chara.columns:
                    merged = pd.DataFrame({metric: results[metric], var: catchments_chara[var]}).dropna()
                else:
                    continue
                
                if len(merged) < 3:
                    continue
                
                corr, pval = stats.pearsonr(merged[metric], merged[var])
                
                if pval < 0.05:
                    significant_results.append({
                        'metric': metric,
                        'metric_label': metric_label,
                        'var': var,
                        'corr': corr,
                        'pval': pval,
                        'n': len(merged),
                        'type': 'meteo'
                    })
                    global_significant_results.append({
                        'metric': metric,
                        'metric_label': metric_label,
                        'var': var,
                        'corr': corr,
                        'pval': pval,
                        'n': len(merged),
                        'type': 'meteo'
                    })
                    
                    plt.figure(figsize=(5,4))
                    sns.scatterplot(x=merged[var], y=merged[metric])
                    z = np.polyfit(merged[var], merged[metric], 1)
                    p = np.poly1d(z)
                    plt.plot(merged[var], p(merged[var]), "r--", alpha=0.8)
                    plt.title(f"{metric_label} vs {var}\nr = {corr:.2f}, p = {pval:.3f}")
                    plt.xlabel(var)
                    plt.ylabel(metric_label)
                    
                    # Highlight significant Mann-Kendall trends
                    pval_col = f"pval_{metric}" if f"pval_{metric}" in results.columns else ("pval" if "pval" in results.columns else None)
                    if pval_col and pval_col in results.columns:
                        sig_idx = results.index[results[pval_col] < 0.05]
                        sig_points = merged.loc[sig_idx.intersection(merged.index)]
                        if not sig_points.empty:
                            plt.scatter(sig_points[var], sig_points[metric], facecolors='none', edgecolors='black', s=100, label='Significant MK trend (p < 0.05)')
                            plt.legend()
                    
                    plt.tight_layout()
                    
                    try:
                        plot_path = metric_out / f"{metric}_vs_{var}.png"
                        plt.savefig(plot_path, dpi=200)
                        print(f"Saved plot to: {plot_path}")
                    except Exception as e:
                        print(f"Error saving plot {plot_path}: {e}")
                    finally:
                        plt.close()
                        
            except Exception as e:
                print(f"Error processing {metric} vs {var}: {e}")
                continue
        
        if significant_results:
            try:
                df = pd.DataFrame(significant_results)
                df.to_csv(metric_out / 'significant_correlations.csv', index=False)
            except Exception as e:
                print(f"Error saving significant correlations for {metric}: {e}")
    
    if global_significant_results:
        try:
            df = pd.DataFrame(global_significant_results)
            df.to_csv(output_folder / 'significant_trend_correlations_all.csv', index=False)
            df.to_excel(output_folder / 'significant_correlations.xlsx', index=False)
        except Exception as e:
            print(f"Error saving global significant correlations: {e}")
    
    return global_significant_results

# Run analysis for all rivers (original analysis)
all_results = run_correlation_analysis(results, catchments_chara, output_folder)

# Run analysis for glacial rivers
glacial_results = run_correlation_analysis(results, catchments_chara, glac_output_folder, 'glacial', 0.1)

# Run analysis for non-glacial rivers
non_glacial_results = run_correlation_analysis(results, catchments_chara, non_glac_output_folder, 'non_glacial', 0.1)

# Create a summary of the split analysis
summary_data = {
    'all_rivers': len(catchments_chara),
    'glacial_rivers': len(catchments_chara[catchments_chara['g_frac'] >= 0.1]),
    'non_glacial_rivers': len(catchments_chara[catchments_chara['g_frac'] < 0.1])
}

summary_df = pd.DataFrame([summary_data])
summary_df.to_csv(split_output_folder / 'analysis_summary.csv', index=False)
print("\n=== Analysis Summary ===")
print(f"Total number of rivers: {summary_data['all_rivers']}")
print(f"Number of glacial rivers (g_frac >= 0.1): {summary_data['glacial_rivers']}")
print(f"Number of non-glacial rivers (g_frac < 0.1): {summary_data['non_glacial_rivers']}") 