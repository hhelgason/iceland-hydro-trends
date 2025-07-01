import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import geopandas as gpd
import pickle

print("Starting analysis...")

plt.rcParams['font.family'] = 'Arial'

from config import (
    OUTPUT_DIR,
    LAMAH_ICE_BASE_PATH,
    CATCHMENT_ATTRIBUTES_FILE
)

# Time periods
time_periods = [(1973, 2023), (1993, 2023)]

# Base paths
figures_path = OUTPUT_DIR / 'timing_trends_figures'
base_folder = OUTPUT_DIR / 'timing_analysis'

print(f"Base folder path: {base_folder}")
print(f"Base folder exists: {base_folder.exists()}")

# Check if timing calculations need to be run
def check_timing_files_exist():
    time_periods = [(1973, 2023), (1993, 2023)]
    metrics = ['freshet', 'centroid', 'peak_flow']
    
    for start_year, end_year in time_periods:
        period_folder = base_folder / f"{start_year}_{end_year}"
        if not period_folder.exists():
            return False
        
        for metric in metrics:
            trends_file = period_folder / f'{metric}_trends_{start_year}_{end_year}_ts_mk.csv'
            if not trends_file.exists():
                print(f"Missing trend file: {trends_file}")
                return False
    
    return True

# Run timing calculations if needed
if not check_timing_files_exist():
    print("Need to run timing calculations first...")
    print("Please run timing_calculations_ts_mk.py first to generate the trend files.")
    sys.exit(1)

print("Found all required trend files. Proceeding with analysis...")

# Read catchment characteristics from GeoPackage
print(f"Catchment attributes file exists: {CATCHMENT_ATTRIBUTES_FILE.exists()}")

catchments_chara = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)
catchments_chara = catchments_chara.set_index('id')

print("Successfully read catchment characteristics")
print("Available columns:", catchments_chara.columns.tolist())

# --- Load and merge meteorological trends for each period ---
# Define meteorological variables and seasons
meteo_vars_list = ['prec', '2m_temp_mean', 'total_et', 'snowfall', 'rainfall']
seasons = ['DJF', 'MAM', 'JJA', 'SON']
meteo_var_labels = {}

# Define paths for both periods
results_paths = {
    '1973': OUTPUT_DIR / "merged_results_dict_1973-2023_june_2025.pkl",
    '1993': OUTPUT_DIR / "merged_results_dict_1993-2023_june_2025.pkl"
}

# Load meteorological trends for each period
for start_year, end_year in time_periods:
    current_results_path = results_paths[str(start_year)]
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

# Variables to analyze based on ESSD paper appendix tables
catchment_vars = {
    # Location and topography
    'area_calc': 'Catchment Area (km²)',
    'elev_mean': 'Mean Elevation (m a.s.l.)',
    'elev_med': 'Median Elevation (m)',
    'elev_std': 'Elevation Standard Deviation (m)',
    'elev_ran': 'Elevation Range (m)',
    'slope_mean': 'Mean Slope (degrees)',
    'asp_mean': 'Mean Aspect (degrees)',
    'elon_ratio': 'Elongation Ratio (-)',
    'strm_dens': 'Stream Density (km/km²)',
    'g_lat': 'Latitude (°N)',
    'g_lon': 'Longitude (°E)',
    
    # Climate indices
    'p_mean': 'Mean Annual Precipitation (mm/year)',
    'p_mean_ERA5L': 'Mean Annual Precipitation ERA5L (mm/year)',
    'pet_mean_ERA5L': 'Mean Annual PET ERA5L (mm/year)',
    'aridity': 'Aridity Index (-)',
    'aridity_ERA5L': 'Aridity Index ERA5L (-)',
    'frac_snow': 'Fraction of Precipitation as Snow (-)',
    'frac_snow_ERA5L': 'Fraction of Precipitation as Snow ERA5L (-)',
    'p_season': 'Precipitation Seasonality (-)',
    'p_season_ERA5L': 'Precipitation Seasonality ERA5L (-)',
    
    # Precipitation timing
    'high_prec_timing': 'High Precipitation Timing (day of year)',
    'high_prec_timing_ERA5L': 'High Precipitation Timing ERA5L (day of year)',
    'low_prec_timing': 'Low Precipitation Timing (day of year)',
    'low_prec_timing_ERA5L': 'Low Precipitation Timing ERA5L (day of year)',
    'high_prec_du': 'High Precipitation Duration (days)',
    'high_prec_du_ERA5L': 'High Precipitation Duration ERA5L (days)',
    'low_prec_du': 'Low Precipitation Duration (days)',
    'low_prec_du_ERA5L': 'Low Precipitation Duration ERA5L (days)',
    'high_prec_fr': 'High Precipitation Frequency (-)',
    'high_prec_fr_ERA5L': 'High Precipitation Frequency ERA5L (-)',
    'lo_prec_fr': 'Low Precipitation Frequency (-)',
    'lo_prec_fr_ERA5L': 'Low Precipitation Frequency ERA5L (-)',
    
    # Land cover
    'bare_fra': 'Bare Ground Fraction (-)',
    'forest_fra': 'Forest Fraction (-)',
    'glac_fra': 'Glacier Fraction (-)',
    #'g_frac': 'Glacier Fraction (alternative) (-)',
    'lake_fra': 'Lake Fraction (-)',
    'wetl_fra': 'Wetland Fraction (-)',
    'urban_fra': 'Urban Fraction (-)',
    'agr_fra': 'Agricultural Fraction (-)',
    
    # Soil properties
    'sand_fra': 'Sand Fraction (-)',
    'silt_fra': 'Silt Fraction (-)',
    'clay_fra': 'Clay Fraction (-)',
    'oc_fra': 'Organic Carbon Fraction (-)',
    'root_dep': 'Root Zone Depth (cm)',
    'soil_tawc': 'Total Available Water Content (mm)',
    'soil_poros': 'Soil Porosity (-)',
    'bedrk_dep': 'Depth to Bedrock (m)',
    
    # Vegetation indices
    'ndvi_max': 'Maximum NDVI (-)',
    'ndvi_min': 'Minimum NDVI (-)',
    'lai_max': 'Maximum LAI (-)',
    'lai_diff': 'LAI Difference (-)',
    'gvf_max': 'Maximum Green Vegetation Fraction (-)',
    'gvf_diff': 'Green Vegetation Fraction Difference (-)',
    
    # Hydrological signatures
    'q_mean': 'Mean Annual Discharge (mm/year)',
    'runoff_ratio': 'Runoff Ratio (-)',
    'baseflow_index_ladson': 'Baseflow Index (-)',
    'hfd_mean': 'Mean Half Flow Date (day of year)',
    #'stream_elas': 'Streamflow Elasticity (-)',
    'slope_fdc': 'Slope of Flow Duration Curve (-)',
    'high_q_dur': 'High Flow Duration (days)',
    'low_q_dur': 'Low Flow Duration (days)',
    'high_q_freq': 'High Flow Frequency (year⁻¹)',
    'low_q_freq': 'Low Flow Frequency (year⁻¹)',
    'Q5': '5th Percentile Flow (mm/day)',
    'Q95': '95th Percentile Flow (mm/day)',
    
    # Glacier characteristics
    'g_area': 'Total Glacier Area (km²)',
    'g_mean_el': 'Glacier Mean Elevation (m)',
    'g_max_el': 'Glacier Maximum Elevation (m)',
    'g_min_el': 'Glacier Minimum Elevation (m)',
    'g_slope': 'Glacier Mean Slope (degrees)',
    #'g_aspect': 'Glacier Mean Aspect (degrees)',
    'g_slopel20': 'Slope of the 20% lowermost area of the glacier (degrees)',
    'g_dom_NI': 'Dominant Glacier Type (-)'
}

# Add meteorological trend columns to catchment_vars
to_add = {k: v for k, v in meteo_var_labels.items() if k in catchments_chara.columns}
catchment_vars.update(to_add)

# Timing metrics
timing_metrics = {
    'freshet': 'Spring Freshet',
    'centroid': 'Centroid of Timing',
    'peak_flow': 'Peak Flow'
}

# Function to read trend files
def read_trend_file(start_year, end_year, trend_type):
    period_folder = base_folder / f"{start_year}_{end_year}"
    trends_file = period_folder / f'{trend_type}_trends_{start_year}_{end_year}_ts_mk.csv'
    print(f"Reading trend file: {trends_file}")
    trends_df = pd.read_csv(trends_file)
    trends_df = trends_df.set_index('Gauge')
    return trends_df

# Function to calculate correlations and p-values
def calculate_correlations(data, x_var, y_var):
    correlation, p_value = stats.pearsonr(data[x_var], data[y_var])
    return correlation, p_value

# Function to create scatter plot with correlation info
def plot_relationship(data, x_var, y_var, ax, title, x_label, y_label):
    # For glacier attributes, only include catchments with g_frac > 0.05
    glacier_attributes = ['g_area', 'g_mean_el', 'g_max_el', 'g_min_el', 'g_slope', 'g_slopel20', 'glac_fra', 'g_frac']
    
    if x_var in glacier_attributes:
        # Get g_frac from catchments_chara
        data = data.copy()
        data['g_frac'] = catchments_chara.loc[data.index, 'g_frac']
        # Filter for glacierized catchments
        data = data[data['g_frac'] > 0.001]
        # Remove g_frac column before further processing
        data = data.drop('g_frac', axis=1)
    
    # Remove any rows with NaN, None, or infinite values
    valid_data = data.replace([np.inf, -np.inf, None], np.nan).dropna(subset=[x_var, y_var])
    
    # Convert columns to numeric, coercing errors to NaN
    valid_data[x_var] = pd.to_numeric(valid_data[x_var], errors='coerce')
    valid_data[y_var] = pd.to_numeric(valid_data[y_var], errors='coerce')
    
    # Drop any rows that became NaN after conversion
    valid_data = valid_data.dropna(subset=[x_var, y_var])
    
    if len(valid_data) < 2:
        print(f"Warning: Not enough valid data points for {title}")
        ax.text(0.5, 0.5, 'Insufficient data\nfor analysis', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return
    
    try:
        # Calculate correlation
        correlation, p_value = calculate_correlations(valid_data, x_var, y_var)
        
        # Create scatter plot
        sns.scatterplot(data=valid_data, x=x_var, y=y_var, ax=ax)
        
        # Add trend line
        try:
            z = np.polyfit(valid_data[x_var], valid_data[y_var], 1)
            p = np.poly1d(z)
            x_range = valid_data[x_var]
            ax.plot(x_range, p(x_range), "r--", alpha=0.8)
        except np.linalg.LinAlgError:
            print(f"Warning: Could not fit trend line for {title}")
        
        # Add correlation info
        text = f'r = {correlation:.2f}\np = {p_value:.3f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    except Exception as e:
        print(f"Error in plotting {title}: {str(e)}")
        ax.text(0.5, 0.5, 'Error in analysis', 
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

# Function to create scatter plots for significant correlations
def plot_significant_correlations(trend_data, catchments_chara, start_year, end_year, plot_folder):
    # Collect all correlations
    all_correlations = []
    
    # Filter catchment_vars to only include meteorological trend columns for this period
    period_catchment_vars = {k: v for k, v in catchment_vars.items() if (not k.startswith('trend_') or k.endswith(f'{start_year}_{end_year}'))}
    
    # Define glacier attributes
    glacier_attributes = ['g_area', 'g_mean_el', 'g_max_el', 'g_min_el', 'g_slope', 'g_slopel20', 'glac_fra', 'g_frac']
    
    # Process catchment characteristics correlations
    for metric in timing_metrics:
        # 1. Correlate with catchment characteristics
        for var, var_label in period_catchment_vars.items():
            # Merge trend data with catchment characteristics
            merged_data = trend_data[metric].merge(
                catchments_chara[[var]], 
                left_index=True, 
                right_index=True
            )
            
            # For glacier attributes, only include catchments with g_frac > 0.001
            if var in glacier_attributes:
                merged_data['g_frac'] = catchments_chara.loc[merged_data.index, 'g_frac']
                merged_data = merged_data[merged_data['g_frac'] > 0.001]
                merged_data = merged_data.drop('g_frac', axis=1)
            
            # Convert columns to numeric, coercing errors to NaN
            merged_data[var] = pd.to_numeric(merged_data[var], errors='coerce')
            merged_data['Trend (days/decade)'] = pd.to_numeric(merged_data['Trend (days/decade)'], errors='coerce')
            merged_data['Mann-Kendall p-value'] = pd.to_numeric(merged_data['Mann-Kendall p-value'], errors='coerce')
            
            # Calculate correlation
            try:
                valid_data = merged_data.replace([np.inf, -np.inf, None], np.nan).dropna(subset=[var, 'Trend (days/decade)', 'Mann-Kendall p-value'])
                if len(valid_data) >= 2:
                    correlation, p_value = calculate_correlations(valid_data, var, 'Trend (days/decade)')
                    all_correlations.append({
                        'metric': metric,
                        'metric_label': timing_metrics[metric],
                        'var': var,
                        'var_label': var_label,
                        'correlation': correlation,
                        'p_value': p_value,
                        'data': valid_data,
                        'type': 'catchment'
                    })
            except Exception as e:
                print(f"Error calculating correlation for {metric} vs {var}: {str(e)}")
        
        # 2. Correlate with meteorological trends
        for var in meteo_vars:
            try:
                # Merge trend data with meteorological trends
                merged_data = trend_data[metric].merge(
                    catchments_chara[[var]], 
                    left_index=True, 
                    right_index=True
                )
                
                # Convert columns to numeric, coercing errors to NaN
                merged_data[var] = pd.to_numeric(merged_data[var], errors='coerce')
                merged_data['Trend (days/decade)'] = pd.to_numeric(merged_data['Trend (days/decade)'], errors='coerce')
                merged_data['Mann-Kendall p-value'] = pd.to_numeric(merged_data['Mann-Kendall p-value'], errors='coerce')
                
                # Calculate correlation
                valid_data = merged_data.replace([np.inf, -np.inf, None], np.nan).dropna(subset=[var, 'Trend (days/decade)', 'Mann-Kendall p-value'])
                if len(valid_data) >= 2:
                    correlation, p_value = calculate_correlations(valid_data, var, 'Trend (days/decade)')
                    all_correlations.append({
                        'metric': metric,
                        'metric_label': timing_metrics[metric],
                        'var': var,
                        'var_label': meteo_var_labels[var],
                        'correlation': correlation,
                        'p_value': p_value,
                        'data': valid_data,
                        'type': 'meteo'
                    })
            except Exception as e:
                print(f"Error calculating correlation for {metric} vs {var}: {str(e)}")
    
    # Filter for significant correlations
    significant_correlations = [c for c in all_correlations if c['p_value'] < 0.05]
    
    if not significant_correlations:
        print(f"\nNo significant correlations found for period {start_year}-{end_year}")
        return
    
    # Create a figure for significant correlations
    n_sig = len(significant_correlations)
    if not significant_correlations:
        print(f"\nNo significant correlations found for period {start_year}-{end_year}")
        return
    
    n_cols = min(3, n_sig)  # Maximum 3 columns
    n_rows = (n_sig + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'Significant Correlations ({start_year}-{end_year})', fontsize=16, y=0.95)
    
    # Convert axes to 1D array if necessary
    if n_sig > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each significant correlation
    for i, corr in enumerate(significant_correlations):
        ax = axes[i]
        data = corr['data']
        
        try:
            # Create base scatter plot
            sns.scatterplot(data=data, x=corr['var'], y='Trend (days/decade)', ax=ax)
            
            # Add trend line
            try:
                z = np.polyfit(data[corr['var']], data['Trend (days/decade)'], 1)
                p = np.poly1d(z)
                x_range = data[corr['var']]
                ax.plot(x_range, p(x_range), "r--", alpha=0.8)
            except np.linalg.LinAlgError:
                print(f"Warning: Could not fit trend line for {corr['metric_label']} vs {corr['var_label']}")
            
            # Add correlation info
            text = f'r = {corr["correlation"]:.2f}\np = {corr["p_value"]:.3f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add circles around significant Mann-Kendall trends
            try:
                mk_significant = data['Mann-Kendall p-value'].astype(float) < 0.05
                significant_mk = data[mk_significant]
                
                if len(significant_mk) > 0:
                    ax.scatter(
                        significant_mk[corr['var']],
                        significant_mk['Trend (days/decade)'],
                        facecolors='none',
                        edgecolors='black',
                        s=100,
                        label='Significant MK trend\n(p < 0.05)',
                        zorder=10  # Ensure circles are drawn on top
                    )
                    ax.legend(loc='best', framealpha=0.9)
            except Exception as e:
                print(f"Warning: Could not highlight significant MK trends for {corr['metric_label']} vs {corr['var_label']}: {str(e)}")
            
            # Add type indicator to title
            type_label = '(Meteo)' if corr['type'] == 'meteo' else '(Catchment)'
            ax.set_title(f"{corr['metric_label']} vs\n{corr['var_label']}\n{type_label}")
            ax.set_xlabel(corr['var_label'])
            ax.set_ylabel('Trend (days/decade)')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting {corr['metric_label']} vs {corr['var_label']}: {str(e)}")
            ax.text(0.5, 0.5, 'Error in plotting', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{corr['metric_label']} vs\n{corr['var_label']}")
            ax.set_xlabel(corr['var_label'])
            ax.set_ylabel('Trend (days/decade)')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save the significant correlations plot
    save_path = plot_folder / f'significant_correlations_{start_year}_{end_year}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path = plot_folder / f'significant_correlations_{start_year}_{end_year}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Print summary of significant correlations
    print(f"\nSignificant correlations for period {start_year}-{end_year}:")
    # Group by type and metric
    for corr_type in ['catchment', 'meteo']:
        type_correlations = [c for c in significant_correlations if c['type'] == corr_type]
        if type_correlations:
            print(f"\n{corr_type.title()} Correlations:")
            for metric in timing_metrics.values():
                metric_correlations = [c for c in type_correlations if c['metric_label'] == metric]
                if metric_correlations:
                    print(f"\n{metric}:")
                    for corr in metric_correlations:
                        print(f"- {corr['var_label']} (r = {corr['correlation']:.2f}, p = {corr['p_value']:.3f})")
                        data = corr['data']
                        mk_significant = data['Mann-Kendall p-value'].astype(float) < 0.05
                        significant_mk = data[mk_significant]
                        print(f"  Number of significant MK trends: {len(significant_mk)}")
                        if len(significant_mk) > 0:
                            print(f"  Stations with significant MK trends: {', '.join(map(str, significant_mk.index.tolist()))}")
                            print(f"  Their MK p-values: {', '.join(map(str, significant_mk['Mann-Kendall p-value'].tolist()))}")

# Create analysis plots for each time period
for start_year, end_year in time_periods:
    print(f"\nAnalyzing period {start_year}-{end_year}")
    period_str = f'{start_year}-{end_year}'
    # Filter catchment_vars to only include meteorological trend columns for this period
    period_catchment_vars = {k: v for k, v in catchment_vars.items() if (not k.startswith('trend_') or k.endswith(f'{start_year}_{end_year}'))}
    
    # Read all trend data for this period
    trend_data = {}
    for metric in timing_metrics:
        trend_data[metric] = read_trend_file(start_year, end_year, metric)
    
    # Create figure with subplots for each catchment characteristic
    n_chars = len(period_catchment_vars)
    n_metrics = len(timing_metrics)
    fig, axes = plt.subplots(n_chars, n_metrics, figsize=(15, 4*n_chars))
    fig.suptitle(f'Timing Trends vs Catchment Characteristics ({start_year}-{end_year})', 
                 fontsize=16, y=0.95)
    
    # For each catchment characteristic
    for i, (var, var_label) in enumerate(period_catchment_vars.items()):
        # For each timing metric
        for j, (metric, metric_label) in enumerate(timing_metrics.items()):
            # Merge trend data with catchment characteristics
            merged_data = trend_data[metric].merge(
                catchments_chara[[var]], 
                left_index=True, 
                right_index=True
            )
            
            # Create scatter plot
            plot_relationship(
                merged_data,
                var,
                'Trend (days/decade)',
                axes[i, j],
                f'{metric_label} vs {var_label}',
                var_label,
                'Trend (days/decade)'
            )
            
            # Highlight significant trends
            try:
                merged_data['Mann-Kendall p-value'] = pd.to_numeric(merged_data['Mann-Kendall p-value'], errors='coerce')
                merged_data[var] = pd.to_numeric(merged_data[var], errors='coerce')
                merged_data['Trend (days/decade)'] = pd.to_numeric(merged_data['Trend (days/decade)'], errors='coerce')
                significant = merged_data[
                    (merged_data['Mann-Kendall p-value'] < 0.05) & 
                    (merged_data[var].notna()) & 
                    (merged_data['Trend (days/decade)'].notna())
                ]
                if not significant.empty:
                    axes[i, j].scatter(
                        significant[var],
                        significant['Trend (days/decade)'],
                        facecolors='none',
                        edgecolors='black',
                        s=100,
                        label='Significant (p < 0.05)'
                    )
                    axes[i, j].legend()
            except Exception as e:
                print(f"Warning: Could not highlight significant trends for {metric_label} vs {var_label}: {str(e)}")
    
    plt.tight_layout()
    
    # Save plots
    plot_folder = figures_path / 'catchment_characteristics_analysis'
    plot_folder.mkdir(parents=True, exist_ok=True)
    
    save_path = plot_folder / f'timing_trends_vs_catchment_chars_{start_year}_{end_year}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Create plots for significant correlations
    plot_significant_correlations(trend_data, catchments_chara, start_year, end_year, plot_folder)

# Create summary tables of correlations
summary_tables = {}
detailed_summary = []

for start_year, end_year in time_periods:
    period_str = f'{start_year}-{end_year}'
    period_catchment_vars = {k: v for k, v in catchment_vars.items() if (not k.startswith('trend_') or k.endswith(f'{start_year}_{end_year}'))}
    period_correlations = []
    for metric in timing_metrics:
        trend_df = read_trend_file(start_year, end_year, metric)
        for var, var_label in period_catchment_vars.items():
            # Merge and calculate correlation
            merged = trend_df.merge(catchments_chara[[var]], left_index=True, right_index=True)
            merged[var] = pd.to_numeric(merged[var], errors='coerce')
            merged['Trend (days/decade)'] = pd.to_numeric(merged['Trend (days/decade)'], errors='coerce')
            merged['Mann-Kendall p-value'] = pd.to_numeric(merged['Mann-Kendall p-value'], errors='coerce')
            valid_data = merged.dropna(subset=[var, 'Trend (days/decade)', 'Mann-Kendall p-value'])
            if len(valid_data) >= 2:
                correlation, p_value = calculate_correlations(valid_data, var, 'Trend (days/decade)')
                if p_value < 0.05:
                    mk_significant = valid_data[valid_data['Mann-Kendall p-value'] < 0.05]
                    significant_stations = mk_significant.index.tolist()
                    period_correlations.append({
                        'Period': period_str,
                        'Timing Metric': timing_metrics[metric],
                        'Catchment Variable': var_label,
                        'Variable Code': var,
                        'Correlation': correlation,
                        'P-value': p_value,
                        'Number of Stations': len(valid_data),
                        'Number of Significant MK Trends': len(mk_significant),
                        'Significant Stations': ', '.join(map(str, significant_stations)) if significant_stations else 'None'
                    })
    if period_correlations:
        summary_tables[period_str] = pd.DataFrame(period_correlations)

# Create detailed summary DataFrame
if detailed_summary:
    detailed_summary_df = pd.DataFrame(detailed_summary)
    # Sort by period, timing metric, and p-value
    detailed_summary_df = detailed_summary_df.sort_values(['Period', 'Timing Metric', 'P-value'])
else:
    detailed_summary_df = pd.DataFrame()

# Save correlation summary tables (only significant correlations)
descriptions = {
    'Period': 'Time period of the analysis (e.g., 1973-2023)',
    'Timing Metric': 'Type of timing analysis (Spring Freshet, Centroid of Timing, or Peak Flow)',
    'Catchment Variable': 'Name of the catchment characteristic',
    'Variable Code': 'Short code for the catchment variable',
    'Correlation': 'Pearson correlation coefficient between the timing trend and catchment variable',
    'P-value': 'Statistical significance of the correlation (values < 0.05 indicate significant correlation)',
    'Number of Stations': 'Total number of stations used in the correlation analysis',
    'Number of Significant MK Trends': 'Number of stations showing significant Mann-Kendall trends',
    'Significant Stations': 'List of station IDs showing significant Mann-Kendall trends'
}
for period, table in summary_tables.items():
    # Filter for significant correlations only
    sig_table = table[table['P-value'] < 0.05].copy()
    if not sig_table.empty:
        # Save as CSV
        save_path = plot_folder / f'correlation_summary_{period}.csv'
        sig_table.to_csv(save_path, index=False)

        # Save as Excel with column descriptions
        excel_path = plot_folder / f'significant_correlations_{period}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame({
                'Column': list(descriptions.keys()),
                'Description': list(descriptions.values())
            }).to_excel(writer, index=False, sheet_name='Column Descriptions')
            sig_table.to_excel(writer, index=False, sheet_name='Significant Correlations')
    else:
        print(f"No significant correlations for period {period}.")

# Save comprehensive summary for all periods
if not detailed_summary_df.empty:
    detailed_summary_path = plot_folder / 'significant_correlations_summary.xlsx'
    with pd.ExcelWriter(detailed_summary_path, engine='openpyxl') as writer:
        pd.DataFrame({
            'Column': list(descriptions.keys()),
            'Description': list(descriptions.values())
        }).to_excel(writer, index=False, sheet_name='Column Descriptions')
        detailed_summary_df.to_excel(writer, index=False, sheet_name='Significant Correlations')
        summary_stats = detailed_summary_df.groupby(['Period', 'Timing Metric']).agg({
            'Catchment Variable': 'count',
            'Number of Significant MK Trends': 'sum'
        }).rename(columns={'Catchment Variable': 'Number of Significant Correlations'})
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    print(f"\nDetailed summary of significant correlations has been saved to:\n- {detailed_summary_path}")
else:
    print("\nNo significant correlations found in any period. No summary file created.")

# Print a summary of significant correlations for each period
for period, table in summary_tables.items():
    print(f"\nSignificant correlations for period {period}:")
    print("=" * 80)
    for metric in timing_metrics.values():
        metric_data = table[table['Timing Metric'] == metric]
        if not metric_data.empty:
            print(f"\n{metric}:")
            for _, row in metric_data.iterrows():
                print(f"- {row['Catchment Variable']} (r = {row['Correlation']:.2f}, p = {row['P-value']:.3f})")
                print(f"  Significant stations: {row['Significant Stations']}")

# --- Additional: Paper Figure for Selected Attributes ---
# Define selected attributes for each period
selected_attrs = {
    '1973-2023': [
        'elev_std', 'elev_mean',
        'ndvi_max', 'ndvi_min', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
        'glac_fra', 'baseflow_index_ladson', 'g_mean_el'
    ],
    '1993-2023': [
        'glac_fra', 'g_slope', 'baseflow_index_ladson', 'g_slopel20'
    ]
}

for period, table in summary_tables.items():
    # Filter for significant correlations only (should already be filtered, but double-check)
    sig_table = table[table['P-value'] < 0.05].copy()
    # Get the selected attributes for this period
    attrs = selected_attrs.get(period, [])
    paperfig_table = sig_table[sig_table['Variable Code'].isin(attrs)]
    if not paperfig_table.empty:
        n = len(paperfig_table)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        # Improved title and position
        fig.suptitle(f'Significant correlations between catchment attribute and trend in timing metric, {period}', fontsize=16, y=1.02)
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, (_, row) in enumerate(paperfig_table.iterrows()):
            ax = axes[i]
            # Get the data for this variable and metric
            metric = row['Timing Metric']
            var = row['Variable Code']
            var_label = row['Catchment Variable']
            # Find the corresponding trend data
            start_year, end_year = map(int, period.split('-'))
            trend_df = read_trend_file(start_year, end_year, [k for k,v in timing_metrics.items() if v==metric][0])
            merged = trend_df.merge(catchments_chara[[var]], left_index=True, right_index=True)
            merged[var] = pd.to_numeric(merged[var], errors='coerce')
            merged['Trend (days/decade)'] = pd.to_numeric(merged['Trend (days/decade)'], errors='coerce')
            merged['Mann-Kendall p-value'] = pd.to_numeric(merged['Mann-Kendall p-value'], errors='coerce')
            valid_data = merged.dropna(subset=[var, 'Trend (days/decade)', 'Mann-Kendall p-value'])
            # Plot
            plot_relationship(
                valid_data,
                var,
                'Trend (days/decade)',
                ax,
                f'{metric} vs {var_label}',
                var_label,
                'Trend (days/decade)'
            )
            # Highlight significant trends
            significant = valid_data[valid_data['Mann-Kendall p-value'] < 0.05]
            if not significant.empty:
                ax.scatter(
                    significant[var],
                    significant['Trend (days/decade)'],
                    facecolors='none',
                    edgecolors='black',
                    s=100,
                    label='Significant (p < 0.05)'
                )
                ax.legend()
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        # Save
        save_path_png = plot_folder / f'significant_correlations_paperfig_{period}.png'
        save_path_pdf = plot_folder / f'significant_correlations_paperfig_{period}.pdf'
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Paper figure for {period} saved as:\n- {save_path_png}\n- {save_path_pdf}")
    else:
        print(f"No significant correlations for selected attributes in period {period}.") 