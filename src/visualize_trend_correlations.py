"""
Script for visualizing trend correlations in streamflow dynamics across Iceland.

This script is part of the analysis for the paper:
"Understanding Changes in Iceland's Streamflow Dynamics in Response to Climate Change"

The script creates correlation heatmaps and spatial maps showing relationships between:
- Streamflow trends (annual and seasonal)
- Catchment characteristics
- Meteorological trends
- Glacier attributes

It produces separate analyses for:
- All rivers
- Glacial rivers (glacier fraction >= 0.1)
- Non-glacial rivers (glacier fraction < 0.1)

The visualizations include:
1. Correlation heatmaps showing significant relationships (p<0.05)
2. Spatial maps showing correlation values across Iceland
3. Combined plots showing both annual and seasonal patterns

Dependencies:
    pandas
    numpy
    matplotlib
    seaborn
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
from pathlib import Path
import geopandas as gpd
import os
from config import (
    OUTPUT_DIR,
    PERIOD,
    CATCHMENT_ATTRIBUTES_FILE,
    ICELAND_SHAPEFILE,
    GLACIER_OUTLINES
)

# --- CONFIG ---
output_folder = OUTPUT_DIR / f'{PERIOD}/correlations'
split_output_folder = output_folder / 'split_glac_and_non_glac_rivers'
glac_output_folder = split_output_folder / 'correlations_glac_rivers'
non_glac_output_folder = split_output_folder / 'correlations_non_glac_rivers'

# Create output directories
output_folder.mkdir(parents=True, exist_ok=True)
split_output_folder.mkdir(parents=True, exist_ok=True)
glac_output_folder.mkdir(parents=True, exist_ok=True)
non_glac_output_folder.mkdir(parents=True, exist_ok=True)

# Create seasonal output directories
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    (output_folder / f'seasonal_{season}').mkdir(parents=True, exist_ok=True)
    (glac_output_folder / f'seasonal_{season}').mkdir(parents=True, exist_ok=True)
    (non_glac_output_folder / f'seasonal_{season}').mkdir(parents=True, exist_ok=True)

# Read catchment characteristics for glacier count
catchments = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)

# Get the actual count of glacierized catchments (g_frac > 0.1)
n_glacier_catchments = len(catchments[catchments['g_frac'] > 0.1])

# Read Iceland shapefile and glacier outlines
iceland_map = gpd.read_file(ICELAND_SHAPEFILE)
glaciers = gpd.read_file(GLACIER_OUTLINES)

def plot_figs(basemap, glaciers, ax, iceland_shapefile_color='gray', glaciers_color='white'):
    """
    Plot Iceland map with glaciers as a base layer for correlation maps.
    
    Parameters
    ----------
    basemap : GeoDataFrame
        GeoDataFrame containing Iceland's base map geometry
    glaciers : GeoDataFrame
        GeoDataFrame containing glacier outlines
    ax : matplotlib.axes.Axes
        Axes object to plot on
    iceland_shapefile_color : str, optional
        Color for Iceland's base map, by default 'gray'
    glaciers_color : str, optional
        Color for glacier outlines, by default 'white'
    """
    # Plot Iceland basemap
    basemap.plot(ax=ax, color=iceland_shapefile_color, edgecolor='darkgray')
    
    # Plot glaciers
    if glaciers is not None:
        glaciers.plot(ax=ax, color=glaciers_color, alpha=0.5)

def plot_correlations_on_map(catchments, correlations, metric, output_folder, title_prefix=""):
    """
    Create spatial maps showing correlation values across Iceland for a specific metric.
    
    Parameters
    ----------
    catchments : GeoDataFrame
        GeoDataFrame containing catchment geometries and attributes
    correlations : DataFrame
        DataFrame containing correlation results
    metric : str
        Name of the metric to plot correlations for
    output_folder : Path
        Directory to save output plots
    title_prefix : str, optional
        Prefix to add to plot titles, by default ""
    """
    # Merge correlations with catchment geometries
    catchments_with_corr = catchments.copy()
    metric_correlations = correlations[correlations['metric'] == metric]
    
    # Create a dictionary of correlations for each variable
    corr_dict = {}
    for _, row in metric_correlations.iterrows():
        corr_dict[row['var']] = row['corr']
    
    # Add correlation values to catchments
    for var in corr_dict.keys():
        # Create a new column for correlations without trying to access the variable column
        catchments_with_corr[f'corr_{var}'] = corr_dict[var]
    
    # Plot a map for each correlation
    for var in corr_dict.keys():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot base map
        plot_figs(iceland_map, glaciers, ax)
        
        # Plot catchments colored by correlation value
        catchments_with_corr.plot(
            column=f'corr_{var}',
            ax=ax,
            legend=True,
            legend_kwds={'label': 'Correlation (r)'},
            cmap='RdBu',
            vmin=-1,
            vmax=1,
            missing_kwds={'color': 'lightgray'}
        )
        
        plt.title(f'{title_prefix}Correlation between {metric} and {var}')
        plt.axis('off')
        
        # Save plot
        plt.savefig(output_folder / f'map_correlation_{metric}_{var}.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_folder / f'map_correlation_{metric}_{var}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

# Define trend_metrics
trend_metrics = {
    # Annual metrics
    'annual_avg_flow_trend_per_decade': 'Trend in mean flow',
    'trend_annual_cv_per_decade': 'Trend in CV',
    'trend_annual_std_per_decade': 'Trend in st. dev.',
    'trend_flashiness_per_decade': 'Trend in flashiness',
    'trend_baseflow_index_per_decade': 'Trend in BFI'
}

# Add seasonal metrics with flattened directory structure
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    trend_metrics.update({
        f'trend_{season}_per_decade': 'Trend in mean flow',
        f'std_{season}_trend_per_decade': 'Trend in st. dev.',
        f'cv_{season}_trend_per_decade': 'Trend in CV',
        f'flashiness_{season}_trend_per_decade': 'Trend in flashiness',
        f'baseflow_index_{season}_trend_per_decade': 'Trend in BFI'
    })

def create_correlation_plots(summary_file, output_folder, title_prefix=""):
    """
    Create correlation heatmaps and maps from trend analysis results.
    
    This function processes correlation results and creates:
    1. Annual correlation heatmaps
    2. Seasonal correlation heatmaps (DJF, MAM, JJA, SON)
    3. Spatial maps showing correlations across Iceland
    4. Combined plots showing both annual and seasonal patterns
    
    Parameters
    ----------
    summary_file : str or Path
        Path to the CSV file containing correlation results
    output_folder : str or Path
        Directory to save output plots
    title_prefix : str, optional
        Prefix to add to plot titles, by default ""
        
    Notes
    -----
    The function handles both catchment characteristics and meteorological trends,
    and creates separate visualizations for significant correlations (p<0.05).
    """
    print(f"\n=== Loading correlation results ===")
    print(f"From file: {summary_file}")
    
    # Check if the file exists
    if not os.path.exists(summary_file):
        print(f"Error: Correlation results file not found: {summary_file}")
        print("Please run trend_correlation_analysis.py first to generate the correlation results.")
        return
    
    global_df = pd.read_csv(summary_file)
    print("\nCorrelation results DataFrame:")
    print(global_df.head())
    print("\nColumns:", global_df.columns.tolist())
    print("\nShape:", global_df.shape)
    
    # Print unique variable names
    print("\nUnique variable names in correlation results:")
    print(sorted(global_df['var'].unique().tolist()))
    print("\nUnique metric names in correlation results:")
    print(sorted(global_df['metric'].unique().tolist()))

    # --- FILTER: Only annual metrics, only ndvi_max for vegetation, omit seasonal and meteo trends ---
    annual_metrics = [
        'annual_avg_flow_trend_per_decade',
        'trend_annual_cv_per_decade',
        'trend_annual_std_per_decade',
        'trend_flashiness_per_decade',
        'trend_baseflow_index_per_decade'
    ]
    metric_labels = {
        'annual_avg_flow_trend_per_decade': 'Trend in mean flow',
        'trend_annual_cv_per_decade': 'Trend in CV',
        'trend_annual_std_per_decade': 'Trend in st. dev.',
        'trend_flashiness_per_decade': 'Trend in flashiness',
        'trend_baseflow_index_per_decade': 'Trend in BFI'
    }

    # Only use ndvi_max for vegetation
    catchment_vars = [
        'area_calc', 'elev_mean', 'elev_std', 'elev_ran', 'slope_mean', 'asp_mean', 'elon_ratio',
        'strm_dens', 'p_mean', 'aridity', 'frac_snow', 'p_season', 'bare_fra', 'forest_fra',
        'lake_fra', 'agr_fra', 'sand_fra', 'silt_fra', 'clay_fra',
        'oc_fra', 'root_dep', 'soil_tawc', 'soil_poros', 'bedrk_dep', 'ndvi_max', 'q_mean', 'runoff_ratio',
        'baseflow_index_ladson', 'g_frac', 'g_lat', 'g_lon', 'g_mean_el', 'g_min_el', 'g_slope', 'g_slopel20', 'glac_fra',  #'hfd_mean', 'slope_fdc', 'Q5', 'Q95',
        # Add annual meteorological trends
        f'trend_prec_{PERIOD}', f'trend_rainfall_{PERIOD}', f'trend_snowfall_{PERIOD}', f'trend_total_et_{PERIOD}', f'trend_2m_temp_mean_{PERIOD}'
    ]

    # Define glacier-related attributes
    glacier_attributes = ['g_frac', 'g_lat', 'g_lon', 'g_mean_el', 'g_min_el', 'g_slope', 'g_slopel20', 'glac_fra']

    # Filter global_df
    filtered = global_df.copy()
    print("\nBefore filtering - unique variables:", sorted(filtered['var'].unique().tolist()))
    
    # For glacier attributes, we'll keep all correlations as they should already be calculated only for glacierized catchments
    filtered = filtered[
        (filtered['metric'].isin(annual_metrics)) &
        (filtered['var'].isin(catchment_vars)) &
        (filtered['type'].isin(['catchment', 'meteo']))
    ]
    
    print("\nAfter filtering - unique variables:", sorted(filtered['var'].unique().tolist()))
    
    # Check for duplicates
    print("\nChecking for duplicate metric-var pairs:")
    duplicates = filtered.groupby(['metric', 'var']).size().reset_index(name='count')
    duplicates = duplicates[duplicates['count'] > 1]
    if not duplicates.empty:
        print("Found duplicate pairs:")
        print(duplicates)
        print("\nTaking first occurrence of each metric-var pair")
        filtered = filtered.drop_duplicates(subset=['metric', 'var'], keep='first')
    else:
        print("No duplicate pairs found")

    # Get n-values for glacier attributes from this season's data
    glacier_n_values = {}
    print("\nChecking glacier attributes in filtered data:")
    for var in glacier_attributes:
        var_data = filtered[filtered['var'] == var]
        print(f"\nVariable: {var}")
        print("Data found:")
        print(var_data[['var', 'n', 'corr', 'pval']] if not var_data.empty else "No data")
        glacier_n_values[var] = var_data['n'].iloc[0] if not var_data.empty else 0

    # Update column_plot_titles_dict to include shorter but descriptive names
    column_plot_titles_dict = {
        # Meteorological Trends
        f'trend_prec_{PERIOD}': 'Trend in precip.',
        f'trend_rainfall_{PERIOD}': 'Trend in rainf.',
        f'trend_snowfall_{PERIOD}': 'Trend in snowf.',
        f'trend_total_et_{PERIOD}': 'Trend in ET',
        f'trend_2m_temp_mean_{PERIOD}': 'Trend in temp.',
        # Topography & Geometry
        'area_calc': 'Area',
        'elev_mean': 'Mean elev.',
        'elev_std': 'Elev. std',
        'elev_ran': 'Elev. range',
        'slope_mean': 'Mean slope',
        'asp_mean': 'Aspect',
        'elon_ratio': 'Elong. ratio',
        'strm_dens': 'Stream dens.',
        # Climate & Hydrometeorology
        'p_mean': 'Mean precip.',
        'aridity': 'Aridity',
        'frac_snow': 'Snow frac.',
        'p_season': 'Precip. seas.',
        # Land Cover & Land Use Fractions
        'bare_fra': 'Bare frac.',
        'forest_fra': 'Forest frac.',
        'lake_fra': 'Lake frac.',
        'agr_fra': 'Agri. frac.',
        # Soil Texture & Organic Matter
        'sand_fra': 'Sand frac.',
        'silt_fra': 'Silt frac.',
        'clay_fra': 'Clay frac.',
        'oc_fra': 'Org. C frac.',
        # Soil & Root Zone Properties
        'root_dep': 'Root depth',
        'soil_tawc': 'Water cap.',
        'soil_poros': 'Porosity',
        'bedrk_dep': 'Bedrock depth',
        # Vegetation Index
        'ndvi_max': 'Max NDVI',
        # Hydrological Signatures
        'q_mean': 'Mean flow',
        'runoff_ratio': 'Runoff ratio',
        'baseflow_index_ladson': 'BFI',
        'hfd_mean': 'Half-flow date',
        'slope_fdc': 'FDC slope',
        'Q5': 'Q5',
        'Q95': 'Q95',
        # Glacier Attributes
        'g_frac': f'Glac. frac. (n={glacier_n_values.get("g_frac", 0)})',
        'g_lat': f'Glac. lat. (n={glacier_n_values.get("g_lat", 0)})',
        'g_lon': f'Glac. lon. (n={glacier_n_values.get("g_lon", 0)})',
        'g_mean_el': f'Glac. mean elev. (n={glacier_n_values.get("g_mean_el", 0)})',
        'g_min_el': f'Glac. min elev. (n={glacier_n_values.get("g_min_el", 0)})',
        'g_slope': f'Glac. slope (n={glacier_n_values.get("g_slope", 0)})',
        'g_slopel20': f'Glac. low slope (n={glacier_n_values.get("g_slopel20", 0)})',
        'glac_fra': f'Glac. frac. (n={glacier_n_values.get("glac_fra", 0)})',
    }

    # Add seasonal meteorological trends with shorter names
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        column_plot_titles_dict.update({
            f'trend_prec_{season}_{PERIOD}': f'Trend in {season} precip.',
            f'trend_rainfall_{season}_{PERIOD}': f'Trend in {season} rainf.',
            f'trend_snowfall_{season}_{PERIOD}': f'Trend in {season} snowf.',
            f'trend_total_et_{season}_{PERIOD}': f'Trend in {season} ET',
            f'trend_2m_temp_mean_{season}_{PERIOD}': f'Trend in {season} temp.'
        })

    # Define ordered_catchment_vars by type
    ordered_catchment_vars = [
        # Meteorological Trends (will be filtered per season)
        f'trend_prec_{PERIOD}', f'trend_rainfall_{PERIOD}', f'trend_snowfall_{PERIOD}', f'trend_total_et_{PERIOD}', f'trend_2m_temp_mean_{PERIOD}',
        f'trend_prec_DJF_{PERIOD}', f'trend_prec_MAM_{PERIOD}', f'trend_prec_JJA_{PERIOD}', f'trend_prec_SON_{PERIOD}',
        f'trend_rainfall_DJF_{PERIOD}', f'trend_rainfall_MAM_{PERIOD}', f'trend_rainfall_JJA_{PERIOD}', f'trend_rainfall_SON_{PERIOD}',
        f'trend_snowfall_DJF_{PERIOD}', f'trend_snowfall_MAM_{PERIOD}', f'trend_snowfall_JJA_{PERIOD}', f'trend_snowfall_SON_{PERIOD}',
        f'trend_total_et_DJF_{PERIOD}', f'trend_total_et_MAM_{PERIOD}', f'trend_total_et_JJA_{PERIOD}', f'trend_total_et_SON_{PERIOD}',
        f'trend_2m_temp_mean_DJF_{PERIOD}', f'trend_2m_temp_mean_MAM_{PERIOD}', f'trend_2m_temp_mean_JJA_{PERIOD}', f'trend_2m_temp_mean_SON_{PERIOD}',
        # Topography & Geometry
        'area_calc', 'elev_mean', 'elev_std', 'elev_ran', 'slope_mean', 'asp_mean', 'elon_ratio', 'strm_dens',
        # Climate & Hydrometeorology
        'p_mean', 'aridity', 'frac_snow', 'p_season',
        # Land Cover & Land Use Fractions
        'bare_fra', 'forest_fra', 'lake_fra', 'agr_fra',
        # Soil Texture & Organic Matter
        'sand_fra', 'silt_fra', 'clay_fra', 'oc_fra',
        # Soil & Root Zone Properties
        'root_dep', 'soil_tawc', 'soil_poros', 'bedrk_dep',
        # Vegetation Index
        'ndvi_max',
        # Hydrological Signatures
        'q_mean', 'runoff_ratio', 'baseflow_index_ladson', 'hfd_mean', 'slope_fdc',
        'high_q_dur', 'low_q_dur', 'high_q_freq', 'low_q_freq', 'Q5', 'Q95',
        # Glacier Attributes
        'g_area', 'g_mean_el', 'g_max_el', 'g_min_el', 'g_slope', 'g_slopel20', 'g_dom_NI', 'glac_fra'
    ]

    # Update column titles with correct n-values for this season
    column_plot_titles = column_plot_titles_dict.copy()
    for var in glacier_attributes:
        base_label = column_plot_titles_dict[var].split(' (n=')[0]  # Get the label without the n-value
        column_plot_titles[var] = f"{base_label} (n={glacier_n_values.get(var, 0)})"

    # Pivot to matrix: rows=metric, columns=var, values=corr
    heatmap_data = filtered.pivot(index='metric', columns='var', values='corr')
    heatmap_data = heatmap_data.reindex(index=annual_metrics, columns=catchment_vars)
    heatmap_data.index = [metric_labels.get(m, m) for m in heatmap_data.index]
    heatmap_data = heatmap_data.dropna(axis=1, how='all')

    # Order columns using ordered_catchment_vars, but only keep seasonal meteorological trends for this season
    ordered_cols = [col for col in ordered_catchment_vars if col in heatmap_data.columns]
    heatmap_data = heatmap_data[ordered_cols]
    col_labels = [column_plot_titles_dict.get(col, col) for col in heatmap_data.columns]

    # --- PLOT HEATMAPS ---
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    n_seasons = len(seasons)

    # First save individual annual heatmap
    plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # Transpose the data for annual heatmap
    heatmap_data_rotated = heatmap_data.T
    
    ax = sns.heatmap(
        heatmap_data_rotated,
        annot=True,
        fmt='.2f',
        cmap='RdBu',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation (r)'},
        annot_kws={'size': 12}
    )
    plt.title(f'{title_prefix}Significant correlations (p<0.05)\nbetween annual streamflow trends and catchment characteristics', pad=20)
    plt.xlabel('Streamflow trend metric', labelpad=15)
    plt.ylabel('Meteorological trends and catchment characteristics', labelpad=15)
    
    # Get the tick labels from the heatmap
    y_labels = [column_plot_titles_dict.get(label.get_text(), label.get_text()) 
                for label in ax.get_yticklabels()]
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Set the labels
    ax.set_yticklabels(y_labels, rotation=0, ha='right')
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add vertical and horizontal dashed lines
    for i in range(1, len(heatmap_data_rotated.columns)):
        ax.axvline(i, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    for j in range(1, len(heatmap_data_rotated.index)):
        ax.axhline(j, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_folder / 'annual_trend_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'annual_trend_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Save individual seasonal heatmaps
    for season in seasons:
        # Filter for seasonal metrics using trend_metrics
        seasonal_metrics = [m for m in trend_metrics.keys() if season in m]
        if not seasonal_metrics:
            continue
        
        # Create seasonal metric labels
        metric_labels_seasonal = {
            f'trend_{season}_per_decade': 'Trend in mean flow',
            f'std_{season}_trend_per_decade': 'Trend in st. dev.',
            f'cv_{season}_trend_per_decade': 'Trend in CV',
            f'flashiness_{season}_trend_per_decade': 'Trend in flashiness',
            f'baseflow_index_{season}_trend_per_decade': 'Trend in BFI'
        }
        
        # Update catchment_vars to include seasonal meteorological trends for this season
        seasonal_catchment_vars = catchment_vars.copy()
        # Remove annual meteorological trends
        seasonal_catchment_vars = [var for var in seasonal_catchment_vars if not any(x in var for x in [f'_{PERIOD}'])]
        # Add seasonal meteorological trends for this season
        seasonal_catchment_vars.extend([
            f'trend_prec_{season}_{PERIOD}',
            f'trend_rainfall_{season}_{PERIOD}',
            f'trend_snowfall_{season}_{PERIOD}',
            f'trend_total_et_{season}_{PERIOD}',
            f'trend_2m_temp_mean_{season}_{PERIOD}'
        ])
        
        # Filter global_df for seasonal metrics
        filtered_seasonal = global_df[
            (global_df['metric'].isin(seasonal_metrics)) &
            (global_df['var'].isin(seasonal_catchment_vars)) &
            (global_df['type'].isin(['catchment', 'meteo']))
        ]
        
        print(f"\nProcessing {season} metrics")
        print(f"Number of rows before duplicate handling: {len(filtered_seasonal)}")
        print("Metrics:", sorted(filtered_seasonal['metric'].unique().tolist()))
        print("Variables:", sorted(filtered_seasonal['var'].unique().tolist()))
        
        # Check for duplicates in seasonal data
        seasonal_duplicates = filtered_seasonal.groupby(['metric', 'var']).size().reset_index(name='count')
        seasonal_duplicates = seasonal_duplicates[seasonal_duplicates['count'] > 1]
        if not seasonal_duplicates.empty:
            print(f"\nFound duplicate pairs in {season} data:")
            print(seasonal_duplicates)
            print("\nTaking first occurrence of each metric-var pair")
            filtered_seasonal = filtered_seasonal.drop_duplicates(subset=['metric', 'var'], keep='first')
        else:
            print(f"\nNo duplicate pairs found in {season} data")
        
        print(f"Number of rows after duplicate handling: {len(filtered_seasonal)}")
        
        if filtered_seasonal.empty:
            print(f"No data for {season}, skipping...")
            continue
            
        # Get n-values for glacier attributes from this season's data
        seasonal_glacier_n_values = {}
        print(f"\nChecking glacier attributes in {season} data:")
        for var in glacier_attributes:
            var_data = filtered_seasonal[filtered_seasonal['var'] == var]
            print(f"\nVariable: {var}")
            print("Data found:")
            print(var_data[['var', 'n', 'corr', 'pval']] if not var_data.empty else "No data")
            seasonal_glacier_n_values[var] = var_data['n'].iloc[0] if not var_data.empty else 0
            
        # Update column titles with correct n-values for this season
        seasonal_column_plot_titles = column_plot_titles_dict.copy()
        for var in glacier_attributes:
            base_label = column_plot_titles_dict[var].split(' (n=')[0]  # Get the label without the n-value
            seasonal_column_plot_titles[var] = f"{base_label} (n={seasonal_glacier_n_values.get(var, 0)})"
        
        # Pivot to matrix: rows=metric, columns=var, values=corr
        heatmap_data_seasonal = filtered_seasonal.pivot(index='metric', columns='var', values='corr')
        heatmap_data_seasonal = heatmap_data_seasonal.reindex(index=seasonal_metrics, columns=seasonal_catchment_vars)
        heatmap_data_seasonal.index = [metric_labels_seasonal.get(m, m) for m in heatmap_data_seasonal.index]
        heatmap_data_seasonal = heatmap_data_seasonal.dropna(axis=1, how='all')
        
        # Order columns using ordered_catchment_vars, but only keep seasonal meteorological trends for this season
        ordered_cols_seasonal = [col for col in ordered_catchment_vars if 
                               (col in heatmap_data_seasonal.columns and 
                                (season in col if '_1973_2023' in col else True))]
        heatmap_data_seasonal = heatmap_data_seasonal[ordered_cols_seasonal]
        col_labels_seasonal = [seasonal_column_plot_titles.get(col, col) for col in heatmap_data_seasonal.columns]
        
        plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        
        # Transpose the seasonal data
        heatmap_data_seasonal_rotated = heatmap_data_seasonal.T
        
        ax_seasonal = sns.heatmap(
            heatmap_data_seasonal_rotated,
            annot=True,
            fmt='.2f',
            cmap='RdBu',
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation (r)'},
            annot_kws={'size': 12}
        )
        plt.title(f'{title_prefix}Significant correlations (p<0.05)\nbetween {season} streamflow trends and catchment characteristics', pad=20)
        plt.xlabel('Streamflow trend metric', labelpad=15)
        plt.ylabel('Meteorological trends and catchment characteristics', labelpad=15)
        
        # Get the tick labels from the heatmap
        y_labels_seasonal = [seasonal_column_plot_titles.get(label.get_text(), label.get_text()) 
                           for label in ax_seasonal.get_yticklabels()]
        x_labels_seasonal = [label.get_text() for label in ax_seasonal.get_xticklabels()]
        
        # Set the labels
        ax_seasonal.set_yticklabels(y_labels_seasonal, rotation=0, ha='right')
        ax_seasonal.set_xticklabels(x_labels_seasonal, rotation=45, ha='right')
        
        # Add vertical and horizontal dashed lines
        for i in range(1, len(heatmap_data_seasonal_rotated.columns)):
            ax_seasonal.axvline(i, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        for j in range(1, len(heatmap_data_seasonal_rotated.index)):
            ax_seasonal.axhline(j, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_folder / f'{season}_trend_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_folder / f'{season}_trend_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    # Now create combined plot with proper labels
    fig = plt.figure(figsize=(max(15, 1.5 * len(heatmap_data.columns)), 4 * (n_seasons + 1)))
    
    # Plot annual heatmap
    plt.subplot(n_seasons + 1, 1, 1)
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdBu',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation (r)'}
    )
    plt.title(f'a) {title_prefix}Significant correlations (p<0.05) between annual streamflow trends and catchment characteristics')
    plt.ylabel('Streamflow trend metric')
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    plt.xlabel('')  # Remove x-label for all but the last subplot
    
    # Add vertical and horizontal dashed lines
    for i in range(1, len(heatmap_data.columns)):
        ax.axvline(i, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    for j in range(1, len(heatmap_data.index)):
        ax.axhline(j, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    # Plot seasonal heatmaps with letters
    subplot_letters = ['b', 'c', 'd', 'e']
    for idx, (season, letter) in enumerate(zip(seasons, subplot_letters), start=1):
        # Filter for seasonal metrics using trend_metrics
        seasonal_metrics = [m for m in trend_metrics.keys() if season in m]
        if not seasonal_metrics:
            continue
        
        # Create seasonal metric labels
        metric_labels_seasonal = {
            f'trend_{season}_per_decade': 'Trend in mean flow',
            f'std_{season}_trend_per_decade': 'Trend in st. dev.',
            f'cv_{season}_trend_per_decade': 'Trend in CV',
            f'flashiness_{season}_trend_per_decade': 'Trend in flashiness',
            f'baseflow_index_{season}_trend_per_decade': 'Trend in BFI'
        }
        
        # Update catchment_vars to include seasonal meteorological trends for this season
        seasonal_catchment_vars = catchment_vars.copy()
        # Remove annual meteorological trends
        seasonal_catchment_vars = [var for var in seasonal_catchment_vars if not any(x in var for x in [f'_{PERIOD}'])]
        # Add seasonal meteorological trends for this season
        seasonal_catchment_vars.extend([
            f'trend_prec_{season}_{PERIOD}',
            f'trend_rainfall_{season}_{PERIOD}',
            f'trend_snowfall_{season}_{PERIOD}',
            f'trend_total_et_{season}_{PERIOD}',
            f'trend_2m_temp_mean_{season}_{PERIOD}'
        ])
        
        # Filter global_df for seasonal metrics
        filtered_seasonal = global_df[
            (global_df['metric'].isin(seasonal_metrics)) &
            (global_df['var'].isin(seasonal_catchment_vars)) &
            (global_df['type'].isin(['catchment', 'meteo']))
        ]
        
        print(f"\nProcessing {season} metrics")
        print(f"Number of rows before duplicate handling: {len(filtered_seasonal)}")
        print("Metrics:", sorted(filtered_seasonal['metric'].unique().tolist()))
        print("Variables:", sorted(filtered_seasonal['var'].unique().tolist()))
        
        # Check for duplicates in seasonal data
        seasonal_duplicates = filtered_seasonal.groupby(['metric', 'var']).size().reset_index(name='count')
        seasonal_duplicates = seasonal_duplicates[seasonal_duplicates['count'] > 1]
        if not seasonal_duplicates.empty:
            print(f"\nFound duplicate pairs in {season} data:")
            print(seasonal_duplicates)
            print("\nTaking first occurrence of each metric-var pair")
            filtered_seasonal = filtered_seasonal.drop_duplicates(subset=['metric', 'var'], keep='first')
        else:
            print(f"\nNo duplicate pairs found in {season} data")
        
        print(f"Number of rows after duplicate handling: {len(filtered_seasonal)}")
        
        if filtered_seasonal.empty:
            print(f"No data for {season}, skipping...")
            continue
            
        # Get n-values for glacier attributes from this season's data
        seasonal_glacier_n_values = {}
        print(f"\nChecking glacier attributes in {season} data:")
        for var in glacier_attributes:
            var_data = filtered_seasonal[filtered_seasonal['var'] == var]
            print(f"\nVariable: {var}")
            print("Data found:")
            print(var_data[['var', 'n', 'corr', 'pval']] if not var_data.empty else "No data")
            seasonal_glacier_n_values[var] = var_data['n'].iloc[0] if not var_data.empty else 0
            
        # Update column titles with correct n-values for this season
        seasonal_column_plot_titles = column_plot_titles_dict.copy()
        for var in glacier_attributes:
            base_label = column_plot_titles_dict[var].split(' (n=')[0]  # Get the label without the n-value
            seasonal_column_plot_titles[var] = f"{base_label} (n={seasonal_glacier_n_values.get(var, 0)})"
        
        # Pivot to matrix: rows=metric, columns=var, values=corr
        heatmap_data_seasonal = filtered_seasonal.pivot(index='metric', columns='var', values='corr')
        heatmap_data_seasonal = heatmap_data_seasonal.reindex(index=seasonal_metrics, columns=seasonal_catchment_vars)
        heatmap_data_seasonal.index = [metric_labels_seasonal.get(m, m) for m in heatmap_data_seasonal.index]
        heatmap_data_seasonal = heatmap_data_seasonal.dropna(axis=1, how='all')
        
        # Order columns using ordered_catchment_vars, but only keep seasonal meteorological trends for this season
        ordered_cols_seasonal = [col for col in ordered_catchment_vars if 
                               (col in heatmap_data_seasonal.columns and 
                                (season in col if '_1973_2023' in col else True))]
        heatmap_data_seasonal = heatmap_data_seasonal[ordered_cols_seasonal]
        col_labels_seasonal = [seasonal_column_plot_titles.get(col, col) for col in heatmap_data_seasonal.columns]
        
        plt.subplot(n_seasons + 1, 1, idx + 1)
        ax_seasonal = sns.heatmap(
            heatmap_data_seasonal,
            annot=True,
            fmt='.2f',
            cmap='RdBu',
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation (r)'}
        )
        plt.title(f'{letter}) {title_prefix}Significant correlations (p<0.05) between {season} streamflow trends and catchment characteristics')
        plt.ylabel('Streamflow trend metric')
        ax_seasonal.set_xticklabels(col_labels_seasonal, rotation=45, ha='right')
        
        # Only add x-label to the last subplot
        if idx == len(seasons):
            plt.xlabel('Meteorological trends and catchment characteristics')
        else:
            plt.xlabel('')
        
        # Add vertical and horizontal dashed lines
        for i in range(1, len(heatmap_data_seasonal.columns)):
            ax_seasonal.axvline(i, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        for j in range(1, len(heatmap_data_seasonal.index)):
            ax_seasonal.axhline(j, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_folder / 'trend_correlation_heatmaps_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder / 'trend_correlation_heatmaps_combined.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Create map plots for each metric
    for metric in annual_metrics:
        plot_correlations_on_map(catchments, filtered, metric, output_folder, title_prefix)

    # Create seasonal map plots
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        seasonal_metrics = [m for m in trend_metrics.keys() if season in m]
        seasonal_filtered = global_df[
            (global_df['metric'].isin(seasonal_metrics)) &
            (global_df['var'].isin(catchment_vars)) &
            (global_df['type'].isin(['catchment', 'meteo']))
        ]
        
        for metric in seasonal_metrics:
            plot_correlations_on_map(
                catchments, 
                seasonal_filtered, 
                metric, 
                output_folder / f'seasonal_{season}',
                f"{title_prefix}{season}: "
            )

# Run visualization for all rivers (original analysis)
create_correlation_plots(
    output_folder / 'significant_trend_correlations_all.csv',
    output_folder,
    title_prefix=f"{PERIOD[:4]}-{PERIOD[5:]}: "
)

# Run visualization for glacial rivers
create_correlation_plots(
    glac_output_folder / 'significant_trend_correlations_all.csv',
    glac_output_folder,
    title_prefix=f"{PERIOD[:4]}-{PERIOD[5:]}: Glacial Rivers (g_frac >= 0.1): "
)

# Run visualization for non-glacial rivers
create_correlation_plots(
    non_glac_output_folder / 'significant_trend_correlations_all.csv',
    non_glac_output_folder,
    title_prefix=f"{PERIOD[:4]}-{PERIOD[5:]}: Non-Glacial Rivers (g_frac < 0.1): "
) 