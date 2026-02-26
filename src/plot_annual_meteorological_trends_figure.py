"""
Plot annual trends in meteorological variables from ERA5-Land.

Creates a 2x5 subplot figure showing:
- Row 1: Trends for 1973-2023
- Row 2: Trends for 1993-2023
- Columns: Temperature, Precipitation, Rainfall, Snowfall, Evapotranspiration

Author: Hordur Bragi Helgason
Date: 2025
"""

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import pickle
from matplotlib.colors import LinearSegmentedColormap
from config import OUTPUT_DIR, LAMAH_ICE_BASE_PATH, ICELAND_SHAPEFILE, GLACIER_SHAPEFILE

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

def place_newline(string):
    """Insert newline at the halfway point of a string."""
    halfway = len(string) // 2
    if ' ' in string:
        before = string.rfind(' ', 0, halfway)
        after = string.find(' ', halfway)
        if before == -1:
            split_point = after if after != -1 else halfway
        elif after == -1:
            split_point = before
        else:
            split_point = before if halfway - before <= after - halfway else after
    else:
        split_point = halfway
    return string[:split_point] + '\n' + string[split_point:]

def plot_figs(basemap, glaciers, ax, iceland_shapefile_color='gray', glaciers_color='white'):
    """Plot Iceland map with glaciers as base layer."""
    basemap.plot(ax=ax, color=iceland_shapefile_color, edgecolor='darkgray')
    if glaciers is not None:
        glaciers.plot(ax=ax, color=glaciers_color, alpha=0.5)
    ax.set_xlim(222375, 765246)
    ax.set_ylim(307671, 697520)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def load_prec_data_for_et_calc():
    """Load precipitation data for calculating ET as percentage of precipitation."""
    print("  Loading precipitation data from LamaH-Ice...")
    meteo_data_path = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm" / "2_timeseries" / "daily" / "meteorological_data"
    
    # Read gauges shapefile to get gauge IDs
    gauges_shp = LAMAH_ICE_BASE_PATH / "D_gauges" / "3_shapefiles" / "gauges.shp"
    gauges = gpd.read_file(gauges_shp)
    gauges['id'] = gauges['id'].astype(int)
    gauge_ids = gauges['id'].tolist()[:107]
    
    # Load precipitation data for all gauges
    combined_df = pd.DataFrame()
    for gauge_id in gauge_ids:
        file_path = meteo_data_path / f"ID_{gauge_id}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, sep=';')
                df['date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].rename(
                    columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
                ))
                df = df.set_index('date')
                if 'prec' in df.columns:
                    combined_df[gauge_id] = df['prec']
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
    
    print(f"  Loaded precipitation data for {len(combined_df.columns)} catchments")
    return combined_df

def main():
    """Main function to create the meteorological trends figure."""
    
    print("=== Creating Annual Meteorological Trends Figure ===\n")
    
    # Define output path
    output_path = OUTPUT_DIR / 'meteorological_trends_figures'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load pickle files
    print("Loading meteorological trend data...")
    pkl_1973 = OUTPUT_DIR / 'merged_results_dict_1973-2023.pkl'
    pkl_1993 = OUTPUT_DIR / 'merged_results_dict_1993-2023.pkl'
    
    with open(pkl_1973, 'rb') as f:
        merged_results_1973 = pickle.load(f)
    
    with open(pkl_1993, 'rb') as f:
        merged_results_1993 = pickle.load(f)
    
    print("Loaded both period datasets")
    
    # Load basemap and glaciers
    print("Loading Iceland basemap and glacier outlines...")
    bmap = gpd.read_file(ICELAND_SHAPEFILE)
    glaciers = gpd.read_file(GLACIER_SHAPEFILE)
    
    # Load precipitation data for ET calculation
    print("Loading precipitation data for ET as % of precipitation calculation...")
    prec_data = load_prec_data_for_et_calc()
    
    # Calculate ET as percentage of precipitation for both periods
    print("\nCalculating ET as percentage of precipitation...")
    for period_start, period_end, merged_results in [('1973', '2023', merged_results_1973), 
                                                      ('1993', '2023', merged_results_1993)]:
        key = f'total_et_{period_start}-{period_end}'
        if key in merged_results:
            # Calculate average annual precipitation for this period
            start_date = f'{period_start}-10-01'
            end_date = f'{period_end}-09-30'
            prec_period = prec_data[start_date:end_date]
            average_ann_prec = prec_period.resample('Y').sum().mean().sort_index()
            
            # Calculate ET as percentage of precipitation
            # Formula: 100 * (ET trend in mm) / (average annual precip in mm)
            et_as_percentage_of_precip = 100 * (
                merged_results[key]['annual_trend_mm'] / average_ann_prec
            )
            
            # Add this as a new column
            merged_results[key]['et_as_percentage_of_precip'] = et_as_percentage_of_precip
            print(f"  Calculated ET as % of precip for {period_start}-{period_end}")
    
    # Define variables to plot and their settings
    variables = ['2m_temp_mean', 'prec', 'rainfall', 'snowfall', 'total_et']
    
    # Define colormaps
    colormaps = {
        'prec': 'Blues',
        'rainfall': 'Blues',
        'snowfall': 'RdBu',
        'total_et': 'Blues',
        '2m_temp_mean': 'YlOrRd'
    }
    
    # Define units
    unit_dict = {
        'prec': '%/decade',
        'rainfall': '%/decade',
        'snowfall': '%/decade',
        'total_et': '% of precip./decade',
        '2m_temp_mean': '°C/decade'
    }
    
    # Define colorbar titles
    colorbar_titles = {
        'prec': 'precipitation',
        'rainfall': 'rainfall',
        'snowfall': 'snowfall',
        'total_et': 'evapotranspiration',
        '2m_temp_mean': '2m temperature'
    }
    
    # Define vmin, vmax for each variable (same for both periods)
    vmin_vmax = {
        '2m_temp_mean': (0.35, 0.45),
        'prec': (0, 5),
        'rainfall': (0, 10),
        'snowfall': (-5, 5),
        'total_et': (0, 1.1)  # ET as % of precipitation
    }
    
    # Use same ranges for second period
    vmin_vmax_1993 = vmin_vmax.copy()
    
    # Create figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    
    # Subplot labels
    labels_row1 = ['a)', 'b)', 'c)', 'd)', 'e)']
    labels_row2 = ['f)', 'g)', 'h)', 'i)', 'j)']
    
    # Plot each variable for both periods
    for col_idx, variable in enumerate(variables):
        print(f"\nProcessing {variable}...")
        
        # Period 1: 1973-2023 (top row)
        ax = axes[0, col_idx]
        key = f'{variable}_1973-2023'
        
        if key in merged_results_1973:
            to_plot = merged_results_1973[key]
            
            # Get vmin, vmax
            vmin, vmax = vmin_vmax.get(variable, (None, None))
            
            # Plot base map
            plot_figs(bmap, glaciers, ax)
            
            # Get colormap
            cmap = plt.get_cmap(colormaps.get(variable, 'viridis'))
            
            # Set colormap properties
            extend = 'neither'
            if variable == 'snowfall':
                extend = 'both'
            elif variable == 'total_et':
                extend = 'min'
                cmap.set_under('red')
            elif variable == '2m_temp_mean':
                extend = 'min'
                cmap.set_under('white')
            
            # Plot data (use special column for ET)
            plot_column = 'et_as_percentage_of_precip' if variable == 'total_et' else 'annual_trend'
            to_plot.plot(
                column=plot_column,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                legend=False,
                markersize=100
            )
            
            # Plot significant points
            significant_points = to_plot[to_plot['pval'] < 0.05]
            ax.plot(
                significant_points.geometry.x,
                significant_points.geometry.y,
                marker='o',
                markersize=14,
                markerfacecolor='none',
                markeredgecolor='k',
                linestyle='none',
                markeredgewidth=1.5
            )
            
            # Add title for first row
            title = colorbar_titles[variable]
            if variable == '2m_temp_mean':
                title = '2m temperature\n(°C/decade)'
            elif variable == 'prec':
                title = 'precipitation\n(%/decade)'
            elif variable == 'rainfall':
                title = 'rainfall (%/decade)'
            elif variable == 'snowfall':
                title = 'snowfall (%/decade)'
            elif variable == 'total_et':
                title = 'evapotranspiration\n(% of precip./decade)'
            
            ax.set_title(title, fontsize=18, pad=10)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            
            cb = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, extend=extend)
            cb.ax.tick_params(labelsize=18)
            
            # Add subplot label
            ax.text(0.02, 0.98, labels_row1[col_idx], transform=ax.transAxes,
                   fontsize=20, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Period 2: 1993-2023 (bottom row)
        ax = axes[1, col_idx]
        key = f'{variable}_1993-2023'
        
        if key in merged_results_1993:
            to_plot = merged_results_1993[key]
            
            # Get vmin, vmax for second period
            vmin, vmax = vmin_vmax_1993.get(variable, (None, None))
            
            # Plot base map
            plot_figs(bmap, glaciers, ax)
            
            # Get colormap
            cmap = plt.get_cmap(colormaps.get(variable, 'viridis'))
            
            # Set colormap properties
            extend = 'neither'
            if variable == 'snowfall':
                extend = 'both'
            elif variable == 'total_et':
                extend = 'min'
                cmap.set_under('red')
            elif variable == '2m_temp_mean':
                extend = 'min'
                cmap.set_under('white')
            
            # Plot data (use special column for ET)
            plot_column = 'et_as_percentage_of_precip' if variable == 'total_et' else 'annual_trend'
            to_plot.plot(
                column=plot_column,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                legend=False,
                markersize=100
            )
            
            # Plot significant points
            significant_points = to_plot[to_plot['pval'] < 0.05]
            ax.plot(
                significant_points.geometry.x,
                significant_points.geometry.y,
                marker='o',
                markersize=14,
                markerfacecolor='none',
                markeredgecolor='k',
                linestyle='none',
                markeredgewidth=1.5
            )
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            
            cb = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, extend=extend)
            cb.ax.tick_params(labelsize=18)
            
            # Add subplot label
            ax.text(0.02, 0.98, labels_row2[col_idx], transform=ax.transAxes,
                   fontsize=20, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add row titles
    axes[0, 0].text(-0.15, 0.5, 'Period 1: 1973-2023', transform=axes[0, 0].transAxes,
                    fontsize=20, fontweight='bold', va='center', ha='center', rotation=90)
    axes[1, 0].text(-0.15, 0.5, 'Period 2: 1993-2023', transform=axes[1, 0].transAxes,
                    fontsize=20, fontweight='bold', va='center', ha='center', rotation=90)
    
    # Add main title
    fig.suptitle('Annual trends in meteorological variables from ERA5-Land',
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = output_path / 'annual_meteorological_trends_2x5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n=== Figure saved to: {output_file} ===")
    
    # Also save as PDF
    output_file_pdf = output_path / 'annual_meteorological_trends_2x5.pdf'
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
    print(f"PDF saved to: {output_file_pdf}")
    
    plt.close()

if __name__ == "__main__":
    main()

