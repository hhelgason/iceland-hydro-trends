"""
Plot seasonal trends in meteorological variables (Temperature and Precipitation).

Creates TWO separate figures:
- Figure 1: 1973-2023 (Temperature left, Precipitation right) - labeled a), b)
- Figure 2: 1993-2023 (Temperature left, Precipitation right) - labeled c), d)
- Each subplot contains a 2x2 grid of the 4 seasons (DJF, MAM, JJA, SON)

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

def plot_seasonal_grid(to_plot, bmap, glaciers, axes_grid, seasonal_colnames, seasonal_pnames, 
                       seasonal_title_names, cmap, vmin, vmax, extend='neither'):
    """
    Plot a 2x2 grid of seasonal trends.
    """
    for i, (col, pname, title, ax) in enumerate(zip(seasonal_colnames, seasonal_pnames, 
                                                      seasonal_title_names, axes_grid.ravel())):
        # Plot base map
        plot_figs(bmap, glaciers, ax)
        
        # Plot data
        to_plot.plot(
            column=col,
            legend=False,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            markersize=60
        )
        
        # Plot significant points
        significant_points = to_plot[to_plot[pname] < 0.05]
        ax.plot(
            significant_points.geometry.x,
            significant_points.geometry.y,
            marker='o',
            markersize=10,
            markerfacecolor='none',
            markeredgecolor='k',
            linestyle='none',
            markeredgewidth=1.2
        )
        
        # Add title
        ax.set_title(title, y=0.9, fontsize=24)

def main():
    """Main function to create the seasonal meteorological trends figures."""
    
    print("=== Creating Seasonal Meteorological Trends Figures (2 separate) ===\n")
    
    # Output path
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
    
    # Define seasonal column names and titles
    seasonal_colnames = ['trend_DJF', 'trend_MAM', 'trend_JJA', 'trend_SON']
    seasonal_pnames = ['pval_DJF', 'pval_MAM', 'pval_JJA', 'pval_SON']
    seasonal_title_names = ['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov']
    
    # Variables to plot
    variables = ['2m_temp_mean', 'prec']
    
    # Define colormaps
    colormaps = {
        'prec': 'RdBu',  # Red (negative) to blue (positive)
        '2m_temp_mean': 'YlOrRd'
    }
    
    # Define units and titles
    units = {
        'prec': '%',
        '2m_temp_mean': '°C'
    }
    
    colorbar_titles = {
        'prec': 'Precipitation',
        '2m_temp_mean': 'Temperature'
    }
    
    # Define colorbar ranges
    print("\nSetting colorbar ranges...")
    global_min_max = {
        '2m_temp_mean': {'vmin': 0, 'vmax': 1},
        'prec': {'vmin': -12, 'vmax': 12}
    }
    
    for variable in variables:
        vmin = global_min_max[variable]['vmin']
        vmax = global_min_max[variable]['vmax']
        print(f"  {variable}: vmin={vmin}, vmax={vmax}")
    
    # Define periods with their data and labels
    periods_info = [
        {
            'name': '1973-2023',
            'data': merged_results_1973,
            'labels': ['a)', 'b)'],
            'filename_suffix': '1973-2023'
        },
        {
            'name': '1993-2023',
            'data': merged_results_1993,
            'labels': ['c)', 'd)'],
            'filename_suffix': '1993-2023'
        }
    ]
    
    # Create a separate figure for each period
    for period_info in periods_info:
        period_name = period_info['name']
        merged_results = period_info['data']
        labels = period_info['labels']
        
        print(f"\n=== Creating figure for period: {period_name} ===")
        
        # Create figure with 1 row, 2 columns (Temperature and Precipitation)
        fig = plt.figure(figsize=(14, 6))
        
        # Create GridSpec for the two variables
        outer_gs = fig.add_gridspec(1, 2, hspace=0.0, wspace=0.15)
        
        for var_idx, variable in enumerate(variables):
            print(f"  Processing variable: {variable}")
            
            # Create inner 2x2 grid for seasons
            inner_gs = outer_gs[0, var_idx].subgridspec(2, 2, hspace=0.05, wspace=0.05)
            
            # Create axes for the 4 seasons
            axes_grid = np.array([[fig.add_subplot(inner_gs[i, j]) for j in range(2)] for i in range(2)])
            
            # Get data
            key = f'{variable}_{period_name}'
            if key not in merged_results:
                print(f"    WARNING: {key} not found in data")
                continue
            
            to_plot = merged_results[key]
            
            # Get colormap and ranges
            cmap = plt.get_cmap(colormaps[variable])
            vmin = global_min_max[variable]['vmin']
            vmax = global_min_max[variable]['vmax']
            
            # Set extend and colormap properties
            if variable == '2m_temp_mean':
                extend = 'min'  # Temperature can go below 0
                cmap.set_under('white')
            else:
                extend = 'both'  # Precipitation can go beyond range
            
            # Plot the 2x2 seasonal grid
            plot_seasonal_grid(
                to_plot, bmap, glaciers, axes_grid,
                seasonal_colnames, seasonal_pnames, seasonal_title_names,
                cmap, vmin, vmax, extend
            )
            
            # Add main title above the 2x2 grid
            title = f"{colorbar_titles[variable]}\n{period_name}"
            fig.text(
                0.25 + var_idx * 0.5,
                0.98,
                title,
                ha='center',
                va='top',
                fontsize=22,
                fontweight='bold'
            )
            
            # Add subplot label
            label = labels[var_idx]
            axes_grid[0, 0].text(
                0.02, 1.05,
                label,
                transform=axes_grid[0, 0].transAxes,
                fontsize=28,
                fontweight='bold',
                va='top',
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Add colorbar below the 2x2 grid
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            
            # Position colorbar (center under each subplot)
            if var_idx == 0:  # Temperature (left)
                x_pos = 0.16
            else:  # Precipitation (right)
                x_pos = 0.60
            
            cbar_ax = fig.add_axes([
                x_pos,  # x position
                0.08,  # y position (below grid)
                0.24,  # width
                0.025   # height (larger for better visibility)
            ])
            
            cb = fig.colorbar(
                sm,
                cax=cbar_ax,
                orientation='horizontal',
                extend=extend
            )
            cb.ax.tick_params(labelsize=22)
            cb.set_label(f'[{units[variable]}/decade]', fontsize=24)
        
        # Save figure
        png_file = output_path / f'seasonal_meteorological_trends_{period_info["filename_suffix"]}.png'
        pdf_file = output_path / f'seasonal_meteorological_trends_{period_info["filename_suffix"]}.pdf'
        
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {png_file}")
        print(f"  Saved: {pdf_file}")
    
    print(f"\n=== All figures saved to: {output_path} ===")

if __name__ == "__main__":
    main()

