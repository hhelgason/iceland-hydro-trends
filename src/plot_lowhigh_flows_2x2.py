"""
Script to plot low and high flow trends in a 2x2 figure.

Creates a figure with:
- Row 1: Period 1973-2023 (low flow, high flow)
- Row 2: Period 1993-2023 (low flow, high flow)

Author: Hordur Bragi Helgason
Date: 2025
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import numpy as np
import seaborn as sns
from pathlib import Path
from config import OUTPUT_DIR, ICELAND_SHAPEFILE, GLACIER_SHAPEFILE, CATCHMENT_ATTRIBUTES_FILE

# Set style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

def plot_figs(basemap, glaciers, ax, iceland_shapefile_color='gray', glaciers_color='white'):
    """Plot the base map of Iceland with glaciers."""
    minx, miny = 222375, 307671
    maxx, maxy = 765246, 697520
    basemap.plot(ax=ax, color=iceland_shapefile_color, edgecolor='darkgray')
    glaciers.plot(ax=ax, facecolor=glaciers_color, edgecolor='none')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def determine_extend(vmin, vmax, vmin_actual, vmax_actual):
    """
    Determine the extend parameter for colorbar based on data range vs. colorbar limits.
    """
    if vmin_actual < vmin and vmax_actual > vmax:
        return 'both'
    elif vmin_actual < vmin:
        return 'min'
    elif vmax_actual > vmax:
        return 'max'
    else:
        return 'neither'

def add_colorbar(fig, colormap, vmin, vmax, label, extend='neither'):
    """Add a colorbar at the bottom of the figure."""
    cax = fig.add_axes([0.25, 0.08, 0.5, 0.025])
    cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax), 
                     orientation='horizontal', extend=extend)
    cb.set_label(label, size=22)
    cb.ax.tick_params(labelsize=20)
    return cb

def generate_heatmap_row(df, trend_col, pval_col, glaciation_threshold=0.05):
    """Generate heatmap data for a single row."""
    positive_cases = (df[trend_col].dropna() > 0).sum()
    negative_cases = (df[trend_col].dropna() < 0).sum()
    positive_significant = ((df[trend_col].dropna() > 0) & (df[pval_col].dropna() < 0.05)).sum()
    negative_significant = ((df[trend_col].dropna() < 0) & (df[pval_col].dropna() < 0.05)).sum()

    # Glaciated basin counts
    positive_glaciated = ((df[trend_col] > 0) & (df['g_frac'] > glaciation_threshold)).sum()
    negative_glaciated = ((df[trend_col] < 0) & (df['g_frac'] > glaciation_threshold)).sum()
    positive_significant_glaciated = ((df[trend_col] > 0) & (df[pval_col] < 0.05) & (df['g_frac'] > glaciation_threshold)).sum()
    negative_significant_glaciated = ((df[trend_col] < 0) & (df[pval_col] < 0.05) & (df['g_frac'] > glaciation_threshold)).sum()

    heatmap_row = {
        'Positive': positive_cases,
        'Negative': negative_cases,
        'Positive significant': positive_significant,
        'Negative significant': negative_significant
    }
    annotation_row = {
        'Positive': f"{positive_cases} ({positive_glaciated})",
        'Negative': f"{negative_cases} ({negative_glaciated})",
        'Positive significant': f"{positive_significant} ({positive_significant_glaciated})",
        'Negative significant': f"{negative_significant} ({negative_significant_glaciated})"
    }
    return heatmap_row, annotation_row

def plot_heatmap_summary(df_1973_lowflow, df_1973_highflow, df_1993_lowflow, df_1993_highflow):
    """Plot heatmap summary of low/high flow trends."""
    
    # Define columns for each metric
    columns = [
        ('low_flow_trend_per_decade', 'low_flow_pval', 'Low Flow (Q10)'),
        ('high_flow_trend_per_decade', 'high_flow_pval', 'High Flow (Q90)')
    ]
    
    # Generate heatmap data for each period
    heatmap_data_1973 = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    annotations_1973 = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    
    heatmap_data_1993 = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    annotations_1993 = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    
    # 1973-2023 period
    for trend_col, pval_col, label in columns:
        if 'low_flow' in trend_col:
            df = df_1973_lowflow
        else:
            df = df_1973_highflow
        heatmap_row, annotation_row = generate_heatmap_row(df, trend_col, pval_col)
        for key in heatmap_data_1973:
            heatmap_data_1973[key].append(heatmap_row[key])
            annotations_1973[key].append(annotation_row[key])
    
    # 1993-2023 period
    for trend_col, pval_col, label in columns:
        if 'low_flow' in trend_col:
            df = df_1993_lowflow
        else:
            df = df_1993_highflow
        heatmap_row, annotation_row = generate_heatmap_row(df, trend_col, pval_col)
        for key in heatmap_data_1993:
            heatmap_data_1993[key].append(heatmap_row[key])
            annotations_1993[key].append(annotation_row[key])
    
    index_labels = [label for _, _, label in columns]
    df_1973_heatmap = pd.DataFrame(heatmap_data_1973, index=index_labels)
    annotations_1973_df = pd.DataFrame(annotations_1973, index=index_labels)
    df_1993_heatmap = pd.DataFrame(heatmap_data_1993, index=index_labels)
    annotations_1993_df = pd.DataFrame(annotations_1993, index=index_labels)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Find max value for consistent color scale
    max_val = max(df_1973_heatmap.values.max(), df_1993_heatmap.values.max())
    
    # Plot 1973-2023 heatmap
    sns.heatmap(df_1973_heatmap, 
                annot=annotations_1973_df, 
                fmt='', 
                cmap='Blues', 
                ax=axes[0], 
                vmin=0, vmax=max_val,
                cbar_kws={'label': 'Number of gauges'})
    axes[0].set_title('1973-2023', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    
    # Plot 1993-2023 heatmap
    sns.heatmap(df_1993_heatmap, 
                annot=annotations_1993_df, 
                fmt='', 
                cmap='Blues', 
                ax=axes[1], 
                vmin=0, vmax=max_val,
                cbar_kws={'label': 'Number of gauges'})
    axes[1].set_title('1993-2023', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    
    plt.suptitle('Summary of Low and High Flow Trends\n(glaciated basins in parentheses)', 
                 fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'lowhigh_flow_trends_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    
    output_path_pdf = OUTPUT_DIR / 'lowhigh_flow_trends_heatmap.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path_pdf}")
    
    plt.close()

def main():
    """Main function to create the 2x2 low/high flow trends figure."""
    
    print("=== Creating Low/High Flow Trends Figure (2x2) ===\n")
    
    print("Loading data...")
    # Load merged gdfs for both periods
    merged_1973 = gpd.read_file(OUTPUT_DIR / 'results_lamah_data' / 'merged_gdf_1973_2023.gpkg')
    merged_1993 = gpd.read_file(OUTPUT_DIR / 'results_lamah_data' / 'merged_gdf_1993_2023.gpkg')
    
    # Restore index from column (matching main.py behavior)
    if 'index' in merged_1973.columns:
        merged_1973 = merged_1973.set_index('index')
    if 'index' in merged_1993.columns:
        merged_1993 = merged_1993.set_index('index')
    
    # Load basemaps
    bmap = gpd.read_file(ICELAND_SHAPEFILE)
    glaciers = gpd.read_file(GLACIER_SHAPEFILE)
    
    # Load catchments for plotting
    catchments = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)
    catchments = catchments.set_index('id')
    
    # Filter to natural gauges
    natural_1973 = merged_1973[merged_1973['degimpact'] != 's']
    natural_1993 = merged_1993[merged_1993['degimpact'] != 's']
    
    # For high flow analysis - no additional filtering
    filtered_1973_highflow = natural_1973
    filtered_1993_highflow = natural_1993
    
    # For low flow analysis - drop gauge 48 (Ufsarlón reservoir affected low flows after ~2008)
    filtered_1973_lowflow = natural_1973.drop([48], errors='ignore')
    filtered_1993_lowflow = natural_1993.drop([48], errors='ignore')
    
    print(f"Period 1973-2023: {len(filtered_1973_highflow)} gauges (high flow), {len(filtered_1973_lowflow)} gauges (low flow)")
    print(f"Period 1993-2023: {len(filtered_1993_highflow)} gauges (high flow), {len(filtered_1993_lowflow)} gauges (low flow)")
    
    # Map parameters (matching plotting.py)
    colormap = 'RdBu'
    iceland_shapefile_color = 'gray'
    glaciers_color = 'white'
    xlim = (222375, 765246)
    ylim = (307671, 697520)
    vmin, vmax = -10.5, 10.5  # Matching plotting.py
    
    # Calculate actual data range for extend parameter
    all_low_trends = pd.concat([
        filtered_1973_lowflow['low_flow_trend_per_decade'],
        filtered_1993_lowflow['low_flow_trend_per_decade']
    ]).dropna()
    all_high_trends = pd.concat([
        filtered_1973_highflow['high_flow_trend_per_decade'],
        filtered_1993_highflow['high_flow_trend_per_decade']
    ]).dropna()
    all_trends = pd.concat([all_low_trends, all_high_trends])
    data_min = all_trends.min()
    data_max = all_trends.max()
    extend = determine_extend(vmin, vmax, data_min, data_max)
    print(f"Data range: {data_min:.2f} to {data_max:.2f}, extend: {extend}")
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Create 2x2 grid with proper spacing
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.08, 
                          left=0.05, right=0.95, top=0.98, bottom=0.12)
    
    # Define subplot parameters (lowflow excludes gauge 48, highflow includes it)
    subplots = [
        (filtered_1973_lowflow, 'low_flow_trend_per_decade', 'low_flow_pval', 
         'Annual low flow (10th percentile)\n1973-2023', 'a)'),
        (filtered_1973_highflow, 'high_flow_trend_per_decade', 'high_flow_pval', 
         'Annual high flow (90th percentile)\n1973-2023', 'b)'),
        (filtered_1993_lowflow, 'low_flow_trend_per_decade', 'low_flow_pval', 
         'Annual low flow (10th percentile)\n1993-2023', 'c)'),
        (filtered_1993_highflow, 'high_flow_trend_per_decade', 'high_flow_pval', 
         'Annual high flow (90th percentile)\n1993-2023', 'd)'),
    ]
    
    print("\nPlotting maps...")
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        data, trend_col, pval_col, title, label = subplots[i]
        
        ax = fig.add_subplot(gs[row, col])
        
        # Plot basemap using plot_figs function (matching plotting.py style)
        plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
        
        # Plot trend data
        data_with_trends = data[data[trend_col].notna()]
        if len(data_with_trends) > 0:
            # Plot points (matching plotting.py markersize)
            data_with_trends.plot(
                column=trend_col,
                ax=ax,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                legend=False,
                s=150,
                zorder=5
            )
            
            # Mark significant trends (matching plotting.py style)
            significant = data_with_trends[data_with_trends[pval_col] < 0.05]
            if len(significant) > 0:
                ax.plot(significant.geometry.x - 100, significant.geometry.y,
                       marker='o', markersize=18, markerfacecolor='none',
                       markeredgecolor='k', linestyle='none', lw=0.5, zorder=6)
            
            # Plot catchment boundaries (matching plotting.py style)
            indices_to_plot = data_with_trends.index.tolist()
            catchments_to_plot = catchments.loc[catchments.index.isin(indices_to_plot)]
            catchments_to_plot.plot(facecolor='none', edgecolor='black', 
                                    ax=ax, zorder=3, lw=0.25)
        
        # Add title using text for precise positioning (inside the axes area)
        ax.text(0.5, 1.02, title, transform=ax.transAxes, fontsize=22, 
                fontweight='bold', ha='center', va='bottom')
        
        # Add subplot label (matching seasonal trends style)
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
               fontsize=28, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add shared colorbar at the bottom (matching plotting.py add_colorbar function)
    cmap_obj = plt.get_cmap(colormap)
    add_colorbar(fig, cmap_obj, vmin, vmax, 'Trend (%/decade)', extend=extend)
    
    # Save
    output_path = OUTPUT_DIR / 'lowhigh_flow_trends_2x2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    output_path_pdf = OUTPUT_DIR / 'lowhigh_flow_trends_2x2.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path_pdf}")
    
    plt.close()
    
    # Generate heatmap summary
    print("\nGenerating heatmap summary...")
    plot_heatmap_summary(filtered_1973_lowflow, filtered_1973_highflow, 
                         filtered_1993_lowflow, filtered_1993_highflow)
    
    print(f"\n=== Figures complete ===")

if __name__ == "__main__":
    main()

