"""
Plotting Functions for Streamflow Trend Analysis in Iceland

This module contains functions for visualizing trends in streamflow data across Iceland.
It is part of the supplementary material for the paper "Understanding Changes in Iceland's Streamflow Dynamics in Response to Climate Change".

The module provides functions for creating:
1. Maps of trend magnitudes and significance
2. Time series plots of various streamflow metrics
3. Seasonal and annual trend visualizations
4. Baseflow and flow sequence analyses

Dependencies:
- matplotlib: Core plotting functionality
- seaborn: Enhanced plotting styles
- pandas: Data manipulation
- geopandas: Spatial data handling
- numpy: Numerical operations

Author: Hordur Bragi Helgason
Institution: Landsvirkjun
Date: 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import geopandas as gpd
import datetime as dt
from pathlib import Path
import os
from config import ICELAND_SHAPEFILE, GLACIER_SHAPEFILE

# Plot limits for different metrics
# Annual mean flow trend limits (percent per decade)
annual_vmin = -6
annual_vmax = 6

# General trend limits (percent per decade)
vmin = -10
vmax = 10

# Low/high flow trend limits (percent per decade)
low_high_flow_vmin = -10.5
low_high_flow_vmax = 10.5

# Standard deviation trend limits (percent per decade)
trend_std_vmin = -20
trend_std_vmax = 20

# Coefficient of variation trend limits (percent per decade)
trend_cv_vmin = -20
trend_cv_vmax = 20

# Flashiness index trend limits (percent per decade)
trend_flashiness_vmin = -20
trend_flashiness_vmax = 20

# Rising/falling sequences trend limits (percent per decade)
trend_rising_falling_vmin = -15
trend_rising_falling_vmax = 15

# Baseflow index trend limits (percent per decade)
trend_baseflow_index_vmin = -4
trend_baseflow_index_vmax = 4

# Set seaborn style for all plots
sns.set()

def determine_extend(vmin, vmax, vmin_actual, vmax_actual):
    """
    Determine the extend parameter for colorbar based on data range vs. colorbar limits.
    
    Args:
        vmin (float): Minimum value for colorbar
        vmax (float): Maximum value for colorbar
        vmin_actual (float): Minimum value in the actual data
        vmax_actual (float): Maximum value in the actual data
    
    Returns:
        str: One of 'both', 'min', 'max', or 'neither' indicating which arrows to show on colorbar
    """
    if vmin_actual < vmin and vmax_actual > vmax:
        extend = 'both'
    elif vmin_actual < vmin:
        extend = 'min'
    elif vmax_actual > vmax:
        extend = 'max'
    else:
        extend = 'neither'
    return extend

def plot_figs(basemap, glaciers, ax, iceland_shapefile_color, glaciers_color):
    """
    Plot the base map of Iceland with glaciers.
    
    Args:
        basemap (GeoDataFrame): GeoDataFrame containing Iceland's outline
        glaciers (GeoDataFrame): GeoDataFrame containing glacier outlines
        ax (matplotlib.axes.Axes): Axes to plot on
        iceland_shapefile_color (str): Color for Iceland's landmass
        glaciers_color (str): Color for glaciers
    """
    minx, miny = 222375, 307671
    maxx, maxy = 765246, 697520
    basemap.plot(ax=ax, color=iceland_shapefile_color, edgecolor='darkgray')
    glaciers.plot(ax=ax, facecolor=glaciers_color, edgecolor='none')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def create_folders(base_folder, start_year, end_year):
    """
    Create folder structure for storing different types of plots.
    
    Creates a hierarchical folder structure for organizing different types of plots:
    - Daily timeseries
    - Annual trends and statistics
    - Seasonal trends
    - Monthly trends
    - Maps and spatial visualizations
    
    Args:
        base_folder (str or Path): Base directory for all output
        start_year (int): Start year of the analysis period
        end_year (int): End year of the analysis period
    
    Returns:
        tuple: Paths to all created directories
    """
    period = f"{start_year}_{end_year}"
    base_path = base_folder / period
    daily_timeseries_path = base_path / 'daily_streamflow_series'
    seasonal_trends_path_mod_ts = base_path / 'seasonal_trend_series_mod_ts'
    monthly_trends_path_mod_ts = base_path / 'monthly_trend_series_mod_ts'
    annual_trends_path = base_path / 'annual_trend_series'
    annual_autocorrelation_path = base_path / 'annual_autocorrelation'
    maps_path = base_path / 'maps'
    raster_trends_path = base_path / 'raster_trends'
    annual_mean_flow_path = annual_trends_path / 'annual_mean_flow'
    annual_cv_path = annual_trends_path / 'annual_cv'
    annual_std_path = annual_trends_path / 'annual_std'
    flashiness_path = annual_trends_path / 'flashiness'
    sequences_path = annual_trends_path / 'sequences'
    baseflow_index_path = annual_trends_path / 'baseflow_index'
    baseflow_series_path = annual_trends_path / 'baseflow_series'
    
    # Create all directories
    for folder_path in [base_path, daily_timeseries_path, annual_autocorrelation_path, maps_path, 
                       raster_trends_path, seasonal_trends_path_mod_ts, annual_trends_path, 
                       monthly_trends_path_mod_ts, annual_mean_flow_path, annual_cv_path, 
                       annual_std_path, flashiness_path, sequences_path, baseflow_index_path, 
                       baseflow_series_path]:
        folder_path.mkdir(parents=True, exist_ok=True)
    
    return (daily_timeseries_path, annual_autocorrelation_path, maps_path, raster_trends_path, 
            seasonal_trends_path_mod_ts, annual_trends_path, monthly_trends_path_mod_ts,
            annual_mean_flow_path, annual_cv_path, annual_std_path, flashiness_path, 
            sequences_path, baseflow_index_path, baseflow_series_path)

def setup_plot(ax, title, xlim, ylim, xticks, yticks, frame_on):
    """
    Set up common plot parameters for consistency across figures.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to configure
        title (str): Plot title
        xlim (tuple): X-axis limits (min, max)
        ylim (tuple): Y-axis limits (min, max)
        xticks (list): X-axis tick positions
        yticks (list): Y-axis tick positions
        frame_on (bool): Whether to show the plot frame
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_frame_on(frame_on)
    ax.set_title(title, size=20, y=0.95)

def add_colorbar(fig, ax, colormap, vmin, vmax, label, extend):
    """
    Add a colorbar to a plot with consistent styling.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to add colorbar to
        ax (matplotlib.axes.Axes): Axes the colorbar relates to
        colormap (str or matplotlib.colors.Colormap): Colormap to use
        vmin (float): Minimum value for colorbar scale
        vmax (float): Maximum value for colorbar scale
        label (str): Colorbar label
        extend (str): How to extend the colorbar ('both', 'min', 'max', or 'neither')
    
    Returns:
        matplotlib.colorbar.Colorbar: The created colorbar
    """
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.07, 0.6, 0.03])
    cb = ColorbarBase(cax, cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax), 
                     orientation='horizontal', extend=extend)
    cb.set_label(label, size=20)
    cb.ax.tick_params(labelsize=20)
    return cb

def plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim):
    """
    Create a base map of Iceland with consistent styling.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        bmap (GeoDataFrame): GeoDataFrame containing Iceland's outline
        glaciers (GeoDataFrame): GeoDataFrame containing glacier outlines
        iceland_shapefile_color (str): Color for Iceland's landmass
        glaciers_color (str): Color for glaciers
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
    """
    plot_figs(bmap, glaciers, ax, iceland_shapefile_color, glaciers_color)
    setup_plot(ax, '', xlim, ylim, [], [], False)

def plot_maps(catchments, which_plots, merged_gdf, start_year, end_year, results, valid_data_dict, invalid_data_dict, maps_path):
    """
    Plot all map figures showing spatial patterns of trends.

    This function creates a series of maps showing the spatial distribution of trends
    in various streamflow metrics across Iceland. Each map includes:
    - Base map of Iceland with glacier outlines
    - Colored points showing trend magnitudes
    - Black circles indicating statistically significant trends (p < 0.05)
    - Catchment boundaries
    - Colorbar showing the trend scale

    Args:
        catchments (GeoDataFrame): GeoDataFrame containing catchment boundaries
        which_plots (dict): Dictionary specifying which plots to create
        merged_gdf (GeoDataFrame): GeoDataFrame containing gauge locations and trend results
        start_year (int): Start year of analysis period
        end_year (int): End year of analysis period
        results (DataFrame): DataFrame containing trend results
        valid_data_dict (dict): Dictionary of valid data for each gauge and metric
        invalid_data_dict (dict): Dictionary of invalid data for each gauge and metric
        maps_path (str or Path): Directory to save the output maps

    The function creates separate maps for:
    - Annual mean flow trends
    - Seasonal mean flow trends
    - Low/high flow trends
    - Standard deviation trends
    - Coefficient of variation trends
    - Flashiness index trends
    - Rising/falling sequence trends
    - Baseflow index trends
    """
    # Debug prints
    print("Available columns in merged_gdf:", merged_gdf.columns.tolist())
    print("Which plots configuration:", which_plots)
    
    # Plot setup
    colormap = 'RdBu'
    iceland_shapefile_color = 'gray'
    glaciers_color = 'white'
    bmap = gpd.read_file(ICELAND_SHAPEFILE)
    glaciers = gpd.read_file(GLACIER_SHAPEFILE)
    xlim = (222375, 765246)
    ylim = (307671, 697520)

    # Create version of merged_gdf that excludes anthropogenically influenced gauges
    natural_mask = merged_gdf['degimpact'] != 's'  # 's' indicates strong anthropogenic influence
    merged_gdf_natural = merged_gdf.loc[natural_mask]

    if which_plots['annual_map']:
        print("Plotting annual map...")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf['annual_avg_flow_trend_per_decade'].min()
        data_max = merged_gdf['annual_avg_flow_trend_per_decade'].max()
        extend = determine_extend(annual_vmin, annual_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf.plot(column='annual_avg_flow_trend_per_decade', legend=False, vmin=annual_vmin, vmax=annual_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf[merged_gdf['pval'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf['annual_avg_flow_trend_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, annual_vmin, annual_vmax, 'Trend in streamflow from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'annual_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_map']:
        print("Plotting seasonal map...")
        columns = ['trend_DJF_per_decade', 'trend_MAM_per_decade', 'trend_JJA_per_decade', 'trend_SON_per_decade']
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(vmin, vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            im = merged_gdf_natural.plot(column=col, legend=False, vmin=vmin, vmax=vmax, ax=ax, cmap=colormap, s=150)
            significant_points = merged_gdf_natural[merged_gdf_natural[f'pval_{col.split("_")[1]}'] < 0.05]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            catchments.loc[merged_gdf_natural[col].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
        add_colorbar(fig, axs[1, 1], colormap, vmin, vmax, 'Trend in streamflow from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'seasonal_trend_ts_mod.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()    

    if which_plots['low_high_flow_map']:
        print("Plotting low/high flow maps...")
        # Drop Fellsá and Eyjabakkafoss (15, 48)
        merged_gdf_filtered = merged_gdf_natural.drop([15, 48], errors='ignore')
        
        # Plot low flow map
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf_filtered['low_flow_trend_per_decade'].min()
        data_max = merged_gdf_filtered['low_flow_trend_per_decade'].max()
        extend = determine_extend(low_high_flow_vmin, low_high_flow_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf_filtered.plot(column='low_flow_trend_per_decade', legend=False, vmin=low_high_flow_vmin, vmax=low_high_flow_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf_filtered[merged_gdf_filtered['low_flow_pval'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_filtered['low_flow_trend_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, low_high_flow_vmin, low_high_flow_vmax, 'Trend in low flow from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'low_flow_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()    
    
        # Plot high flow map
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf_filtered['high_flow_trend_per_decade'].min()
        data_max = merged_gdf_filtered['high_flow_trend_per_decade'].max()
        extend = determine_extend(low_high_flow_vmin, low_high_flow_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf_filtered.plot(column='high_flow_trend_per_decade', legend=False, vmin=low_high_flow_vmin, vmax=low_high_flow_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf_filtered[merged_gdf_filtered['high_flow_pval'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_filtered['high_flow_trend_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, low_high_flow_vmin, low_high_flow_vmax, 'Trend in high flow from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'high_flow_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['annual_std_map']:
        print("Plotting annual std map...")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf_natural['trend_annual_std_per_decade'].min()
        data_max = merged_gdf_natural['trend_annual_std_per_decade'].max()
        extend = determine_extend(trend_std_vmin, trend_std_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf_natural.plot(column='trend_annual_std_per_decade', legend=False, vmin=trend_std_vmin, vmax=trend_std_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf_natural[merged_gdf_natural['pval_annual_std'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_natural['trend_annual_std_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, trend_std_vmin, trend_std_vmax, 'Trend in annual std. dev. from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'annual_std_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['annual_cv_map']:
        print("Plotting annual CV map...")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf_natural['trend_annual_cv_per_decade'].min()
        data_max = merged_gdf_natural['trend_annual_cv_per_decade'].max()
        extend = determine_extend(trend_cv_vmin, trend_cv_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf_natural.plot(column='trend_annual_cv_per_decade', legend=False, vmin=trend_cv_vmin, vmax=trend_cv_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf_natural[merged_gdf_natural['pval_annual_cv'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_natural['trend_annual_cv_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, trend_cv_vmin, trend_cv_vmax, 'Trend in annual CV from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'annual_cv_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['flashiness_map']:
        print("Plotting flashiness map...")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf_natural['trend_flashiness_per_decade'].min()
        data_max = merged_gdf_natural['trend_flashiness_per_decade'].max()
        extend = determine_extend(trend_flashiness_vmin, trend_flashiness_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf_natural.plot(column='trend_flashiness_per_decade', legend=False, vmin=trend_flashiness_vmin, vmax=trend_flashiness_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf_natural[merged_gdf_natural['pval_flashiness'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_natural['trend_flashiness_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, trend_flashiness_vmin, trend_flashiness_vmax, 'Trend in flashiness index from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'flashiness_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['baseflow_index_map']:
        print("Plotting baseflow index map...")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        data_min = merged_gdf_natural['trend_baseflow_index_per_decade'].min()
        data_max = merged_gdf_natural['trend_baseflow_index_per_decade'].max()
        extend = determine_extend(trend_baseflow_index_vmin, trend_baseflow_index_vmax, data_min, data_max)
        plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im = merged_gdf_natural.plot(column='trend_baseflow_index_per_decade', legend=False, vmin=trend_baseflow_index_vmin, vmax=trend_baseflow_index_vmax, ax=ax, cmap=colormap, s=150)
        significant_points = merged_gdf_natural[merged_gdf_natural['pval_baseflow_index'] < 0.05]
        ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_natural['trend_baseflow_index_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
        add_colorbar(fig, ax, colormap, trend_baseflow_index_vmin, trend_baseflow_index_vmax, 'Trend in baseflow index from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'baseflow_index_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_baseflow_index_map']:
        print("Plotting seasonal baseflow index map...")
        columns = ['baseflow_index_DJF_trend_per_decade', 'baseflow_index_MAM_trend_per_decade', 'baseflow_index_JJA_trend_per_decade', 'baseflow_index_SON_trend_per_decade']
        # Check if seasonal baseflow index columns exist
        if not any(col in merged_gdf_natural.columns for col in columns):
            print("Warning: Seasonal baseflow index trend columns not found. Skipping seasonal baseflow index map.")
            return
        
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(trend_baseflow_index_vmin, trend_baseflow_index_vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            im = merged_gdf_natural.plot(column=col, legend=False, vmin=trend_baseflow_index_vmin, vmax=trend_baseflow_index_vmax, ax=ax, cmap=colormap, s=150)
            # Fix the p-value column name
            pval_col = f'baseflow_index_{col.split("_")[2]}_pval'  # Extract season (DJF/MAM/JJA/SON) from trend column
            significant_points = merged_gdf_natural[merged_gdf_natural[pval_col] < 0.05]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            catchments.loc[merged_gdf_natural[col].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
        add_colorbar(fig, axs[1, 1], colormap, trend_baseflow_index_vmin, trend_baseflow_index_vmax, 'Trend in baseflow index from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'seasonal_baseflow_index_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['rising_falling_map']:
        print("Plotting rising/falling sequences map...")
        # Create a figure with two subplots for rising and falling sequences
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        # Plot rising sequences
        data_min_rising = merged_gdf_natural['trend_rising_seq_per_decade'].min()
        data_max_rising = merged_gdf_natural['trend_rising_seq_per_decade'].max()
        extend_rising = determine_extend(trend_rising_falling_vmin, trend_rising_falling_vmax, data_min_rising, data_max_rising)
        plot_map(ax1, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im1 = merged_gdf_natural.plot(column='trend_rising_seq_per_decade', legend=False, vmin=trend_rising_falling_vmin, vmax=trend_rising_falling_vmax, ax=ax1, cmap=colormap, s=150)
        significant_points = merged_gdf_natural[merged_gdf_natural['pval_rising_seq'] < 0.05]
        ax1.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_natural['trend_rising_seq_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax1, zorder=3, lw=0.25)
        ax1.set_title('Rising Sequences')
        
        # Plot falling sequences
        data_min_falling = merged_gdf_natural['trend_falling_seq_per_decade'].min()
        data_max_falling = merged_gdf_natural['trend_falling_seq_per_decade'].max()
        extend_falling = determine_extend(trend_rising_falling_vmin, trend_rising_falling_vmax, data_min_falling, data_max_falling)
        plot_map(ax2, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
        im2 = merged_gdf_natural.plot(column='trend_falling_seq_per_decade', legend=False, vmin=trend_rising_falling_vmin, vmax=trend_rising_falling_vmax, ax=ax2, cmap=colormap, s=150)
        significant_points = merged_gdf_natural[merged_gdf_natural['pval_falling_seq'] < 0.05]
        ax2.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
        catchments.loc[merged_gdf_natural['trend_falling_seq_per_decade'].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax2, zorder=3, lw=0.25)
        ax2.set_title('Falling Sequences')
        
        # Add colorbars
        add_colorbar(fig, ax1, colormap, trend_rising_falling_vmin, trend_rising_falling_vmax, 'Trend in rising sequences from %s-%s (%%/decade)' % (start_year, end_year), extend_rising)
        add_colorbar(fig, ax2, colormap, trend_rising_falling_vmin, trend_rising_falling_vmax, 'Trend in falling sequences from %s-%s (%%/decade)' % (start_year, end_year), extend_falling)
        
        save_path = os.path.join(maps_path, 'rising_falling_sequences_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_std_map']:
        print("Plotting seasonal std map...")
        columns = ['std_DJF_trend_per_decade', 'std_MAM_trend_per_decade', 'std_JJA_trend_per_decade', 'std_SON_trend_per_decade']
        # Check if columns exist
        if not all(col in merged_gdf_natural.columns for col in columns):
            print("Warning: Seasonal std trend columns not found. Skipping seasonal std map.")
            return
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(trend_std_vmin, trend_std_vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            im = merged_gdf_natural.plot(column=col, legend=False, vmin=trend_std_vmin, vmax=trend_std_vmax, ax=ax, cmap=colormap, s=150)
            pval_col = f'std_{col.split("_")[1]}_pval'
            # Only plot significance circles for points that have both a significant p-value AND a valid trend value
            significant_points = merged_gdf_natural[(merged_gdf_natural[pval_col] < 0.05) & (~merged_gdf_natural[col].isna())]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            catchments.loc[merged_gdf_natural[col].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
        add_colorbar(fig, axs[1, 1], colormap, trend_std_vmin, trend_std_vmax, 'Trend in seasonal std. dev. from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'seasonal_std_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_cv_map']:
        print("Plotting seasonal CV map...")
        columns = ['cv_DJF_trend_per_decade', 'cv_MAM_trend_per_decade', 'cv_JJA_trend_per_decade', 'cv_SON_trend_per_decade']
        # Check if columns exist
        if not all(col in merged_gdf_natural.columns for col in columns):
            print("Warning: Seasonal CV trend columns not found. Skipping seasonal CV map.")
            return
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(trend_cv_vmin, trend_cv_vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            
            # Only plot points and catchments for gauges with valid trend values
            valid_trends_mask = ~merged_gdf_natural[col].isna()
            valid_trends_gdf = merged_gdf_natural[valid_trends_mask]
            
            # Plot the trend values
            im = valid_trends_gdf.plot(column=col, legend=False, vmin=trend_cv_vmin, vmax=trend_cv_vmax, ax=ax, cmap=colormap, s=150)
            
            # Plot significance indicators only for points with both valid trends and significant p-values
            pval_col = f'cv_{col.split("_")[1]}_pval'
            significant_points = valid_trends_gdf[valid_trends_gdf[pval_col] < 0.05]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            
            # Plot catchment boundaries only for gauges with valid trend values
            catchments.loc[valid_trends_gdf.index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
        add_colorbar(fig, axs[1, 1], colormap, trend_cv_vmin, trend_cv_vmax, 'Trend in seasonal CV from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'seasonal_cv_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_flashiness_map']:
        print("Plotting seasonal flashiness map...")
        columns = ['flashiness_DJF_trend_per_decade', 'flashiness_MAM_trend_per_decade', 'flashiness_JJA_trend_per_decade', 'flashiness_SON_trend_per_decade']
        # Check if columns exist
        if not all(col in merged_gdf_natural.columns for col in columns):
            print("Warning: Seasonal flashiness trend columns not found. Skipping seasonal flashiness map.")
            return
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(trend_flashiness_vmin, trend_flashiness_vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            im = merged_gdf_natural.plot(column=col, legend=False, vmin=trend_flashiness_vmin, vmax=trend_flashiness_vmax, ax=ax, cmap=colormap, s=150)
            pval_col = f'flashiness_{col.split("_")[1]}_pval'
            significant_points = merged_gdf_natural[merged_gdf_natural[pval_col] < 0.05]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            catchments.loc[merged_gdf_natural[col].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
        add_colorbar(fig, axs[1, 1], colormap, trend_flashiness_vmin, trend_flashiness_vmax, 'Trend in seasonal flashiness from %s-%s (%%/decade)' % (start_year, end_year), extend)
        save_path = os.path.join(maps_path, 'seasonal_flashiness_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_rising_falling_map']:
        print("Plotting seasonal rising/falling sequences map...")
        for sequence_type in ['rising', 'falling']:
            columns = [f'{sequence_type}_seq_DJF_trend_per_decade', f'{sequence_type}_seq_MAM_trend_per_decade', 
                      f'{sequence_type}_seq_JJA_trend_per_decade', f'{sequence_type}_seq_SON_trend_per_decade']
            # Check if columns exist
            if not all(col in merged_gdf_natural.columns for col in columns):
                print(f"Warning: Seasonal {sequence_type} sequences trend columns not found. Skipping seasonal {sequence_type} sequences map.")
                continue
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(trend_rising_falling_vmin, trend_rising_falling_vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            im = merged_gdf_natural.plot(column=col, legend=False, vmin=trend_rising_falling_vmin, vmax=trend_rising_falling_vmax, ax=ax, cmap=colormap, s=150)
            pval_col = f'{sequence_type}_seq_{col.split("_")[2]}_pval'
            significant_points = merged_gdf_natural[merged_gdf_natural[pval_col] < 0.05]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            catchments.loc[merged_gdf_natural[col].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
            add_colorbar(fig, axs[1, 1], colormap, trend_rising_falling_vmin, trend_rising_falling_vmax, f'Trend in seasonal {sequence_type} sequences from %s-%s (%%/decade)' % (start_year, end_year), extend)
            save_path = os.path.join(maps_path, f'seasonal_{sequence_type}_sequences_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if which_plots['seasonal_low_high_flow_map']:
        print("Plotting seasonal low/high flow maps...")
        for flow_type in ['low', 'high']:
            columns = [f'{flow_type}_flow_DJF_trend_per_decade', f'{flow_type}_flow_MAM_trend_per_decade', 
                      f'{flow_type}_flow_JJA_trend_per_decade', f'{flow_type}_flow_SON_trend_per_decade']
            # Check if columns exist
            if not all(col in merged_gdf_natural.columns for col in columns):
                print(f"Warning: Seasonal {flow_type} flow trend columns not found. Skipping seasonal {flow_type} flow map.")
                continue
        data_min = merged_gdf_natural[columns].min().min()
        data_max = merged_gdf_natural[columns].max().max()
        extend = determine_extend(low_high_flow_vmin, low_high_flow_vmax, data_min, data_max)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(hspace=-0, wspace=0)
        for i, (col, ax) in enumerate(zip(columns, axs.ravel())):
            plot_map(ax, bmap, glaciers, iceland_shapefile_color, glaciers_color, xlim, ylim)
            im = merged_gdf_natural.plot(column=col, legend=False, vmin=low_high_flow_vmin, vmax=low_high_flow_vmax, ax=ax, cmap=colormap, s=150)
            pval_col = f'{flow_type}_flow_{col.split("_")[2]}_pval'
            significant_points = merged_gdf_natural[merged_gdf_natural[pval_col] < 0.05]
            ax.plot(significant_points.geometry.x - 100, significant_points.geometry.y, marker='o', markersize=18, markerfacecolor='none', markeredgecolor='k', linestyle='none', lw='0.5')
            catchments.loc[merged_gdf_natural[col].dropna().index].plot(facecolor='none', edgecolor='black', ax=ax, zorder=3, lw=0.25)
            ax.set_title(['Dec-Feb', 'Mar-May', 'Jun-Aug', 'Sep-Nov'][i], y=0.9, fontsize=35)
            add_colorbar(fig, axs[1, 1], colormap, low_high_flow_vmin, low_high_flow_vmax, f'Trend in seasonal {flow_type} flow from %s-%s (%%/decade)' % (start_year, end_year), extend)
            save_path = os.path.join(maps_path, f'seasonal_{flow_type}_flow_trends.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_timeseries(catchments, which_plots, merged_gdf, start_year, end_year, results, valid_data_dict, invalid_data_dict, daily_timeseries_path, annual_trends_path, seasonal_trends_path_mod_ts, monthly_trends_path_mod_ts, annual_mean_flow_path, annual_cv_path, annual_std_path, flashiness_path, sequences_path, baseflow_index_path):
    """
    Plot time series figures for all gauges and metrics.

    This function creates time series plots showing the temporal evolution of various
    streamflow metrics for each gauge. Each plot includes:
    - Scatter points showing annual/seasonal values
    - Trend line (solid for significant trends, dashed for non-significant)
    - Trend magnitude and p-value in the legend
    - Invalid data points shown in light blue if present

    Args:
        catchments (GeoDataFrame): GeoDataFrame containing catchment boundaries
        which_plots (dict): Dictionary specifying which plots to create
        merged_gdf (GeoDataFrame): GeoDataFrame containing gauge metadata
        start_year (int): Start year of analysis period
        end_year (int): End year of analysis period
        results (DataFrame): DataFrame containing trend results
        valid_data_dict (dict): Dictionary of valid data for each gauge and metric
        invalid_data_dict (dict): Dictionary of invalid data for each gauge and metric
        daily_timeseries_path (Path): Directory for daily timeseries plots
        annual_trends_path (Path): Directory for annual trend plots
        seasonal_trends_path_mod_ts (Path): Directory for seasonal trend plots
        monthly_trends_path_mod_ts (Path): Directory for monthly trend plots
        annual_mean_flow_path (Path): Directory for annual mean flow plots
        annual_cv_path (Path): Directory for annual CV plots
        annual_std_path (Path): Directory for annual std plots
        flashiness_path (Path): Directory for flashiness plots
        sequences_path (Path): Directory for sequence plots
        baseflow_index_path (Path): Directory for baseflow index plots
    """
    for gauge in results.index:
        # Annual mean flow
        if which_plots.get('annual_series', False):
            plt.figure(figsize=(10, 6))
            plt.suptitle('Gauge %s, %s %s' % (gauge, merged_gdf.loc[gauge]['river'], merged_gdf.loc[gauge]['name']))
            valid_data = valid_data_dict.get((str(gauge), 'annual'))
            
            if valid_data is not None:
                print(f'Plotting annual series for gauge {gauge}')
                plt.scatter(valid_data.index, valid_data.values, label=None)
                trend_years = pd.date_range(valid_data.index[0], valid_data.index[-1], freq='YE')
                # Use trend and intercept from results
                trend = results.loc[gauge]['annual_avg_flow_trend']
                trend_per_decade = results.loc[gauge]['annual_avg_flow_trend_per_decade']
                intercept = results.loc[gauge]['annual_intercept']
                if results.loc[gauge]['pval'] < 0.05:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept, ls='-', c='r', 
                            label='%s %% per decade, pval = %.3f' % (np.round(trend_per_decade, 1), results.loc[gauge]['pval']))
                else:
                    plt.plot(trend_years, trend * np.arange(len(trend_years)) + intercept, ls='--', c='r', 
                            label='%s %% per decade, pval = %.3f' % (np.round(trend_per_decade, 1), results.loc[gauge]['pval']))
            else:
                invalid_data = invalid_data_dict.get((str(gauge), 'annual'))
                if invalid_data is not None and len(invalid_data) > 0:
                    plt.scatter(invalid_data.index, invalid_data.values, color='lightblue', label='Trend not calculated due to missing data')
                else:
                    plt.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', transform=plt.gca().transAxes)
            
            plt.xlabel('Year')
            plt.ylabel('Yearly avg. flow (m3/s)')
            
            # Adjust tight_layout to avoid compatibility issues
            try:
                plt.tight_layout()
            except UserWarning:
                print("Warning: tight_layout may not work correctly with this figure.")

            # Ensure legend is created only if there are valid labels
            handles, labels = plt.gca().get_legend_handles_labels()
            if labels:
                plt.legend()
                
            save_name = '%s.png' % str(gauge)
            save_path = os.path.join(annual_mean_flow_path, save_name)
            plt.savefig(save_path)
            plt.close()

        # Annual metrics
        for metric, metric_path in zip(
            ['annual_std', 'annual_cv', 'flashiness', 'rising_seq', 'falling_seq', 'baseflow_index'],
            [annual_std_path, annual_cv_path, flashiness_path, sequences_path, sequences_path, baseflow_index_path]
        ):
            if which_plots.get(f'{metric}_series', False):
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                plt.suptitle(f'{metric.replace("_", " ").title()} - Gauge {gauge}, {merged_gdf.loc[gauge]["river"]} {merged_gdf.loc[gauge]["name"]}')
                plot_metric_timeseries(
                    gauge=gauge,
                    metric=metric,
                    valid_data_dict=valid_data_dict,
                    invalid_data_dict=invalid_data_dict,
                    results=results,
                    ylabel=f'{metric.replace("_", " ").title()} (m3/s)',
                    save_path=metric_path,
                    ax=None  # Set to None to trigger saving in plot_metric_timeseries
                )
                plt.tight_layout()
                save_name = f'{gauge}_{metric}.png'
                plt.savefig(os.path.join(metric_path, save_name))
                plt.close()

        if which_plots.get('rising_falling_series', False):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(f'Flow Sequences - Gauge {gauge}, {merged_gdf.loc[gauge]["river"]} {merged_gdf.loc[gauge]["name"]}')
            
            plot_metric_timeseries(
                gauge=gauge,
                metric='rising_seq',
                valid_data_dict=valid_data_dict,
                invalid_data_dict=invalid_data_dict,
                results=results,
                ylabel='Rising Sequences Count',
                save_path=sequences_path,
                ax=ax1
            )
            plot_metric_timeseries(
                gauge=gauge,
                metric='falling_seq',
                valid_data_dict=valid_data_dict,
                invalid_data_dict=invalid_data_dict,
                results=results,
                ylabel='Falling Sequences Count',
                save_path=sequences_path,
                ax=ax2
            )
            
            plt.tight_layout()
            save_path = os.path.join(sequences_path, f'{gauge}_sequences.png')
            plt.savefig(save_path)
            plt.close()

        if which_plots.get('seasonal_series', False):
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                plt.figure()
                valid_data = valid_data_dict.get((str(gauge), season))
                if valid_data is not None:
                    plt.scatter(valid_data.index, valid_data.values, label=None)
                    # Handle both datetime and integer indices
                    if isinstance(valid_data.index, pd.DatetimeIndex):
                        years = valid_data.index.year
                    else:
                        years = valid_data.index
                    trend = results.loc[gauge][f'trend_{season}']
                    intercept = results.loc[gauge][f'intercept_{season}']
                    if results.loc[gauge][f'pval_{season}'] < 0.05:
                        plt.plot(valid_data.index, intercept + trend * (years - years[0]), ls='-', c='r',
                                 label=f'{trend:.3f} m³/s per year, pval = {results.loc[gauge][f"pval_{season}"]:.3f}')
                    else:
                        plt.plot(valid_data.index, intercept + trend * (years - years[0]), ls='--', c='r',
                                 label=f'{trend:.3f} m³/s per year, pval = {results.loc[gauge][f"pval_{season}"]:.3f}')
                else:
                    invalid_data = invalid_data_dict.get((str(gauge), season))
                    if invalid_data is not None and len(invalid_data) > 0:
                        plt.scatter(invalid_data.index, invalid_data.values, color='lightblue', label='Trend not calculated due to missing data')
                    else:
                        plt.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', transform=plt.gca().transAxes)
                plt.title(f'Gauge {gauge} - {season} average flow')
                plt.xlabel('Year')
                plt.ylabel('Seasonal avg. flow (m3/s)')
                plt.legend()
                plt.tight_layout()
                save_name = f'{gauge}_{season}.png'
                save_path = os.path.join(seasonal_trends_path_mod_ts, save_name)
                plt.savefig(save_path)
                plt.close()

        if which_plots.get('seasonal_cv_series', False):
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                plt.figure(figsize=(10, 6))
                valid_data = valid_data_dict.get((str(gauge), f'cv_{season}'))
                if valid_data is not None:
                    plt.scatter(valid_data.index, valid_data.values, label=None)
                    # Handle both datetime and integer indices
                    if isinstance(valid_data.index, pd.DatetimeIndex):
                        years = valid_data.index.year
                    else:
                        years = valid_data.index
                    trend = results.loc[gauge][f'cv_{season}_trend']
                    intercept = results.loc[gauge][f'cv_{season}_intercept']
                    
                    if results.loc[gauge][f'cv_{season}_pval'] < 0.05:
                        plt.plot(valid_data.index, intercept + trend * (years - years[0]), ls='-', c='r',
                                label=f'{trend:.3f} per year, pval = {results.loc[gauge][f"cv_{season}_pval"]:.3f}')
                    else:
                        plt.plot(valid_data.index, intercept + trend * (years - years[0]), ls='--', c='r',
                                label=f'{trend:.3f} per year, pval = {results.loc[gauge][f"cv_{season}_pval"]:.3f}')
                else:
                    invalid_data = invalid_data_dict.get((str(gauge), f'cv_{season}'))
                    if invalid_data is not None and len(invalid_data) > 0:
                        plt.scatter(invalid_data.index, invalid_data.values, color='lightblue', label='Trend not calculated due to missing data')
                    else:
                        plt.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', transform=plt.gca().transAxes)
                
                plt.title(f'Gauge {gauge} - {season} Coefficient of Variation')
                plt.xlabel('Year')
                plt.ylabel('Seasonal CV')
                plt.legend()
                plt.tight_layout()
                
                save_name = f'{gauge}_cv_{season}.png'
                save_path = os.path.join(seasonal_trends_path_mod_ts, save_name)
                plt.savefig(save_path)
                plt.close()

        if which_plots.get('seasonal_sequences_series', False):
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                fig.suptitle(f'Seasonal Flow Sequences - Gauge {gauge}, {merged_gdf.loc[gauge]["river"]} {merged_gdf.loc[gauge]["name"]} - {season}')
                
                # Plot rising sequences
                valid_data_rising = valid_data_dict.get((str(gauge), f'rising_{season}'))
                if valid_data_rising is not None:
                    ax1.scatter(valid_data_rising.index, valid_data_rising.values, label=None)
                    # Handle both datetime and integer indices
                    if isinstance(valid_data_rising.index, pd.DatetimeIndex):
                        years = valid_data_rising.index.year
                    else:
                        years = valid_data_rising.index
                    start_year = years[0]
                    trend = results.loc[gauge][f'rising_{season}_trend']
                    intercept = results.loc[gauge][f'rising_{season}_intercept']
                    pval = results.loc[gauge][f'rising_{season}_pval']
                    trendline = intercept + trend * (years - start_year)
                    if pval < 0.05:
                        ax1.plot(valid_data_rising.index, trendline, ls='-', c='r',
                                label=f'{trend:.2f} counts/year, pval = {pval:.3f}')
                    else:
                        ax1.plot(valid_data_rising.index, trendline, ls='--', c='r',
                                label=f'{trend:.2f} counts/year, pval = {pval:.3f}')
                else:
                    invalid_data = invalid_data_dict.get((str(gauge), f'rising_{season}'))
                    if invalid_data is not None and len(invalid_data) > 0:
                        ax1.scatter(invalid_data.index, invalid_data.values, color='lightblue', label='Trend not calculated due to missing data')
                    else:
                        ax1.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', transform=ax1.transAxes)
                
                ax1.set_title('Rising Sequences')
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Count')
                ax1.legend()
                
                # Plot falling sequences
                valid_data_falling = valid_data_dict.get((str(gauge), f'falling_{season}'))
                if valid_data_falling is not None:
                    ax2.scatter(valid_data_falling.index, valid_data_falling.values, label=None)
                    # Handle both datetime and integer indices
                    if isinstance(valid_data_falling.index, pd.DatetimeIndex):
                        years = valid_data_falling.index.year
                    else:
                        years = valid_data_falling.index
                    start_year = years[0]
                    trend = results.loc[gauge][f'falling_{season}_trend']
                    intercept = results.loc[gauge][f'falling_{season}_intercept']
                    pval = results.loc[gauge][f'falling_{season}_pval']
                    trendline = intercept + trend * (years - start_year)
                    if pval < 0.05:
                        ax2.plot(valid_data_falling.index, trendline, ls='-', c='r',
                                label=f'{trend:.2f} counts/year, pval = {pval:.3f}')
                    else:
                        ax2.plot(valid_data_falling.index, trendline, ls='--', c='r',
                                label=f'{trend:.2f} counts/year, pval = {pval:.3f}')
                else:
                    invalid_data = invalid_data_dict.get((str(gauge), f'falling_{season}'))
                    if invalid_data is not None and len(invalid_data) > 0:
                        ax2.scatter(invalid_data.index, invalid_data.values, color='lightblue', label='Trend not calculated due to missing data')
                    else:
                        ax2.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', transform=ax2.transAxes)
                
                ax2.set_title('Falling Sequences')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Count')
                ax2.legend()
                
                plt.tight_layout()
                save_name = f'{gauge}_sequences_{season}.png'
                save_path = os.path.join(sequences_path, save_name)
                plt.savefig(save_path)
                plt.close()

def plot_metric_timeseries(gauge, metric, valid_data_dict, invalid_data_dict, results, ylabel, save_path, ax=None, plot_trend=True):
    """
    Plot time series for a specific metric at a given gauge.

    This function creates a single time series plot for one metric at one gauge.
    It handles both valid and invalid data points, and includes trend lines when
    appropriate.

    Args:
        gauge (int): Gauge ID
        metric (str): Name of the metric to plot (e.g., 'annual_std', 'flashiness')
        valid_data_dict (dict): Dictionary of valid data points
        invalid_data_dict (dict): Dictionary of invalid data points
        results (DataFrame): DataFrame containing trend results
        ylabel (str): Y-axis label
        save_path (Path): Directory to save the plot
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure
        plot_trend (bool, optional): Whether to plot trend line. Defaults to True
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    
    valid_data = valid_data_dict.get((str(gauge), metric))
    try:
        if valid_data is not None:
            # Plot only valid data points
            ax.scatter(valid_data.index, valid_data.values, label=None)
            
            if plot_trend and not valid_data.empty:
                trend = results.loc[gauge][f'trend_{metric}']  # Raw trend from theilslopes
                intercept = results.loc[gauge][f'intercept_{metric}']  # Intercept from theilslopes
                trend_per_decade = results.loc[gauge][f'trend_{metric}_per_decade']  # For display only
                pval = results.loc[gauge][f'pval_{metric}']
                
                # Create trend line only over the range of valid data
                valid_indices = valid_data.index
                # Handle both datetime and integer indices
                if isinstance(valid_indices, pd.DatetimeIndex):
                    years = valid_indices.year
                else:
                    years = valid_indices
                
                # Calculate trend values using raw trend and intercept from theilslopes
                x_values = np.arange(len(years))  # Use zero-based index as theilslopes does
                trend_values = intercept + trend * x_values
                
                if pval < 0.05:
                    ax.plot(valid_indices, trend_values, ls='-', c='r', 
                           label=f"{trend_per_decade:.1f} % per decade, p={pval:.3f}")
                else:
                    ax.plot(valid_indices, trend_values, ls='--', c='r',
                           label=f"{trend_per_decade:.1f} % per decade, p={pval:.3f}")
        else:
            invalid_data = invalid_data_dict.get((str(gauge), metric))
            if invalid_data is not None:
                ax.scatter(invalid_data.index, invalid_data.values, 
                         color='lightblue', label='Trend not calculated due to missing data')
            else:
                ax.text(0.5, 0.5, 'No data available',
                       horizontalalignment='center',
                       transform=ax.transAxes)
                    
    except Exception as e:
        print(f"Error plotting {metric} for gauge {gauge}: {str(e)}")
        ax.text(0.5, 0.5, 'Error plotting data',
               horizontalalignment='center',
               transform=ax.transAxes)
    
    # Set correct y-axis label based on metric
    if metric == 'flashiness':
        ax.set_ylabel('Flashiness index (unitless)')
    else:
        ax.set_ylabel(ylabel)
    
    ax.set_xlabel('Year')
    if ax.get_legend() is None and len(ax.get_lines()) > 0:  # Only add legend if there are plotted lines
        ax.legend()
    
    if ax is None:  # If we created a new figure
        plt.tight_layout()
        save_name = f'{gauge}_{metric}.png'
        plt.savefig(os.path.join(save_path, save_name))
        plt.close()

def plot_peak_flow_timing_maps(merged_gdf, catchments, peak_results, output_path, start_year, end_year):
    """
    Plot maps showing peak flow timing trends and statistics.

    Creates two maps:
    1. Trend magnitude map showing the rate of change in peak flow timing
    2. Significance map showing locations of significant changes

    Args:
        merged_gdf (GeoDataFrame): GeoDataFrame with gauge locations and timing results
        catchments (GeoDataFrame): GeoDataFrame with catchment boundaries
        peak_results (DataFrame): DataFrame with peak flow timing results
        output_path (Path): Directory to save output figures
        start_year (int): Start year of analysis period
        end_year (int): End year of analysis period

    The maps include:
    - Base map of Iceland
    - Colored points showing trend magnitudes or significance
    - Catchment boundaries
    - Legend explaining the colors
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Plot trend map
    fig, ax = plt.subplots(figsize=(12, 8))
    catchments.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.5)
    
    # Plot points colored by trend
    scatter = merged_gdf.plot(column='slope_peak', ax=ax, 
                            cmap='RdBu_r', legend=True,
                            legend_kwds={'label': 'Trend (days/year)'})
    
    # Add title and labels
    plt.title(f'Peak Flow Timing Trends ({start_year}-{end_year})')
    plt.axis('equal')
    
    # Save figure
    plt.savefig(os.path.join(output_path, 'peak_flow_timing_trends.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot significance map
    fig, ax = plt.subplots(figsize=(12, 8))
    catchments.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.5)
    
    # Create categorical color map for significance
    colors = ['red', 'blue', 'grey']
    categories = ['Significant Increase', 'Significant Decrease', 'No Significant Trend']
    
    # Determine significance categories
    def get_significance_category(row):
        if row['pvalue_peak'] > 0.05:
            return 'No Significant Trend'
        elif row['slope_peak'] > 0:
            return 'Significant Increase'
        else:
            return 'Significant Decrease'
    
    merged_gdf['significance'] = merged_gdf.apply(get_significance_category, axis=1)
    
    # Create legend handles
    legend_elements = [plt.scatter([], [], c=color, label=cat) 
                      for color, cat in zip(colors, categories)]
    
    # Plot points
    for cat, color in zip(categories, colors):
        mask = merged_gdf['significance'] == cat
        if mask.any():
            merged_gdf[mask].plot(ax=ax, color=color, label=cat)
    
    # Add legend
    plt.legend(handles=legend_elements)
    
    # Add title
    plt.title(f'Peak Flow Timing Trend Significance ({start_year}-{end_year})')
    plt.axis('equal')
    
    # Save figure
    plt.savefig(os.path.join(output_path, 'peak_flow_timing_significance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_timing_distributions(timing_stats, output_path):
    """
    Plot timing distributions for all gauges.

    Creates histograms showing the distribution of flow timing metrics
    (peak flow, centroid, or spring freshet) across all gauges.

    Args:
        timing_stats (dict): Dictionary with timing statistics for each gauge
        output_path (Path): Directory to save output figures

    The plots include:
    - Histogram of timing values
    - X-axis showing day of water year
    - Title indicating the timing metric type
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Combine all timing data
    all_data = []
    for gauge, stats in timing_stats.items():
        if 'PeakFlowDay' in stats.columns:  # Peak flow timing
            timing_data = stats['PeakFlowDay']
            timing_type = 'Peak Flow'
        elif 'CentroidDay' in stats.columns:  # Centroid timing
            timing_data = stats['CentroidDay']
            timing_type = 'Centroid'
        else:  # Spring freshet timing
            timing_data = stats['FreshetDay']
            timing_type = 'Spring Freshet'
        
        all_data.extend(timing_data)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_data, bins=50, edgecolor='black')
    plt.xlabel('Day of Water Year')
    plt.ylabel('Frequency')
    plt.title(f'{timing_type} Timing Distribution')
    
    # Save figure
    plt.savefig(os.path.join(output_path, f'{timing_type.lower().replace(" ", "_")}_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_timing_series(timing_data, timing_results, df_raw, df_filled, output_path, gauge_info=None):
    """
    Plot timing series for all gauges.

    Creates time series plots showing how flow timing metrics change over time
    for each gauge.

    Args:
        timing_data (dict): Dictionary with timing data for each gauge
        timing_results (DataFrame): DataFrame with timing trend results
        df_raw (DataFrame): DataFrame with raw streamflow data
        df_filled (DataFrame): DataFrame with filled streamflow data
        output_path (Path): Directory to save output figures
        gauge_info (DataFrame, optional): DataFrame with gauge metadata

    Each plot includes:
    - Scatter points showing annual timing values
    - Trend line (if significant)
    - Gauge information in title
    - Legend with trend magnitude and significance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    for gauge in timing_data.keys():
        # Get gauge data
        gauge_timing = timing_data[gauge]
        
        # Determine timing type based on column names
        if 'PeakFlowDay' in gauge_timing.columns:
            day_col = 'PeakFlowDay'
            date_col = 'PeakFlowDate'
            timing_type = 'Peak Flow'
        elif 'CentroidDay' in gauge_timing.columns:
            day_col = 'CentroidDay'
            date_col = 'CentroidDate'
            timing_type = 'Centroid'
        else:
            day_col = 'FreshetDay'
            date_col = 'FreshetDate'
            timing_type = 'Spring Freshet'
        
        # Get trend data
        slope = timing_results.loc[gauge, f'slope_{timing_type.lower().replace(" ", "_")}']
        pvalue = timing_results.loc[gauge, f'pvalue_{timing_type.lower().replace(" ", "_")}']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot timing points
        ax.scatter(gauge_timing['WaterYear'], gauge_timing[day_col], 
                  color='blue', alpha=0.6, label='Annual Timing')
        
        # Calculate and plot trend line
        years = gauge_timing['WaterYear'].values
        if len(years) > 0:
            # Calculate trend line using mean-centered approach
            mean_year = np.mean(years)
            mean_day = np.mean(gauge_timing[day_col])
            trend_years = np.array([years[0], years[-1]])
            trend_days = mean_day + slope * (trend_years - mean_year)
            
            # Plot trend line
            ax.plot(trend_years, trend_days, color='red', linestyle='--', 
                   label=f'Trend: {slope:.2f} days/year (p={pvalue:.3f})')
        
        # Add title and labels
        title = f'{timing_type} Timing - Gauge {gauge}'
        if gauge_info is not None and gauge in gauge_info.index:
            title += f'\n{gauge_info.loc[gauge, "name"]}'
        plt.title(title)
        plt.xlabel('Water Year')
        plt.ylabel('Day of Water Year')
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(output_path, f'{gauge}_{timing_type.lower().replace(" ", "_")}_timing.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_trendfigs(catchments, which_plots, merged_gdf, start_year, end_year, results, valid_data_dict, invalid_data_dict, daily_timeseries_path, annual_autocorrelation_path, maps_path, raster_trends_path, seasonal_trends_path_mod_ts, annual_trends_path, monthly_trends_path_mod_ts, annual_mean_flow_path, annual_cv_path, annual_std_path, flashiness_path, sequences_path, baseflow_index_path):
    """
    Main plotting function that delegates to specific plotting functions based on configuration.

    This function serves as the central coordinator for all plotting operations. It checks
    the which_plots dictionary to determine which plots to create and calls the appropriate
    specialized plotting functions.

    Args:
        catchments (GeoDataFrame): GeoDataFrame containing catchment boundaries
        which_plots (dict): Dictionary specifying which plots to create
        merged_gdf (GeoDataFrame): GeoDataFrame containing gauge locations and results
        start_year (int): Start year of analysis period
        end_year (int): End year of analysis period
        results (DataFrame): DataFrame containing trend results
        valid_data_dict (dict): Dictionary of valid data for each gauge and metric
        invalid_data_dict (dict): Dictionary of invalid data for each gauge and metric
        daily_timeseries_path (Path): Directory for daily timeseries plots
        annual_autocorrelation_path (Path): Directory for autocorrelation plots
        maps_path (Path): Directory for map plots
        raster_trends_path (Path): Directory for raster trend plots
        seasonal_trends_path_mod_ts (Path): Directory for seasonal trend plots
        annual_trends_path (Path): Directory for annual trend plots
        monthly_trends_path_mod_ts (Path): Directory for monthly trend plots
        annual_mean_flow_path (Path): Directory for annual mean flow plots
        annual_cv_path (Path): Directory for annual CV plots
        annual_std_path (Path): Directory for annual std plots
        flashiness_path (Path): Directory for flashiness plots
        sequences_path (Path): Directory for sequence plots
        baseflow_index_path (Path): Directory for baseflow index plots

    The function handles two main categories of plots:
    1. Maps (if any of these are requested):
       - Annual mean flow
       - Seasonal mean flow
       - Low/high flow
       - Standard deviation
       - CV
       - Flashiness
       - Rising/falling sequences
       - Baseflow index

    2. Time series (if any of these are requested):
       - Annual series
       - Seasonal series
       - Monthly series
       - Low/high flow series
       - Various metric series (std, CV, flashiness, etc.)

    Error handling:
    - Each category of plots is wrapped in a try-except block
    - If an error occurs, a warning is printed with the full traceback
    - The function continues to process other categories even if one fails
    """
    # Plot maps if any map plots are requested
    map_plots = ['annual_map', 'seasonal_map', 'low_high_flow_map', 'annual_std_map', 
                 'annual_cv_map', 'flashiness_map', 'rising_falling_map',
                 'baseflow_index_map', 'seasonal_baseflow_index_map', 'monthly_map']
    
    if any(which_plots.get(plot, False) for plot in map_plots):
        try:
            plot_maps(catchments, which_plots, merged_gdf, start_year, end_year, results, 
                     valid_data_dict, invalid_data_dict, maps_path)
        except Exception as e:
            import traceback
            print(f"Warning: Error while plotting maps: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())

    # Plot timeseries if any timeseries plots are requested
    series_plots = ['annual_series', 'seasonal_series', 'monthly_series', 'low_flow_series',
                    'high_flow_series', 'annual_std_series', 'annual_cv_series',
                    'flashiness_series', 'rising_falling_series', 'baseflow_index_series']
    
    if any(which_plots.get(plot, False) for plot in series_plots):
        try:
            plot_timeseries(catchments, which_plots, merged_gdf, start_year, end_year, results,
                           valid_data_dict, invalid_data_dict, daily_timeseries_path,
                           annual_trends_path, seasonal_trends_path_mod_ts,
                           monthly_trends_path_mod_ts, annual_mean_flow_path, annual_cv_path, 
                           annual_std_path, flashiness_path, sequences_path, baseflow_index_path)
        except Exception as e:
            import traceback
            print(f"Warning: Error while plotting timeseries: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())