import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
import os
import geopandas as gpd
from config import (
    OUTPUT_DIR,
    CATCHMENT_ATTRIBUTES_FILE
)

def generate_heatmap_data_with_glaciation(df, columns, glaciation_threshold=0.05):
    """Generate heatmap data with glaciated basin counts."""
    heatmap_data = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    annotations = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}

    for trend_col, pval_col, label in columns:
        total_cases = df[trend_col].notna().sum()

        # General counts
        positive_cases = (df[trend_col].dropna() > 0).sum()
        negative_cases = (df[trend_col].dropna() < 0).sum()
        positive_significant = ((df[trend_col].dropna() > 0) & (df[pval_col].dropna() < 0.05)).sum()
        negative_significant = ((df[trend_col].dropna() < 0) & (df[pval_col].dropna() < 0.05)).sum()

        # Glaciated basin counts
        positive_glaciated = ((df[trend_col] > 0) & (df['g_frac'] > glaciation_threshold)).sum()
        negative_glaciated = ((df[trend_col] < 0) & (df['g_frac'] > glaciation_threshold)).sum()
        positive_significant_glaciated = ((df[trend_col] > 0) & (df[pval_col] < 0.05) & (df['g_frac'] > glaciation_threshold)).sum()
        negative_significant_glaciated = ((df[trend_col] < 0) & (df[pval_col] < 0.05) & (df['g_frac'] > glaciation_threshold)).sum()

        # Append general counts
        heatmap_data['Positive'].append(positive_cases)
        heatmap_data['Negative'].append(negative_cases)
        heatmap_data['Positive significant'].append(positive_significant)
        heatmap_data['Negative significant'].append(negative_significant)

        # Append annotations with glaciated basin counts
        annotations['Positive'].append(f"{positive_cases} ({positive_glaciated})")
        annotations['Negative'].append(f"{negative_cases} ({negative_glaciated})")
        annotations['Positive significant'].append(f"{positive_significant} ({positive_significant_glaciated})")
        annotations['Negative significant'].append(f"{negative_significant} ({negative_significant_glaciated})")

    # Use dynamic index based on column labels
    index_labels = [label for _, _, label in columns]
    return pd.DataFrame(heatmap_data, index=index_labels), pd.DataFrame(annotations, index=index_labels)

def plot_trend_heatmaps(df_1973, df_1993, columns, metric_name, savepath):
    """Plot heatmaps for a specific metric."""
    # Generate heatmap data and annotations
    df_1973_heatmap, annotations_1973 = generate_heatmap_data_with_glaciation(df_1973, columns)
    df_1993_heatmap, annotations_1993 = generate_heatmap_data_with_glaciation(df_1993, columns)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot first heatmap (1973-2023)
    sns.heatmap(df_1973_heatmap, 
                annot=annotations_1973, 
                fmt='', 
                cmap='Blues', 
                ax=axes[0], 
                cbar_kws={'label': 'Number of cases'})
    axes[0].set_title(f'a) Trends in {metric_name}, 1973-2023', fontsize=22)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot second heatmap (1993-2023)
    sns.heatmap(df_1993_heatmap, 
                annot=annotations_1993, 
                fmt='', 
                cmap='Greens', 
                ax=axes[1], 
                cbar_kws={'label': 'Number of cases'})
    axes[1].set_title(f'b) Trends in {metric_name}, 1993-2023', fontsize=22)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()
    
    # Save figures
    plt.savefig(os.path.join(savepath, f'trends_summary_heatmap_{metric_name.lower().replace(" ", "_")}.png'), dpi=300)
    plt.savefig(os.path.join(savepath, f'trends_summary_heatmap_{metric_name.lower().replace(" ", "_")}.pdf'), dpi=300)
    plt.close()

def main():
    # Set style parameters
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 20
    rcParams['axes.titlesize'] = 20
    rcParams['axes.labelsize'] = 18
    rcParams['xtick.labelsize'] = 18
    rcParams['ytick.labelsize'] = 18
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Define paths and periods
    timing_analysis_dir = OUTPUT_DIR / 'timing_analysis'
    periods = ['1973_2023', '1993_2023']
    
    # Create output directory for heatmaps
    savepath = timing_analysis_dir / 'trend_summary_heatmaps_ts_mk'
    savepath.mkdir(parents=True, exist_ok=True)
    
    # Read timing results files for both periods
    results_1973 = {
        'freshet': pd.read_csv(timing_analysis_dir / '1973_2023' / 'freshet_trends_1973_2023_ts_mk.csv'),
        'centroid': pd.read_csv(timing_analysis_dir / '1973_2023' / 'centroid_trends_1973_2023_ts_mk.csv'),
        'peak': pd.read_csv(timing_analysis_dir / '1973_2023' / 'peak_flow_trends_1973_2023_ts_mk.csv')
    }
    
    results_1993 = {
        'freshet': pd.read_csv(timing_analysis_dir / '1993_2023' / 'freshet_trends_1993_2023_ts_mk.csv'),
        'centroid': pd.read_csv(timing_analysis_dir / '1993_2023' / 'centroid_trends_1993_2023_ts_mk.csv'),
        'peak': pd.read_csv(timing_analysis_dir / '1993_2023' / 'peak_flow_trends_1993_2023_ts_mk.csv')
    }
    
    # Read catchment attributes for glacier fraction
    catchment_attrs = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)
    catchment_attrs = catchment_attrs.set_index('id')
    catchment_attrs_selected = catchment_attrs[['g_frac']]
    
    # Add glacier fraction to results
    for period_results in [results_1973, results_1993]:
        for metric_results in period_results.values():
            metric_results['g_frac'] = catchment_attrs_selected.loc[metric_results['Gauge'].astype(int)].g_frac.values
    
    # Define metrics and their columns
    timing_metrics = {
        'Streamflow Timing': [
            ('Trend (days/decade)', 'Mann-Kendall p-value', 'Freshet Timing'),
            ('Trend (days/decade)', 'Mann-Kendall p-value', 'Centroid Timing'),
            ('Trend (days/decade)', 'Mann-Kendall p-value', 'Peak Flow Timing')
        ]
    }
    
    # Generate heatmaps for timing metrics
    for metric_name, columns in timing_metrics.items():
        print(f"Generating heatmap for {metric_name}...")
        
        # Combine results for each timing metric into single dataframes
        df_1973_combined = pd.DataFrame()
        df_1993_combined = pd.DataFrame()
        
        for (trend_col, pval_col, label), metric_key in zip(columns, ['freshet', 'centroid', 'peak']):
            df_1973_combined = pd.concat([
                df_1973_combined,
                results_1973[metric_key][['Gauge', trend_col, pval_col, 'g_frac']].rename(
                    columns={trend_col: f'{label} {trend_col}', pval_col: f'{label} {pval_col}'}
                )
            ])
            
            df_1993_combined = pd.concat([
                df_1993_combined,
                results_1993[metric_key][['Gauge', trend_col, pval_col, 'g_frac']].rename(
                    columns={trend_col: f'{label} {trend_col}', pval_col: f'{label} {pval_col}'}
                )
            ])
        
        # Create columns list for heatmap
        heatmap_columns = [
            (f'{label} {trend_col}', f'{label} {pval_col}', label)
            for trend_col, pval_col, label in columns
        ]
        
        plot_trend_heatmaps(df_1973_combined, df_1993_combined, heatmap_columns, metric_name, savepath)
        print(f"Saved heatmap for {metric_name} to {savepath}")

if __name__ == "__main__":
    main() 