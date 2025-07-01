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
    periods = ['1973_2023', '1993_2023']
    
    # Create output directory for heatmaps
    savepath = OUTPUT_DIR / 'trend_summary_heatmaps'
    savepath.mkdir(parents=True, exist_ok=True)
    
    # Read results files
    results_1973 = pd.read_csv(OUTPUT_DIR / 'results_lamah_data' / f'results_{periods[0]}.csv', sep=';', index_col=0)
    results_1993 = pd.read_csv(OUTPUT_DIR / 'results_lamah_data' / f'results_{periods[1]}.csv', sep=';', index_col=0)
    
    # Read catchment attributes
    catchment_attrs = gpd.read_file(CATCHMENT_ATTRIBUTES_FILE)
    catchment_attrs = catchment_attrs.set_index('id')
    catchment_attrs_selected = catchment_attrs[['g_frac', 'baseflow_index_ladson']]
    
    # Convert index to string to match results index
    results_1973.index = results_1973.index.astype(str)
    results_1993.index = results_1993.index.astype(str)
    catchment_attrs_selected.index = catchment_attrs_selected.index.astype(str)
    
    # Merge attributes with results
    results_1973 = results_1973.join(catchment_attrs_selected)
    results_1993 = results_1993.join(catchment_attrs_selected)
    
    # Define metrics and their columns
    annual_metrics = {
        'Annual Flow': [
            ('annual_avg_flow_trend_per_decade', 'pval', 'Annual'),
            ('trend_JJA_per_decade', 'pval_JJA', 'Summer (JJA)'),
            ('trend_JAS_per_decade', 'pval_JAS', 'Summer (JAS)'),
            ('trend_SON_per_decade', 'pval_SON', 'Fall (SON)'),
            ('trend_DJF_per_decade', 'pval_DJF', 'Winter (DJF)'),
            ('trend_MAM_per_decade', 'pval_MAM', 'Spring (MAM)')
        ],
        'Flow Variability': [
            ('trend_annual_std_per_decade', 'pval_annual_std', 'Annual St.Dev.'),
            ('std_JJA_trend_per_decade', 'std_JJA_pval', 'Summer (JJA) St.Dev.'),
            ('std_JAS_trend_per_decade', 'std_JAS_pval', 'Summer (JAS) St.Dev.'),
            ('std_SON_trend_per_decade', 'std_SON_pval', 'Fall St.Dev.'),
            ('std_DJF_trend_per_decade', 'std_DJF_pval', 'Winter St.Dev.'),
            ('std_MAM_trend_per_decade', 'std_MAM_pval', 'Spring St.Dev.')
        ],
        'Coefficient of Variation': [
            ('trend_annual_cv_per_decade', 'pval_annual_cv', 'Annual CV'),
            ('cv_JJA_trend_per_decade', 'cv_JJA_pval', 'Summer (JJA) CV'),
            ('cv_JAS_trend_per_decade', 'cv_JAS_pval', 'Summer (JAS) CV'),
            ('cv_SON_trend_per_decade', 'cv_SON_pval', 'Fall CV'),
            ('cv_DJF_trend_per_decade', 'cv_DJF_pval', 'Winter CV'),
            ('cv_MAM_trend_per_decade', 'cv_MAM_pval', 'Spring CV')
        ],
        'Flashiness Index': [
            ('trend_flashiness_per_decade', 'pval_flashiness', 'Annual Flashiness'),
            ('flashiness_JJA_trend_per_decade', 'flashiness_JJA_pval', 'Summer (JJA) Flashiness'),
            ('flashiness_JAS_trend_per_decade', 'flashiness_JAS_pval', 'Summer (JAS) Flashiness'),
            ('flashiness_SON_trend_per_decade', 'flashiness_SON_pval', 'Fall Flashiness'),
            ('flashiness_DJF_trend_per_decade', 'flashiness_DJF_pval', 'Winter Flashiness'),
            ('flashiness_MAM_trend_per_decade', 'flashiness_MAM_pval', 'Spring Flashiness')
        ],
        'Baseflow Index': [
            ('trend_baseflow_index_per_decade', 'pval_baseflow_index', 'Annual BFI'),
            ('baseflow_index_JJA_trend_per_decade', 'baseflow_index_JJA_pval', 'Summer (JJA) BFI'),
            ('baseflow_index_JAS_trend_per_decade', 'baseflow_index_JAS_pval', 'Summer (JAS) BFI'),
            ('baseflow_index_SON_trend_per_decade', 'baseflow_index_SON_pval', 'Fall BFI'),
            ('baseflow_index_DJF_trend_per_decade', 'baseflow_index_DJF_pval', 'Winter BFI'),
            ('baseflow_index_MAM_trend_per_decade', 'baseflow_index_MAM_pval', 'Spring BFI')
        ]
    }
    
    # Generate heatmaps for each metric
    for metric_name, columns in annual_metrics.items():
        print(f"Generating heatmap for {metric_name}...")
        plot_trend_heatmaps(results_1973, results_1993, columns, metric_name, savepath)

if __name__ == "__main__":
    main() 