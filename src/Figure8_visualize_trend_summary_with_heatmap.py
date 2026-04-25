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
    CATCHMENT_ATTRIBUTES_FILE,
    GAUGES_TO_KEEP
)


def _win_long_path_file(path: str) -> str:
    """Normalize a file path for writing on Windows so paths can exceed the MAX_PATH (260) limit."""
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    p = os.path.normpath(path)
    if p.startswith("\\\\") and not p.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def generate_heatmap_row(df, trend_col, pval_col, glaciation_threshold=0.05):
    """Generate heatmap data for a single row."""
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

def generate_heatmap_data_with_glaciation(df, columns, glaciation_threshold=0.05):
    """Generate heatmap data with glaciated basin counts."""
    heatmap_data = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    annotations = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}

    for trend_col, pval_col, label in columns:
        heatmap_row, annotation_row = generate_heatmap_row(df, trend_col, pval_col, glaciation_threshold)
        
        for key in heatmap_data:
            heatmap_data[key].append(heatmap_row[key])
            annotations[key].append(annotation_row[key])

    # Use dynamic index based on column labels
    index_labels = [label for _, _, label in columns]
    return pd.DataFrame(heatmap_data, index=index_labels), pd.DataFrame(annotations, index=index_labels)

def generate_annual_flow_heatmap_data(df_annual, df_seasonal, columns, glaciation_threshold=0.05):
    """Generate heatmap data for Annual Flow with different filtering for annual vs seasonal rows.
    
    Uses df_annual (includes GAUGES_TO_KEEP) for the 'Annual' row, and df_seasonal 
    (excludes strongly influenced gauges) for seasonal rows.
    """
    heatmap_data = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}
    annotations = {'Positive': [], 'Negative': [], 'Positive significant': [], 'Negative significant': []}

    for trend_col, pval_col, label in columns:
        # Use df_annual for 'Annual' row, df_seasonal for all other rows
        if label == 'Annual':
            df = df_annual
        else:
            df = df_seasonal
            
        heatmap_row, annotation_row = generate_heatmap_row(df, trend_col, pval_col, glaciation_threshold)
        
        for key in heatmap_data:
            heatmap_data[key].append(heatmap_row[key])
            annotations[key].append(annotation_row[key])

    # Use dynamic index based on column labels
    index_labels = [label for _, _, label in columns]
    return pd.DataFrame(heatmap_data, index=index_labels), pd.DataFrame(annotations, index=index_labels)

def plot_trend_heatmaps(
    df_1973,
    df_1993,
    columns,
    metric_name,
    savepath,
    *,
    panel_letters: tuple[str, str] = ("a", "b"),
    title_entity: str | None = None,
):
    """Plot heatmaps for a specific metric.

    ``metric_name`` is used in output filenames. If ``title_entity`` is set, it is
    used in the subplot titles (e.g. "temperature"); otherwise ``metric_name`` is used.
    ``panel_letters`` are the two subfigure labels (default ``a, b``).
    """
    title_word = title_entity if title_entity is not None else metric_name
    os.makedirs(savepath, exist_ok=True)
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
    axes[0].set_title(
        f'{panel_letters[0]}) Trends in {title_word}, 1973-2023', fontsize=22
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot second heatmap (1993-2023)
    sns.heatmap(df_1993_heatmap, 
                annot=annotations_1993, 
                fmt='', 
                cmap='Greens', 
                ax=axes[1], 
                cbar_kws={'label': 'Number of cases'})
    axes[1].set_title(
        f'{panel_letters[1]}) Trends in {title_word}, 1993-2023', fontsize=22
    )
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()
    
    # Save figures (use long-path form on Windows so long OneDrive + repo paths can exceed 260 chars)
    out_png = _win_long_path_file(
        os.path.join(savepath, f'trends_summary_heatmap_{metric_name.lower().replace(" ", "_")}.png')
    )
    out_pdf = _win_long_path_file(
        os.path.join(savepath, f'trends_summary_heatmap_{metric_name.lower().replace(" ", "_")}.pdf')
    )
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

def plot_annual_flow_heatmaps(df_1973_annual, df_1973_seasonal, df_1993_annual, df_1993_seasonal, 
                               columns, metric_name, savepath):
    """Plot heatmaps for Annual Flow metric with different filtering for annual vs seasonal rows.
    
    Uses df_annual (includes GAUGES_TO_KEEP) for the 'Annual' row, and df_seasonal 
    (excludes strongly influenced gauges) for seasonal rows.
    """
    # Generate heatmap data and annotations
    df_1973_heatmap, annotations_1973 = generate_annual_flow_heatmap_data(
        df_1973_annual, df_1973_seasonal, columns)
    df_1993_heatmap, annotations_1993 = generate_annual_flow_heatmap_data(
        df_1993_annual, df_1993_seasonal, columns)

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
    out_png = _win_long_path_file(
        os.path.join(savepath, f'trends_summary_heatmap_{metric_name.lower().replace(" ", "_")}.png')
    )
    out_pdf = _win_long_path_file(
        os.path.join(savepath, f'trends_summary_heatmap_{metric_name.lower().replace(" ", "_")}.pdf')
    )
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
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
    catchment_attrs_selected = catchment_attrs[['g_frac', 'baseflow_index_ladson', 'degimpact']]
    
    # Convert index to string to match results index
    results_1973.index = results_1973.index.astype(str)
    results_1993.index = results_1993.index.astype(str)
    catchment_attrs_selected.index = catchment_attrs_selected.index.astype(str)
    
    # Merge attributes with results
    results_1973 = results_1973.join(catchment_attrs_selected)
    results_1993 = results_1993.join(catchment_attrs_selected)
    
    # Filter out strongly influenced gauges for most metrics (degimpact != 's')
    natural_mask_1973 = results_1973['degimpact'] != 's'
    natural_mask_1993 = results_1993['degimpact'] != 's'
    
    results_1973_natural = results_1973.loc[natural_mask_1973]
    results_1993_natural = results_1993.loc[natural_mask_1993]
    
    # For Annual Flow metrics, include GAUGES_TO_KEEP (7 and 102) despite strong influence
    # because upstream reservoirs don't significantly alter total annual flows
    gauges_to_keep_str = [str(g) for g in GAUGES_TO_KEEP]
    annual_mask_1973 = (results_1973['degimpact'] != 's') | (results_1973.index.isin(gauges_to_keep_str))
    annual_mask_1993 = (results_1993['degimpact'] != 's') | (results_1993.index.isin(gauges_to_keep_str))
    
    results_1973_annual = results_1973.loc[annual_mask_1973]
    results_1993_annual = results_1993.loc[annual_mask_1993]
    
    # For low/high flow, use the same filtering as in plotting.py:
    # Start with natural (degimpact != 's'), then drop gauge IDs 15 and 48
    results_1973_lowhigh = results_1973_natural.drop(['15', '48'], errors='ignore')
    results_1993_lowhigh = results_1993_natural.drop(['15', '48'], errors='ignore')
    
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
        ],
        'High and Low Flows': [
            ('low_flow_trend_per_decade', 'low_flow_pval', 'Low Flow (Q10)'),
            ('high_flow_trend_per_decade', 'high_flow_pval', 'High Flow (Q90)')
        ]
    }
    
    # Generate heatmaps for each metric
    for metric_name, columns in annual_metrics.items():
        print(f"Generating heatmap for {metric_name}...")
        # For low/high flows, use only uninfluenced gauges (degimpact == 'u')
        # For Annual Flow: include GAUGES_TO_KEEP (7 and 102) for annual row only, 
        #                  exclude them for seasonal rows
        # For other metrics, use natural (non-strongly-influenced) gauges
        if metric_name == 'High and Low Flows':
            plot_trend_heatmaps(results_1973_lowhigh, results_1993_lowhigh, columns, metric_name, savepath)
        elif metric_name == 'Annual Flow':
            plot_annual_flow_heatmaps(results_1973_annual, results_1973_natural,
                                      results_1993_annual, results_1993_natural,
                                      columns, metric_name, savepath)
        else:
            plot_trend_heatmaps(results_1973_natural, results_1993_natural, columns, metric_name, savepath)

if __name__ == "__main__":
    main() 