from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import os
import geopandas as gpd
from config import (
    OUTPUT_DIR,
    CATCHMENT_ATTRIBUTES_CSV,
    GAUGES_SHAPEFILE,
    MANUSCRIPT_FIGURES_PATH
)
plt.rcParams['font.family'] = 'Arial' 

# Here we calculate anomalies with respect to the mean of the period 2000-2010

# Define path to save plots
save_path = MANUSCRIPT_FIGURES_PATH

# Read the catchment characteristics - Extract area_calc and human influence
catchments_chara = pd.read_csv(CATCHMENT_ATTRIBUTES_CSV, sep=';')
catchments_chara = catchments_chara.set_index('id')

# Load gauge names from shapefile
gauges_gdf = gpd.read_file(GAUGES_SHAPEFILE)
gauges_gdf['id'] = gauges_gdf['id'].astype(int)
gauges_gdf = gauges_gdf.set_index('id')

# Read annual streamflow averages (already calculated with proper water year filtering)
path_to_streamflow = OUTPUT_DIR / "annual_streamflow_averages_longterm.csv"
df_streamflow_all_adj_ann = pd.read_csv(path_to_streamflow, index_col=0, parse_dates=True)

# Drop gauge with ID 9 (Syðri-Bægisá River, due to inhomogeneity in the series)
if '9' in df_streamflow_all_adj_ann.columns:
    df_streamflow_all_adj_ann = df_streamflow_all_adj_ann.drop(columns='9')
    print("Dropped gauge 9 (Syðri-Bægisá) due to inhomogeneity")
else:
    print("Gauge 9 not found in dataset (may have been filtered out earlier)")

# Drop gauge 43 (Jökulsá á Dal) due to gauge relocation in 2007 changing catchment area
# from 1964 km2 to 1662 km2, which creates an artificial reduction in the anomaly plot
if '43' in df_streamflow_all_adj_ann.columns:
    df_streamflow_all_adj_ann = df_streamflow_all_adj_ann.drop(columns='43')
    print("Dropped gauge 43 (Jökulsá á Dal) due to gauge relocation in 2007")
else:
    print("Gauge 43 not found in dataset (may have been filtered out earlier)")

# Define the fixed reference period for computing the mean
reference_period = ('2000-10-01', '2010-09-30')
reference_data = df_streamflow_all_adj_ann.loc[reference_period[0]:reference_period[1]]

# Filter out gauges with less than 8 years of data during reference period
min_years_in_reference = 8
gauges_to_keep = []
gauges_removed_reference = []

for col in reference_data.columns:
    valid_years = reference_data[col].notna().sum()
    if valid_years >= min_years_in_reference:
        gauges_to_keep.append(col)
    else:
        gauges_removed_reference.append((col, valid_years))

print(f"Filtering gauges based on reference period ({reference_period[0]} to {reference_period[1]}):")
print(f"  Kept {len(gauges_to_keep)} gauges (≥{min_years_in_reference} years in reference period)")
print(f"  Removed {len(gauges_removed_reference)} gauges (<{min_years_in_reference} years in reference period)")

if gauges_removed_reference:
    print(f"\n  Removed gauges:")
    for gauge_id, years in sorted(gauges_removed_reference, key=lambda x: int(x[0])):
        print(f"    Gauge {gauge_id}: {years} years in reference period")

# Filter the dataframe
df_streamflow_all_adj_ann = df_streamflow_all_adj_ann[gauges_to_keep]
reference_data = reference_data[gauges_to_keep]

# Calculate mean for each gauge over the fixed period
reference_means = reference_data.mean()

window_size = 5  # Define the rolling window size
min_periods = 4  # Allow up to 1 missing year in the 5-year window

df_streamflow_all_adj_ann_rolling = df_streamflow_all_adj_ann.rolling(
    window=window_size, center=True, min_periods=min_periods
).mean()

# Function to generate gauge labels dynamically
def get_gauge_label(gauge_id):
    """Generate a label for a gauge ID using river and station name."""
    if gauge_id in gauges_gdf.index:
        river = gauges_gdf.loc[gauge_id, 'river'] if 'river' in gauges_gdf.columns else ''
        name = gauges_gdf.loc[gauge_id, 'name'] if 'name' in gauges_gdf.columns else ''
        # Use river name if available, otherwise use station name
        if river and name:
            return f"{river} ({gauge_id})"
        elif river:
            return f"{river} ({gauge_id})"
        elif name:
            return f"{name} ({gauge_id})"
    return f"Gauge {gauge_id}"

# Define sorting by glaciation percentage
sortby = 'glac_fra'
fontsize = 18
title_fontsize = 22
xtick_labelsize = 18 # Size for the x-axis tick labels

# Calculate deviations from the reference period mean (in percentages)
deviations_percent = ((df_streamflow_all_adj_ann_rolling - reference_means) / reference_means) * 100

deviations_percent = deviations_percent['1949-10-01':'2023-09-30']

# Create diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True).reversed()
cmap.set_bad(color='#404040')  # Set bad color for NaN values

# Sort the streamflow gauges by glaciation percentage
listi = catchments_chara[sortby].loc[[int(i) for i in deviations_percent.columns]].sort_values().index.tolist()
listi_str = [str(i) for i in listi]
deviations_percent = deviations_percent[listi_str]

# Create figure
#fig, ax = plt.subplots(figsize=(12, 8))
fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

# Plot heatmap
im = ax.imshow(deviations_percent.T, cmap=cmap, aspect='auto', vmin=-30, vmax=30, interpolation='none')

# Modify x-tick positions and labels to show only specific years divisible by 5
selected_years = deviations_percent.index.year
selected_years = selected_years[selected_years % 10 == 0]  # Show only years divisible by 5

# Get positions corresponding to `selected_years`
xtick_positions = np.where(np.isin(deviations_percent.index.year, selected_years))[0]
xtick_labels = [str(year) for year in selected_years]

# Set x-ticks and labels only for selected years
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, fontsize=xtick_labelsize)

# Set y-ticks for Streamflow Gauge ID
ytick_positions = np.arange(len(deviations_percent.columns))
listi_int = [int(i) for i in listi_str]
ytick_labels = [get_gauge_label(i) for i in listi_int]

ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels, fontsize=fontsize)
# ax.set_ylabel('Streamflow Gauge ID', fontsize=fontsize)

# Create a secondary y-axis for glaciation percentages
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())

# Get glaciation values corresponding to the streamflow gauge IDs
glaciation_values = 100 * catchments_chara.loc[[int(i) for i in deviations_percent.columns]]['glac_fra'].sort_values()
glaciation_values = glaciation_values.astype(int)

# Set y-ticks for glaciation values
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(glaciation_values, fontsize=fontsize)

# Adjust the labelpad to bring the label closer to the heatmap
ax2.set_ylabel('Watershed glaciation (%)', fontsize=fontsize, labelpad=2)  # Decrease labelpad to move the label closer

# Adjust colorbar positioning and extend both sides
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend='both')  # Set extend='both'
cbar.set_label('Anomalies relative to 2000–2010 mean (%)', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

# Set the title and labels
plt.title('Annual streamflow anomalies by watershed across Iceland', fontsize=title_fontsize)
ax.set_xlabel('Water year', fontsize=fontsize)

# Save the figure as PDF (with _longterm suffix to avoid overwriting)
pdf_save_path = os.path.join(save_path, 'Figure2_streamflow_anomalies_longterm_.pdf')
plt.savefig(pdf_save_path, dpi=300, format='pdf', bbox_inches='tight')

# Save the figure as PNG (with _longterm suffix to avoid overwriting)
png_save_path = os.path.join(save_path, 'Figure2_streamflow_anomalies_longterm_.png')
plt.savefig(png_save_path, dpi=300, format='png', bbox_inches='tight')

# Display the plot
plt.show()
