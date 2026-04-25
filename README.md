# iceland-hydro-trends

This repository contains the code used for the paper:

**"Understanding changes in Iceland’s streamflow dynamics in response to climate change"**  
*Published in Hydrology and Earth System Sciences, 2026.*

---

## Overview

This project analyzes long-term streamflow trends across Iceland using the LamaH-Ice dataset. It includes scripts to compute streamflow metrics, assess trends, and relate those trends to catchment characteristics and climate drivers.

---

## Getting Started

### 1. Download the LamaH-Ice Dataset

Download the **daily version** of the LamaH-Ice dataset (`lamah_ice.zip`) from HydroShare:

🔗 [https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/](https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/)

Unzip the file and make note of the path to the extracted data.

---

### 2. Create a Conda Environment

Create the environment from `environment.yml` (conda packages plus `pip` extras such as `pymannkendall` and `openpyxl`):

```bash
conda env create -f environment.yml
conda activate iceland-hydro-trends
```

To refresh after pulling updates: `conda env update -f environment.yml --prune`

---

### 3. Configure Paths

Edit the `config.py` file in the `src/` directory to set:

- The path to the extracted **LamaH-Ice dataset** (`LAMAH_ICE_BASE_PATH`)
- The output directory for results and figures (`OUTPUT_DIR`; default is `paper_repro_output/` in the repo)
- `START_YEAR` and `END_YEAR` for the period `main.py` and other scripts use. If reproducing results from the paper, run `main.py` **twice**: with `START_YEAR` = 1973 and with `START_YEAR` = 1993 (and `END_YEAR` = 2023), changing `config.py` between runs.
- **Bundled basemaps:** `ICELAND_SHAPEFILE` and `GLACIER_SHAPEFILE` point to `island_isn93.shp` and `2019_glacier_outlines.shp` under the repo’s `data/` folder (include all sidecar files: `.shp`, `.shx`, `.dbf`, `.prj`, etc.).

`OUTPUT_DIR` is where most tables and model outputs go. Some figure scripts write to `manuscript_figures/` (`MANUSCRIPT_FIGURES_PATH` under the same output root).

---

### 4. Run the Scripts

Navigate to the `src/` directory:

```bash
cd src
```

#### Streamflow data order (important)

These steps must run in this order:

1. **`pre_process_streamflow_measurements_from_LamaH_Ice.py`** — builds `paper_repro_output/cleaned_streamflow_data/cleaned_streamflow_data.csv` from LamaH daily discharge files.
2. **`calculate_annual_and_seasonal_averages_for_longterm_analysis.py`** — reads that cleaned CSV, optionally merges `data/Jokulsa_a_dal_river_longterm_series.csv` for gauge 43, and writes long-term annual/seasonal average tables (used with the climate-index analysis). **Do not run this before pre-processing.**
3. **`main.py`** — full streamflow trend analysis and maps; it also reads the cleaned CSV. For paper periods, **run it twice** with `START_YEAR` = 1973 and then 1993 in `config.py` (and matching `END_YEAR`).

#### Figures 2 and 3 (Figure 1 is from a notebook in `notebooks/`)

Use the long-term averages from step 2 where needed for indices. Typical order:

| Script | Description |
|--------|-------------|
| `plot_Figure2_raster_anomalies.py` | Figure 2 |
| `calculate_climate_indices_correlation_with_streamflow.py` | Prepares data for Figure 3 |
| `plot_climate_indices_correlation_analysis_AO_NAO.py` | Figure 3 |

**Meteorological trends (outputs used by Figure 4, Figure 5, and met heatmap summaries)**  
First compute merged result pickles from LamaH-Ice daily met CSVs and the per-basin snowfall data under `data/` (see `config` / `data/.gitignore`). Then run the figure scripts (defaults read those pickles from `OUTPUT_DIR`).

| Script | Description |
|--------|-------------|
| `calculate_trends_in_meteorological_vars.py` | **Computes** `merged_results_dict_1973-2023.pkl` and `merged_results_dict_1993-2023.pkl` (met variables, incl. rain/snow from the snowfall pickle) |
| `plot_annual_meteorological_trends_figure.py` | **Plots** annual met trends (Figure 4); default output under `manuscript_figures/` |
| `plot_seasonal_meteorological_trends_2periods.py` | **Plots** seasonal met trends (Figure 5); default output under `manuscript_figures/` |
| `plot_meteorological_trends_heatmap_summary.py` | Optional: count/summary heatmaps of met trends (reads the same merged pickles) |

**Other analyses**

| Script | Description |
|--------|-------------|
| `plot_lowhigh_flows_2x2.py` | High and low flow trends (Figure 10) |
| `Figure8_visualize_trend_summary_with_heatmap.py` | Figure 8: Streamflow trend summary heatmaps |
| `trend_correlation_analysis.py` | Correlates streamflow trends with catchment and met trends |
| `visualize_trend_correlations.py` | Correlation heatmaps and maps from trend correlations |
| `calculate_trends_in_streamflow_timing_metrics.py` | Trends in streamflow timing (e.g. freshet, peak timing) |
| `timing_metric_trend_correlation_analysis.py` | Timing trends vs. climate / catchment |
| `visualize_timing_trends_with_heatmap.py` | Heatmaps of timing-trend summaries |
| `create_manuscript_tables.py` | Tables for manuscript / supplement |

---

## Notebooks

The `notebooks/` folder contains Jupyter notebooks used to generate the rest of the figures for the manuscript. These are primarily for visualization and post-processing. Core computations are handled in the scripts listed above. The conda environment includes `jupyter`, `ipykernel`, and `xarray` for those notebooks.

---

## License

This project is open-source under the MIT License.

---

## Contact

For questions, please contact [@hhelgason](https://github.com/hhelgason).
