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

Create the environment from `environment.yml` (see also `requirements.txt` if you use `pip`):

```bash
conda env create -f environment.yml
conda activate iceland-hydro-trends
```

---

### 3. Configure Paths

Edit the `config.py` file in the `src/` directory to set:

- The path to the extracted **LamaH-Ice dataset**
- The output directory for results and figures
- Specify `START_YEAR` and `END_YEAR`. If reproducing results from the paper, first run `main.py` (see below) with `START_YEAR = 1973`, then again with `START_YEAR = 1993` (in `config.py`).

`OUTPUT_DIR` in `config` (by default `paper_repro_output/` inside the repo) is where most tables and model outputs go. Some figure scripts also write to `manuscript_figures/` under that same output root (`MANUSCRIPT_FIGURES_PATH`).

---

### 4. Run the Scripts

Navigate to the `src/` directory:

```bash
cd src
```

**Figures 2 and 3** (Figure 1 is produced from a notebook in `notebooks/`)

| Script | Description |
|--------|-------------|
| `calculate_annual_and_seasonal_averages_for_longterm_analysis.py` | Long-term streamflow means (inputs for the climate–indices work) |
| `plot_Figure2_raster_anomalies.py` | Figure 2 |
| `calculate_climate_indices_correlation_with_streamflow.py` | Prepares data for Figure 3 |
| `plot_climate_indices_correlation_analysis_AO_NAO.py` | Figure 3 |

**Streamflow trend analysis (core pipeline)**

| Script | Description |
|--------|-------------|
| `pre_process_streamflow_measurements_from_LamaH_Ice.py` | Pre-processes daily streamflow to cleaned CSV, etc. |
| `main.py` | Trend analysis: annual/seasonal streamflow, CV, flashiness, baseflow, etc. **Run twice**: `START_YEAR` = 1973 and 1993 in `config.py` |

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

The `notebooks/` folder contains Jupyter notebooks used to generate the rest of the figures for the manuscript. These are primarily for visualization and post-processing. Core computations are handled in the scripts listed above.

---

## License

This project is open-source under the MIT License.

---

## Contact

For questions, please contact [@hhelgason](https://github.com/hhelgason).
