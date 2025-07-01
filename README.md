# iceland-hydro-trends

This repository contains the code used for the paper:

**"Understanding Changes in Iceland’s Streamflow Dynamics in Response to Climate Change"**  
*Submitted to Hydrology and Earth System Sciences, 2024.*

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

Create a new environment and install the required packages using `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate iceland-hydro-trends
```

---

### 3. Configure Paths

Edit the `config.py` file in the `src/` directory to set:

- The path to the extracted **LamaH-Ice dataset**
- The output directory for results and figures

---

### 4. Run the Scripts

Navigate to the `src/` directory:

```bash
cd src
```

Then run the desired analysis scripts:

| Script | Description |
|--------|-------------|
| `pre_process_streamflow_measurements_from_LamaH_Ice.py` | Pre-processes daily streamflow measurements |
| `main.py` | Calculates trends in annual and seasonal average streamflow, coefficient of variation (CV), flashiness index, baseflow index, and more |
| `Figure8_visualize_trend_summary_with_heatmap.py` | Code to generate Figure 8: A heatmap to summarize trend results |
| `trend_correlation_analysis.py` | Correlates streamflow trends with catchment attributes and meteorological trends |
| `visualize_trend_correlations.py` | Creates correlation heatmaps for trend correlation results |
| `calculate_trends_in_streamflow_timing_metrics.py` | Computes trends in streamflow timing metrics (e.g., center of mass, timing of high/low flows) |
| `timing_metric_trend_correlation_analysis.py` | Analyzes relationships between timing trends and climate drivers or catchment features |
| `visualize_timing_trends_with_heatmap.py` | Creates heatmaps to summarize timing trends results |

---

## Notebooks

The `notebooks/` folder contains Jupyter notebooks used to generate figures for the manuscript. These are primarily for visualization and post-processing. Core computations are handled in the scripts listed above.

---

## License

This project is open-source under the MIT License.

---

## Contact

For questions, please contact [@hhelgason](https://github.com/hhelgason).
