"""Configuration settings for the streamflow data processing."""

import gzip
import pickle
from pathlib import Path

# Repository folder (contains ``src/``). Outputs default to ``paper_repro_output`` inside it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"

# Base path to the LamaH-Ice dataset
# Users should modify this path to point to their local copy of the dataset
LAMAH_ICE_BASE_PATH = Path(r"")

# Output directory for processed data and figures (edit if you want results elsewhere)
OUTPUT_DIR = _REPO_ROOT / "paper_repro_output"

# Path to the cleaned streamflow data
STREAMFLOW_DATA_PATH = OUTPUT_DIR / "cleaned_streamflow_data" / "cleaned_streamflow_data.csv"

# Analysis period configuration
START_YEAR = 1993
END_YEAR = 2023
PERIOD = f"{START_YEAR}_{END_YEAR}"

# Data quality thresholds
MISSING_DATA_THRESHOLD = 0.8  # Maximum fraction of missing data allowed
WITHIN_YEAR_COVERAGE_THRESHOLD = 0.9  # Minimum fraction of data required within a year

# Path to catchment attributes file used for plotting
CATCHMENT_ATTRIBUTES_FILE = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/1_attributes/Catchment_and_gauge_attributes_used_for_plotting.gpkg"

# Path to catchment attributes CSV (used for sorting by glaciation, etc.)
CATCHMENT_ATTRIBUTES_CSV = LAMAH_ICE_BASE_PATH / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"

# Path to gauges shapefile (used for gauge names and locations)
GAUGES_SHAPEFILE = LAMAH_ICE_BASE_PATH / "D_gauges/3_shapefiles/gauges.shp"

# Paths to Iceland shapefile and glacier outlines (bundled under ``data/`` in this repo)
ICELAND_SHAPEFILE = _DATA_DIR / "island_isn93.shp"
GLACIER_SHAPEFILE = _DATA_DIR / "2019_glacier_outlines.shp"

# Alias for scripts that import ``GLACIER_OUTLINES``
GLACIER_OUTLINES = GLACIER_SHAPEFILE

# Path to save manuscript figures
MANUSCRIPT_FIGURES_PATH = OUTPUT_DIR / "manuscript_figures"

# List of gauges to keep despite strong human influence
# These gauges are kept for annual trend analysis because upstream reservoirs
# do not significantly alter the total annual flows
GAUGES_TO_KEEP = [
    102,  # Þjórsá Þjórsártún
    7     # Blanda við Löngumýri
]

# Gauges to remove due to known data quality issues
GAUGES_TO_REMOVE = [
    33,  # Hrafnkelsdalsá
    43,  # Hálslón Reservoir
    9,   # Syðri-Bægisá
    13,  # Elliðaár
    78,   # Smyrlabjargaá
    96   # Álftafitjakvísl
]

# --- Caravan / LamaH per-basin daily DataFrames: ``lamahice_<id>`` -> DataFrame
# Prefer uncompressed ``daily_dfs_snowfall_runoff.p`` (local; may include ``snowmelt_sum``) if present;
# else ``daily_dfs_snowfall_runoff.p.gz`` (smaller, ``snowfall_sum`` only) for the repo.
SNOWFALL_RUNOFF_PICKLE_P = _DATA_DIR / "daily_dfs_snowfall_runoff.p"
SNOWFALL_RUNOFF_PICKLE_P_GZ = _DATA_DIR / "daily_dfs_snowfall_runoff.p.gz"


def resolve_snowfall_runoff_pickle_path() -> Path:
    """Uncompressed pickle if present, otherwise gzip (Git-friendly) copy."""
    if SNOWFALL_RUNOFF_PICKLE_P.is_file():
        return SNOWFALL_RUNOFF_PICKLE_P
    if SNOWFALL_RUNOFF_PICKLE_P_GZ.is_file():
        return SNOWFALL_RUNOFF_PICKLE_P_GZ
    raise FileNotFoundError(
        f"Need either {SNOWFALL_RUNOFF_PICKLE_P} or {SNOWFALL_RUNOFF_PICKLE_P_GZ} under data/"
    )


def load_snowfall_runoff_pickle(path: Path | None = None) -> dict:
    """Load dict of per-basin DataFrames. Supports plain ``.p`` and gzip ``.p.gz``."""
    path = path or resolve_snowfall_runoff_pickle_path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".gz" or path.name.endswith(".p.gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)
