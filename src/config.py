"""Configuration settings for the streamflow data processing."""

from pathlib import Path

# Base path to the LamaH-Ice dataset
# Users should modify this path to point to their local copy of the dataset
LAMAH_ICE_BASE_PATH = Path(r"")

# Output directory for processed data
# By default, saves in a 'data' subdirectory of the project
OUTPUT_DIR = Path(r"")

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

# Paths to Iceland shapefile and glacier outlines
ICELAND_SHAPEFILE = Path(r'../data/island_isn93.shp')
GLACIER_SHAPEFILE = Path(r'../data/2019_glacier_outlines.shp')

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