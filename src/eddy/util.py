__all__ = ["DATA_DIR", "OUT_DIR", "LOG_DIR", "load_natural_earth_countries", "load_natural_earth_states"]

import os
import pathlib
import logging
import geopandas as gpd


DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
OUT_DIR = pathlib.Path(__file__).parent.parent.parent / "out"
LOG_DIR = pathlib.Path(__file__).parent.parent.parent / "log"

dirs = [DATA_DIR, OUT_DIR, LOG_DIR]
for _dir in dirs:
    if not _dir.exists():
        os.makedirs(_dir)

log_files = {
    'eddy_data': LOG_DIR / 'eddy-data.log',
    'eddy_processing': LOG_DIR / 'eddy-processing.log',
}

logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for logger, log_file in log_files.items():
    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write('')  # Create an empty log file
    logger = logging.getLogger(logger)
    logger.setLevel(logging.DEBUG)
    logger_fh = logging.FileHandler(log_file)
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logger_formatter)
    logger.addHandler(logger_fh)


# TODO: find a home for natural earth data functions
# TODO: download Natural Earth data if not present
def load_natural_earth_countries():
    return gpd.read_file(DATA_DIR / 'NaturalEarth' / 'ne_10m_admin_0_map_subunits')
def load_natural_earth_states():
    return gpd.read_file(DATA_DIR / 'NaturalEarth' / 'ne_10m_admin_1_states_provinces')

