__all__ = [
    'DATA_DIR', 'OUT_DIR', 'LOG_DIR',
    'BASE_CRS', 'DISTANCE_CRS', 'PLOTTING_CRS',
    'logger_data', 'logger_processing'
]

import os
import pathlib
import logging

BASE_CRS = 'EPSG:3857'  # Web Mercator projection for mapping and distance calculations (units in meters)
#BASE_CRS = 'EPSG:4326'  # WGS 84 projection for mapping and distance calculations (units in degrees)
DISTANCE_CRS = 'EPSG:3857'  # Web Mercator projection for distance calculations (units in meters)
PLOTTING_CRS = 'EPSG:4326'  # WGS 84 projection for plotting (units in degrees)


DATA_DIR = pathlib.Path(__file__).parent.parent.parent / 'data'
OUT_DIR = pathlib.Path(__file__).parent.parent.parent / 'out'
LOG_DIR = pathlib.Path(__file__).parent.parent.parent / 'log'

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

logger_data = logging.getLogger('eddy_data')
logger_processing = logging.getLogger('eddy_processing')