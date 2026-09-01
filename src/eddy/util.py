__all__ = ["DATA_DIR", "OUT_DIR"]

import os
import pathlib
import logging


DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
OUT_DIR = pathlib.Path(__file__).parent.parent.parent / "out"
LOG_DIR = pathlib.Path(__file__).parent.parent.parent / "log"

dirs = [DATA_DIR, OUT_DIR, LOG_DIR]
for _dir in dirs:
    if not _dir.exists():
        os.makedirs(_dir)

if not (LOG_DIR / 'eddy.log').exists():
    with open(LOG_DIR / 'eddy.log', 'w') as f:
        f.write('')  # Create an empty log file

logger_data = logging.getLogger('eddy_data')
logger_wind = logging.getLogger('eddy_wind')
logger_fire = logging.getLogger('eddy_fire')
logger_drone = logging.getLogger('eddy_drone')

logger_fh = logging.FileHandler(LOG_DIR / 'eddy.log')
logger_fh.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_fh.setFormatter(logger_formatter)


for logger in [logger_data, logger_wind, logger_fire, logger_drone]:
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logger_fh)
