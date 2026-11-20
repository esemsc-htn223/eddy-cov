__all__ = [
    'DATA_DIR', 'OUT_DIR', 'LOG_DIR',
    'BASE_CRS', 'DISTANCE_CRS', 'PLOTTING_CRS',
    'logger_data', 'logger_processing',
    'BASEMAP_DEFAULT',
    'VON_KARMAN', 'GRAVITY',
    'standardise_df', '_standardise_df',
    'smallest_file',
    'var_metadata', 'var_dim_size', 'var_n_stations', 'var_n_samples',
    'average_over_time'
]

import os
import pathlib
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xyzservices.providers as xyz
from netCDF4 import Dataset

# CRS
BASE_CRS = 'EPSG:3857'  # Web Mercator projection for mapping and distance calculations (units in meters)
DISTANCE_CRS = 'EPSG:3857'  # Web Mercator projection for distance calculations (units in meters)
PLOTTING_CRS = 'EPSG:4326'  # WGS 84 projection for plotting (units in degrees)

# Physical constants
VON_KARMAN = 0.4
GRAVITY = 9.81

# Dirs
DATA_DIR = pathlib.Path(__file__).parent.parent.parent / 'data'
OUT_DIR = pathlib.Path(__file__).parent.parent.parent / 'out'
LOG_DIR = pathlib.Path(__file__).parent.parent.parent / 'log'

dirs = [DATA_DIR, OUT_DIR, LOG_DIR]
for _dir in dirs:
    if not _dir.exists():
        os.makedirs(_dir)


# Logging
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



# Plotting
BASEMAP_DEFAULT = xyz.CartoDB.Positron(api_key=os.getenv('CARTO_API_KEY', None))


# Helpers
def standardise_df(func) -> pd.DataFrame:
    """
    Decorator that standardises column names of an output DataFrame to snake_case. Does nothing if the output is not a DataFrame.
    """
    def inner(*args, **kwargs):
        df = func(*args, **kwargs)
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, [list, tuple, set]):
                out = []
                for i, item in enumerate(df):
                    if isinstance(item, pd.DataFrame):
                        out.append(_standardise_df(item))
                    else:
                        out.append(item)
                return type(df)(out)
            return df
        return _standardise_df(df)
    return inner

def _standardise_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Standardise a DataFrame:
    1. Convert column names to snake_case.
    2. Set GeoDataFrame geometry and CRS if applicable.

    In general, use of this function is discouraged in favour of the `@standardise_df` decorator, 
    which automatically applies this function to the output of a DataFrame-returning function.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to standardise.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame, with the above standardisations applied.
    '''
    df = df.convert_dtypes().copy()
    cols = df.columns
    # account for camelCase
    new_cols = {}
    for col in cols:
        if ' ' in col or '-' in col or '_' in col or col.isupper():
            # non-pascalCase
            new_cols[col] = col.lower().replace(' ', '_').replace('-', '_')
        else:
            # pascalCase
            new_cols[col] = ''.join('_' + c.lower() if c.isupper() else c for c in col).lstrip('_').replace(' ', '_').replace('-', '_')
        new_cols[col] = new_cols[col].replace('__', '_').replace('(', '').replace(')', '')

    if isinstance(df, gpd.GeoDataFrame):
        df = df.rename(columns=new_cols)
        if 'geometry' in df.columns:
            df.set_geometry('geometry', inplace=True)
        df.to_crs(BASE_CRS, inplace=True)
    
    return df.rename(columns=new_cols)


def smallest_file(dir_path: pathlib.Path, pattern: str = '*') -> pathlib.Path:
    '''
    Find the smallest file in a directory matching a glob pattern.

    Parameters
    ----------
    dir_path : pathlib.Path
        The directory to search.
    pattern : str, optional
        Glob pattern the file must match, e.g. '*.nc'. Defaults to '*' (any file).

    Returns
    -------
    pathlib.Path
        The path to the smallest matching file.
    '''
    if not isinstance(dir_path, pathlib.Path):
        dir_path = pathlib.Path(dir_path)

    files = [f for f in dir_path.glob(pattern) if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in directory: {dir_path}")

    return min(files, key=lambda f: f.stat().st_size)


def var_metadata(ds: Dataset, var_name: str|None = None) -> dict:
    '''
    Get the metadata for a variable in a netCDF dataset.

    Parameters
    ----------
    ds : netCDF4.Dataset
        The netCDF dataset to inspect.
    var_name : str or None
        The name of the variable to get the metadata for. If None (default), returns
        metadata for all variables in the dataset.

    Returns
    -------
    dict
        A dictionary containing the metadata for the variable(s),
        including its dimensions, shape, data type, and any attributes.
    '''
    if var_name is not None and var_name not in ds.variables:
        raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")

    if var_name is None:
        return {var: var_metadata(ds, var) for var in ds.variables}

    var = ds.variables[var_name]
    return {
        'dimensions': var.dimensions,
        'shape': var.shape,
        'dtype': var.dtype,
        'is_dimension': var_name in ds.dimensions,
        'attributes': {attr: getattr(var, attr) for attr in var.ncattrs()}
    }


def var_dim_size(ds: Dataset, dim_name: str, var_name: str|None = None) -> int|dict:
    '''
    Get the size of a dimension, for variables that have it.

    Parameters
    ----------
    ds : netCDF4.Dataset
        The netCDF dataset to inspect.
    dim_name : str
        The name of the dimension to measure, e.g. 'station' or 'sample'.
    var_name : str or None
        The variable to measure the dimension for. If None (default), returns a dictionary
        covering all variables in the dataset.

    Returns
    -------
    int or dict
        The size of `dim_name` for the given variable, or a mapping of variable name to size.
        Variables without that dimension have a size of 0.
    '''
    if dim_name not in ds.dimensions:
        raise ValueError(f"Dimension '{dim_name}' not found in dataset dimensions: {list(ds.dimensions.keys())}.")

    if var_name is not None:
        if var_name not in ds.variables:
            raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")
        return ds.dimensions[dim_name].size if dim_name in ds.variables[var_name].dimensions else 0

    return {
        var_: ds.dimensions[dim_name].size if dim_name in ds.variables[var_].dimensions else 0
        for var_ in ds.variables
    }


def var_n_stations(ds: Dataset, var_name: str|None = None) -> int|dict:
    '''The number of stations for a variable, or for all variables. See `var_dim_size`.'''
    return var_dim_size(ds, 'station', var_name)


def var_n_samples(ds: Dataset, var_name: str|None = None) -> int|dict:
    '''The number of samples for a variable, or for all variables. See `var_dim_size`.'''
    return var_dim_size(ds, 'sample', var_name)


def average_over_time(
        time: np.ndarray, var: np.ndarray, period: str|np.timedelta64|pd.Timedelta
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Average a timeseries over fixed-length, epoch-anchored time bins.

    Parameters
    ----------
    time : np.ndarray
        Timestamps for each row of `var`, in ascending order.
    var : np.ndarray
        Values to average. Only the leading (time) axis is reduced.
    period : str or np.timedelta64 or pd.Timedelta
        Length of each averaging bin, e.g. '30m', '1h' or np.timedelta64(30, 'm').

    Returns
    -------
    np.ndarray, np.ndarray
        The start time of each bin, and the mean of `var` within it. Bins containing no valid
        data are NaN (masked, if `var` is a masked array).
    '''

    period = pd.Timedelta(period).to_timedelta64().astype('timedelta64[ns]')
    if period <= np.timedelta64(0, 'ns'):
        raise ValueError(f"period must be a positive duration, got {period}.")

    bin_index = (time.astype('datetime64[ns]').astype('int64')
                 // period.astype('int64'))  # anchored on the Unix epoch
    bin_starts, first_row = np.unique(bin_index, return_index=True)

    was_masked = np.ma.isMaskedArray(var)
    values = np.ma.filled(var, np.nan).astype(np.float64)

    is_valid = ~np.isnan(values)
    totals = np.add.reduceat(np.where(is_valid, values, 0.0), first_row, axis=0)
    counts = np.add.reduceat(is_valid, first_row, axis=0)

    with np.errstate(invalid='ignore'):
        means = np.where(counts > 0, totals / counts, np.nan)

    if was_masked:
        means = np.ma.masked_invalid(means)

    return bin_starts * period + np.datetime64(0, 'ns'), means
