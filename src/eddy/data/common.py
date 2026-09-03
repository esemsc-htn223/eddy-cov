__all__ = [
    'BASE_CRS',
    'standardise_df', '_standardise_df',
    'load_natural_earth_countries', 'load_natural_earth_states'
]

import pandas as pd
import geopandas as gpd

from eddy.util import DATA_DIR, BASE_CRS

def standardise_df(func) -> pd.DataFrame:
    """
    Decorator that standardises column names of an output DataFrame to snake_case.
    """
    def inner(*args, **kwargs):
        df = func(*args, **kwargs)
        return _standardise_df(df)
    return inner

def _standardise_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Standardise a DataFrame:
    1. Convert column names to snake_case.
    2. Set GeoDataFrame geometry and CRS if applicable.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to standardise.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame, with the above standardisations applied.
    '''
    df = df.copy()
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

@standardise_df
def load_natural_earth_countries() -> gpd.GeoDataFrame:
    '''Load the Natural Earth 10m country borders as a GeoDataFrame.'''
    return gpd.read_file(DATA_DIR / 'NaturalEarth' / 'ne_10m_admin_0_map_subunits')

@standardise_df
def load_natural_earth_states() -> gpd.GeoDataFrame:
    '''Load the Natural Earth 10m state and province borders as a GeoDataFrame.'''
    return gpd.read_file(DATA_DIR / 'NaturalEarth' / 'ne_10m_admin_1_states_provinces')

