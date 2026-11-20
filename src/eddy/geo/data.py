__all__ = ['load_natural_earth_countries', 'load_natural_earth_states']

import geopandas as gpd

from eddy.util import DATA_DIR, standardise_df


@standardise_df
def load_natural_earth_countries() -> gpd.GeoDataFrame:
    '''Load the Natural Earth 10m country borders as a GeoDataFrame.'''
    return gpd.read_file(DATA_DIR / 'natural-earth' / 'ne_10m_admin_0_map_subunits')

@standardise_df
def load_natural_earth_states() -> gpd.GeoDataFrame:
    '''Load the Natural Earth 10m state and province borders as a GeoDataFrame.'''
    return gpd.read_file(DATA_DIR / 'natural-earth' / 'ne_10m_admin_1_states_provinces')
