from typing import Literal
import pandas as pd
import geopandas as gpd

from eddy.util import DATA_DIR
from eddy.data.common import standardise_df

WIND_DIR = DATA_DIR / 'wind'


def _download_wind_data() -> None:
    """
    Download wind data from the Global Wind Power Tracker Excel file and save it to the local directory.
    """
    raise NotImplementedError("Wind data download function not implemented yet. Please download the data manually from https://globalenergymonitor.org/projects/global-wind-power-tracker/ and place it in the 'wind' directory.")

@standardise_df
def load_wind_farm_locs(source: Literal['gem'] = 'gem') -> gpd.GeoDataFrame:
    '''
    Load wind farm locations from the Global Wind Power Tracker Excel file and return it as a GeoDataFrame.

    Parameters
    ----------
    source : {'gem'}, optional
        The source of the wind farm data. Currently, only 'gem' (Global Energy Monitor) 
        is supported. Defaults to 'gem'.
        - 'gem': Load wind farm locations from the locally saved Global Wind Power Tracker Excel file.
        Contains wind farm locations from the Global Energy Monitor (GEM) database. Aims to include all
        wind farms greater than 10 MW globally, with some smaller projects included. See https://globalenergymonitor.org/projects/global-wind-power-tracker/.

    Returns
    -------
    gpd.GeoDataFrame
    '''
    if source.lower() == 'gem':
        df_wind = pd.read_excel(WIND_DIR / 'Global-Wind-Power-Tracker-February-2026.xlsx', sheet_name = 'Data')
        gdf_wind = gpd.GeoDataFrame(df_wind, geometry=gpd.points_from_xy(df_wind['Longitude'], df_wind['Latitude'], crs='EPSG:4326'))
        gdf_wind['project'] = gdf_wind['Project Name'].astype(str) + ', phase ' + gdf_wind['Phase Name'].astype(str)
        gdf_wind.rename(columns = {'GEM phase ID': 'gem_phase_id'}, inplace = True)
        gdf_wind.set_index('gem_phase_id', inplace = True)
    else:
        raise ValueError(f"Unsupported source '{source}'. Currently, only 'gem' is supported.")
    return gdf_wind