import pandas as pd
import geopandas as gpd

from eddy.util import DATA_DIR
from eddy.data.common import standardise_df

WIND_DIR = DATA_DIR / 'wind'


def _download_wind_data() -> None:
    """
    Download wind data from the Global Wind Power Tracker Excel file and save it to the local directory.
    """
    raise NotImplementedError("Wind data download function not implemented yet. Please download the data manually and place it in the 'wind' directory.")

@standardise_df
def load_wind_locations():
    '''
    Load wind farm locations from the Global Wind Power Tracker Excel file and return it as a GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
    '''
    df_wind = pd.read_excel(WIND_DIR / 'Global-Wind-Power-Tracker-February-2026.xlsx', sheet_name = 'Data')
    gdf_wind = gpd.GeoDataFrame(df_wind, geometry=gpd.points_from_xy(df_wind['Longitude'], df_wind['Latitude'], crs='EPSG:4326'))
    gdf_wind['project'] = gdf_wind['Project Name'].astype(str) + ', phase ' + gdf_wind['Phase Name'].astype(str)
    return gdf_wind