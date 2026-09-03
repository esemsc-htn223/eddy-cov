import geopandas as gpd
import requests

from eddy.util import DATA_DIR
from eddy.data.common import standardise_df

OFFSET_DIR = DATA_DIR / 'offsets'
if not OFFSET_DIR.exists():
    OFFSET_DIR.mkdir(parents=True, exist_ok=True)


def _download_offset_uk() -> None:
    """
    Download UK woodland carbon code offset projects as a GeoJSON file and save it to the local directory.
    """
    url = 'https://services9.arcgis.com/RCPJF8Z8BrfjscvL/arcgis/rest/services/Woodland_Carbon_Code_Projects_Read_Only_View/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson'
    gdf = gpd.read_file(url)
    gdf.to_file(OFFSET_DIR / 'offset_uk.geojson', driver='GeoJSON')

@standardise_df
def load_offset_uk(*, force_download: bool = False) -> gpd.GeoDataFrame:
    """
    Load UK woodland carbon code offset projects as a GeoDataFrame. If the data is not found locally, it will be downloaded from the source.

    Parameters
    ----------
    force_download : bool, optional
        If True, the data will be downloaded from the source even if it exists locally. Default is False.
    
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the UK woodland carbon code offset projects.
    """
    if force_download or not (OFFSET_DIR / 'offset_uk.geojson').exists():
        _download_offset_uk()
    
    gdf = gpd.read_file(OFFSET_DIR / 'offset_uk.geojson').set_index('OBJECTID')
    class_mapping = {}
    for item in _load_offset_uk_metadata()['types']:
        class_mapping[item['id']] = {
            'name': item['name'],
            'subclasses': {_sub['code']: _sub['name'] for _sub in item['domains']['Sub_Class']['codedValues']}
        }

    gdf['class_name'] = gdf['Class'].map(lambda x: class_mapping.get(x, {}).get('name', None)).astype('category')
    gdf['subclass_name'] = gdf[['Class', 'Sub_Class']].apply(
        lambda row: class_mapping.get(row['Class'], {}).get('subclasses', {}).get(row['Sub_Class'], None)
        , axis=1
        ).astype('category')

    gdf[['Country', 'Project_status', 'RAG_Status']] = gdf[['Country', 'Project_status', 'RAG_Status']].astype('category')
    gdf.convert_dtypes()
    gdf[['Class', 'Sub_Class']] = gdf[['Class', 'Sub_Class']].astype('int8')

    gdf = gdf.rename(columns={col: col.lower() for col in gdf.columns})
    # swap class and class_name, and subclass and subclass_name
    gdf = gdf.rename(columns={'class': 'class_id', 'sub_class': 'subclass_id', 'class_name': 'class', 'subclass_name': 'subclass'})
    first_cols = ['country', 'project_id', 'project_name', 'class', 'subclass']
    gdf = gdf[first_cols + gdf.drop(columns=first_cols).columns.tolist()]
    return gdf


def _load_offset_uk_metadata():
    '''
    Loads the metadata for UK woodland carbon code offset projects from the API.
    '''
    url = 'https://services9.arcgis.com/RCPJF8Z8BrfjscvL/ArcGIS/rest/services/Woodland_Carbon_Code_Projects_Read_Only_View/FeatureServer/0?f=pjson'

    return requests.get(url).json()