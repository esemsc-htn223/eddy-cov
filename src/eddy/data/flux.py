__all__ = [
    'FLUX_DIR', 'FLUXNET_DIR', 'TERN_DIR', 'AMERIFLUX_DIR',
    'load_fluxnet_snapshot', 'load_fluxnet_sites'
]

import fluxnet_shuttle as shuttle
import pandas as pd
import geopandas as gpd
import pathlib
from typing import Literal
import zipfile
import warnings

import os
import datetime

from eddy.util import DATA_DIR, logger_data
from eddy.data.common import standardise_df, _standardise_df

FLUX_DIR = DATA_DIR / 'flux'
FLUXNET_DIR = FLUX_DIR / 'FLUXNET'
TERN_DIR = FLUX_DIR / 'TERN'
AMERIFLUX_DIR = FLUX_DIR / 'AmeriFlux'

_dirs = [FLUX_DIR, FLUXNET_DIR, TERN_DIR, AMERIFLUX_DIR]
for _dir in _dirs:
    if not _dir.exists():
        os.makedirs(_dir)

def latest_fluxnet_snapshot_path():
    '''Find the latest FLUXNET snapshot file on disk, or raise an error if none are found.'''
    files = [f for f in FLUXNET_DIR.iterdir() if f.is_file() and f.suffix == '.csv' and f.name.startswith('fluxnet_shuttle_snapshot_')]
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in directory {FLUXNET_DIR}.")
    
    # sort by datetime in filename
    latest_file = max(files, key=lambda f: datetime.datetime.strptime(f.name.split('_')[-1].replace('.csv', ''), '%Y%m%dT%H%M%S'))
    return FLUXNET_DIR / latest_file

async def load_fluxnet_snapshot(*, return_type: Literal['gdf', 'df', 'path'] = 'gdf', force_download:bool=False) -> gpd.GeoDataFrame | pd.DataFrame | pathlib.Path:
    '''
    Loads the latest FLUXNET snapshot from disk, or downloads it if none have been downloaded.
    Must be run in an async context (`async with` or `await load_fluxnet_snapshot()`) due to
    fluxnet_shuttle requirements.

    Parameters
    ----------
    return_type : Literal['gdf', 'df', 'path'], optional
        The type of object to return. If 'gdf', returns a GeoDataFrame. If 'df', returns a regular pandas DataFrame. If 'path', returns the path to the snapshot file. Default is 'gdf'.
    force_download : bool, optional
        If True, forces a download of the latest FLUXNET snapshot even if one is already present on disk. \
        Default is False.
    
    Returns
    -------
    GeoDataFrame or DataFrame
    '''
    return_type = return_type.lower()
    if return_type not in ['gdf', 'df', 'path']:
        raise ValueError(f"Invalid return_type '{return_type}'. Must be one of 'gdf', 'df', or 'path'.")

    if force_download:
        logger_data.info('DOWNLOADING FLUXNET SNAPSHOT - force_download=True')
        snapshot_filepath = await shuttle.listall(output_dir = FLUXNET_DIR)
    else:
        try:
            snapshot_filepath = latest_fluxnet_snapshot_path()
            logger_data.info(f'LOADED FLUXNET SNAPSHOT - loaded latest file with fname="{snapshot_filepath}" from disk')
        except FileNotFoundError as e:
            print(f'No local snapshot file found, downloading from FLUXNET.')
            logger_data.info('DOWNLOADING FLUXNET SNAPSHOT - no local file found')
            snapshot_filepath = await shuttle.listall(output_dir = FLUXNET_DIR)

    snapshot_date = datetime.datetime.strptime(snapshot_filepath.name.split('_')[-1].replace('.csv', ''), '%Y%m%dT%H%M%S')
    print(f'Loaded FLUXNET snapshot from {snapshot_date}')

    if return_type == 'path':
        return snapshot_filepath
    out = _standardise_df(pd.read_csv(snapshot_filepath))
    if return_type == 'df':
        return out
    out = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out['location_long'], out['location_lat'], crs = 'EPSG:4326'))
    out = _standardise_df(out)  # ensure geometry and CRS are set correctly
    return out


def _find_local_fluxnet_site_files(sites: list[str], 
                                   search_dir: pathlib.Path = FLUXNET_DIR / 'sites', 
                                   return_type: Literal['folder', 'zip'] = 'folder'):
    '''
    Finds local FLUXNET site files in the specified directory. Returns a dictionary with two keys:
    - 'found_paths': a dict mapping site ids to pathlib.Path objects for the found site files
    - 'missing_sites': a list of site names that were not found in the directory
    '''
    if return_type not in ['folder', 'zip']:
        raise ValueError(f"Invalid return_type '{return_type}'. Must be one of 'folder' or 'zip'.")
    if not isinstance(search_dir, pathlib.Path):
        search_dir = pathlib.Path(search_dir)
    if not search_dir.is_dir():
        raise NotADirectoryError(f"Directory {search_dir} does not exist.")

    out = {'found_paths': {}, 'missing_sites': []}
    if return_type == 'folder':
        out['found_paths'] = {site: search_dir / site for site in sites if (search_dir / site).is_dir()}
    elif return_type == 'zip':
        out['found_paths'] = {}
        for site in sites:
            out['found_paths'].update({site: search_dir / f for f in search_dir.iterdir() if site in f.stem and (search_dir / f).is_file() and f.suffix == '.zip'})
    out['missing_sites'] = [site for site in sites if not any(site in f.stem for f in out['found_paths'].values())]
    return out

async def get_fluxnet_site_dirs(
        sites:list[str]|str, *, 
        snapshot_filepath:pathlib.Path | Literal['latest'] = 'latest', 
        force_download:bool=False
    ) -> dict[str, 'FLUXNETSite'|None]:
    '''
    Gets the paths of the folders containing FLUXNET site data for the specified sites. 
    If the site files are not found locally, they will be downloaded from the FLUXNET snapshot file.
    Can be used to download latest data using `force_download=True`.
    
    Parameters
    ----------
    sites : list of str
        A list of site ids to load.
    snapshot_filepath : pathlib.Path or 'latest', optional
        The path to the FLUXNET snapshot file to use for downloading site files.
        If 'latest', the function will use the latest snapshot file on disk (if available). 
        Default is 'latest'.
    force_download : bool, optional
        If True, forces a download of the specified FLUXNET site files even if they are already present on disk. 
        Default is False.
    extract_zip : bool, optional
        If True, extracts the downloaded .zip files. Default is True.
        
    Returns
    -------
    dict of str: pathlib.Path
        A dictionary with site ids as keys and paths to the loaded or downloaded FLUXNET site
    '''

    if snapshot_filepath == 'latest':
        snapshot_filepath = latest_fluxnet_snapshot_path()
    else:
        try:
            snapshot_filepath = pathlib.Path(snapshot_filepath)
        except Exception as e:
            logger_data.error(f"Error occurred while converting snapshot_filepath to pathlib.Path: {e}")
            raise e
            

    sites_dir = FLUXNET_DIR / 'sites'
    if not os.path.isdir(sites_dir):
        os.makedirs(sites_dir)
    
    if force_download:
        logger_data.info(f"FORCE DOWNLOADING FLUXNET SITE DATA - {sites}")
        sites_to_download = sites
        out = {}
    else:
        search_results = _find_local_fluxnet_site_files(sites, search_dir = sites_dir, return_type = 'folder')
        if len(search_results['missing_sites']) > 0:
            logger_data.info(f"SITE DATA NOT FOUND LOCALLY - {search_results['missing_sites']}")
        sites_to_download = search_results['missing_sites']
        out = search_results['found_paths']
        
    if sites_to_download:
        logger_data.info(f"DOWNLOADING FLUXNET SITE DATA - {sites_to_download}")
        fnames = await shuttle.download(sites_to_download, snapshot_file=snapshot_filepath, output_dir = sites_dir)
        for site in sites_to_download:
            candidate_files = [fname for fname in fnames if site in fname]
            if len(candidate_files) == 1:
                out[site] = sites_dir / candidate_files[0]
            elif len(candidate_files) == 0:
                logger_data.warning(f"COULD NOT DOWNLOAD FLUXNET SITE DATA - {site} not found in snapshot file")
                out[site] = None
            else:
                logger_data.warning(f"MULTIPLE FILES FOUND FOR FLUXNET SITE - {site} found in snapshot file: {candidate_files}. Using the first one.")
                out[site] = sites_dir / candidate_files[0]

    # extract zip files
    for site_id, zip_file_or_folder in out.items():
        if zip_file_or_folder is None:
            continue  # skip if no file was found for this site
        elif zip_file_or_folder.is_dir():
            out[site_id] = FLUXNETSite.from_folder(zip_file_or_folder)
        elif zip_file_or_folder.suffix == '.zip':
            out[site_id] = FLUXNETSite.from_zip(zip_file_or_folder)
        else:
            logger_data.warning(f"UNEXPECTED FILE TYPE FOR FLUXNET SITE - {zip_file_or_folder} is not a .zip file or a directory. Skipping.")
            out[site_id] = None

    return out


class FLUXNETSite:
    '''
    A FLUXNET site object that contains the site id, path to the site folder, and a methods to load the data.
    
    Attributes
    ----------
    site_id : str
        The site id of the FLUXNET site.
    site_folder : pathlib.Path
        The path to the folder containing the site data.
    metadata_path : pathlib.Path
        The path to the metadata file for the site.
    network_code : str
        The network code of the FLUXNET site.
    start_year : int
        The start year of the FLUXNET site data.
    end_year : int
        The end year of the FLUXNET site data.
    oneflux_version : str
        The major version of the OneFlux used to process the site data, e.g. 'v2.2'.
    release_version : str
        The release version of the FLUXNET site data.
    data_available : bool
        Whether the site folder exists and contains data files.
    metadata : pandas.DataFrame
        The metadata file (BIF) for the site as a pandas DataFrame.
    bif : pandas.DataFrame
        The BIF file for the site as a pandas DataFrame. Alias for `metadata`.
    
    Methods
    -------
    download(snapshot_filepath:pathlib.Path | Literal['latest']='latest', force_download:bool=False)
        Download the site data from the FLUXNET snapshot file.
    from_folder(site_folder:pathlib.Path)
        Create a FLUXNET site object from a folder containing the site data.
        Class method, returning a new instance of the class.
    from_download(site_id:str, snapshot_filepath:pathlib.Path|Literal['latest']='latest', force_download:bool=False)
        Create a FLUXNET site object by downloading the site data from the FLUXNET snapshot file.
        Class method, returning a new instance of the class.
    fluxmet(time_resolution: Literal['HH', 'DD', 'WW', 'YY'] ='HH')
        Load the continuous flux and meteorological data (FLUXMET) for the site at the specified time resolution.
    bifvarinfo(time_resolution: Literal['HH', 'DD', 'WW', 'YY'] ='HH')
        Load the variable information (BIFVARINFO) for the site at the specified time resolution.
    era5(time_resolution: Literal['HH', 'DD', 'WW', 'YY'] ='HH')
        Load the ERA5 reanalysis data for the site at the specified time resolution.
    
    '''

    def __init__(self, site_id:str):
        '''
        Create a FLUXNET site object.

        Parameters
        ----------
        site_id : str
            The site id of the FLUXNET site to load. Must be a valid site id in the latest 
            FLUXNET snapshot file.
        '''

        latest_snapshot = pd.read_csv(latest_fluxnet_snapshot_path())
        if site_id not in latest_snapshot['site_id'].values:
            raise ValueError(f"Could not find site id '{site_id}' in the latest FLUXNET snapshot file - either update the snapshot file, or check the site id.")

        self.site_id = site_id
        self.id = site_id  # alias for self.site_id
        self.site_folder = FLUXNET_DIR / 'sites' / site_id
        if not self.data_available:
            warnings.warn(f"Data for site '{site_id}' not found in {self.site_folder}. Try downloading with `await site.download()`.")
        else:
            self.__finish_init__()  # initialise metadata if data is available

    def __finish_init__(self):
        '''Initialisation to occur after data has been made available.'''
        if not self.data_available:
            raise FileNotFoundError(f"No data files found for site '{self.site_id}' in folder '{self.site_folder}'. Try downloading with `await site.download()`.")
        self.metadata_path = next(self.site_folder.glob(f'*_BIF_*'), None)  # only one BIF file per site
        if self.metadata_path is None:
            raise FileNotFoundError(f"No metadata file found for site '{self.site_id}' in folder '{self.site_folder}'.")

        # extract metadata from the filename
        self.network_code = self.metadata_path.stem.split('_')[0]

        yrs = self.metadata_path.stem.split('_')[4].split('-')
        self.start_year = int(yrs[0])
        self.end_year = int(yrs[1])

        self.oneflux_version = self.metadata_path.stem.split('_')[5]
        self.release_version = self.metadata_path.stem.split('_')[6]
        return self  # return self to allow for method chaining
    
    async def download(self, snapshot_filepath:pathlib.Path | Literal['latest'] = 'latest', force_download:bool=False):
        '''
        Download the site data from the FLUXNET snapshot file. If the site data is already present on disk, 
        it will not be downloaded again unless `force_download` is set to True.

        Parameters
        ----------
        snapshot_filepath : pathlib.Path or 'latest', optional
            The path to the FLUXNET snapshot file to use for downloading site files.
            If 'latest', the function will use the latest snapshot file on disk (if available). 
            Default is 'latest'.
        force_download : bool, optional
            If True, forces a download of the specified FLUXNET site files even if they are already present on disk. 
            Default is False.
        '''
        out = await get_fluxnet_site_dirs(self.site_id, snapshot_filepath=snapshot_filepath, force_download=force_download)
        if self.site_id not in out or out[self.site_id] is None:
            raise FileNotFoundError(f"Could not download site data for site '{self.site_id}'.")
        self.__finish_init__()
        return self  # return self to allow for method chaining

    @classmethod
    def from_zip(cls, zip_file:pathlib.Path) -> 'FLUXNETSite':
        '''
        Create a FLUXNET site object from a .zip file containing the site data. The .zip file must contain a BIF file.

        Parameters
        ----------
        zip_file : pathlib.Path
            The path to the .zip file containing the site data.
        
        Returns
        -------
        FLUXNETSite
        '''
        if not isinstance(zip_file, pathlib.Path):
            zip_file = pathlib.Path(zip_file)
        if not zip_file.is_file():
            raise FileNotFoundError(f"File {zip_file} does not exist.")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            bif_file = next((f for f in zip_ref.namelist() if '_BIF_' in f), None)
            if bif_file is None:
                raise FileNotFoundError(f"No BIF file found in .zip file {zip_file}.")
            site_id = bif_file.split('_')[1]
            zip_ref.extractall(FLUXNET_DIR / 'sites' / site_id)  # extract to the sites folder

        return cls(site_id)

    @classmethod
    def from_folder(cls, site_folder:pathlib.Path) -> 'FLUXNETSite':
        '''
        Create a FLUXNET site object from a folder containing the site data. The folder must contain a BIF file.

        Parameters
        ----------
        site_folder : pathlib.Path
            The path to the folder containing the site data.
        
        Returns
        -------
        FLUXNETSite
        '''
        if not isinstance(site_folder, pathlib.Path):
            site_folder = pathlib.Path(site_folder)
        if not site_folder.is_dir():
            raise NotADirectoryError(f"Directory {site_folder} does not exist.")
        
        bif_file = next(site_folder.glob(f'*_BIF_*'), None)
        if bif_file is None:
            raise FileNotFoundError(f"No BIF file found in directory {site_folder}.")
        
        site_id = bif_file.stem.split('_')[1]
        return cls(site_id)

    @classmethod
    async def from_download(cls, site_id:str, snapshot_filepath:pathlib.Path | Literal['latest'] = 'latest', force_download:bool=False) -> 'FLUXNETSite':
        '''
        Create a FLUXNET site object by downloading the site data from the FLUXNET snapshot file.

        Parameters
        ----------
        site_id : str
            The site id of the FLUXNET site to load. Must be a valid site id in the latest 
            FLUXNET snapshot file.
        snapshot_filepath : pathlib.Path or 'latest', optional
            The path to the FLUXNET snapshot file to use for downloading site files.
            If 'latest', the function will use the latest snapshot file on disk (if available). 
            Default is 'latest'.
        force_download : bool, optional
            If True, forces a download of the specified FLUXNET site files even if they are already present on disk. 
            Default is False.
        
        Returns
        -------
        FLUXNETSite
        '''
        out = await get_fluxnet_site_dirs(site_id, snapshot_filepath=snapshot_filepath, force_download=force_download)
        if site_id not in out or out[site_id] is None:
            raise FileNotFoundError(f"Could not download site data for site '{site_id}'.")
        return cls(site_id)

    @classmethod
    def multiple(cls, site_ids:list[str]) -> tuple['FLUXNETSite']:
        '''
        Create multiple FLUXNET site objects from a list of site ids.

        Parameters
        ----------
        site_ids : list of str
            A list of site ids of the FLUXNET sites to load. Must be valid site ids in the latest 
            FLUXNET snapshot file.
        
        Returns
        -------
        tuple of FLUXNETSite
        '''
        return tuple([cls(site_id) for site_id in site_ids])

    @property
    def data_available(self) -> bool:
        '''Check if the site folder exists and contains data files.'''
        if not self.site_folder.exists():
            return False
        data_files = list(self.site_folder.glob(f'*_FLUXNET_*'))
        return len(data_files) > 0

    @property
    def metadata(self) -> pd.DataFrame:
        '''The metadata file (BIF) for the site as a pandas DataFrame.'''
        if not self.data_available:
            raise FileNotFoundError(f"No data files found for site '{self.site_id}' in folder '{self.site_folder}'. Try downloading with `await site.download()`.")
        return pd.read_csv(self.metadata_path)

    @property
    def bif(self) -> pd.DataFrame:
        '''The BIF file for the site as a pandas DataFrame. Wrapper for self.metadata.'''
        return self.metadata

    def _data_file_path(self, 
                        data_type:Literal['FLUXMET', 'BIF', 'BIFLUX', 'BIFMET', 'BIFLUXMET'], 
                        time_resolution: Literal['HH', 'DD', 'WW', 'YY'] = 'HH'
                        ) -> pathlib.Path:
        '''
        Get the path to the data file for the specified data type and time resolution.

        Parameters
        ----------
        data_type : str
            The type of data to load. Must be one of 'FLUXMET', 'BIF', 'BIFLUX', 'BIFMET', 'BIFLUXMET'.
        time_resolution : Literal['HH', 'DD', 'WW', 'YY'], optional
            The time resolution of the data to load. Must be one of 'HH' (half-hourly), 'DD' (daily), 
            'WW' (weekly), or 'YY' (yearly). Default is 'HH'.

        Returns
        -------
        pathlib.Path
            The path to the data file.
        '''

        if not self.data_available:
            raise FileNotFoundError(f"No data files found for site '{self.site_id}' in folder '{self.site_folder}'. Try downloading with `await site.download()`.")

        data_type = data_type.upper()
        time_resolution = time_resolution.upper()

        # allow passing of literal names - all filenames start with FLUXNET_
        if data_type.startswith('FLUXNET_'):
            data_type = data_type.replace('FLUXNET_', '')

        if data_type not in ['FLUXMET', 'BIFVARINFO', 'ERA5', 'BIF']:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be one of 'FLUXMET', 'BIFVARINFO', 'ERA5', or 'BIF'.")
        if time_resolution not in ['HH', 'DD', 'WW', 'YY']:
            raise ValueError(f"Invalid time_resolution '{time_resolution}'. Must be one of 'HH', 'DD', 'WW', or 'YY'.")

        if data_type == 'BIF':
            return self.metadata_path
        
        # find the file in the site folder
        file_name = f'{self.network_code}_{self.site_id}_FLUXNET_{data_type}_{time_resolution}_{self.start_year}-{self.end_year}_{self.oneflux_version}_{self.release_version}.csv'
        if not (self.site_folder / file_name).exists():
            raise FileNotFoundError(f"No {data_type} file found for site '{self.site_id}' with time resolution '{time_resolution}' in folder '{self.site_folder}'.\nExpected file name: '{file_name}'")
        return self.site_folder / file_name

    def fluxmet(self, time_resolution: Literal['HH', 'DD', 'WW', 'YY'] = 'HH'):
        '''
        Load the continuous flux and meteorological data (FLUXMET) for the site at 
        the specified time resolution.

        Parameters
        ----------
        time_resolution : Literal['HH', 'DD', 'WW', 'YY'], optional
            The time resolution of the data to load. Must be one of 'HH' (half-hourly), 'DD' (daily), 
            'WW' (weekly), or 'YY' (yearly). Default is 'HH'.
        '''
        csv_path = self._data_file_path('FLUXMET', time_resolution)
        return pd.read_csv(csv_path)

    def bifvarinfo(self, time_resolution: Literal['HH', 'DD', 'WW', 'YY'] = 'HH'):
        '''
        Load the variable information (BIFVARINFO) for the site.
        '''
        csv_path = self._data_file_path('BIFVARINFO', time_resolution)
        return pd.read_csv(csv_path)

    def era5(self, time_resolution: Literal['HH', 'DD', 'WW', 'YY'] = 'HH'):
        '''
        Load the ERA5 reanalysis data for the site at the specified time resolution.
        '''
        csv_path = self._data_file_path('ERA5', time_resolution)
        return pd.read_csv(csv_path)

    @property
    def data_policy_and_license(self) -> str:
        '''The contents of the data policy file (DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt) for the site as a string.'''
        return (self.site_folder / 'DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt').read_text()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"FLUXNETSite(site_id='{self.site_id}', data_available={self.data_available})"


