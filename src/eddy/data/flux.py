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

def _get_latest_fluxnet_snapshot_path():
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
            snapshot_filepath = _get_latest_fluxnet_snapshot_path()
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


def _find_local_fluxnet_site_files(sites, search_dir):
    '''
    Finds local FLUXNET site files in the specified directory. Returns a dictionary with two keys:
    - 'found_paths': a list of pathlib.Path objects for the found site files
    - 'missing_sites': a list of site names that were not found in the directory
    '''
    if not isinstance(search_dir, pathlib.Path):
        search_dir = pathlib.Path(search_dir)
    out = {}
    if not search_dir.is_dir():
        raise NotADirectoryError(f"Directory {search_dir} does not exist.")
    out['found_paths'] = [search_dir / f for f in search_dir.iterdir() if any(site in f.stem for site in sites) and not (search_dir / f).is_dir()]
    out['missing_sites'] = [site for site in sites if not any(site in f.stem for f in out['found_paths'])]
    return out

async def load_fluxnet_sites(sites:list[str], snapshot_filepath:pathlib.Path, *, 
                             force_download:bool=False, extract_zip:bool = True):
    '''
    Loads the specified FLUXNET site files from disk, or downloads them if they are not found locally.
    
    Parameters
    ----------
    sites : list of str
        A list of site names to load.
    snapshot_filepath : pathlib.Path
        The path to the FLUXNET snapshot file to use for downloading site files.
    force_download : bool, optional
        If True, forces a download of the specified FLUXNET site files even if they are already present on disk. Default is False.
    extract_zip : bool, optional
        If True, extracts the downloaded .zip files. Default is True.
        
    Returns
    -------
    list of pathlib.Path
        A list of paths to the loaded or downloaded FLUXNET site .zip files or folders (if extracted).
    '''
    if not isinstance(snapshot_filepath, pathlib.Path):
        snapshot_filepath = pathlib.Path(snapshot_filepath)
    
    sites_dir = FLUXNET_DIR / 'sites'
    if not os.path.isdir(sites_dir):
            os.makedirs(sites_dir)
    
    if force_download:
        logger_data.info(f"FORCE DOWNLOADING FLUXNET SITE DATA - {sites}")
        sites_to_download = sites
        out = []
    else:
        search_results = _find_local_fluxnet_site_files(sites, search_dir = sites_dir)
        if len(search_results['missing_sites']) > 0:
            logger_data.info(f"SITE DATA NOT FOUND LOCALLY - {search_results['missing_sites']}")
        sites_to_download = search_results['missing_sites']
        out = search_results['found_paths']
        
    if sites_to_download:
        logger_data.info(f"DOWNLOADING FLUXNET SITE DATA - {sites_to_download}")
        fnames = await shuttle.download(sites_to_download, snapshot_file=snapshot_filepath, output_dir = sites_dir)
        out.extend([sites_dir / fname for fname in fnames])

    if extract_zip:
        for i, zip_file in enumerate(out):
            site_folder = zip_file.with_name(zip_file.stem.split('_')[1])  # all fluxnet files comply with this structure
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(site_folder)
            logger_data.info(f"EXTRACTED FLUXNET SITE DATA - {zip_file} to {site_folder}")
            out[i] = site_folder  # Update the path to point to the extracted folder
    return out