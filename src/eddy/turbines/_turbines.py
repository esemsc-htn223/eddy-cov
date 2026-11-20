__all__ = ['ec_neighbours', 'offset_neighbours']

import pandas as pd
import geopandas as gpd

from eddy.util import DISTANCE_CRS
from eddy.geo import neighbours


def _operational(gdf_wind: gpd.GeoDataFrame, date: pd.Timestamp) -> gpd.GeoDataFrame:
    '''Wind farms operating in the year of `date`. Farms with no retirement year are treated as still operating.'''
    return gdf_wind.loc[
        (gdf_wind['start_year'] <= date.year)
        & ((gdf_wind['retired_year'] >= date.year) | gdf_wind['retired_year'].isna())
    ]


def ec_neighbours(
        gdf_wind: gpd.GeoDataFrame,
        gdf_ec: gpd.GeoDataFrame,
        max_match_dist: int = 100_000,
        *,
        date: pd.Timestamp|None = None,
        min_height: int|None = None,
        bounds: tuple|None = None,
        crs: str = DISTANCE_CRS,
        site_id_col: str = 'site_id',
        height_col: str = 'height',
        overlap: bool|int = True
    ) -> gpd.GeoDataFrame:
    '''
    Match wind farms to every eddy covariance tower within a specified distance.

    Parameters
    ----------
    gdf_wind : geopandas.GeoDataFrame
        GeoDataFrame containing wind farm locations and attributes.
    gdf_ec : geopandas.GeoDataFrame
        GeoDataFrame containing eddy covariance tower locations and attributes.
    max_match_dist : int
        Maximum haversine distance (in meters) between a wind farm and an eddy covariance tower. Defaults to 100,000 m.
    date : pd.Timestamp, optional
        Date to filter wind farms and eddy covariance towers by their operational status.
        If None, all are included.
    min_height : int, optional
        If specified, `gdf_ec` must contain `height_col`, and the output gains the tower's
        'ec_height_max', 'ec_height_min' and 'ec_sensor_heights'.
    bounds : tuple, optional
        Bounds to apply to the matching in the form (minx, miny, maxx, maxy), in `crs` units.
        If None (default), all towers and wind farms are considered.
    crs : str, optional
        Coordinate reference system of the output and of `bounds`. Defaults to `eddy.DISTANCE_CRS`.
    site_id_col : str, optional
        Column name in `gdf_ec` that contains the unique site IDs for the eddy covariance towers. Defaults to 'site_id'.
    height_col : str, optional
        Column name in `gdf_ec` that contains the measurement heights of the eddy covariance towers. Defaults to 'height'.
        This column is required if `min_height` is specified.
    overlap : bool or int, optional
        If True (default), only return matches where the wind farm and eddy covariance tower have overlapping
        operational periods. If an int, only return matches overlapping by at least that many years.
        If False, return all matches.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per wind farm-tower pair, in the layout of `eddy.geo.neighbours` with prefixes 'wind' and 'ec',
        plus 'n_overlap_yrs', 'overlap' and 'overlap_yrs'.
    '''

    # check for required columns
    if site_id_col not in gdf_ec.columns:
        raise ValueError(f"gdf_ec must contain a '{site_id_col}' column with the tower site IDs.")
    if min_height is not None and height_col not in gdf_ec.columns:
        raise ValueError(f"gdf_ec must contain a '{height_col}' column with the tower heights.")
    if not isinstance(overlap, (int, bool)):
        raise ValueError('overlap must be either a boolean for any overlap (True) or ignore overlap (False), or a number of years at a minimum the towers and wind')

    if date is not None:
        gdf_wind = _operational(gdf_wind, date)
        gdf_ec = gdf_ec.loc[(gdf_ec['first_year'] <= date.year) & (gdf_ec['last_year'] >= date.year)]

    if min_height is not None:
        gdf_ec_grouped = gdf_ec.dissolve(
            by = site_id_col, aggfunc = None,
            **{
                'height_max': (height_col, 'max'),
                'height_min': (height_col, 'min'),
                'n_sensors': (site_id_col, 'count'),
                'sensor_heights': (height_col, lambda x: list(x)),
                'first_year': ('first_year', 'min'),
                'last_year': ('last_year', 'max')
            }
        )
    else:
        gdf_ec_grouped = gdf_ec.dissolve(
            by = site_id_col, aggfunc = None,
            **{
                'n_sensors': (site_id_col, 'count'),
                'first_year': ('first_year', 'min'),
                'last_year': ('last_year', 'max')
            }
        )

    gdf_nbrs = neighbours(
        gdf_wind, gdf_ec_grouped, max_match_dist,
        left_prefix = 'wind', right_prefix = 'ec', bounds = bounds, crs = crs
    )
    wind_id_col = gdf_nbrs.columns[0]

    # Calculate temporal overlaps -- TODO: include EC tower measurement periods: for ameriflux, this is in the BADM: gdf_nbrs = gdf_nbrs.join(gdf_ameriflux_badm[['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = site_id_col)
    fill_na_far_future = lambda x: 9999 if pd.isna(x) else x

    gdf_nbrs['n_overlap_yrs'] = gdf_nbrs.apply(
        lambda row: max(0, min(fill_na_far_future(row['wind_retired_year']), row['ec_last_year']) - max(fill_na_far_future(row['wind_start_year']), row['ec_first_year'])),
        axis = 1
    ).astype(int)

    gdf_nbrs['overlap'] = gdf_nbrs['n_overlap_yrs'] > 0
    gdf_nbrs['overlap_yrs'] = gdf_nbrs.apply(
        lambda row: list(range(max(fill_na_far_future(row['wind_start_year']), row['ec_first_year']), min(fill_na_far_future(row['wind_retired_year']), row['ec_last_year']))),
        axis = 1
    )

    if overlap == True:
        gdf_nbrs = gdf_nbrs.loc[gdf_nbrs['overlap']]
    elif isinstance(overlap, int):
        gdf_nbrs = gdf_nbrs.loc[gdf_nbrs['n_overlap_yrs'] >= overlap]

    first_cols = [wind_id_col, 'wind_project', site_id_col, 'dist_m', 'n_overlap_yrs', 'wind_capacity_mw', 'wind_start_year', 'wind_retired_year', 'ec_first_year', 'ec_last_year', 'overlap', 'overlap_yrs']
    first_cols = [col for col in first_cols if col in gdf_nbrs.columns]
    other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
    return gdf_nbrs[first_cols + other_cols]


def offset_neighbours(
        gdf_wind: gpd.GeoDataFrame,
        gdf_offset: gpd.GeoDataFrame,
        *,
        crs: str = DISTANCE_CRS,
        bounds: tuple|None = None,
        date: pd.Timestamp|None = None,
        max_match_dist: int = 10_000,
    ) -> gpd.GeoDataFrame:
    '''
    Match wind farms to every carbon offset project within a given distance.

    Parameters
    ----------
    gdf_wind : geopandas.GeoDataFrame
        GeoDataFrame containing wind farm locations.
    gdf_offset : geopandas.GeoDataFrame
        GeoDataFrame containing carbon offset project boundaries.
    crs : str, optional
        Coordinate reference system of the output and of `bounds`. Defaults to `eddy.DISTANCE_CRS`.
    bounds : tuple, optional
        Bounds to apply to the matching in the form (minx, miny, maxx, maxy), in `crs` units.
        If None (default), all wind farms and projects are considered.
    date : pd.Timestamp, optional
        Date to filter wind farms and carbon offset projects by their operational status.
        Wind farms with no retirement year are treated as still operating.
    max_match_dist : int, optional
        Matching radius, in metres: every offset project whose boundary lies within this
        haversine distance of a wind farm is matched to it. Defaults to 10,000 m.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per wind farm-offset project pair, in the layout of `eddy.geo.neighbours` with prefixes
        'wind' and 'offset'. 'dist_m' is to the nearest vertex of the project boundary, and 0 for farms inside it.
    '''

    if date is not None:
        date = pd.Timestamp(date)
        gdf_wind = _operational(gdf_wind, date)
        gdf_offset = gdf_offset.loc[gdf_offset['start_date'] <= date]

    gdf_matches = neighbours(
        gdf_wind, gdf_offset, max_match_dist,
        left_prefix = 'wind', right_prefix = 'offset', bounds = bounds, crs = crs
    )

    first_cols = [
        *gdf_matches.columns[:2],
        'wind_project_name', 'wind_phase_name', 'wind_capacity_mw',
        'offset_project_id', 'offset_project_name', 'offset_class', 'offset_subclass',
        'dist_m', 'n_offset_within', 'n_wind_within'
    ]
    first_cols = [col for col in first_cols if col in gdf_matches.columns]
    other_cols = [col for col in gdf_matches.columns if col not in first_cols]
    return gdf_matches[first_cols + other_cols]
