__all__ = ['neighbours']

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from sklearn.neighbors import BallTree

from eddy.util import DISTANCE_CRS

RADIANS_TO_METERS = 6.371e6  # approximate mean radius of the Earth in meters


def _pairs_within(gdf_left: gpd.GeoDataFrame, gdf_right: gpd.GeoDataFrame, max_dist_m: float) -> pd.DataFrame:
    '''
    Every (left, right) pair within `max_dist_m`, as positional indices with their haversine separation in metres.
    Left geometries must be points. Right geometries may be points or polygons: polygon distances are to the
    nearest boundary vertex, and are 0 where the left point lies inside the polygon.
    '''
    left_lonlat = shapely.get_coordinates(gdf_left.geometry.to_crs('EPSG:4326').values)
    right_lonlat, right_pos = shapely.get_coordinates(gdf_right.geometry.to_crs('EPSG:4326').values, return_index=True)

    ball_tree = BallTree(np.deg2rad(right_lonlat[:, ::-1]), metric='haversine')
    vertex_matches, distances_rad = ball_tree.query_radius(
        np.deg2rad(left_lonlat[:, ::-1]),
        r=max_dist_m / RADIANS_TO_METERS,  # query must be in radians to match lat/lon units
        return_distance=True
    )
    left_ind = np.repeat(np.arange(len(vertex_matches)), [len(ms) for ms in vertex_matches])
    vertex_ind = np.concatenate(vertex_matches) if len(vertex_matches) > 0 else np.empty(0, dtype=int)
    distances_rad = np.concatenate(distances_rad) if len(distances_rad) > 0 else np.empty(0, dtype=float)

    pairs = pd.DataFrame({
        'left_pos': left_ind,
        'right_pos': right_pos[vertex_ind],
        'dist_m': distances_rad * RADIANS_TO_METERS
    })

    is_polygon = gdf_right.geom_type.isin(['Polygon', 'MultiPolygon']).to_numpy()
    if is_polygon.any():
        inside = gpd.sjoin(
            gpd.GeoDataFrame({'left_pos': np.arange(len(gdf_left))}, geometry=gdf_left.geometry.values),
            gpd.GeoDataFrame({'right_pos': np.flatnonzero(is_polygon)}, geometry=gdf_right.geometry.values[is_polygon]),
            how='inner', predicate='within'
        )
        pairs = pd.concat([pairs, pd.DataFrame({
            'left_pos': inside['left_pos'].to_numpy(),
            'right_pos': inside['right_pos'].to_numpy(),
            'dist_m': 0.0
        })])

    return pairs.groupby(['left_pos', 'right_pos'], as_index=False)['dist_m'].min()


def _prefixed(
        gdf: gpd.GeoDataFrame, positions: np.ndarray, prefix: str, *, taken: str|None = None
    ) -> tuple[pd.DataFrame, str]:
    '''
    Rows of `gdf` at `positions`, with the index as an id column and every other column prefixed.
    The id column is named after the index, or '<prefix>_id' if the index is unnamed or its name is `taken`.
    '''
    id_col = gdf.index.name if gdf.index.name not in (None, taken) else f'{prefix}_id'
    df = gdf.iloc[positions].rename_axis(id_col).reset_index()
    df = df.rename(columns={col: f'{prefix}_{col}' for col in df.columns if col != id_col})
    return df, id_col


def neighbours(
        gdf_left: gpd.GeoDataFrame,
        gdf_right: gpd.GeoDataFrame,
        max_dist_m: float,
        *,
        left_prefix: str,
        right_prefix: str,
        bounds: tuple|None = None,
        crs: str = DISTANCE_CRS
    ) -> gpd.GeoDataFrame:
    '''
    Match every left point to every right geometry within a given haversine distance.

    Parameters
    ----------
    gdf_left : geopandas.GeoDataFrame
        Point geometries to match from, e.g. wind farms.
    gdf_right : geopandas.GeoDataFrame
        Point or polygon geometries to match to, e.g. eddy covariance towers or offset projects.
        Polygon distances are to the nearest boundary vertex, and 0 for points inside the polygon.
    max_dist_m : float
        Matching radius, in metres.
    left_prefix, right_prefix : str
        Prefixes for the columns of each side in the output, e.g. 'wind' and 'ec'.
    bounds : tuple, optional
        Bounds to apply before matching, in the form (minx, miny, maxx, maxy) in `crs` units.
        If None (default), all geometries are considered.
    crs : str, optional
        Coordinate reference system of the output and of `bounds`. Defaults to `eddy.DISTANCE_CRS`.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per (left, right) pair, indexed by 'match_id' ('<left id>_<right id>') and sorted by 'dist_m'.
        Columns: the left and right ids (named after each input's index, or '<prefix>_id' if unnamed or the
        same on both sides), 'dist_m', 'n_<right_prefix>_within'
        (right matches per left row), 'n_<left_prefix>_within' (left matches per right row), then every input
        column prefixed with its side's prefix. '<left_prefix>_geometry' is the active geometry, and
        '<right_prefix>_geometry' holds the right geometry.
    '''
    gdf_left = gdf_left.to_crs(crs)
    gdf_right = gdf_right.to_crs(crs)

    if bounds is not None:
        bounds = np.array(bounds)
        gdf_left = gdf_left.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        gdf_right = gdf_right.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    pairs = _pairs_within(gdf_left, gdf_right, max_dist_m)

    df_left, left_id = _prefixed(gdf_left, pairs['left_pos'].to_numpy(), left_prefix)
    df_right, right_id = _prefixed(gdf_right, pairs['right_pos'].to_numpy(), right_prefix, taken=left_id)
    ids = pd.DataFrame({left_id: df_left.pop(left_id), right_id: df_right.pop(right_id)})
    ids['dist_m'] = pairs['dist_m'].to_numpy()
    ids[f'n_{right_prefix}_within'] = ids.groupby(left_id)['dist_m'].transform('size')
    ids[f'n_{left_prefix}_within'] = ids.groupby(right_id)['dist_m'].transform('size')

    gdf_matches = gpd.GeoDataFrame(
        pd.concat([ids, df_left, df_right], axis=1),
        geometry=f'{left_prefix}_geometry', crs=crs
    )
    gdf_matches.index = gdf_matches[left_id].astype(str) + '_' + gdf_matches[right_id].astype(str)
    gdf_matches.index.name = 'match_id'

    return gdf_matches.sort_values(by='dist_m')
