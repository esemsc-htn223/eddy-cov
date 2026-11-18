import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import folium
from dotenv import load_dotenv

from eddy.data.common import load_natural_earth_countries, load_natural_earth_states
GDF_COUNTRIES = load_natural_earth_countries()
GDF_STATES = load_natural_earth_states()


def wind_farm_ec_neighbours(
        gdf_wind: gpd.GeoDataFrame, 
        gdf_ec: gpd.GeoDataFrame, 
        max_match_dist: int = 100_000, 
        *,
        date: pd.Timestamp|None = None,
        min_height: int|None = None, 
        bounds: tuple|None = None,
        crs: str = 'EPSG:3857',
        site_id_col: str = 'site_id', 
        height_col: str = 'height',
        overlap: bool|int = True
    ):
    '''
    Match wind turbine locations to eddy covariance towers within a specified distance.

    Parameters
    ----------
    gdf_wind : geopandas.GeoDataFrame
        GeoDataFrame containing wind turbine locations and attributes.
    gdf_ec : geopandas.GeoDataFrame
        GeoDataFrame containing eddy covariance tower locations and attributes.
    max_match_dist : int
        Maximum distance (in meters) to consider a wind turbine as a match for an eddy covariance tower. Defaults to 100,000 m. If None, all turbines are considered regardless of distance.
    date : pd.Timestamp, optional
        Date to filter wind turbines and eddy covariance towers by their operational status.
        If None, all turbines are included.
    min_height : int, optional
        Minimum height of eddy covariance towers to include. If None, all towers are included. Defaults to None.
        If specified, `gdf_ec` must contain the specified `height_col` with the measurement heights.
    bounds : tuple, optional
        Bounds to apply to the matching in the form (minx, miny, maxx, maxy). If None (default), all towers and turbines are considered. 
        If specified, only towers and turbines within the bounds are considered.
    crs : str, optional
        Coordinate reference system to use for matching. Defaults to 'EPSG:3857' (Web Mercator).
    site_id_col : str, optional
        Column name in `gdf_ec` that contains the unique site IDs for the eddy covariance towers. Defaults to 'site_id'.
    height_col : str, optional
        Column name in `gdf_ec` that contains the measurement heights of the eddy covariance towers. Defaults to 'height'. 
        This column is required if `min_height` is specified.
    overlap : bool, optional
        If True, only return matches where the wind turbine and eddy covariance tower have overlapping operational periods. Defaults to True.
    group_wind_by_loc : bool, optional
        If True, wind farms are grouped by their gem_location_id. If False (default), returns rows for each phase of the wind farm project.


    Returns
    -------
    geopandas.GeoDataFrame, optional
        The GeoDataFrame of matched wind turbines and eddy covariance towers.
    '''

    # check for required columns
    if site_id_col not in gdf_ec.columns:
        raise ValueError(f"gdf_ec must contain a '{site_id_col}' column with the tower site IDs.")
    if min_height is not None and height_col not in gdf_ec.columns:
        raise ValueError(f"gdf_ec must contain a '{height_col}' column with the tower heights.")
    if not isinstance(overlap, (int, bool)):
        raise ValueError('overlap must be either a boolean for any overlap (True) or ignore overlap (False), or a number of years at a minimum the towers and wind')

    gdf_wind = gdf_wind.to_crs(crs).copy()
    gdf_ec = gdf_ec.to_crs(crs).copy()

    # slice by time and space
    if date is not None:
        gdf_wind = gdf_wind.loc[(gdf_wind['start_year'] <= date.year) & (gdf_wind['retired_year'].fillna(9999) >= date.year)]  # retired_year = na when it has never retired - so treat this as the far future
        gdf_ec = gdf_ec.loc[(gdf_ec['first_year'] <= date.year) & (gdf_ec['last_year'] >= date.year)]

    if bounds:
        bounds = np.array(bounds)
        gdf_wind = gdf_wind.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        gdf_ec = gdf_ec.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    if min_height is not None:
        gdf_ec_grouped = gdf_ec.dissolve(
            by = site_id_col, aggfunc = None, 
            **{
                'eddy_height_max': (height_col, 'max'), 
                'eddy_height_min': (height_col, 'min'), 
                'eddy_n_sensors': (site_id_col, 'count'),
                'eddy_sensor_heights': (height_col, lambda x: list(x)),
                'eddy_first_year': ('first_year', 'min'),
                'eddy_last_year': ('last_year', 'max')
            }
        )
    else:
        gdf_ec_grouped = gdf_ec.dissolve(
            by = site_id_col, aggfunc = None,
            **{
                'eddy_n_sensors': (site_id_col, 'count'),
                'eddy_first_year': ('first_year', 'min'),
                'eddy_last_year': ('last_year', 'max')
            }
        )

    gdf_ec_grouped['eddy_geometry'] = gdf_ec_grouped['geometry']
    wind_cols = ['project', 'capacity_mw', 'start_year', 'retired_year', 'gem_location_id', 
                 'project_name', 'phase_name', 'installation_type', 'status', 'location_accuracy', 'geometry']

    RADIANS_TO_METERS = 6.371e6  # approximate mean radius of the Earth in meters

    ball_tree = BallTree(
        np.deg2rad(
            np.reshape(gdf_wind.to_crs('EPSG:4326').geometry.apply(lambda geom: (geom.y, geom.x)).to_list(), (-1, 2))
        ), 
        metric='haversine'
    )
    wind_matches, distances_rad = ball_tree.query_radius(
        np.deg2rad(
            np.reshape(gdf_ec_grouped.to_crs('EPSG:4326').geometry.apply(lambda geom: (geom.y, geom.x)).to_list(), (-1, 2))
        ), 
        r=max_match_dist / RADIANS_TO_METERS,  # query must be in radians to match lat/lon units
        return_distance=True
    )
    ec_ind = np.repeat(np.arange(len(distances_rad)), [len(ds) for ds in distances_rad])
    wind_ind = np.concatenate(wind_matches) if len(wind_matches) > 0 else np.empty(0, dtype = int)  # w ind
    distances_rad = np.concatenate(distances_rad) if len(distances_rad) > 0 else np.empty(0, dtype=float)

    # collect results into a gdf
    gdf_nbrs = gdf_wind.iloc[wind_ind][wind_cols].rename(columns={col: f'wind_{col}' for col in wind_cols}).reset_index()
    wind_id_col = gdf_nbrs.columns[0]

    gdf_nbrs['dist_m'] = distances_rad * RADIANS_TO_METERS
    gdf_nbrs[site_id_col] = gdf_ec_grouped.iloc[ec_ind].index.to_numpy()
    gdf_nbrs.index = gdf_nbrs[wind_id_col].astype(str) + '_' + gdf_nbrs[site_id_col].astype(str)
    gdf_nbrs = gdf_nbrs.join(gdf_ec_grouped, on = site_id_col, how = 'left')
    gdf_nbrs.set_geometry('wind_geometry', inplace = True)

    del gdf_ec_grouped, ball_tree  # free memory


    # Calculate temporal overlaps -- TODO: include EC tower measurement periods: for ameriflux, this is in the BADM: gdf_nbrs = gdf_nbrs.join(gdf_ameriflux_badm[['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = site_id_col)
    fill_na_far_future = lambda x: 9999 if pd.isna(x) else x

    gdf_nbrs['n_overlap_yrs'] = gdf_nbrs.apply(
        lambda row: max(0, min(fill_na_far_future(row['wind_retired_year']), row['eddy_last_year']) - max(fill_na_far_future(row['wind_start_year']), row['eddy_first_year'])),
        axis = 1
    ).astype(int)

    gdf_nbrs['overlap'] = gdf_nbrs['n_overlap_yrs'] > 0
    gdf_nbrs['overlap_yrs'] = gdf_nbrs.apply(
        lambda row: list(range(max(fill_na_far_future(row['wind_start_year']), row['eddy_first_year']), min(fill_na_far_future(row['wind_retired_year']), row['eddy_last_year']))),
        axis = 1
    )

    if overlap == True:
        gdf_nbrs = gdf_nbrs.loc[gdf_nbrs['overlap']]
    elif isinstance(overlap, int):
        gdf_nbrs = gdf_nbrs.loc[gdf_nbrs['n_overlap_yrs'] >= overlap]

    first_cols = [wind_id_col, 'wind_project', site_id_col, 'dist_m', 'n_overlap_yrs', 'wind_capacity_mw', 'wind_start_year', 'wind_retired_year', 'eddy_first_year', 'eddy_last_year', 'overlap', 'overlap_yrs']
    other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
    gdf_nbrs = gdf_nbrs[first_cols + other_cols]
    gdf_nbrs.index.name = 'match_id'

    return gdf_nbrs.sort_values(by = 'dist_m')


def plot_wind_farm_ec_neighbours(
        gdf_matches: gpd.GeoDataFrame,
        *,
        min_height: int|None = None, 
        gdf_borders: gpd.GeoDataFrame = GDF_STATES,
        crs: str = 'EPSG:3857',
        pad: int = 10_000,
        marker_wind: str = '1', marker_size_wind: int = 100, marker_width_wind: float = 0.75, colour_wind: str = 'blue',
        marker_size_ec: int = 25, marker_ec: str = '^', colour_ec: str = 'green',
        alpha: float = 0.5, 
        figsize = (7, 5),
        bounds: tuple|None = None,
        scale_bar: bool = True
    ):
    '''
    Plot the nearest wind turbines to eddy covariance towers, within a maximum distance, and with a minimum height for the towers.

    Parameters
    ----------
    gdf_matches : geopandas.GeoDataFrame
        GeoDataFrame containing the matched wind turbines and eddy covariance towers, as returned by `wind_farm_ec_neighbours`.
    min_height : int, optional
            Minimum height of eddy covariance towers to include in the plot. If None, all towers are included. Defaults to None.
            If specified, `gdf_matches` must contain a 'Height' column with the tower heights.
    gdf_borders : geopandas.GeoDataFrame, optional
        GeoDataFrame containing country/state borders to plot in the background. Defaults to Natural Earth 10m states/provinces.
    crs : str, optional
        Coordinate reference system to use for plotting. Defaults to 'EPSG:3857' (Web Mercator), which has units of meters.
    pad : int, optional
        Padding (in the units of `crs`) around the matched points to set the plot bounds. Defaults to 10,000 m.
    marker_wind : str, optional
        Marker style for wind turbines. Defaults to '1'.
    marker_size_wind : int, optional
        Size of the marker for wind turbines. Defaults to 100.
    marker_width_wind : float, optional
        Width of the marker for wind turbines. Defaults to 0.75.
    colour_wind : str, optional
        Color of the marker for wind turbines. Defaults to 'blue'.
    marker_size_ec : int, optional
        Size of the marker for eddy covariance towers. Defaults to 25.
    marker_ec : str, optional
        Marker style for eddy covariance towers. Defaults to '^'.
    colour_ec : str, optional
        Color of the marker for eddy covariance towers. Defaults to 'green'.
    alpha : float, optional
        Transparency of the markers. Defaults to 0.5.
    return_nbrs_gdf : bool, optional
        If True, return the GeoDataFrame of matched wind turbines and eddy covariance towers. Defaults to False.
    figsize : tuple, optional
        Size of the figure to create. Defaults to (7, 5).
    bounds : tuple, optional
        Bounds of the plot in the form (minx, miny, maxx, maxy). If None (default), the bounds are determined from the matched points. 
        Padding is applied if specified.
    scale_bar : bool, optional
        If True, add a scale bar to the plot. Defaults to True.

    Returns
    -------
    matplotlib.axes.Axes
    '''

    fig, ax = plt.subplots(1,1, figsize = figsize)

    if min_height is not None:
        height_ind = gdf_matches['height_max'] >= min_height
    else:
        height_ind = gdf_matches.index

    # apply bounds, adding padding if specified
    if bounds is None:
        bounds = gdf_matches.loc[height_ind].total_bounds
    else:
        bounds = np.array(bounds)
    bounds[0:2] = bounds[0:2] - pad
    bounds[2:4] = bounds[2:4] + pad

    gdf_matches.loc[height_ind].to_crs(crs).plot(  # wind
        color = colour_wind,
        ax = ax, 
        markersize = marker_size_wind, alpha = alpha, 
        marker = marker_wind, linewidth = marker_width_wind
    )
    gdf_matches.set_geometry('eddy_geometry').loc[height_ind].to_crs(crs).plot(  # eddy covariance
        color = colour_ec, 
        ax = ax, 
        legend = False, 
        markersize = marker_size_ec, alpha = alpha,
        marker = marker_ec
    )

    gdf_borders.to_crs(crs).plot(ax=ax, facecolor = 'None', linewidth = 0.25, autolim = False, zorder = 0)

    # lines
    gdf_matches.loc[height_ind].apply(lambda row: LineString([row['geometry'].centroid, row['eddy_geometry'].centroid]) if not pd.isnull(row[['eddy_geometry', 'geometry']].values).any() else None, axis = 1).plot(
        linewidth = 0.1, color = 'grey',
        ax = ax
    )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.legend(
        ['Wind Farm', 'EC Tower'],
        #loc = 'lower right',
        loc = 'best',
        markerscale = 1,
        fontsize = 10,
        frameon = True,
        fancybox = True,
        framealpha = 0.5,
    )
    if scale_bar:
        ax.add_artist(ScaleBar(1))  # add scale bar to give a gauge of distance
    ax.set_axis_off()

    return ax


def explore_wind_farm_ec_neighbours(
        gdf_matches: gpd.GeoDataFrame, 
        *, 
        gdf_borders: gpd.GeoDataFrame|None = None,
        crs = 'EPSG:3857',
        colour_wind = 'blue', 
        colour_ec = 'green', 
        marker_size_wind = 100, 
        marker_width_wind = 0.75, 
        marker_size_ec = 25, 
        marker_width_ec = 0.5,
        alpha = 0.5,
        tiles = 'OpenTopoMap'
) -> folium.Map:
    '''
    Plot the nearest wind turbines to eddy covariance towers on an interactive folium map.

    Parameters
    ----------
    gdf_matches : geopandas.GeoDataFrame
        GeoDataFrame containing the matched wind turbines and eddy covariance towers, as returned by `wind_farm_ec_neighbours`.
    gdf_borders : geopandas.GeoDataFrame | None, optional
        GeoDataFrame containing country/state borders to plot in the background. Defaults to Natural Earth 10m states/provinces.
        If None, no borders are plotted.
    crs : str, optional
        Coordinate reference system to use for plotting. Defaults to 'EPSG:3857' (Web Mercator), which has units of meters.
    colour_wind : str, optional
        Color of the marker for wind turbines. Defaults to 'blue'.
    colour_ec : str, optional
        Color of the marker for eddy covariance towers. Defaults to 'green'.
    marker_size_wind : int, optional
        Size of the marker for wind turbines. Defaults to 100.
    marker_width_wind : float, optional
        Width of the marker for wind turbines. Defaults to 0.75.
    marker_size_ec : int, optional
        Size of the marker for eddy covariance towers. Defaults to 25.
    marker_width_ec : float, optional
        Width of the marker for eddy covariance towers. Defaults to 0.5.
    alpha : float, optional
        Transparency of the markers. Defaults to 0.5.
    tiles : str, optional
        Tile set to use for the folium map. Defaults to 'CartoDB positron'.
    '''
    load_dotenv()

    if gdf_borders is None:
        m = folium.Map(tiles = tiles)
    else:
        m = gdf_borders.to_crs(crs).explore(color = 'None', style_kwds = {'color': 'black', 'weight': 0.5}, tiles = tiles, tooltip = False)
    gdf_matches.to_crs(crs).rename(
        columns={
            'site_id': 'Neighbour Site ID', 
            'dist_m': 'Dist to EC Neighbour (m)',
            'project_name': 'Project Name',
            'capacity_mw': 'Capacity (MW)',
            'start_year': 'Start year',
            'retired_year': 'Retired year'
        }
    ).explore(
        m = m, 
        color = colour_wind, 
        marker_type = folium.Marker(
            icon = folium.Icon(color = 'blue', icon = 'wind', prefix = 'fa')
        ), 
        marker_kwds = {
            'radius': marker_size_wind/10, 
            'weight': marker_width_wind, 
            'fill_opacity': alpha
        }, 
        tooltip = ['Project Name', 'Capacity (MW)', 'Dist to EC Neighbour (m)', 'Neighbour Site ID', 'Start year', 'Retired year']
    )

    desired_cols_rename = {  # all the cols we would like to have
        'site_id': 'Site ID',
        'flux_measurements_date_start': 'EC Measurements Start',
        'flux_measurements_date_end': 'EC Measurements End',
        'height_max': 'Max Height',
        'height_min': 'Min Height',
        'n_sensors': 'Number of Sensors'
    }

    # the cols we actually have
    actual_cols_rename = {col: desired_cols_rename[col] for col in desired_cols_rename if col in gdf_matches.columns}

    gdf_matches.set_geometry('eddy_geometry').to_crs(crs).rename(
        columns=actual_cols_rename
    ).explore(
        m = m,
        color = colour_ec,
        marker_type = folium.Marker(
            icon = folium.Icon(color = 'green', icon = 'tower-broadcast', prefix = 'fa')
        ),
        marker_kwds = {
            'radius': marker_size_ec/10, 
            'weight': marker_width_ec, 
            'fill_opacity': alpha
        },
        tooltip = [col for col in actual_cols_rename.values()]
    )

    geom_lat_lon = gdf_matches['geometry'].to_crs(crs)
    geom_af_lat_lon = gdf_matches['eddy_geometry'].to_crs(crs)

    # create lines between tubines and nearest towers
    for i in range(len(gdf_matches)):
        geom = geom_lat_lon.iloc[i]
        geom_af = geom_af_lat_lon.iloc[i]
        if pd.isna([geom_af, geom]).any():
            # skip if either geometry is NaN
            continue

        dist = gdf_matches['dist_m'].iloc[i] / 1000

        locs = [
            [geom.y, geom.x], 
            [geom_af.y, geom_af.x]
        ]
        line = folium.PolyLine(
            locations = locs,
            color = 'grey', weight = 10,
            #dash_array = '6',
            opacity = 0.25,
            tooltip = f'{dist:.1f} km'
        )
        line.add_to(m)
    return m

