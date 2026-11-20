__all__ = ['plot_ec_neighbours', 'explore_ec_neighbours', 'explore_offset_neighbours']

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import LineString
from shapely.ops import nearest_points

from eddy.util import DISTANCE_CRS, PLOTTING_CRS, BASEMAP_DEFAULT
from eddy.geo.data import load_natural_earth_states


def plot_ec_neighbours(
        gdf_matches: gpd.GeoDataFrame,
        *,
        min_height: int|None = None, 
        gdf_borders: gpd.GeoDataFrame|None = None,
        crs: str = DISTANCE_CRS,
        pad: int = 10_000,
        marker_wind: str = '1', marker_size_wind: int = 100, marker_width_wind: float = 0.75, colour_wind: str = 'blue',
        marker_size_ec: int = 25, marker_ec: str = '^', colour_ec: str = 'green',
        alpha: float = 0.5, 
        figsize = (7, 5),
        bounds: tuple|None = None,
        scale_bar: bool = True
    ):
    '''
    Plot wind farm to eddy covariance tower matches, optionally only for towers above a minimum height.

    Parameters
    ----------
    gdf_matches : geopandas.GeoDataFrame
        GeoDataFrame containing the matched wind turbines and eddy covariance towers, as returned by `eddy.turbines.ec_neighbours`.
    min_height : int, optional
            Minimum height of eddy covariance towers to include in the plot. If None, all towers are included. Defaults to None.
            If specified, `gdf_matches` must contain 'ec_height_max', i.e. `ec_neighbours` was run with `min_height`.
    gdf_borders : geopandas.GeoDataFrame, optional
        GeoDataFrame containing country/state borders to plot in the background. If None (default), Natural Earth 10m states/provinces.
    crs : str, optional
        Coordinate reference system to use for plotting. Defaults to `eddy.DISTANCE_CRS`, which has units of meters.
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

    if gdf_borders is None:
        gdf_borders = load_natural_earth_states()

    fig, ax = plt.subplots(1,1, figsize = figsize)

    if min_height is not None:
        height_ind = gdf_matches['ec_height_max'] >= min_height
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
    gdf_matches.set_geometry('ec_geometry').loc[height_ind].to_crs(crs).plot(  # eddy covariance
        color = colour_ec, 
        ax = ax, 
        legend = False, 
        markersize = marker_size_ec, alpha = alpha,
        marker = marker_ec
    )

    gdf_borders.to_crs(crs).plot(ax=ax, facecolor = 'None', linewidth = 0.25, autolim = False, zorder = 0)

    # lines
    gdf_matches.loc[height_ind].apply(lambda row: LineString([row['wind_geometry'].centroid, row['ec_geometry'].centroid]) if not pd.isnull(row[['ec_geometry', 'wind_geometry']].values).any() else None, axis = 1).plot(
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


def explore_ec_neighbours(
        gdf_matches: gpd.GeoDataFrame, 
        *, 
        gdf_borders: gpd.GeoDataFrame|None = None,
        crs = DISTANCE_CRS,
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
        GeoDataFrame containing the matched wind turbines and eddy covariance towers, as returned by `eddy.turbines.ec_neighbours`.
    gdf_borders : geopandas.GeoDataFrame | None, optional
        GeoDataFrame containing country/state borders to plot in the background. If None (default), no borders are plotted.
    crs : str, optional
        Coordinate reference system to use for plotting. Defaults to `eddy.DISTANCE_CRS`, which has units of meters.
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
            'wind_project_name': 'Project Name',
            'wind_capacity_mw': 'Capacity (MW)',
            'wind_start_year': 'Start year',
            'wind_retired_year': 'Retired year'
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
        'ec_flux_measurements_date_start': 'EC Measurements Start',
        'ec_flux_measurements_date_end': 'EC Measurements End',
        'ec_height_max': 'Max Height',
        'ec_height_min': 'Min Height',
        'ec_n_sensors': 'Number of Sensors'
    }

    # the cols we actually have
    actual_cols_rename = {col: desired_cols_rename[col] for col in desired_cols_rename if col in gdf_matches.columns}

    gdf_matches.set_geometry('ec_geometry').to_crs(crs).rename(
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

    geom_lat_lon = gdf_matches['wind_geometry'].to_crs(crs)
    geom_af_lat_lon = gdf_matches['ec_geometry'].to_crs(crs)

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


def _turbine_icon(colour: str, size: int) -> folium.DivIcon:
    '''
    A turbine icon :). Turbine's base is at the position of the marker.

    Parameters
    ----------
    colour : str
        The CSS colour for the icon, e.g. 'blue' or '#1f6feb'.
    size : int
        Height and width of the icon (the icon is square), in pixels.

    Returns
    -------
    folium.DivIcon
    '''
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{colour}" stroke-width="2" stroke-linecap="round">'
        '<line x1="12" y1="12" x2="12" y2="23"/>'      # mast
        '<line x1="12" y1="12" x2="12" y2="3.5"/>'     # blade, up
        '<line x1="12" y1="12" x2="4.6" y2="16.2"/>'   # blade, lower left
        '<line x1="12" y1="12" x2="19.4" y2="16.2"/>'  # blade, lower right
        f'<circle cx="12" cy="12" r="1.4" fill="{colour}" stroke="none"/>'  # hub
        '</svg>'
    )
    return folium.DivIcon(
        html = svg,
        icon_size = (size, size),
        icon_anchor = (size / 2, size * 23 / 24)
    )



def explore_offset_neighbours(
        gdf_matches: gpd.GeoDataFrame,
        *,
        gdf_base: gpd.GeoDataFrame|None = None,
        colour_wind: str = 'blue',
        colour_offset: str = 'green',
        colour_link: str = 'grey',
        marker_size_wind: int = 22,
        alpha: float = 0.5,
        simplify_m: float|None = None,
        tiles: str = BASEMAP_DEFAULT,
        link_size: int = 1
    ) -> folium.Map:
    '''
    Plot wind turbine to carbon offset project matches on an interactive folium map.

    Turbines are drawn as markers and offset projects as filled regions, with lines
    connecting the turbine to the nearest point of the offset project.


    Parameters
    ----------
    gdf_matches : geopandas.GeoDataFrame
        Output of `eddy.turbines.offset_neighbours`. Must have the wind farm id as its first column,
        wind farm geometry in 'wind_geometry', offset project geometry in 'offset_geometry',
        and the separation distance in 'dist_m'.
    gdf_base : geopandas.GeoDataFrame, optional
        Boundaries (e.g. countries) drawn underneath the matches.
    colour_wind, colour_offset, colour_link : str, optional
        Colours of the turbine markers, offset regions, and their connecting lines.
        Accepts any valid CSS colour, e.g. 'blue' or '#1f6feb'.
    marker_size_wind : int, optional
        Height of the turbine markers, in pixels.
    alpha : float, optional
        Fill opacity of the offset regions.
    simplify_m : float, optional
        Tolerance, in the units of the CRS of `gdf_matches` (metres for a projected CRS),
        used to simplify the offset regions before drawing them, reducing output map size.
        Note this does not affect the accuracy of the connecting lines/other distances.
    tiles : str, optional
        Basemap tiles.

    Returns
    -------
    folium.Map
    '''

    gdf = gdf_matches.reset_index(drop = True)
    gdf['turbine_key'] = gdf[gdf_matches.columns[0]]
    crs = gdf.crs

    geom_wind = gpd.GeoSeries(gdf['wind_geometry'].to_numpy(), crs = crs)
    geom_offset = gpd.GeoSeries(gdf['offset_geometry'].to_numpy(), crs = crs)

    valid = (geom_wind.notna() & geom_offset.notna()).to_numpy()
    gdf = gdf[valid].reset_index(drop = True)
    geom_wind = geom_wind[valid].reset_index(drop = True)
    geom_offset = geom_offset[valid].reset_index(drop = True)

    if gdf.empty:
        raise ValueError('no matches with both a turbine and an offset project geometry to plot')

    # projects are identified by id where possible, and by their geometry otherwise
    gdf['offset_key'] = gdf['offset_project_id'] if 'offset_project_id' in gdf.columns else geom_offset.to_wkb()
    dist_km = (gdf['dist_m'] / 1000).round(2)

    # lines running from the turbine to the nearest point of the project
    lines = gpd.GeoSeries(
        [LineString(nearest_points(wind, offset)) for wind, offset in zip(geom_wind, geom_offset)],
        crs = crs
    ).to_crs(PLOTTING_CRS)

    geom_wind = geom_wind.to_crs(PLOTTING_CRS)
    if simplify_m:
        geom_offset = geom_offset.simplify(simplify_m)
    geom_offset = geom_offset.to_crs(PLOTTING_CRS)

    # remove duplicates to avoid multiple tooltips for the same turbine or offset project
    nearest = gdf.sort_values('dist_m')
    rows_wind = nearest.drop_duplicates(subset = 'turbine_key').index
    rows_offset = nearest.drop_duplicates(subset = 'offset_key').index

    def tooltip_frame(rows, cols, geom):
        '''Frame of the named columns, over the matches in `rows`, ready to be drawn.'''
        cols = {col: name for col, name in cols.items() if col in gdf.columns}
        return gpd.GeoDataFrame(
            {
                name: gdf.loc[rows, col].astype(str) if isinstance(gdf[col].dtype, pd.CategoricalDtype)
                else gdf.loc[rows, col]
                for col, name in cols.items()
            },
            geometry = geom.loc[rows], crs = PLOTTING_CRS
        )

    # ---- boundaries underneath ----
    bounds = pd.concat([geom_wind, geom_offset]).total_bounds
    pad = 0.5
    if gdf_base is not None:
        m = gdf_base.to_crs(PLOTTING_CRS).cx[
            bounds[0] - pad:bounds[2] + pad, bounds[1] - pad:bounds[3] + pad
        ].explore(
            color = 'None', style_kwds = {'color': 'black', 'weight': 0.5},
            tiles = tiles, tooltip = False, name = 'Boundaries'
        )
    else:
        m = folium.Map(tiles = tiles)
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # ---- offset projects ----
    gdf_offset_plot = tooltip_frame(
        rows_offset,
        {
            'offset_project_name': 'Offset Project',
            'offset_project_id': 'Offset Project ID',
            'offset_class': 'Offset Class',
            'offset_subclass': 'Offset Subclass',
            'offset_country': 'Country',
            'offset_project_status': 'Project Status',
            'offset_area_ha': 'Area (ha)',
            'n_wind_within': 'Turbines Within Radius'
        },
        geom_offset
    )
    gdf_offset_plot['Nearest Turbine (km)'] = dist_km.loc[rows_offset]
    if 'wind_capacity_mw' in gdf.columns:
        gdf_offset_plot['Capacity Within Radius (MW)'] = gdf.loc[rows_offset, 'offset_key'].map(
            gdf.groupby('offset_key', observed = True)['wind_capacity_mw'].sum()
        ).round(1)

    gdf_offset_plot.explore(
        m = m,
        color = colour_offset,
        style_kwds = {'color': colour_offset, 'weight': 1, 'fillOpacity': alpha},
        tooltip = list(gdf_offset_plot.columns.drop('geometry')),
        name = 'Carbon offset projects'
    )

    # ---- wind turbines ----
    gdf_wind_plot = tooltip_frame(
        rows_wind,
        {
            'wind_project_name': 'Turbine Project',
            'wind_phase_name': 'Phase',
            'wind_capacity_mw': 'Capacity (MW)',
            'wind_start_year': 'Start year',
            'wind_retired_year': 'Retired year',
            'n_offset_within': 'Offsets Within Radius',
            'offset_project_name': 'Nearest Offset Project',
            'offset_subclass': 'Nearest Offset Subclass'
        },
        geom_wind
    )
    gdf_wind_plot['Nearest Offset (km)'] = dist_km.loc[rows_wind]

    gdf_wind_plot.explore(
        m = m,
        marker_type = folium.Marker(icon = _turbine_icon(colour_wind, marker_size_wind)),
        tooltip = list(gdf_wind_plot.columns.drop('geometry')),
        name = 'Wind turbines'
    )

    # ---- one line per match ----
    gdf_links_plot = tooltip_frame(
        gdf.index,
        {
            'wind_project_name': 'Turbine Project',
            'wind_phase_name': 'Phase',
            'offset_project_name': 'Offset Project',
            'offset_subclass': 'Offset Subclass'
        },
        lines
    )
    gdf_links_plot['Separation (km)'] = dist_km

    gdf_links_plot.explore(
        m = m,
        color = colour_link,
        style_kwds = {'color': colour_link, 'weight': link_size, 'opacity': 0.25},
        tooltip = list(gdf_links_plot.columns.drop('geometry')),
        name = 'Turbine - offset links'
    )

    folium.LayerControl().add_to(m)
    return m
