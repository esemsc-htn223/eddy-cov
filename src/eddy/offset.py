__all__ = ['wind_offset_matches', 'plot_wind_offset_folium']

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import LineString
from shapely.ops import nearest_points

from eddy.util import DISTANCE_CRS, PLOTTING_CRS, BASEMAP_DEFAULT


def wind_offset_matches(
        gdf_wind: gpd.GeoDataFrame,
        gdf_offset: gpd.GeoDataFrame,
        *,
        crs: str = DISTANCE_CRS,
        bounds: tuple|None = None,
        date: pd.Timestamp|None = None,
        max_match_dist: int = 10_000,
    ) -> gpd.GeoDataFrame:
    '''
    Match wind turbines to every carbon offset project within a given distance of them.

    Parameters
    ----------
    gdf_wind : geopandas.GeoDataFrame
        GeoDataFrame containing wind turbine locations.
    gdf_offset : geopandas.GeoDataFrame
        GeoDataFrame containing carbon offset project locations.
    crs : str, optional
        Coordinate reference system to use for matching. Defaults to `eddy.DISTANCE_CRS`.
        Should be a projected CRS (e.g., UTM) to ensure accurate distance calculations.
    bounds : tuple, optional
        Bounds to apply to the matching in the form (minx, miny, maxx, maxy), in `crs` units.
        If None (default), all turbines and projects are considered.
    date : pd.Timestamp, optional
        Date to filter wind turbines and carbon offset projects by their operational status.
        Turbines with no retirement year are treated as still operating.
    max_match_dist : int, optional
        Matching radius, in metres: every offset project whose boundary lies within this
        distance of a turbine is matched to it. Defaults to 10,000 m.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per wind farm-offset project pair within `max_match_dist`. 
        Wind farm location held in 'geometry', while 'geometry_offset' holds the offset project geometry.
        'dist_m' the farm-to-project separation, 'n_offsets_within' the number of projects matched to
        that turbine and 'n_turbines_within' the number of turbines matched to that project.
    '''

    gdf_wind = gdf_wind.to_crs(crs)
    gdf_offset = gdf_offset.to_crs(crs)

    if date is not None:
        date = pd.Timestamp(date)
        # turbines with no retirement year are still operating
        gdf_wind = gdf_wind.loc[
            (gdf_wind['start_year'] <= date.year)
            & ((gdf_wind['retired_year'] >= date.year) | gdf_wind['retired_year'].isna())
        ]
        gdf_offset = gdf_offset.loc[gdf_offset['start_date'] <= date]

    if bounds is not None:
        bounds = np.array(bounds)
        gdf_wind = gdf_wind.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        gdf_offset = gdf_offset.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    # keep the offset geometry alongside the turbine geometry, which the join makes active
    gdf_offset = gdf_offset.copy()
    gdf_offset['geometry_offset'] = gdf_offset['geometry']
    gdf_offset['_offset_key'] = np.arange(len(gdf_offset))  # identifies a project once joined

    # every turbine-project pair within the radius, rather than nearest neighbours only
    gdf_matches = gdf_wind.sjoin(
        gdf_offset,
        how = 'inner', predicate = 'dwithin', distance = max_match_dist,
        lsuffix = 'wind', rsuffix = 'offset'
    )

    # separation of each pair, turbine to project boundary
    gdf_matches['dist_m'] = gdf_matches.geometry.distance(
        gpd.GeoSeries(gdf_matches['geometry_offset'].to_numpy(), index = gdf_matches.index, crs = crs),
        align = False
    )

    # how crowded each side of the pairing is
    gdf_matches['n_offsets_within'] = gdf_matches.groupby(level = 0)['dist_m'].transform('size')
    gdf_matches['n_turbines_within'] = gdf_matches.groupby('_offset_key')['dist_m'].transform('size')
    gdf_matches = gdf_matches.drop(columns = '_offset_key')

    first_cols = [
        'project_name_wind', 'phase_name', 'capacity_mw',
        'project_id', 'project_name_offset', 'class', 'subclass',
        'dist_m', 'n_offsets_within', 'n_turbines_within'
    ]
    first_cols = [col for col in first_cols if col in gdf_matches.columns]
    other_cols = [col for col in gdf_matches.columns if col not in first_cols]
    gdf_matches = gdf_matches[first_cols + other_cols]

    return gdf_matches


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


def plot_wind_offset_folium(
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
        Output of `wind_offset_matches`. Must have turbine geometry in 'geometry', 
        offset project geometry in 'geometry_offset', and the separation distance in 'dist_m'.
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

    gdf = gdf_matches.reset_index(names = 'turbine_key')
    crs = gdf.crs

    geom_wind = gpd.GeoSeries(gdf['geometry'].to_numpy(), crs = crs)
    geom_offset = gpd.GeoSeries(gdf['geometry_offset'].to_numpy(), crs = crs)

    valid = (geom_wind.notna() & geom_offset.notna()).to_numpy()
    gdf = gdf[valid].reset_index(drop = True)
    geom_wind = geom_wind[valid].reset_index(drop = True)
    geom_offset = geom_offset[valid].reset_index(drop = True)

    if gdf.empty:
        raise ValueError('no matches with both a turbine and an offset project geometry to plot')

    # projects are identified by id where possible, and by their geometry otherwise
    gdf['offset_key'] = gdf['project_id'] if 'project_id' in gdf.columns else geom_offset.to_wkb()
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
            'project_name_offset': 'Offset Project',
            'project_id': 'Offset Project ID',
            'class': 'Offset Class',
            'subclass': 'Offset Subclass',
            'country': 'Country',
            'project_status': 'Project Status',
            'area_ha': 'Area (ha)',
            'n_turbines_within': 'Turbines Within Radius'
        },
        geom_offset
    )
    gdf_offset_plot['Nearest Turbine (km)'] = dist_km.loc[rows_offset]
    if 'capacity_mw' in gdf.columns:
        gdf_offset_plot['Capacity Within Radius (MW)'] = gdf.loc[rows_offset, 'offset_key'].map(
            gdf.groupby('offset_key', observed = True)['capacity_mw'].sum()
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
            'project_name_wind': 'Turbine Project',
            'phase_name': 'Phase',
            'capacity_mw': 'Capacity (MW)',
            'start_year': 'Start year',
            'retired_year': 'Retired year',
            'n_offsets_within': 'Offsets Within Radius',
            'project_name_offset': 'Nearest Offset Project',
            'subclass': 'Nearest Offset Subclass'
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
            'project_name_wind': 'Turbine Project',
            'phase_name': 'Phase',
            'project_name_offset': 'Offset Project',
            'subclass': 'Offset Subclass'
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
