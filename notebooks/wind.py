import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eddy - Wind
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd

    from matplotlib import pyplot as plt
    from matplotlib_scalebar.scalebar import ScaleBar
    import altair as alt
    import folium

    from shapely.geometry import LineString, MultiPoint, Point, box
    from shapely.ops import nearest_points

    import os

    import marimo as mo
    import numpy as np
    import json
    import datetime
    import requests
    import sys
    import time
    from tqdm import tqdm

    import xyzservices.providers as xyz
    from dotenv import load_dotenv
    load_dotenv()
    return (
        LineString,
        Point,
        ScaleBar,
        box,
        folium,
        gpd,
        mo,
        nearest_points,
        np,
        os,
        pd,
        plt,
        xyz,
    )


@app.cell
def _():
    import eddy

    # get dirs too
    from eddy.data.flux import FLUX_DIR, FLUXNET_DIR, TERN_DIR, AMERIFLUX_DIR
    from eddy.util import DATA_DIR, OUT_DIR
    from eddy.data import _standardise_df as standardise_df


    return FLUX_DIR, OUT_DIR, eddy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _(eddy):
    # geographic data
    GDF_COUNTRIES = eddy.data.load_natural_earth_countries()
    GDF_STATES = eddy.data.load_natural_earth_states()

    #_uk_bbox = (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085)
    UK_BBOX = {
        'min_lon': -10.0,
        'max_lon': 2.0,
        'min_lat': 49.0,
        'max_lat': 61.0
    }
    return GDF_COUNTRIES, GDF_STATES, UK_BBOX


@app.cell
async def _(eddy):
    # flux tower locations
    gdf_fluxnet = await eddy.data.flux.load_fluxnet_snapshot()
    return (gdf_fluxnet,)


@app.cell
def _(GDF_COUNTRIES, eddy, mo):
    # wind turbine locations
    gdf_wind = eddy.data.wind.load_wind_locations()

    ax = gdf_wind.to_crs(eddy.PLOTTING_CRS).plot(column = 'capacity_mw', figsize = (10,20), legend = False, markersize = 0.1, alpha = 0.1)
    GDF_COUNTRIES.to_crs(eddy.PLOTTING_CRS).plot(ax=ax, facecolor = 'None', linewidth = 0.25)
    ax.set_axis_off()
    mo.vstack([
        ax,
        gdf_wind
    ])
    return (gdf_wind,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantifying the relevance of EC to wind turbines
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. How close are wind turbines to EC towers?
        - Are towers close enough to turbines to be in a wind wake?
            - Allows us to quantify the effect
    2. How close are wind turbines to offset sites?
        - Are offset sites within range of wind wakes?
            - Allows us to quantify risk, if any
        - Good to classify offset by vegetation type - some may see some benefit from wind wakes (e.g. agricultural), while others will be limited
    3. Are towers located near or on *likely turbine sites*?
        - proposed/planned/constructing sites
        - theoretically good sites for wind turbines?
        - Allows us to quantufy opportunities
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. How close are wind turbines to EC towers?
    """)
    return


@app.cell
def _(gpd, np):
    def wind_ec_matches(
            gdf_wind: gpd.GeoDataFrame, gdf_ec: gpd.GeoDataFrame, 
            *,
            site_id_col: str = 'Site_ID', height_col: str = 'Height',
            min_height: int|None = None, max_match_dist: int = 100_000, 
            crs: str = 'EPSG:3857',
            bounds: tuple|None = None
        ):
        '''
        Match wind turbine locations to their nearest eddy covariance tower location.

        Parameters
        ----------
        gdf_wind : geopandas.GeoDataFrame
            GeoDataFrame containing wind turbine locations and attributes.
        gdf_ec : geopandas.GeoDataFrame
            GeoDataFrame containing eddy covariance tower locations and attributes.
        min_height : int, optional
            Minimum height of eddy covariance towers to include. If None, all towers are included. Defaults to None.
            If specified, `gdf_ec` must contain the specified `height_col` with the measurement heights.
        crs : str, optional
            Coordinate reference system to use for matching. Defaults to 'EPSG:3857' (Web Mercator).
        max_match_dist : int, optional
            Maximum distance (in meters) to consider a wind turbine as a match for an eddy covariance tower. Defaults to 100,000 m.
        bounds : tuple, optional
            Bounds to apply to the matching in the form (minx, miny, maxx, maxy). If None (default), all towers and turbines are considered. 
            If specified, only towers and turbines within the bounds are considered.

        Returns
        -------
        geopandas.GeoDataFrame, optional
            The GeoDataFrame of matched wind turbines and eddy covariance towers.
        '''

        if bounds:
            bounds = np.array(bounds)
            gdf_wind_wkg = gdf_wind.to_crs(crs).cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
            gdf_ec_wkg = gdf_ec.to_crs(crs).cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
        else:
            gdf_wind_wkg = gdf_wind.to_crs(crs).copy()
            gdf_ec_wkg = gdf_ec.to_crs(crs).copy()

        if site_id_col not in gdf_ec.columns:
            raise ValueError(f"gdf_ec must contain a '{site_id_col}' column with the tower site IDs.")

        if min_height is not None:
            if height_col not in gdf_ec.columns:
                raise ValueError(f"gdf_ec must contain a '{height_col}' column with the tower heights.")

            gdf_ec_grouped = gdf_ec_wkg.dissolve(
                by = site_id_col, aggfunc = None, 
                **{
                    'height_max': (height_col, 'max'), 
                    'height_min': (height_col, 'min'), 
                    'n_sensors': (site_id_col, 'count'),
                    'sensor_heights': (height_col, lambda x: list(x))
                }
            )
            gdf_ec_grouped['geometry_eddy'] = gdf_ec_grouped['geometry']
            gdf_ec_grouped = gdf_ec_grouped[gdf_ec_grouped['height_min'] >= min_height]
            gdf_nbrs = gdf_wind_wkg[['project_name', 'phase_name', 'capacity_mw', 'geometry', 'start_year', 'retired_year', 'project']].sjoin_nearest(
                gdf_ec_grouped[['height_max', 'height_min', 'n_sensors', 'geometry_eddy', 'geometry']],
                how = 'left', distance_col = 'dist_m',
                max_distance = max_match_dist
            )
            del gdf_ec_grouped
        else:
            gdf_ec_grouped = gdf_ec_wkg.dissolve(
                by = site_id_col, aggfunc = None,
                **{
                    'n_sensors': (site_id_col, 'count'),
                }
            )
            gdf_ec_grouped['geometry_eddy'] = gdf_ec_grouped['geometry']
            gdf_nbrs = gdf_wind_wkg[['project_name', 'phase_name', 'capacity_mw', 'geometry', 'start_year', 'retired_year', 'project']].sjoin_nearest(
                gdf_ec_grouped[['n_sensors', 'geometry_eddy', 'geometry']],
                how = 'inner', distance_col = 'dist_m',
                max_distance = max_match_dist
            )
            del gdf_ec_grouped

        first_cols = ['project_name', 'phase_name', 'capacity_mw', site_id_col, 'dist_m']
        other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
        gdf_nbrs = gdf_nbrs[first_cols + other_cols]

        # TODO: time alignment of wind turbine operation and EC tower measurement periods
        #gdf_nbrs = gdf_nbrs.join(gdf_ameriflux_badm[['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = site_id_col)

        return gdf_nbrs

    return (wind_ec_matches,)


@app.cell
def _(GDF_STATES, LineString, ScaleBar, folium, gpd, np, pd, plt):
    def plot_wind_ec_matches(
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
            **kwargs
        ):
        '''
        Plot the nearest wind turbines to eddy covariance towers, within a maximum distance, and with a minimum height for the towers.

        Parameters
        ----------
        gdf_matches : geopandas.GeoDataFrame
            GeoDataFrame containing the matched wind turbines and eddy covariance towers, as returned by `wind_ec_matches`.
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
        **kwargs
            Additional keyword arguments to pass to the plotting functions.

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
        gdf_matches.set_geometry('geometry_eddy').loc[height_ind].to_crs(crs).plot(  # eddy covariance
            color = colour_ec, 
            ax = ax, 
            legend = False, 
            markersize = marker_size_ec, alpha = alpha,
            marker = marker_ec
        )

        gdf_borders.to_crs(crs).plot(ax=ax, facecolor = 'None', linewidth = 0.25, autolim = False, zorder = 0)

        # lines
        gdf_matches.loc[height_ind].apply(lambda row: LineString([row['geometry'].centroid, row['geometry_eddy'].centroid]) if not pd.isnull(row[['geometry_eddy', 'geometry']].values).any() else None, axis = 1).plot(
            linewidth = 0.1, color = 'grey',
            ax = ax
        )
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.legend(
            ['Wind Farm', 'EC Tower'],
            loc = 'lower right',
            #loc = 'best',
            markerscale = 1,
            fontsize = 10,
            frameon = True,
            fancybox = True,
            framealpha = 0.5,
        )
        ax.add_artist(ScaleBar(1))  # add scale bar to give a gauge of distance
        ax.set_axis_off()

        return ax


    def plot_wind_ec_matches_folium(
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
            GeoDataFrame containing the matched wind turbines and eddy covariance towers, as returned by `wind_ec_matches`.
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
    
        gdf_matches.set_geometry('geometry_eddy').to_crs(crs).rename(
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
        geom_af_lat_lon = gdf_matches['geometry_eddy'].to_crs(crs)

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



    return plot_wind_ec_matches, plot_wind_ec_matches_folium


@app.cell
def _(gdf_matches, os, plot_wind_ec_matches_folium, xyz):
    provider = xyz.Stadia.StamenTerrain(api_key=os.getenv('STADIA_API_KEY'))
    provider["url"] = provider["url"] + f"?api_key={os.getenv('STADIA_API_KEY')}"

    _m = plot_wind_ec_matches_folium(
        gdf_matches.loc[gdf_matches['site_id'].str.startswith('UK')], 
        tiles = provider
    )
    _m.show_in_browser()
    return


@app.cell
def _(gdf_matches):
    gdf_matches
    return


@app.cell
def _(gdf_fluxnet, gdf_wind, mo, plot_wind_ec_matches, wind_ec_matches):
    gdf_matches = wind_ec_matches(gdf_wind, gdf_fluxnet, max_match_dist = 1000_000, site_id_col = 'site_id').sort_values('dist_m')

    _ax = plot_wind_ec_matches(gdf_matches.loc[gdf_matches['site_id'].str.startswith('UK-')], pad = 50000, marker_wind = 'None')


    mo.vstack([
        _ax,
        (gdf_matches['dist_m'] / 1000).rename('dist_km').describe()
    ], align = 'center')
    return (gdf_matches,)


@app.cell
def _(gdf_matches):
    gdf_matches.loc[gdf_matches['dist_m'] <= 100_000]
    return


@app.cell
def _(gdf_fluxnet, gdf_matches):
    _site_dists = gdf_matches.groupby('site_id').agg(
        n_wind_turbines = ('Project Name', 'count'),
        min_dist_m = ('dist_m', 'min'),
        max_dist_m = ('dist_m', 'max'),
    ).sort_values('min_dist_m')

    gdf_fluxnet.join(_site_dists, on = 'site_id').sort_values('min_dist_m')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. How close are offset sites to wind turbines
    """)
    return


@app.cell
def _(GDF_COUNTRIES, UK_BBOX, eddy, gdf_wind, plt):
    gdf_offset_uk = eddy.data.offset.load_offset_uk()

    _fig, _ax = plt.subplots(figsize=(10, 10))
    gdf_offset_uk.plot(facecolor='green', ax = _ax)
    gdf_wind.plot(ax = _ax, color = 'blue', markersize = 10, alpha = 0.5, marker = '1', linewidth = 0.25)

    GDF_COUNTRIES.plot(ax=_ax, linewidth=0.25, facecolor='None')
    _ax.set_axis_off()
    _ax.set_xlim(UK_BBOX['min_lon'], UK_BBOX['max_lon'])
    _ax.set_ylim(UK_BBOX['min_lat'], UK_BBOX['max_lat'])
    _ax
    return (gdf_offset_uk,)


@app.cell
def _(gpd, np):
    def wind_offset_matches(
            gdf_wind: gpd.GeoDataFrame, gdf_offset: gpd.GeoDataFrame,
            *,
            max_match_dist: int = 100_000, 
            crs: str = 'EPSG:3857',
            bounds: tuple|None = None
        ):
        '''
        Match wind turbines to their nearest carbon offset project location.

        Parameters
        ----------
        gdf_wind : geopandas.GeoDataFrame
            GeoDataFrame containing wind turbine locations.
        gdf_offset : geopandas.GeoDataFrame
            GeoDataFrame containing carbon offset project locations.
        max_match_dist : int, optional
            Maximum distance (in meters) to consider a wind turbine as a match for a carbon offset project. Defaults to 100,000 m.
        crs : str, optional
            Coordinate reference system to use for matching. Defaults to 'EPSG:3857' (Web Mercator).
            Should be a projected CRS (e.g., UTM) to ensure accurate distance calculations.
        bounds : tuple, optional
            Bounds to apply to the matching in the form (minx, miny, maxx, maxy). If None (default), all turbines and projects are considered. 
            If specified, only turbines and projects within the bounds are considered.

        Returns
        -------
        geopandas.GeoDataFrame
            The GeoDataFrame of matched wind turbines and carbon offset projects.
        '''

        if bounds:
            bounds = np.array(bounds)
            gdf_wind_wkg = gdf_wind.to_crs(crs).cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
            gdf_offset_wkg = gdf_offset.to_crs(crs).cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
        else:
            gdf_wind_wkg = gdf_wind.to_crs(crs).copy()
            gdf_offset_wkg = gdf_offset.to_crs(crs).copy()

            gdf_offset_wkg['geometry_offset'] = gdf_offset_wkg['geometry']
            gdf_nbrs = gdf_wind_wkg[['Project Name', 'Phase Name', 'Capacity (MW)', 'geometry', 'Start year', 'Retired year', 'project']].sjoin_nearest(
                gdf_offset_wkg,
                how = 'inner', distance_col = 'dist_m',
                max_distance = max_match_dist
            )

        first_cols = ['Project Name', 'Phase Name', 'Capacity (MW)', 'project_id', 'project_name', 'dist_m', 'class', 'subclass']
        other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
        gdf_nbrs = gdf_nbrs[first_cols + other_cols]

        return gdf_nbrs


    return (wind_offset_matches,)


@app.cell
def _(LineString, folium, gpd, nearest_points, pd):
    def plot_wind_offset_folium(
            gdf_matches, *,
            gdf_base=None,
            colour_wind='blue', colour_offset='green',
            marker_size_wind=100, marker_width_wind=0.75,
            alpha=0.5, simplify_m=None, tiles='CartoDB positron'
    ):
        '''
        Plot wind turbine to nearest carbon offset project matches on an interactive folium map.

        Turbines are shown as markers, offset projects as filled regions, and each match as a
        line connecting the turbine to the closest point of its offset project.

        Parameters
        ----------
        gdf_matches : geopandas.GeoDataFrame
            Output of `wind_offset_matches` - turbine geometry in 'geometry', offset geometry
            in 'geometry_offset' and the separation in 'dist_m'.
        gdf_base : geopandas.GeoDataFrame, optional
            Boundaries (e.g. countries) drawn underneath the matches, clipped to their extent.
        colour_wind, colour_offset : str, optional
            Colours of the turbine markers and the offset regions.
        marker_size_wind, marker_width_wind : float, optional
            Size and outline width of the turbine markers.
        alpha : float, optional
            Fill opacity of the turbine markers and offset regions.
        simplify_m : float, optional
            Tolerance (in the units of the CRS of `gdf_matches`, typically metres) used to simplify
            the offset regions before plotting. Reduces the size of the map considerably; the
            connecting lines and distances are always computed from the full-resolution geometry.
        tiles : str, optional
            Basemap tiles.

        Returns
        -------
        folium.Map
        '''

        _gdf = gdf_matches.reset_index(drop=True)
        _crs = _gdf.crs

        # geometries of both ends of each match, in lat/lon for plotting
        _geom_wind = gpd.GeoSeries(_gdf['geometry'].to_numpy(), crs=_crs)
        _geom_offset = gpd.GeoSeries(_gdf['geometry_offset'].to_numpy(), crs=_crs)
        _valid = _geom_wind.notna() & _geom_offset.notna()

        _gdf = _gdf[_valid.to_numpy()].reset_index(drop=True)
        _geom_wind = _geom_wind[_valid].reset_index(drop=True)
        _geom_offset = _geom_offset[_valid].reset_index(drop=True)

        # line from each turbine to the nearest point of its offset project (i.e. the matched distance)
        _lines = gpd.GeoSeries(
            [LineString(nearest_points(_w, _o)) for _w, _o in zip(_geom_wind, _geom_offset)],
            crs=_crs
        ).to_crs(epsg=4326)

        _geom_wind = _geom_wind.to_crs(epsg=4326)
        if simplify_m:
            _geom_offset = _geom_offset.simplify(simplify_m)
        _geom_offset = _geom_offset.to_crs(epsg=4326)

        _dist_km = (_gdf['dist_m'] / 1000).round(2)

        # per offset project, summarise the turbines matched to it
        _stats = _gdf.groupby('project_id', observed=True).agg(
            n_turbines=('dist_m', 'size'),
            nearest_turbine_km=('dist_m', lambda _s: round(_s.min() / 1000, 2)),
            capacity_mw=('Capacity (MW)', 'sum')
        )

        # ---- base layer
        if gdf_base is not None:
            _bounds = pd.concat([_geom_wind, _geom_offset]).total_bounds
            _pad = 0.5
            m = gdf_base.cx[
                _bounds[0] - _pad:_bounds[2] + _pad, _bounds[1] - _pad:_bounds[3] + _pad
            ].explore(
                color='None', style_kwds={'color': 'black', 'weight': 0.5},
                tiles=tiles, tooltip=False, name='Boundaries'
            )
        else:
            m = folium.Map(tiles=tiles)

        # ---- offset projects (regions)
        _offset_cols = {
            'project_name': 'Offset Project',
            'project_id': 'Offset Project ID',
            'class': 'Offset Class',
            'subclass': 'Offset Subclass',
            'country': 'Country',
            'project_status': 'Project Status',
            'area_ha': 'Area (ha)'
        }
        _offset_cols = {_k: _v for _k, _v in _offset_cols.items() if _k in _gdf.columns}

        gdf_offset_plot = gpd.GeoDataFrame(
            {_v: _gdf[_k].astype(str) if str(_gdf[_k].dtype) == 'category' else _gdf[_k] for _k, _v in _offset_cols.items()},
            geometry=_geom_offset, crs='EPSG:4326'
        )
        gdf_offset_plot['Matched Turbines'] = _gdf['project_id'].map(_stats['n_turbines']).to_numpy()
        gdf_offset_plot['Nearest Turbine (km)'] = _gdf['project_id'].map(_stats['nearest_turbine_km']).to_numpy()
        gdf_offset_plot['Matched Capacity (MW)'] = _gdf['project_id'].map(_stats['capacity_mw']).round(1).to_numpy()
        # one region per offset project, however many turbines matched to it
        gdf_offset_plot = gdf_offset_plot.drop_duplicates(subset='Offset Project ID')

        gdf_offset_plot.explore(
            m=m,
            color=colour_offset,
            style_kwds={'color': colour_offset, 'weight': 1, 'fillOpacity': alpha},
            tooltip=list(gdf_offset_plot.columns.drop('geometry')),
            name='Carbon offset projects'
        )

        # ---- wind turbines (markers)
        _wind_cols = {
            'Project Name': 'Turbine Project',
            'Phase Name': 'Phase',
            'Capacity (MW)': 'Capacity (MW)',
            'Start year': 'Start year',
            'Retired year': 'Retired year',
            'project_name': 'Nearest Offset Project',
            'class': 'Nearest Offset Class',
            'subclass': 'Nearest Offset Subclass'
        }
        _wind_cols = {_k: _v for _k, _v in _wind_cols.items() if _k in _gdf.columns}

        gdf_wind_plot = gpd.GeoDataFrame(
            {_v: _gdf[_k].astype(str) if str(_gdf[_k].dtype) == 'category' else _gdf[_k] for _k, _v in _wind_cols.items()},
            geometry=_geom_wind, crs='EPSG:4326'
        )
        gdf_wind_plot['Dist to Offset (km)'] = _dist_km.to_numpy()

        gdf_wind_plot.explore(
            m=m,
            color=colour_wind,
            marker_type=folium.Marker(
                icon=folium.Icon(color=colour_wind, icon='wind', prefix='fa')
            ),
            marker_kwds={
                'radius': marker_size_wind / 10,
                'weight': marker_width_wind,
                'fill_opacity': alpha
            },
            tooltip=list(gdf_wind_plot.columns.drop('geometry')),
            name='Wind turbines'
        )

        # ---- connecting lines
        _links = folium.FeatureGroup(name='Turbine - offset links', show=True)
        for _i, _line in enumerate(_lines):
            folium.PolyLine(
                locations=[(_lat, _lon) for _lon, _lat in _line.coords],
                color='grey', weight=10, opacity=0.25,
                tooltip=(
                    f"{_dist_km.iloc[_i]:.2f} km<br>"
                    f"{_gdf['project'].iloc[_i]}<br>"
                    f"{_gdf['project_name'].iloc[_i]} ({_gdf['subclass'].iloc[_i]})"
                )
            ).add_to(_links)
        _links.add_to(m)

        folium.LayerControl().add_to(m)
        return m




    return (plot_wind_offset_folium,)


@app.cell
def _(gdf_offset_uk, gdf_wind, wind_offset_matches):
    gdf_wo_matches = wind_offset_matches(gdf_wind, gdf_offset_uk, max_match_dist = 100_000).sort_values('dist_m')
    gdf_wo_matches
    return (gdf_wo_matches,)


@app.cell
def _(GDF_COUNTRIES, gdf_wo_matches, plot_wind_offset_folium):
    m_wo = plot_wind_offset_folium(gdf_wo_matches, gdf_base = GDF_COUNTRIES, simplify_m = 25)
    m_wo.show_in_browser()
    return


@app.cell
def _(gdf_offset_uk):
    gdf_offset_uk.groupby(['project_id', 'project_name']).agg(
        n_rows = ('project_id', 'size'),
        n_classes = ('class', 'nunique'),
        n_subclasses = ('subclass', 'nunique'),
    )
    return


@app.cell
def _(gdf_offset_uk):
    gdf_offset_uk.loc[gdf_offset_uk['project_id'] == '103000000004406']
    return


@app.cell
def _(gdf_wo_matches):
    WAKE_DIST_M = 5_000  # distance a wind turbine wake can extend downwind, in metres
    df_offset_within_wake = gdf_wo_matches.loc[gdf_wo_matches['dist_m'] <= WAKE_DIST_M].groupby('project_name').agg(
        area_ha = ('area_ha', 'sum'),
        offset_classes = ('class', lambda _s: ', '.join(sorted(_s.unique()))),
        offset_subclasses = ('subclass', lambda _s: ', '.join(sorted(_s.unique()))),
        n_wind_farms = ('Project Name', 'count'),
        min_farm_dist_km = ('dist_m', lambda _s: round(_s.min() / 1000, 2)),
        mean_farm_dist_km = ('dist_m', lambda _s: round(_s.mean() / 1000, 2)),
        max_farm_dist_km = ('dist_m', lambda _s: round(_s.max() / 1000, 2)),
    ).sort_values('min_farm_dist_km')
    return WAKE_DIST_M, df_offset_within_wake


@app.cell
def _(WAKE_DIST_M, df_offset_within_wake, mo, plt):
    _ALPHA = 0.25

    fig, _ax_scatter = plt.subplots(1, 1, figsize=(10,5), sharex=True)


    _ax_scatter.scatter(
        df_offset_within_wake['area_ha'],
        df_offset_within_wake['min_farm_dist_km'], 
        s = df_offset_within_wake['n_wind_farms'] * 10, alpha = _ALPHA
    )
    _ax_scatter.set_xlabel('Offset project area (ha)')
    _ax_scatter.set_ylabel('Dist. to nearest wind turbine (km)')
    _ax_scatter.set_title(f'UK carbon offset projects within {WAKE_DIST_M / 1000:.1f} km of a wind farm')
    _ax_scatter.legend(
        handles = [
            plt.scatter([], [], s = 10, color = 'tab:blue', alpha = _ALPHA, label = '1 turbine'),
            plt.scatter([], [], s = 50, color = 'tab:blue', alpha = _ALPHA, label = '5 turbines'),
            plt.scatter([], [], s = 100, color = 'tab:blue', alpha = _ALPHA, label = '10 turbines'),
        ],
        title = '$n$ farms matched to offset project',
        loc = 'upper right', borderaxespad = 0.
    )
    _ax_scatter.grid()


    mo.vstack([
        mo.hstack([
            fig,
            mo.md(f'''
        **Total area:** {df_offset_within_wake['area_ha'].sum():,.0f} ha
            '''),

        ], align = 'center', justify = 'space-around'),
        df_offset_within_wake
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Risk of wind turbines to carbon offset projects
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Is carbon drawdown affected by wind wakes?
        1. What's the theory?
            - What defines a wind wake?
                - Do these things affect carbon flux?
                - Any other notable effects that should be measureable?
        2. Can we see this theory play out in data from the towers that are closest to wind turbines?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. What's the theory?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    See notes [here](/Users/hugoneely/Documents/4-work/1-current/PhD/Notes/projects/carbon-quantification/-eddy-notes.md)

    Summary:
    - Yes, specific effects depend upon vegetation type. Main effects are increased vapour defecit (plants dry out more readily), and nighttime warming due to disrpution of nighttime temperature inversion.
        - For agricultural land this could be beneficial (higher nighttime temps, longer growing seasons, more CO$_2$ for growth due to greater mixing)
        - For forests this can cause trees to grow more stout - throws off estimations of tree biomass? Drying effects can limit growing
        - Largest impacts are on peatlands and offsets that rely upon fungi and
    - Wakes fundamentally disrupt the assumptions required for EC processing.
        - Flatness - wind farms infrastructure means sites aren't flat
        - Turbulence needs to be horizontally homogenous - wind wakes aren't
        - Wakes cause lateral transport of scalar fluxes to be greater than vertical - this seems the most fundamental to me.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Can we see signal in EC data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Theoretically, maybe though this may be due to more noise in the data than signal
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Opportunities provided by EC towers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Can towers be used to estimate wind resource at turbine height for new turbines?
        1. At their own location
        2. At a nearby location
    2. Can EC towers be used to regionally downscale climate data, such as ERA5?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Archive code - to be sorted into headings above
    """)
    return


@app.cell
def _(FLUX_DIR, gpd, pd):
    _cols = ['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END', 'LOCATION_DATE_START', 'LOCATION_COMMENT', 'LOCATION_LAT', 'LOCATION_LONG', 'LOCATION_ELEV']
    # read only the values we want from the flat-structured excel file
    df_ameriflux_badm = pd.read_excel(
        FLUX_DIR / 'AmeriFlux' / 'AMF_AA-Flx_BIF_CCBY4_20260527.xlsx', 
        usecols = ['SITE_ID', 'VARIABLE', 'DATAVALUE']
    ).loc[lambda _df: _df['VARIABLE'].isin(_cols)]

    df_ameriflux_badm = df_ameriflux_badm.pivot_table(index = 'SITE_ID', columns = 'VARIABLE', values = 'DATAVALUE', aggfunc = 'first')  # pivot to non-flat format
    df_ameriflux_badm = df_ameriflux_badm[_cols]  # re-order

    gdf_ameriflux_badm = gpd.GeoDataFrame(df_ameriflux_badm, geometry = gpd.points_from_xy(df_ameriflux_badm['LOCATION_LONG'], df_ameriflux_badm['LOCATION_LAT']))
    gdf_ameriflux_badm.set_crs('EPSG:4326', inplace = True)
    del df_ameriflux_badm
    gdf_ameriflux_badm
    return (gdf_ameriflux_badm,)


@app.cell
def _(FLUX_DIR, gdf_ameriflux_badm, gpd, pd):
    #df_ameriflux = pd.read_excel(FLUX_DIR / 'AmeriFlux' / 'AMF_AA-Flx_BIF_CCBY4_20260527.xlsx')
    df_ameriflux = pd.read_csv(FLUX_DIR / 'AmeriFlux' / 'BASE_MeasurementHeight_20260527.csv')
    df_ameriflux['var_base'] = df_ameriflux['Variable'].str.split('(_\d)+', regex=True).apply(lambda x: x[0])

    df_ameriflux = df_ameriflux.join(
        pd.read_csv(FLUX_DIR / 'AmeriFlux' / 'flux-met_processing_variables_20260618.csv', index_col = 1), on = 'var_base'
    ).sort_values(['Site_ID', 'Variable'])
    df_ameriflux = df_ameriflux.loc[df_ameriflux['Type'] == 'MET_WIND']
    gdf_ameriflux = gpd.GeoDataFrame(df_ameriflux.join(gdf_ameriflux_badm[['geometry', 'FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = 'Site_ID').sort_values(['Site_ID', 'Variable'])).reset_index(drop = True)
    gdf_ameriflux.set_crs('EPSG:4326', inplace = True)
    del df_ameriflux
    gdf_ameriflux
    return (gdf_ameriflux,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wind-EC matching
    """)
    return


@app.cell
def _(GDF_STATES, folium, pd):
    def plot_nearest_turbines_folium(
            gdf_nbrs, *, 
            colour_wind = 'blue', colour_ec = 'green', marker_size_wind = 100, marker_width_wind = 0.75, marker_size_ec = 25, alpha = 0.5
    ):

        m = GDF_STATES.explore(color = 'None', style_kwds = {'color': 'black', 'weight': 0.5}, tiles = 'CartoDB positron', tooltip = False)
        gdf_nbrs.rename(
            columns={
                'site_id': 'Neighbour Site ID', 
                'dist_m': 'Dist to EC Neighbour (m)'
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

        gdf_nbrs.set_geometry('geometry_eddy').rename(
            columns={
                'Site_ID': 'Site ID',
                'FLUX_MEASUREMENTS_DATE_START': 'EC Measurements Start',
                'FLUX_MEASUREMENTS_DATE_END': 'EC Measurements End',
                'height_max': 'Max Height',
                'height_min': 'Min Height',
                'n_sensors': 'Number of Sensors'
            }
        ).explore(
            m = m,
            color = colour_ec,
            marker_type = folium.Marker(
                icon = folium.Icon(color = 'green', icon = 'tower-broadcast', prefix = 'fa')
            ),
            marker_kwds = {
                'radius': marker_size_ec/10, 
                'weight': 0.5, 
                'fill_opacity': alpha
            },
            tooltip = ['Site ID', 'Max Height', 'Min Height', 'Number of Sensors', 'EC Measurements Start', 'EC Measurements End']
        )

        _geom_lat_lon = gdf_nbrs['geometry'].to_crs(epsg = 4326)
        _geom_af_lat_lon = gdf_nbrs['geometry_eddy'].to_crs(epsg = 4326)

        for _i in range(len(gdf_nbrs)):
            _geom = _geom_lat_lon.iloc[_i]
            _geom_af = _geom_af_lat_lon.iloc[_i]
            if pd.isna([_geom_af, _geom]).any():
                # skip if either geometry is NaN
                continue

            _dist = gdf_nbrs['dist_m'].iloc[_i] / 1000

            _locs = [
                [_geom.y, _geom.x], 
                [_geom_af.y, _geom_af.x]
            ]
            _line = folium.PolyLine(
                locations = _locs,
                color = 'grey', weight = 10,
                #dash_array = '6',
                opacity = 0.25,
                tooltip = f'{_dist:.1f} km'
            )
            _line.add_to(m)
        return m

    #m = plot_nearest_turbines_folium(gdf_nbrs.loc[gdf_nbrs['Site_ID'] == 'US-PFa'], colour_wind = 'blue', colour_ec = 'green', marker_size_wind = 100, marker_width_wind = 0.75, marker_size_ec = 25, alpha = 0.5)
    #m.show_in_browser()
    return


@app.cell
def _(OUT_DIR, m):
    m.save(OUT_DIR / 'wind-ec-neighbouurs-ameriflux.html')
    return


@app.cell
def _(
    GDF_STATES,
    colour_ec,
    colour_wind,
    folium,
    gdf_nbrs,
    marker_size_ec,
    marker_size_wind,
    marker_width_wind,
    pd,
):
    na_ind = gdf_nbrs[['geometry', 'geometry_eddy']].notna().all(axis = 1)
    ALPHA = 0.5
    _m = GDF_STATES.explore(color = 'None', style_kwds = {'color': 'black', 'weight': 0.5}, tiles = 'CartoDB positron', tooltip = False)
    gdf_nbrs[na_ind].rename(
        columns={
            'Site_ID': 'Neighbour Site ID', 
            'dist_m': 'Dist to EC Neighbour (m)'
        }
    ).explore(
        m = _m, 
        color = colour_wind, 
        marker_type = folium.Marker(
            icon = folium.Icon(color = 'blue', icon = 'wind', prefix = 'fa')
        ), 
        marker_kwds = {
            'radius': marker_size_wind/10, 
            'weight': marker_width_wind, 
            'fill_opacity': ALPHA
        }, 
        tooltip = ['Project Name', 'Capacity (MW)', 'Dist to EC Neighbour (m)', 'Neighbour Site ID', 'Start year', 'Retired year']
    )

    gdf_nbrs[na_ind].set_geometry('geometry_eddy').rename(
        columns={
            'Site_ID': 'Site ID',
            'FLUX_MEASUREMENTS_DATE_START': 'EC Measurements Start',
            'FLUX_MEASUREMENTS_DATE_END': 'EC Measurements End'
        }
    ).explore(
        m = _m,
        color = colour_ec,
        marker_type = folium.Marker(
            icon = folium.Icon(color = 'green', icon = 'tower-broadcast', prefix = 'fa')
        ),
        marker_kwds = {
            'radius': marker_size_ec/10, 
            'weight': 0.5, 
            'fill_opacity': ALPHA
        },
        tooltip = ['Site ID', 'Height', 'EC Measurements Start', 'EC Measurements End']
    )

    _geom_lat_lon = gdf_nbrs.loc[na_ind, 'geometry'].to_crs(epsg = 4326)
    _geom_af_lat_lon = gdf_nbrs.loc[na_ind, 'geometry_eddy'].to_crs(epsg = 4326)

    for _i in range(na_ind.sum()):
        _geom = _geom_lat_lon.iloc[_i]
        _geom_af = _geom_af_lat_lon.iloc[_i]
        if pd.isna([_geom_af, _geom]).any():
            continue
        _locs = [
            [_geom.y, _geom.x], 
            [_geom_af.y, _geom_af.x]
        ]
        _line = folium.PolyLine(
            locations = _locs,
            color = 'grey', weight = 2,
            dash_array = '3',
            opacity = 0.5
        )
        _line.add_to(_m)

    _m.show_in_browser()
    return


@app.cell
def _(gdf_wind):
    gdf_wind.loc[gdf_wind['Project Name'] == 'JD wind farm']
    return


@app.cell
def _(gdf_nbrs):
    gdf_nbrs.groupby('Site_ID').agg(
        **{
            'height': ('Height', 'first'),
            'n_turbines': ('project', 'nunique'),
            'mean_dist_m': ('dist_m', 'mean'),
            'min_dist_m': ('dist_m', 'min'),
            'max_dist_m': ('dist_m', 'max'),
        }
    )
    return


@app.cell
def _(gdf_nbrs):
    gdf_nbrs.loc[gdf_nbrs['Site_ID'].notna(), 'Site_ID'].value_counts()
    return


@app.cell
def _(gdf_nbrs):
    gdf_nbrs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measurement heights
    """)
    return


@app.cell
def _(gdf_ameriflux):
    gdf_ameriflux.plot.box(column = 'Height', by = 'var_base', figsize = (8,15), vert = False, grid = True).values[0]
    return


@app.cell
def _(gdf_ameriflux):
    gdf_ameriflux.loc[gdf_ameriflux['height_max'] > 75]
    return


@app.cell
def _(GDF_COUNTRIES, gdf_ameriflux):
    _ax = gdf_ameriflux.loc[gdf_ameriflux['height_max'] > 5].dissolve(by = 'Site_ID', aggfunc = {'Height': 'max'}).plot(
        column = 'Height',
        color = 'red',
        #cmap = 'Reds',
        legend = True,
        #legend_kwds = {'label': "Maximum Measurement Height (m)", 'orientation': "horizontal"},
        figsize = (10, 20),
        markersize = 'Height',
        alpha = 0.1
    )
    GDF_COUNTRIES.plot(ax=_ax, facecolor = 'None', linewidth = 0.25)
    _ax.set_axis_off()
    _bounds = gdf_ameriflux.total_bounds
    _pad = 0.1
    _ax.set_xbound(_bounds[0] - _pad * (_bounds[2] - _bounds[0]), _bounds[2] + _pad * (_bounds[2] - _bounds[0]))
    _ax.set_ybound(_bounds[1] - _pad * (_bounds[3] - _bounds[1]), _bounds[3] + _pad * (_bounds[3] - _bounds[1]))
    _ax
    return


@app.cell
def _(GDF_COUNTRIES, gdf_ameriflux, plt):


    _bounds = gdf_ameriflux.total_bounds
    _heights = [0, 10, 25, 50, 75]

    _fig, _axs = plt.subplots(1, len(_heights), figsize = (4 * len(_heights), 10))


    _total_towers = len(gdf_ameriflux.dissolve(by = 'Site_ID', aggfunc = {'Height': 'max'}))

    for _i, _ax in enumerate(_axs):
        _n_towers = len(gdf_ameriflux.loc[gdf_ameriflux['height_max'] > _heights[_i]].dissolve(by = 'Site_ID', aggfunc = {'Height': 'max'}))

        gdf_ameriflux.loc[gdf_ameriflux['height_max'] > _heights[_i]].dissolve(by = 'Site_ID', aggfunc = {'Height': 'max'}).plot(
            column = 'Height',
            color = 'red',
            #cmap = 'Reds',
            legend = True,
            #legend_kwds = {'label': "Maximum Measurement Height (m)", 'orientation': "horizontal"},
            figsize = (10, 20),
            markersize = 1,
            alpha = 0.3,
            ax = _ax
        )
        GDF_COUNTRIES.plot(ax=_ax, facecolor = 'None', linewidth = 0.25)
        _ax.set_axis_off()
        _pad = 0.1
        _ax.set_xbound(_bounds[0] - _pad * (_bounds[2] - _bounds[0]), _bounds[2] + _pad * (_bounds[2] - _bounds[0]))
        _ax.set_ybound(_bounds[1] - _pad * (_bounds[3] - _bounds[1]), _bounds[3] + _pad * (_bounds[3] - _bounds[1]))
        _ax.set_title(f'Max Height > {_heights[_i]} m\n{_n_towers} / {_total_towers} towers', fontsize = 12)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial distribution of EC towers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AmeriFlux
    """)
    return


@app.cell
def _(GDF_COUNTRIES, box, gdf_ameriflux, gpd, np, plt):
    # grid data
    DEFAULT_AGGS = {
        'H_mean': ('Height', 'mean'),
        'H_std': ('Height', 'std'),
        'H_max': ('Height', 'max'),
        'H_median': ('Height', 'median'),
        'H_min': ('Height', 'min'),
        'H_count': ('Height', 'count'),
        'sites': ('Site_ID', 'nunique')
    }

    def grid_data(gdf: gpd.GeoDataFrame, n_cells: int = 65, aggs: dict = DEFAULT_AGGS):
        for agg_name, (col, func) in aggs.items():
            if col not in gdf.columns:
                raise ValueError(f"Column {col} is not a valid column for aggregation. Must be one the columns in gdf - see gdf.columns.")

        xmin, ymin, xmax, ymax = gdf.total_bounds

        cell_size = (xmax - xmin) / n_cells

        grid_cells = []
        for x0 in np.arange(xmin, xmax + cell_size, cell_size):
            for y0 in np.arange(ymin, ymax + cell_size, cell_size):
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                grid_cells.append(box(x0, y0, x1, y1))


        return gpd.GeoDataFrame(
            gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs = gdf.crs).sjoin(
                gdf, 
                how = 'left', 
                predicate = 'contains'
            ).groupby(
                by = 'geometry'
            ).agg(**aggs).reset_index(),
            crs = gdf.crs
        )

    def plot_grid_data(
        gdf: gpd.GeoDataFrame, col: str, 
        *, 
        cmap: str = None, legend_label: str = None, legend_kwds: dict = None,
        ax = None, bounds = None, pad = 0.1, 
        plot_coastline = True, **plot_kwargs
    ):
        if col not in gdf.columns:
            raise ValueError(f"Column {col} is not a valid column for plotting. Must be one the columns in gdf - see gdf.columns.")
        if bounds is None:
            bounds = gdf.total_bounds
        if pad > 0:
            bounds[0] -= pad * (bounds[2] - bounds[0])
            bounds[1] -= pad * (bounds[3] - bounds[1])
            bounds[2] += pad * (bounds[2] - bounds[0])
            bounds[3] += pad * (bounds[3] - bounds[1])

        if legend_kwds:
            legend_kwds = {**{'label': legend_label, 'orientation': "horizontal", 'shrink': 0.6}, **legend_kwds}
        else:
            legend_kwds = {'label': legend_label, 'orientation': "horizontal"}

        if ax is None:
            fig, ax = plt.subplots(figsize = (10, 10))

        plot_kwargs = {**{'legend': True, 'edgecolor': 'k', 'linewidth': 0.4}, **plot_kwargs}
        gdf.plot(
            column = col,
            cmap = cmap,
            legend_kwds = legend_kwds,
            ax = ax,
            **plot_kwargs
        )
        if plot_coastline:
            GDF_COUNTRIES.plot(ax=ax, facecolor = 'None', edgecolor = 'k', alpha = 0.5, linewidth = 0.25)
        ax.set_axis_off()
        ax.set_xbound(bounds[0], bounds[2])
        ax.set_ybound(bounds[1], bounds[3])
        return ax



    _n_cells = 65
    gdf_ameriflux_cells = grid_data(gdf_ameriflux, n_cells = _n_cells, aggs = DEFAULT_AGGS)#.replace({0: np.nan})

    # plot :)
    _fig, (_ax1, _ax2) = plt.subplots(1,2, figsize = (20,10))

    plot_grid_data(
        gdf_ameriflux_cells.replace({0: np.nan}), 
        col = 'sites', 
        cmap = 'Greens',
        legend_label = "N Sites",
        ax = _ax1,
    )
    plot_grid_data(
        gdf_ameriflux_cells.replace({0: np.nan}), 
        col = 'H_max', 
        cmap = 'Purples',
        legend_label = "Max. Measurement Height (m)",
        ax = _ax2,
    )
    _bnds = gdf_ameriflux_cells.total_bounds
    _cell_size = (_bnds[2] - _bnds[0]) / _n_cells
    _fig.suptitle('Gridded AmeriFlux Tower Data', fontsize = 16)
    _fig.text(0.5, 0.93, rf'Cell Size: ${_cell_size:.2f}°\times{_cell_size:.2f}° \approx {_cell_size * 111:.2f} km \times {_cell_size * 111:.2f}km$', ha='center', fontsize = 12)
    _ax1.set_title('Number of Sites per Grid Cell', fontsize = 12)
    _ax2.set_title('Maximum Measurement Height (m)', fontsize = 12)
    _fig
    return grid_data, plot_grid_data


@app.cell
def _(gdf):
    gdf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### FLUXNET
    """)
    return


@app.cell
def _(gdf_fluxnet, grid_data, np, plot_grid_data, plt):
    _n_cells = 300
    gdf_fluxnet_cells = grid_data(
        gdf_fluxnet, 
        n_cells = _n_cells,
        aggs = {
            'sites': ('site_id', 'nunique'),
            'source': ('data_hub', 'first')
        }
    ).replace({0: np.nan})

    _fig, _ax = plt.subplots(figsize = (15, 10))
    plot_grid_data(
        gdf_fluxnet_cells,
        col = 'sites',
        cmap = 'Greens',
        legend_label = "N Sites", legend_kwds = {'shrink': 0.6},
        ax = _ax,
        linewidth = 0.2, 
    )

    _bnds = gdf_fluxnet_cells.total_bounds
    _cell_size = (_bnds[2] - _bnds[0]) / _n_cells
    _fig.suptitle('Gridded FLUXNET Tower Data', fontsize = 16)
    _fig.text(0.5, 0.93, rf'Cell Size: ${_cell_size:.2f}°\times{_cell_size:.2f}° \approx {_cell_size * 111:.2f} km \times {_cell_size * 111:.2f}km$', ha='center', fontsize = 12)
    _ax.set_title('Number of Sites per Grid Cell', fontsize = 12)
    return (gdf_fluxnet_cells,)


@app.cell
def _(Point, folium, gdf_fluxnet_cells):
    def get_width_and_length(geom):
        coords = list(geom.exterior.coords)

        # Distance between first two points (Width)
        width = Point(coords[0]).distance(Point(coords[1]))
        # Distance between second and third points (Length)
        length = Point(coords[1]).distance(Point(coords[2]))
        return min(width, length), max(width, length)
    _dimensions = gdf_fluxnet_cells.to_crs(epsg = 3857).geometry.apply(get_width_and_length)

    gdf_fluxnet_cells["approx_width_km"], gdf_fluxnet_cells["approx_length_km"] = zip(*_dimensions)
    gdf_fluxnet_cells[['approx_width_km', 'approx_length_km']] = gdf_fluxnet_cells[['approx_width_km', 'approx_length_km']] / 1000

    gdf_fluxnet_cells['approx_area_km2'] = gdf_fluxnet_cells.to_crs(epsg = 3857).area / 1000000


    _m = folium.Map(location = [0, 0], zoom_start = 2, tiles = 'CartoDB positron', control_scale = True)
    folium.GeoJson(
        gdf_fluxnet_cells.loc[gdf_fluxnet_cells['sites'] > 0],
        tooltip = folium.GeoJsonTooltip(fields = ['sites', 'source', 'approx_area_km2', 'approx_width_km', 'approx_length_km'], aliases = ['N Sites', 'Data Hub', 'Approximate Area (km²)', 'Approximate Width (km)', 'Approximate Length (km)'], localize = True)
    ).add_to(_m)
    _m.show_in_browser()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of sensors per tower
    """)
    return


@app.cell
def _(gdf_ameriflux):
    gdf_ameriflux
    return


@app.cell
def _(gdf_ameriflux):
    list(gdf_ameriflux.loc[(gdf_ameriflux['Site_ID'] == 'AR-Bal') & (gdf_ameriflux['var_base'] == 'WD'), 'Instrument_Model'].unique())
    return


@app.cell
def _(gdf_ameriflux, gpd):
    gdf_af_sensors = gpd.GeoDataFrame(
        gdf_ameriflux.groupby(['Site_ID', 'var_base']).agg(
            {
                'Variable':'count', 
                'Description': 'first', 
                'Height': lambda x: x.to_list(), 
                'Instrument_Model': lambda x: x.to_list(),
                'Units':'first',
                'geometry': 'first'
            }
        ).rename(columns = {'Variable': 'n_sensors', 'Description': 'desc', 'Height': 'heights', 'Units': 'units', 'Instrument_Model': 'instruments'}).reset_index()
    )
    gdf_af_sensors['instruments_n_unique'] = gdf_af_sensors['instruments'].apply(lambda x: len(set(x)))
    gdf_af_sensors['heights_n_unique'] = gdf_af_sensors['heights'].apply(lambda x: len(set(x)))

    gdf_af_sensors
    return (gdf_af_sensors,)


@app.cell
def _(gdf_af_sensors):
    _ax = gdf_af_sensors.loc[gdf_af_sensors['var_base'] == 'WS', 'n_sensors'].hist(bins = 20)
    _ax.set_xlabel('N. Wind Sensors')
    _ax.set_ylabel('Freq.')
    _ax.set_xticks(range(0, gdf_af_sensors.loc[gdf_af_sensors['var_base'] == 'WS', 'n_sensors'].max() + 1))
    _ax.set_title('Distribution of Number of Wind Sensors per AmeriFlux Site')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wind extrapolation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    todo:
      - method for determining the validity of each extrapolation method
        - Determine conditions for each tower
      - site-independent ML extrapolation?
        - read more!
    ---

    To extrapolate I need to:
    1. Determine validity of methods:
        - Terrain type
            - LiDAR?
            - Satellite?
            - Manual, from description?
                - Worst, most tiring, and least rigorous option.
        - Time of available data (some methods depend upon prior years data)
    2. Determine availability of parameters
        - Some require

    Before I extrapolate, I'd like to understand:
    - How useful would extrapolated EC wind speeds be? Specifically, for:
        - Climate models
        - Weather prediction
        - Wind turbines
            - Could turbines be built near to EC towers?
                - Does this go completely counter to what EC towers are used for? Would this be removing the land-management scheme, and instead replacing it with an industrial project?
                    - I guess the argument for my thesis would be that its a hedging scheme for land management schemes, and makes them more attractive to business.
                    - Is there an amount of time after which an EC tower's measurements are effectively redundant? Does CO2 sequestration stabilise? My imagining would be for certain forest types maybe? And only on certain time scales (at which the forest can be seen as approximately constant)
        - Wildfire prediction?
            - Better understanding of natural disasters could be highly beneficial for land management schemes in high-risk areas
                - Need understanding of:
                    - what a high-risk area is
                    - if land management schemes are likely to be here (or near here)
            - Would this improve the fine-grained prediction of disaster management?
    - How EC data is already used!
        - Am I assuming they're in a silo, but really they're the swiss-army pens of atmospheric science that it seems they could be?
        - Are they used in:
            - Reanalysis datasets
            - NWP
            - Wind speed datasets
    """)
    return


@app.cell
def _(np, pd):
    def instrument_option(ws_measurement_heights: list, target_height_m: float = 80, strict: bool = False) -> int:
        '''
        Determine the instrument option, as defined in Gaulieri (2019).

        1. A single anemometer at a single (lower) height.
        2. Two anemometers at a single (lower) height.
        3. One lower anemometer, and one at the target height.

        Parameters
        ----------
        ws_measurement_heights : list
            List of wind speed measurement heights (in meters).
        target_height_m : float, optional
            Target height for wind speed measurement (default is 80 m).
        strict : bool, optional
            If True, only return option 3 if there is an anemometer exactly at the target height. If False, return option 3 if there is an anemometer at or above the target height.

        Returns
        -------
        int
            Instrument option (1, 2, or 3) based on the provided measurement heights, or 0 if no heights are available.
        '''
        if pd.isna(ws_measurement_heights).all() or not ws_measurement_heights:
            return 0

        unique_heights = set(ws_measurement_heights)
        unique_heights.discard(np.nan)  # Remove NaN values if present - should be at least 1 value left after this, per the above return 0 check
        unique_heights.discard(None) 
        unique_heights.discard(pd.NA)


        if len(ws_measurement_heights) == 1:
            return 1
        elif len(ws_measurement_heights) >= 2:
            if len(unique_heights) == 1:
                return 2

            if strict:
                if target_height_m in ws_measurement_heights:
                    return 3
                else:
                    return 2
            else:
                if (target_height_m <= np.array(ws_measurement_heights)).any():
                    return 3
                else:
                    return 2
        else:
            return 0

    return (instrument_option,)


@app.cell
def _(gdf_af_sensors, instrument_option):
    gdf_af_sensors['instrument_option'] = gdf_af_sensors.apply(lambda row: instrument_option(row['heights'], target_height_m = 80, strict = False) if row['var_base'] == 'WS' else None, axis = 1)
    gdf_af_sensors
    return


@app.cell
def _(gdf_af_sensors):
    gdf_af_sensors.loc[(gdf_af_sensors['var_base'] == 'WS') & (gdf_af_sensors['geometry'].notna()), 'instrument_option'].value_counts().sort_index()
    return


@app.cell
def _(gdf_af_sensors):
    gdf_af_sensors.loc[gdf_af_sensors['instrument_option'] == 3]
    return


if __name__ == "__main__":
    app.run()
