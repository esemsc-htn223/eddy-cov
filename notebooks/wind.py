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
    import pathlib
    from netCDF4 import Dataset

    from typing import Literal


    import xyzservices.providers as xyz
    from dotenv import load_dotenv
    load_dotenv()
    return (
        Dataset,
        LineString,
        Literal,
        ScaleBar,
        folium,
        gpd,
        mo,
        nearest_points,
        np,
        pathlib,
        pd,
        plt,
    )


@app.cell
def _():
    import eddy

    # get dirs too
    from eddy.data.flux import FLUX_DIR, FLUXNET_DIR, TERN_DIR, AMERIFLUX_DIR
    from eddy.util import DATA_DIR, OUT_DIR
    from eddy.data import _standardise_df as standardise_df


    return FLUX_DIR, eddy


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
def _(gpd, np, pd):
    def wind_ec_matches(
            gdf_wind: gpd.GeoDataFrame, 
            gdf_ec: gpd.GeoDataFrame, 
            *,
            date: pd.Timestamp|None = None,
            min_height: int|None = None, 
            bounds: tuple|None = None,
            crs: str = 'EPSG:3857',
            site_id_col: str = 'site_id', 
            height_col: str = 'height',
            max_match_dist: int = 100_000, 
            only_overlap: bool = False
        ):
        '''
        Match wind turbine locations to their nearest eddy covariance tower location.

        Parameters
        ----------
        gdf_wind : geopandas.GeoDataFrame
            GeoDataFrame containing wind turbine locations and attributes.
        gdf_ec : geopandas.GeoDataFrame
            GeoDataFrame containing eddy covariance tower locations and attributes.
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
        max_match_dist : int, optional
            Maximum distance (in meters) to consider a wind turbine as a match for an eddy covariance tower. Defaults to 100,000 m.


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

        gdf_wind = gdf_wind.to_crs(crs).copy()
        gdf_ec = gdf_ec.to_crs(crs).copy()

        # slice by time and space
        if date is not None:
            gdf_wind = gdf_wind.loc[(gdf_wind['start_year'] <= date.year) & (gdf_wind['retired_year'] >= date.year)]
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
        wind_cols = ['project', 'capacity_mw', 'start_year', 'retired_year', 'project_name', 'phase_name', 'installation_type', 'status', 'location_accuracy', 'geometry']
        gdf_nbrs = gdf_wind[
            wind_cols
        ].rename(columns={col: f'wind_{col}' for col in wind_cols if col != 'geometry'}).sjoin_nearest(
            gdf_ec_grouped,
            how = 'inner', distance_col = 'dist_m',
            max_distance = max_match_dist
        )
        del gdf_ec_grouped  # free memory

        gdf_nbrs['wind_ec_overlap'] = gdf_nbrs.apply(
            lambda row: max(0, min(row['wind_retired_year'], row['eddy_last_year']) - max(row['wind_start_year'], row['eddy_first_year'])),
            axis = 1
        ).astype(bool)

        if only_overlap:
            gdf_nbrs = gdf_nbrs.loc[gdf_nbrs['wind_ec_overlap']]

        first_cols = ['wind_project', site_id_col, 'dist_m', 'wind_capacity_mw', 'wind_start_year', 'wind_retired_year', 'eddy_first_year', 'eddy_last_year', 'wind_ec_overlap']
        other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
        gdf_nbrs = gdf_nbrs[first_cols + other_cols]

        # TODO: time alignment of wind turbine operation and EC tower measurement periods
        #gdf_nbrs = gdf_nbrs.join(gdf_ameriflux_badm[['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = site_id_col)

        return gdf_nbrs.sort_values(by = 'dist_m')

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
            scale_bar: bool = True,
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
        scale_bar : bool, optional
            If True, add a scale bar to the plot. Defaults to True.
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



    return (plot_wind_ec_matches,)


@app.cell
def _(
    GDF_COUNTRIES,
    gdf_fluxnet,
    gdf_wind,
    mo,
    plot_wind_ec_matches,
    wind_ec_matches,
):
    match_dist = 20_000  # m


    gdf_matches = wind_ec_matches(
        gdf_wind, gdf_fluxnet, 
        max_match_dist = match_dist,  # m
        only_overlap = True
    )

    _ax = plot_wind_ec_matches(
        gdf_matches,
        pad = 500000,
        gdf_borders = GDF_COUNTRIES,
        scale_bar = False
    )


    mo.vstack([
        mo.md(f'### Wind Turbine to EC Tower Matches (max distance = {match_dist/1000:.0f} km)'),
        _ax,
        # gdf_matches,
        (gdf_matches['dist_m'] / 1000).rename('dist_km').describe().round(1).to_frame().T[['count', 'mean', '50%', 'min', 'max']].convert_dtypes()
    ], align = 'center')
    return (gdf_matches,)


@app.cell
def _(gdf_matches):
    gdf_matches
    return


@app.cell
def _(gdf_matches, gdf_wind):
    gdf_wind.loc[gdf_wind['project'].isin(gdf_matches['wind_project'])]
    return


@app.cell
def _(gdf_matches):
    gdf_matches.loc[gdf_matches['site_id'].str.startswith('DE-')]['wind_project'].unique().tolist()
    return


@app.cell
async def _(eddy, gdf_matches):
    site_files = await eddy.data.flux.get_fluxnet_site_dirs(
        gdf_matches['site_id'].unique().tolist(), 
    )

    fs_ruc = site_files['DE-RuC']
    fs_hai = site_files['DE-Hai']
    fs_myb = site_files['US-Myb']
    fs_hdn = site_files['DE-Hdn']
    fs_wjs = site_files['US-Wjs']
    fs_rum = site_files['DE-RuM']
    return (fs_ruc,)


@app.cell
def _(fs_ruc):
    fs_ruc.fluxmet()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. How close are wind turbines to offset sites?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Definitions for Woodland Carbon Code RAG ratings](https://www.woodlandcarboncode.org.uk/4-verification#:~:text=units%2E-,Green)
    """)
    return


@app.cell
def _(GDF_COUNTRIES, UK_BBOX, eddy, gdf_wind, plt):
    gdf_offset_uk = eddy.data.offset.load_offset_uk()

    _fig, _ax = plt.subplots(figsize=(10, 10))
    gdf_offset_uk.to_crs(eddy.PLOTTING_CRS).plot(facecolor='green', ax = _ax)
    gdf_wind.to_crs(eddy.PLOTTING_CRS).plot(ax = _ax, color = 'blue', markersize = 10, alpha = 0.5, marker = '1', linewidth = 0.25)

    GDF_COUNTRIES.to_crs(eddy.PLOTTING_CRS).plot(ax=_ax, linewidth=0.25, facecolor='None')
    _ax.set_axis_off()
    _ax.set_xlim(UK_BBOX['min_lon'], UK_BBOX['max_lon'])
    _ax.set_ylim(UK_BBOX['min_lat'], UK_BBOX['max_lat'])
    _ax
    return (gdf_offset_uk,)


@app.cell
def _(gpd, np, pd):
    def wind_offset_matches(
            gdf_wind: gpd.GeoDataFrame, 
            gdf_offset: gpd.GeoDataFrame,
            *,
            crs: str = 'EPSG:3857',
            bounds: tuple|None = None,
            date: pd.Timestamp|None = None,
            max_match_dist: int = 100_000, 
        ):
        '''
        Match wind turbines to their nearest carbon offset project location.

        Parameters
        ----------
        gdf_wind : geopandas.GeoDataFrame
            GeoDataFrame containing wind turbine locations.
        gdf_offset : geopandas.GeoDataFrame
            GeoDataFrame containing carbon offset project locations.
        crs : str, optional
            Coordinate reference system to use for matching. Defaults to 'EPSG:3857' (Web Mercator).
            Should be a projected CRS (e.g., UTM) to ensure accurate distance calculations.
        bounds : tuple, optional
            Bounds to apply to the matching in the form (minx, miny, maxx, maxy). If None (default), all turbines and projects are considered. 
            If specified, only turbines and projects within the bounds are considered.
        date : pd.Timestamp, optional
            Date to filter wind turbines and carbon offset projects by their operational status.
        max_match_dist : int, optional
            Maximum distance (in meters) to consider a wind turbine as a match for a carbon offset project. Defaults to 100,000 m.



        Returns
        -------
        geopandas.GeoDataFrame
            The GeoDataFrame of matched wind turbines and carbon offset projects.
        '''

        gdf_wind = gdf_wind.to_crs(crs).copy()
        gdf_offset = gdf_offset.to_crs(crs).copy()
        if date is not None:
            gdf_wind = gdf_wind.loc[(gdf_wind['start_year'] <= date.year) & (gdf_wind['retired_year'] >= date.year)]
            gdf_offset = gdf_offset.loc[(gdf_offset['start_date'] <= date)]
        if bounds is not None:
            bounds = np.array(bounds)
            gdf_wind = gdf_wind.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
            gdf_offset = gdf_offset.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
        else:
            gdf_offset['geometry_offset'] = gdf_offset['geometry']
            gdf_nbrs = gdf_wind[['Project Name', 'Phase Name', 'Capacity (MW)', 'geometry', 'Start year', 'Retired year', 'project']].sjoin_nearest(
                gdf_offset,
                how = 'inner', distance_col = 'dist_m',
                max_distance = max_match_dist
            )

        first_cols = ['Project Name', 'Phase Name', 'Capacity (MW)', 'project_id', 'project_name', 'dist_m', 'class', 'subclass']
        other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
        gdf_nbrs = gdf_nbrs[first_cols + other_cols]

        return gdf_nbrs


    return


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
            capacity_mw=('capacity_mw', 'sum')
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
            'project_name': 'Turbine Project',
            'phase_name': 'Phase',
            'capacity_mw': 'Capacity (MW)',
            'start_year': 'Start year',
            'retired_year': 'Retired year',
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




    return


@app.cell
def _(eddy, gdf_offset_uk, gdf_wind):
    gdf_wo_matches = eddy.offset.wind_offset_matches(gdf_wind, gdf_offset_uk, max_match_dist = 10_000).sort_values('dist_m')
    gdf_wo_matches
    return (gdf_wo_matches,)


@app.cell
def _(eddy, gdf_wo_matches):
    m_wo = eddy.offset.plot_wind_offset_folium(
        gdf_wo_matches, gdf_base = None, simplify_m = 25,
        colour_wind = 'lightblue'
    )
    m_wo.show_in_browser()
    return


@app.cell
def _(eddy, folium, gdf_wind):
    _m = gdf_wind.to_crs(eddy.PLOTTING_CRS).explore(
     tiles = eddy.util.BASEMAP_DEFAULT
    )
    # plot a site at 50.797400664303844, 0.13260808955314488
    _m.add_child(folium.Marker(
        location = [50.797400664303844, 0.13260808955314488],
        popup = 'winery',
        icon = folium.Icon(color = 'red', icon = 'wine-glass', prefix = 'fa')
    ))
    _m.show_in_browser()
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
    Can compare to CWEX data - custom built for measurement of wind wakes using EC.

    Blog: https://wiki.ucar.edu/pages/viewrecentblogposts.action?key=cwex11

    Wind farm:
    - Story County wind farm, Phase 1
        - De/re-commissioned in 2019.
    """)
    return


@app.cell
def _(Dataset, mo):
    _info_accordion = {}
    _ds_test = Dataset('/Users/hugoneely/Documents/4-work/1-current/PhD/repos/eddy-cov/data/flux/NCAR/CWEX/cwex_20110816_20.nc')

    for _var_name, _var in _ds_test.variables.items():
        _long_name = getattr(_var, "long_name", "")
        _key = f'**{_var_name}**{("- " + _long_name) if _long_name else ""}'


        _info_accordion[_key] = {
            'dtype': _var.dtype,
            'shape': str(_var.shape) if _var.shape else '',
            'dimensions': str(_var.dimensions) if _var.dimensions else '',
            'units': getattr(_var, "units", ""),
        }
    mo.accordion(_info_accordion, multiple = True)
    return


@app.cell
def _(Dataset, Literal, Path):
    def var_n_stations(ds: Dataset, var_name: str|None = None) -> int|dict:
        """
        Get a dictionary mapping variable names to the number of stations they have in the dataset.

        Parameters
        ----------
        ds : netCDF4.Dataset
            The netCDF dataset to inspect.
        var_name : str or None
            The name of the variable to get the number of stations for. If None, returns a dictionary with all variables.

        Returns
        -------
        int or dict
            If a specific variable name is provided, returns the number of stations for that variable.
            Otherwise, returns a dictionary where keys are variable names and values are the number of stations for that variable.
            If a variable does not have a 'station' dimension, its value will be 0.
        """
        if var_name is not None:
            if var_name not in ds.variables:
                raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")
            if 'station' in ds.variables[var_name].dimensions:
                return ds.dimensions['station'].size
            else:
                return 0
    
        n_stations = {}
        for var_ in ds.variables:
            if 'station' in ds.variables[var_].dimensions:
                n_stations[var_] = ds.dimensions['station'].size
            else:
                n_stations[var_] = 0
        return n_stations

    def var_n_samples(ds: Dataset, var_name: str|None = None) -> int|dict:
        """
        Get a dictionary mapping variable names to the number of samples they have in the dataset.

        Parameters
        ----------
        ds : netCDF4.Dataset
            The netCDF dataset to inspect.
        var_name : str or None
            The name of the variable to get the number of samples for. If None, returns a dictionary with all variables.

        Returns
        -------
        int or dict
            If a specific variable name is provided, returns the number of samples for that variable.
            Otherwise, returns a dictionary where keys are variable names and values are the number of samples for that variable.
            If a variable does not have a 'sample' dimension, its value will be 0.
        """
        if var_name is not None:
            if var_name not in ds.variables:
                raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")
            if 'sample' in ds.variables[var_name].dimensions:
                return ds.dimensions['sample'].size
            else:
                return 0

        n_samples = {}
        for var_name in ds.variables:
            if 'sample' in ds.variables[var_name].dimensions:
                n_samples[var_name] = ds.dimensions['sample'].size
            else:
                n_samples[var_name] = 0
        return n_samples

    def var_metadata(ds: Dataset, var_name: str|None = None) -> dict:
        """
        Get the metadata for a variable in the dataset.

        Parameters
        ----------
        ds : netCDF4.Dataset
            The netCDF dataset to inspect.
        var_name : str or None
            The name of the variable to get the metadata for. If None (default), returns 
            metadata for all variables in the dataset.

        Returns
        -------
        dict
            A dictionary containing the metadata for the variable(s), 
            including its dimensions, shape, data type, and any attributes.
        """
        if var_name is not None and var_name not in ds.variables:
            raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")

        if var_name is None:
            metadata = {}
            for var in ds.variables:
                metadata[var] = var_metadata(ds, var)
            return metadata
    
        var = ds.variables[var_name]
        metadata = {
            'dimensions': var.dimensions,
            'shape': var.shape,
            'dtype': var.dtype,
            'is_dimension': var_name in ds.dimensions,
            'attributes': {attr: getattr(var, attr) for attr in var.ncattrs()}
        }
        return metadata

    def smallest_ds(dir_path: Path, return_type: Literal['path', 'contents'] = 'contents') -> Path|Dataset:
        """
        Find the smallest netCDF dataset in a directory based on file size.

        Parameters
        ----------
        dir_path : Path
            The path to the directory containing netCDF files.
        return_type : Literal['path', 'contents'], optional
            The type of the returned value. If 'path', returns the path to the smallest netCDF file.
            If 'contents', returns the contents of the smallest netCDF file as a Dataset.

        Returns
        -------
        Path or Dataset
            The path to the smallest netCDF file in the directory, or its contents as a Dataset, 
            depending on the `return_type` parameter.
        """
        nc_files = list(dir_path.glob('*.nc'))
        if not nc_files:
            raise FileNotFoundError(f"No netCDF files found in directory: {dir_path}")
    
        smallest_file = min(nc_files, key=lambda f: f.stat().st_size)
        if return_type == 'contents':
            return Dataset(smallest_file)
        else:
            return smallest_file

    return smallest_ds, var_metadata


@app.cell
def _(Dataset, Literal, Path, np, pathlib, pd):
    def get_timeseries(
            ds: Dataset,
            var_name: str,
            *,
            sample: int|Literal['all'] = 'all',
            station: int|Literal['NCAR1', 'NCAR2', 'NCAR3', 'NCAR4', 'all'] = 'all',
            return_type: Literal['masked_array', 'data_with_nan', 'mask'] = 'data_with_nan',
        ):
        '''
        Get a time series of a variable from a netCDF dataset.

        Parameters
        ----------
        ds : netCDF4.Dataset
            The netCDF dataset to extract the time series from.
        var_name : str
            The name of the variable to extract. Must be a variable in the dataset.
        sample : int or 'all', optional
            The sample index to extract. If 'all', all samples will be extracted. Defaults to 0.
        station : int or str, optional
            The station index or name to extract. 
            If an integer, it is used as the index into the station dimension of the variable.
            If a string, it must be one of 'NCAR1', 'NCAR2', 'NCAR3', 'NCAR4', or 'all'.
            If 'all', all stations will be extracted. Defaults to 'NCAR1'.
            Integer indices as 1-based, so passing 1 is equivalent to passing 'NCAR1'.
        return_type : str, optional
            The type of the returned time series. Must be one of 'masked_array', 'data_with_nan', or 'mask'.
            Defaults to 'data_with_nan'.


        Returns
        -------
        numpy.masked_array
            The time series of the specified variable for the specified sample and station.
            Shape depends on if 'all' is specified for sample or station. If 'all' is specified for both, the shape will be (time, station, sample).
        '''

        if var_name in ds.dimensions:
            raise ValueError(f"Variable name '{var_name}' is a dimension in the dataset, not a variable.")
        if var_name not in ds.variables:
            raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")

        if 'time' not in ds.variables[var_name].dimensions:
            raise ValueError(f"Cannot make a timeseries for variable '{var_name}' because it does not have a 'time' dimension. Its dimensions are: {ds.variables[var_name].dimensions}.")

        if isinstance(station, str) and station != 'all':
            station_map = {
                'NCAR1': 1,
                'NCAR2': 2,
                'NCAR3': 3,
                'NCAR4': 4
            }
            if station not in station_map:
                raise ValueError(f"Invalid station name '{station}'. Must be one of {list(station_map.keys())}.")
            station_index = station_map[station] - 1  # convert to 0-based index
        elif isinstance(station, int):
            station_index = station - 1  # convert to 0-based index
        elif station == 'all':
            station_index = slice(None)  # select all stations
        else:
            raise TypeError(f"Station must be an int or str, got {type(station)}.")

        if isinstance(station_index, int) and (station_index < 0 or station_index >= ds.dimensions['station'].size):
            raise IndexError(f"Station index {station_index + 1} is out of bounds for dimension 'station' with size {ds.dimensions['station'].size}.")

        if isinstance(sample, int) and (sample < 0 or sample >= ds.dimensions['sample'].size):
            raise IndexError(f"Sample index {sample} is out of bounds for dimension 'sample' with size {ds.dimensions['sample'].size}.")
        elif sample == 'all':
            sample = slice(None)  # select all samples

        # determine which indexers we need to use based on the variable's dimensions
        indexers = []
        for dim in ds.variables[var_name].dimensions:
            if dim == 'time':
                indexers.append(slice(None))  # select all time steps
            elif dim == 'station':
                indexers.append(station_index)
            elif dim == 'sample':
                indexers.append(sample)

        return_type = return_type.lower()
        if return_type == 'masked_array':
            return ds[var_name][tuple(indexers)]
        elif return_type == 'mask':
            return ds[var_name][tuple(indexers)].mask
        elif return_type == 'data_with_nan':
            return ds[var_name][tuple(indexers)].filled(np.nan)
        else:
            raise ValueError(f"Invalid return_type '{return_type}'. Must be one of 'masked_array', 'data_with_nan', or 'mask'.")

    def get_all_timeseries(
            dir_path: Path,
            var_name: str,
            *,
            sample: int|Literal['all'] = 'all',
            station: int|Literal['NCAR1', 'NCAR2', 'NCAR3', 'NCAR4', 'all'] = 'all',
            return_type: Literal['masked_array', 'data_with_nan', 'mask'] = 'data_with_nan',
            return_time: bool = False
    ) -> tuple[np.ndarray, np.ndarray]|np.ndarray:
        '''
        Get the timeseries of a variable from all netCDF files in a directory.
        Returns both the timestamps and the variable values as numpy arrays.

        Parameters
        ----------
        dir_path : str or Path
            Path to the directory containing the netCDF files.
        var_name : str
            The name of the variable to extract. Must be a variable in the dataset.
        sample : int or 'all', optional
            The sample index to extract (from 0 to 19). If 'all', all samples will be extracted. Defaults to 'all'.
        station : int or str, optional
            The station index or name to extract.
            If an integer, it is used as the index into the station dimension of the variable.
            If a string, it must be one of 'NCAR1', 'NCAR2', 'NCAR3', 'NCAR4', or 'all'.
            If 'all', all stations will be extracted. Defaults to 'NCAR1'.
            Integer indices are 1-based, so passing 'NCAR1' is equivalent to passing 1.
        return_type : str, optional
            The type of the returned time series. Must be one of 'masked_array', 'data_with_nan', or 'mask'.
            Defaults to 'data_with_nan'.
        return_time : bool, optional
            Whether to return the timestamps corresponding to the variable values. Defaults to True.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            A tuple containing two numpy arrays, containing:
            1. The timestamps corresponding to the variable values.
            2. The variable values for the specified sample and station.
        or np.ndarray
            If return_time is True, returns a tuple of two numpy arrays as described above.
            If return_time is False, returns only the variable values as a single numpy array.
        '''

        if return_time:
            out_time = np.array([], dtype='datetime64[ns]')
        out_var = np.array([], dtype=np.float32)  # all CWEX timeseries are float32

        is_first = True  # create output arrays from first output, to ensure shape is correct
        for file_path in sorted(pathlib.Path(dir_path).glob('*.nc')):
            if return_time:
                # filenames in format cwex_YYYYMMDD_HH.nc
                date_str = file_path.stem.split('_')[1]
                hr_str = file_path.stem.split('_')[2]
                start_time = pd.to_datetime(date_str + hr_str, format='%Y%m%d%H').to_datetime64()

            ds = Dataset(file_path)
            if return_time:
                new_time = start_time + (ds['time'][:] * 1000).astype('timedelta64[ms]')  # [ms] conversion required as some time values are half-seconds
            new_var = get_timeseries(ds, var_name=var_name, sample=sample, station=station, return_type=return_type)
        

            if is_first:  # create
                is_first = False
                if return_time:
                    out_time = new_time
                out_var = new_var
            else:  # append
                if return_time:
                    out_time = np.append(out_time, new_time, axis=0)
                out_var = np.append(out_var, new_var, axis=0)

        if return_time:
            return out_time, out_var
        else:
            return out_var

    def get_all_time(dir_path: Path) -> np.ndarray:
        """
        Get the time values from all netCDF files in a directory.

        Parameters
        ----------
        dir_path : Path
            Path to the directory containing the netCDF files.

        Returns
        -------
        np.ndarray
            A numpy array containing the time values from all netCDF files in the directory.
        """
        out_time = np.array([], dtype='datetime64[ns]')

        for file_path in sorted(pathlib.Path(dir_path).glob('*.nc')):
            # filenames in format cwex_YYYYMMDD_HH.nc
            date_str = file_path.stem.split('_')[1]
            hr_str = file_path.stem.split('_')[2]
            start_time = pd.to_datetime(date_str + hr_str, format='%Y%m%d%H').to_datetime64()

            ds = Dataset(file_path)
            out_time = np.append(out_time, start_time + (ds['time'][:] * 1000).astype('timedelta64[ms]'), axis=0)

        return out_time

    return get_all_time, get_all_timeseries


@app.cell
def _(FLUX_DIR, smallest_ds, var_metadata):
    CWEX_DIR = FLUX_DIR / 'NCAR' / 'CWEX'

    cwex_metadata = var_metadata(smallest_ds(CWEX_DIR), var_name=None)
    return CWEX_DIR, cwex_metadata


@app.cell
def _(CWEX_DIR, cwex_metadata, get_all_timeseries, plt):
    _co2_times, _co2_data = get_all_timeseries(
        CWEX_DIR,
        var_name = 'co2_4_5m',
        return_type = 'data_with_nan',
        return_time = True,
        sample = 0,
        station = 'NCAR1'
    )

    _fig, _ax = plt.subplots(figsize=(10, 3))

    _ax.scatter(
        _co2_times, _co2_data, 
        marker = '.', s = 1, alpha = 0.5,
        linewidth = 0.25, color = 'tab:green'
    )
    _ax.set_xlabel('Time')
    _ax.set_ylabel(f'[CO$_2$] / ${cwex_metadata["co2_4_5m"]["attributes"]["units"]}$')
    _ax.tick_params(axis = 'x', rotation = 90)
    _ax.grid()

    _ax
    return


@app.cell
def _(Literal, Path, get_all_timeseries, np):
    def get_TKE(
            dir_path: Path,
            station: int|Literal['NCAR1', 'NCAR2', 'NCAR3', 'NCAR4', 'all'] = 'all',
        ) -> np.ndarray:
        '''
        Calculate the Turbulent Kinetic Energy (TKE) from the wind velocity components.
        TKE is estimated as 0.5 * (var(u) + var(v) + var(w)), where var(u), var(v), and var(w) are the variances of the 
        wind velocity in the x, y, and z directions, respectively.

        Parameters
        ----------
        dir_path : Path
            Path to the directory containing the netCDF files.
        station : int or str, optional
            The station index or name to extract.
            If an integer, it is used as the index into the station dimension of the variable.
            If a string, it must be one of 'NCAR1', 'NCAR2', 'NCAR3', 'NCAR4', or 'all'.
            If 'all', all stations will be extracted. Defaults to 'all'.

        Returns
        -------
        np.ndarray
            A numpy array containing the TKE values for the specified station(s) across all time steps.
        '''

        is_first = True  # create output arrays from first output, to ensure shape is correct
        for var_name in ['u_4_5m', 'v_4_5m', 'w_4_5m']:
            velocity_component = get_all_timeseries(
                dir_path,
                var_name=var_name,
                return_type='data_with_nan',
                return_time=False,
                sample='all',
                station=station
            )
            if is_first:
                is_first = False
                out = np.nanvar(velocity_component, axis = 1, ddof=1)  # variance across samples for each time step and station
            else:
                out += np.nanvar(velocity_component, axis = 1, ddof=1)
        return 0.5 * out  # TKE = 0.5 * (u'^2 + v'^2 + w'^2)

    return (get_TKE,)


@app.cell
def _(CWEX_DIR, get_TKE, get_all_time):
    t = get_all_time(CWEX_DIR)
    tke = get_TKE(CWEX_DIR, station = 'all')
    return t, tke


@app.cell
def _(np, plt, t, tke):
    _fig, _ax = plt.subplots(4,1, figsize=(10, 12), sharex=True)

    _colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for _i in range(4):
        _ax[_i].scatter(
            t, tke[:, _i],
            marker = '.', s = 1, alpha = 0.5,
            linewidth = 0.25, color = _colours[_i]
        )

        # fill with grey where data is nan
        _ax[_i].fill_between(
            t, 0, 1, where = np.isnan(tke[:, _i]),
            color = 'lightgrey', alpha = 0.9, transform = _ax[_i].get_xaxis_transform()
        )

        _ax[_i].set_ylabel(f'TKE$_{{{_i+1}}}$ / m$^2$ s$^{-2}$')
        #_ax[_i].grid()
        _ax[_i].set_title(f'NCAR{_i+1}')
        _ax[_i].set_xlim(t[0], t[-1])

    _ax[-1].set_xlabel('Time')
    _ax[-1].tick_params(axis = 'x', rotation = 45)
    

    _fig
    return


@app.cell
def _(CWEX_DIR, gpd, pd):
    gdf_cwex = pd.concat([
        gpd.read_file(CWEX_DIR / 'cwex_ncar.kml')[['Name', 'geometry']],  # from NCAR CWEX KML file
        gpd.read_file(CWEX_DIR / 'cwex_turbines.kml')[['Name', 'geometry']]  # from own google earth mapping 
    ]).rename(columns = {'Name': 'id'}).set_index('id').sort_index()
    gdf_cwex.index = gdf_cwex.index.str.lower()
    gdf_cwex['type'] = gdf_cwex.index.map(lambda x: 'eddy tower' if x.startswith('ncar') else 'wind turbine').astype('category')
    gdf_cwex
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


if __name__ == "__main__":
    app.run()
