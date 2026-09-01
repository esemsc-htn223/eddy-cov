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

    import os

    import marimo as mo
    import numpy as np
    import json
    import datetime
    import requests
    import sys
    import time
    from tqdm import tqdm


    return LineString, Point, ScaleBar, box, folium, gpd, mo, np, pd, plt


@app.cell
def _():
    import eddy
    from eddy.data import DATA_DIR, FLUXNET_DIR, TERN_DIR, FLUX_DIR
    from eddy.util import OUT_DIR


    return DATA_DIR, FLUX_DIR, OUT_DIR, eddy


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
    return GDF_COUNTRIES, GDF_STATES


@app.cell
async def _(eddy):
    # flux tower locations
    gdf_fluxnet = await eddy.data.load_fluxnet_snapshot()
    return (gdf_fluxnet,)


@app.cell
def _(DATA_DIR, GDF_COUNTRIES, gpd, mo, pd):
    # wind turbine locations
    _df_wind = pd.read_excel(DATA_DIR / 'wind' / 'Global-Wind-Power-Tracker-February-2026.xlsx', sheet_name = 'Data')
    gdf_wind = gpd.GeoDataFrame(_df_wind, geometry=gpd.points_from_xy(_df_wind['Longitude'], _df_wind['Latitude']))
    gdf_wind.set_crs('EPSG:4326', inplace = True)
    gdf_wind['project'] = gdf_wind['Project Name'].astype(str) + ', phase ' + gdf_wind['Phase Name'].astype(str)
    del _df_wind

    ax = gdf_wind.plot(column = 'Capacity (MW)', figsize = (10,20), legend = False, markersize = 0.1, alpha = 0.1)
    GDF_COUNTRIES.plot(ax=ax, facecolor = 'None', linewidth = 0.25)
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
        - Are towers close enough to offset sites to be in a wind wake?
            - Allows us to quantify risk, if any
    2. Are towers located near or on *likely turbine sites*?
        - proposed/planned/constructing sites
        - theoretically good sites for wind turbines?
        - Allows us to quantufy opportunities
    """)
    return


@app.cell
def _():
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
        - What's the theory?
            - What defines a wind wake?
                - Do these things affect carbon flux?
                - Any other notable effects that should be measureable?
        - Can we see this theory play out in data from the towers that are closest to wind turbines?
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
def _(
    GDF_STATES,
    LineString,
    ScaleBar,
    gdf_ameriflux,
    gdf_wind,
    gpd,
    mo,
    np,
    pd,
    plt,
):
    def plot_wind_ec_matches(
            gdf_wind: gpd.GeoDataFrame, gdf_ec: gpd.GeoDataFrame, 
            *,
            site_id_col: str = 'Site_ID', height_col: str = 'Height',
            gdf_borders: gpd.GeoDataFrame = GDF_STATES,
            min_height: int|None = None, max_match_dist: int = 100_000, 
            crs: str = 'EPSG:3857',
            pad: int = 10_000,
            marker_wind: str = '1', marker_size_wind: int = 100, marker_width_wind: float = 0.75, colour_wind: str = 'blue',
            marker_size_ec: int = 25, marker_ec: str = '^', colour_ec: str = 'green',
            alpha: float = 0.5, 
            return_nbrs_gdf: bool = False,
            figsize = (7, 5),
            bounds: tuple|None = None,
            label_state_names: bool = False,
            **kwargs
        ):
        '''
        Plot the nearest wind turbines to eddy covariance towers, within a maximum distance, and with a minimum height for the towers.

        Parameters
        ----------
        gdf_wind : geopandas.GeoDataFrame
            GeoDataFrame containing wind turbine locations and attributes.
        gdf_ec : geopandas.GeoDataFrame
            GeoDataFrame containing eddy covariance tower locations and attributes.
        gdf_borders : geopandas.GeoDataFrame, optional
            GeoDataFrame containing country/state borders to plot in the background. Defaults to Natural Earth 10m admin 1 states/provinces.
        min_height : int, optional
            Minimum height of eddy covariance towers to include in the plot. If None, all towers are included. Defaults to None.
            If specified, `gdf_ec` must contain a 'Height' column with the tower heights.
        crs : str, optional
            Coordinate reference system to use for plotting. Defaults to 'EPSG:3857' (Web Mercator).
        max_match_dist : int, optional
            Maximum distance (in meters) to consider a wind turbine as a match for an eddy covariance tower. Defaults to 100,000 m.
        pad : int, optional
            Padding (in meters) around the matched points to set the plot bounds. Defaults to 10,000 m.
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
        label_state_names : bool, optional
            If True, label the state names on the plot. Defaults to False.
        **kwargs
            Additional keyword arguments to pass to the plotting functions.
        '''

        if site_id_col not in gdf_ec.columns:
            raise ValueError(f"gdf_ec must contain a '{site_id_col}' column with the tower site IDs.")

        if min_height is not None:
            if height_col not in gdf_ec.columns:
                raise ValueError(f"gdf_ec must contain a '{height_col}' column with the tower heights.")

            gdf_ec_grouped = gdf_ec.dissolve(
                by = site_id_col, aggfunc = None, 
                **{
                    'height_max': (height_col, 'max'), 
                    'height_min': (height_col, 'min'), 
                    'n_sensors': (site_id_col, 'count'),
                    'sensor_heights': (height_col, lambda x: list(x))
                }
            ).to_crs(crs)
            gdf_ec_grouped['geometry_eddy'] = gdf_ec_grouped['geometry']
            gdf_nbrs = gdf_wind[['Project Name', 'Phase Name', 'Capacity (MW)', 'geometry', 'Start year', 'Retired year', 'project']].to_crs(crs).sjoin_nearest(
                gdf_ec_grouped[['height_max', 'height_min', 'n_sensors', 'geometry_eddy', 'geometry']],
                how = 'left', distance_col = 'dist_m',
                max_distance = max_match_dist
            )
            if gdf_nbrs.empty:
                raise ValueError(f"No matches found within {max_match_dist} m for eddy covariance towers with max height >= {min_height} m.")
            del gdf_ec_grouped
        else:
            gdf_ec_grouped = gdf_ec.dissolve(
                by = site_id_col, aggfunc = None,
                **{
                    'n_sensors': (site_id_col, 'count'),
                }
            ).to_crs(crs)
            gdf_ec_grouped['geometry_eddy'] = gdf_ec_grouped['geometry']
            gdf_nbrs = gdf_wind[['Project Name', 'Phase Name', 'Capacity (MW)', 'geometry', 'Start year', 'Retired year', 'project']].to_crs(crs).sjoin_nearest(
                gdf_ec_grouped[['n_sensors', 'geometry_eddy', 'geometry']],
                how = 'inner', distance_col = 'dist_m',
                max_distance = max_match_dist
            )
            if gdf_nbrs['geometry_eddy'].isnull().all():
                raise ValueError(f"No matches found within {max_match_dist} m for eddy covariance towers.")
            del gdf_ec_grouped

        first_cols = ['Project Name', 'Phase Name', 'Capacity (MW)', site_id_col, 'dist_m']
        other_cols = [col for col in gdf_nbrs.columns if col not in first_cols]
        gdf_nbrs = gdf_nbrs[first_cols + other_cols]

        # TODO: time alignment of wind turbine operation and EC tower measurement periods
        #gdf_nbrs = gdf_nbrs.join(gdf_ameriflux_badm[['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = site_id_col)

        fig, ax = plt.subplots(1,1, figsize = figsize)

        if min_height is not None:
            height_ind = gdf_nbrs['height_max'] >= min_height
        else:
            height_ind = gdf_nbrs.index

        # apply bounds, adding padding if specified
        if bounds is None:
            bounds = gdf_nbrs.loc[height_ind].total_bounds
        else:
            bounds = np.array(bounds)
        bounds[0:2] = bounds[0:2] - pad
        bounds[2:4] = bounds[2:4] + pad

        gdf_nbrs.loc[height_ind].plot(  # wind
            color = colour_wind,
            ax = ax, 
            markersize = marker_size_wind, alpha = alpha, 
            marker = marker_wind, linewidth = marker_width_wind
        )
        gdf_nbrs.set_geometry('geometry_eddy').loc[height_ind].plot(  # eddy covariance
            color = colour_ec, 
            ax = ax, 
            legend = False, 
            markersize = marker_size_ec, alpha = alpha,
            marker = marker_ec
        )

        GDF_STATES.to_crs(crs).plot(ax=ax, facecolor = 'None', linewidth = 0.5, autolim = False, zorder = 0)
        if label_state_names:
            for _, row in GDF_STATES.to_crs(crs).iterrows():
                if bounds[0] <= row['geometry'].centroid.x <= bounds[2] and bounds[1] <= row['geometry'].centroid.y <= bounds[3]:
                    ax.annotate(
                    row['name'], xy = row['geometry'].centroid.coords[0],
                    horizontalalignment = 'center', fontsize = 8, color = 'black', fontfamily = 'monospace'
                    )

        # lines
        gdf_nbrs.loc[height_ind].apply(lambda row: LineString([row['geometry'].centroid, row['geometry_eddy'].centroid]) if not pd.isnull(row[['geometry_eddy', 'geometry']].values).any() else None, axis = 1).plot(
            linewidth = 0.1, color = 'grey',
            ax = ax
        )
        ax.legend(
            ['Wind Turbine', 'Eddy Covariance Tower'],
            #loc = 'lower right',
            loc = 'best',
            markerscale = 1,
            fontsize = 10,
            frameon = True,
            fancybox = True,
            framealpha = 0.5,
        )
        ax.add_artist(ScaleBar(1))  # add scale bar to give a gauge of distance
        ax.set_axis_off()


        if return_nbrs_gdf:
            return ax, gdf_nbrs
        else:
            return ax

    _ax, gdf_nbrs = plot_wind_ec_matches(
        gdf_wind, gdf_ameriflux,
        gdf_borders = GDF_STATES,
        min_height = 50,
        crs = 'ESRI:102003',
        return_nbrs_gdf = True,
        label_state_names = True,
    )

    _ax.set_title(f'Eddy Covariance Towers (max height >= 50 m) and Their Nearest Wind Turbines (within 100 km)', fontsize = 12)
    height_ind = gdf_nbrs['height_max'] >= 50

    mo.vstack([
        mo.hstack([
            _ax,
            mo.vstack([
                mo.md(f'### Summary Stats of Distances (m) Between EC Towers (max height >= 50 m) and Nearest Wind Turbines'),
                gdf_nbrs[height_ind]['dist_m'].describe().astype(int)
            ], align = 'center')
        ], align = 'center', justify = 'space-around', gap = 1),
        gdf_nbrs['dist_m'].describe().astype(int)
    ])
    return gdf_nbrs, plot_wind_ec_matches


@app.cell
def _(gdf_fluxnet, gdf_wind, mo, plot_wind_ec_matches):
    _ax, gdf_nbrs_test = plot_wind_ec_matches(
        gdf_wind, gdf_fluxnet.loc[gdf_fluxnet['site_id'].str.startswith('UK-')],
        site_id_col = 'site_id',
        max_match_dist = 30_000,
        #bounds = (-13000000, 4000000, -7000000, 6000000)
        return_nbrs_gdf = True,
        pad = 1_000
    )

    mo.vstack([
        mo.hstack([
            _ax,
            mo.vstack([
                mo.md(f'### Summary Stats of Distances (m) Between UK EC Towers and Nearest Wind Turbines'),
                gdf_nbrs_test['dist_m'].describe().astype(int)
            ], align = 'center')
        ], align = 'center', justify = 'space-around', gap = 1),
        gdf_nbrs_test['dist_m'].describe().astype(int)
    ])
    return


@app.cell
def _(GDF_STATES, folium, gdf_nbrs, pd):
    def plot_nearest_turbines_folium(
            gdf_nbrs, *, 
            colour_wind = 'blue', colour_ec = 'green', marker_size_wind = 100, marker_width_wind = 0.75, marker_size_ec = 25, alpha = 0.5
    ):

        m = GDF_STATES.explore(color = 'None', style_kwds = {'color': 'black', 'weight': 0.5}, tiles = 'CartoDB positron', tooltip = False)
        gdf_nbrs.rename(
            columns={
                'Site_ID': 'Neighbour Site ID', 
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

    m = plot_nearest_turbines_folium(gdf_nbrs.loc[gdf_nbrs['Site_ID'] == 'US-PFa'], colour_wind = 'blue', colour_ec = 'green', marker_size_wind = 100, marker_width_wind = 0.75, marker_size_ec = 25, alpha = 0.5)
    m.show_in_browser()
    return (m,)


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
