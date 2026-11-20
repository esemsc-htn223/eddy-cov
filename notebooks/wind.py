import marimo

__generated_with = "0.25.2"
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
    from sklearn.neighbors import BallTree

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
    return Literal, folium, mo, np, plt


@app.cell
def _():
    import eddy

    # get dirs too
    from eddy.flux.data import FLUX_DIR, FLUXNET_DIR, TERN_DIR, AMERIFLUX_DIR
    from eddy.util import DATA_DIR, OUT_DIR
    from eddy.util import standardise_df


    return (eddy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _(eddy):
    # geographic data
    GDF_COUNTRIES = eddy.geo.data.load_natural_earth_countries()
    GDF_STATES = eddy.geo.data.load_natural_earth_states()

    #_uk_bbox = (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085)
    UK_BBOX = {
        'min_lon': -10.0,
        'max_lon': 2.0,
        'min_lat': 49.0,
        'max_lat': 61.0
    }
    return GDF_COUNTRIES, UK_BBOX


@app.cell
async def _(eddy):
    # flux tower locations
    gdf_fluxnet = await eddy.flux.data.load_fluxnet_snapshot()
    return (gdf_fluxnet,)


@app.cell
def _(GDF_COUNTRIES, eddy, mo):
    # wind turbine locations
    gdf_wind = eddy.turbines.data.load_locs()

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.1 Global ballpark - GEM wind data
    """)
    return


@app.cell
def _(GDF_COUNTRIES, eddy, gdf_fluxnet, gdf_wind, mo, plt):
    match_dist = 35_000  # m
    min_years = 2

    gdf_matches = eddy.turbines.ec_neighbours(
        gdf_wind, gdf_fluxnet, 
        max_match_dist = match_dist,  # m
        overlap = min_years  # 2 years minimum 
    )

    _ax = eddy.turbines.plot.plot_ec_neighbours(
        gdf_matches,
        pad = 500000,
        gdf_borders = GDF_COUNTRIES,
        scale_bar = False,
        figsize = (15, 7)
    )

    _fig, _ax_hist  = plt.subplots(1,1, figsize = (7,5))
    (gdf_matches['dist_m']/1000).plot.hist(bins = 250, ax = _ax_hist)


    mo.vstack([
        mo.md(f'### Wind Turbine to EC Tower Matches (max dist. = {match_dist/1000:.0f}km, min overlap = {min_years}yrs)'),
        _ax,
        mo.hstack([
            _ax_hist,
            (gdf_matches['dist_m'] / 1000).rename('dist (km)').describe().round(1).to_frame().T[['count', 'mean', '50%', 'min', 'max']].convert_dtypes()
        ], align = 'center')

    ], align = 'center')
    return (gdf_matches,)


@app.cell
def _(gdf_matches):
    gdf_matches
    return


@app.cell
def _(gdf_matches):
    gdf_matches.groupby('site_id').agg(
        n_wind_matches = ('wind_project', 'count'),
        n_below_5km = ('dist_m', lambda x: (x < 5000).sum()),
        min_dist_m = ('dist_m', 'min'),
        max_dist_m = ('dist_m', 'max'),
        mean_dist_m = ('dist_m', 'mean'),
        median_dist_m = ('dist_m', 'median'),
    ).sort_values('min_dist_m')
    return


@app.cell
def _(gdf_matches, gdf_wind):
    gdf_wind.loc[gdf_wind['project'].isin(gdf_matches['wind_project'])]
    return


@app.cell
def _(gdf_matches):
    gdf_matches.loc[gdf_matches['site_id'] == 'DE-RuC']
    return


@app.cell
def _(GDF_COUNTRIES, eddy, gdf_matches, mo):

    _df = gdf_matches.loc[gdf_matches['site_id'].str.startswith('UK')].sort_values('dist_m')
    mo.vstack([
        eddy.turbines.plot.plot_ec_neighbours(
            _df,
            pad = 500000,
            gdf_borders = GDF_COUNTRIES,
            scale_bar = False,
            figsize = (15, 7)
        ),
        _df,
        _df.groupby('site_id').agg(
            n_wind_matches = ('wind_project', 'count'),
            n_below_5km = ('dist_m', lambda x: (x < 5000).sum()),
            min_dist_m = ('dist_m', 'min'),
            max_dist_m = ('dist_m', 'max')
        )
    ])

    return


@app.cell
def _(GDF_COUNTRIES, eddy, gdf_matches):
    _ax = eddy.turbines.plot.plot_ec_neighbours(
        gdf_matches.loc[gdf_matches['site_id'] == 'DE-RuC'],
        pad = 750000,
        gdf_borders = GDF_COUNTRIES,
        scale_bar = False,
        figsize = (15, 7)
    )

    _ax.set_title('Farms close to DE-RuC EC tower (25 matches)')
    return


@app.cell
async def _(eddy):
    # site_files = await eddy.flux.data.get_fluxnet_site_dirs(
    #     gdf_matches['site_id'].unique().tolist(), 
    # )

    site_files = await eddy.flux.data.get_fluxnet_site_dirs(['DE-RuC', 'UK-Gnn'])

    fs_de_ruc = site_files['DE-RuC']
    fs_uk_gnn = site_files['UK-Gnn']
    return (fs_de_ruc,)


@app.cell
def _(fs_de_ruc):
    fs_de_ruc.fluxmet()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1.2 Local wind data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sources:
    - **Germany**: MaStR
    - **US**: Wind Turbine Database?
    - **UK**: DUKES?
    """)
    return


@app.cell
def _():
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
    - [Definitions for Woodland Carbon Code RAG ratings](https://www.woodlandcarboncode.org.uk/4-verification#:~:text=units%2E-,Green)

    - [Offset Map](https://offsetmap.com/?surface=projects&map=20.13847%2C-64.6875%2C2)
        - Middleman that charges fees, but states the sources. Useful for determining where to get data on offsets!
    """)
    return


@app.cell
def _(GDF_COUNTRIES, UK_BBOX, eddy, gdf_wind, plt):
    gdf_offset_uk = eddy.offset.data.load_uk()

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
def _(eddy, gdf_offset_uk, gdf_wind):
    gdf_wo_matches = eddy.turbines.offset_neighbours(gdf_wind, gdf_offset_uk, max_match_dist = 10_000).sort_values('dist_m')
    gdf_wo_matches
    return (gdf_wo_matches,)


@app.cell
def _(eddy, gdf_wo_matches):
    m_wo = eddy.turbines.plot.explore_offset_neighbours(
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
    df_offset_within_wake = gdf_wo_matches.loc[gdf_wo_matches['dist_m'] <= WAKE_DIST_M].groupby('offset_project_name').agg(
        area_ha = ('offset_area_ha', 'sum'),
        offset_classes = ('offset_class', lambda _s: ', '.join(sorted(_s.unique()))),
        offset_subclasses = ('offset_subclass', lambda _s: ', '.join(sorted(_s.unique()))),
        n_wind_farms = ('wind_project_name', 'count'),
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
def _():
    from eddy.flux.cwex import CWEX

    cwex = CWEX()
    return (cwex,)


@app.cell
def _(cwex, mo, np, plt):
    from matplotlib.image import imread

    _tmin = np.datetime64('2011-07-02T00:00:00')
    _tmax = np.datetime64('2011-08-17T00:00:00')

    _ax1 = cwex.plot_windrose(
        1, samples = 'all',
        instrument = 'vane',
        t_min=_tmin, t_max=_tmax,
    )

    _im = imread(cwex.dir_path / 'cwex_ncar1_windrose.png')
    _fig, _ax2 = plt.subplots(figsize=(7, 7))
    _ax2.imshow(_im)
    _ax2.set_axis_off()

    mo.hstack([_ax1, _ax2], align = 'center', justify = 'space-around')
    return


app._unparsable_cell(
    r"""
    cwex.plot_site(
        t_min = np.datetime64('2011-07-02T00:00:00'),
        t_max = np.datetime64('2011-08-17T00:00:00'),
        colour_by = 'co2_4_5m'
    )/
    """,
    name="_"
)


@app.cell
def _(cwex):
    wd, ws = cwex.wind(station = 1)
    return wd, ws


@app.cell
def _(np, plt, wd, ws):
    _fig, _ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    _ax.scatter(
        np.radians(wd), ws, s = 1, alpha = 0.2
    )
    _ax.set_theta_zero_location('N')
    _ax.set_theta_direction(-1)
    _ax


    _WIDTH = 5  # degrees
    _ax.axvspan(
        np.radians(180-_WIDTH/2), np.radians(180+_WIDTH/2), color = 'red', alpha = 0.2
    )

    _index = (wd >= 180-_WIDTH/2) & (wd <= 180+_WIDTH/2)

    _ax.scatter(
        np.radians(wd[_index]), ws[_index], s = 2, alpha = 1, color = 'red'
    )
    return


@app.cell
def _(cwex):
    cwex.metadata
    return


@app.cell
def _(Literal):
    def candidate_wakes(
            tower: Literal['NCAR1']
        ):
        '''
        Return times in which it is likely that the given tower is in the wake of a wind turbine.
        Determined by the wind direction being within a given width of the direction from the turbine to the tower.
        Wind speed threshold may also be applied.


        '''

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Anisotropy-based MOST can help!
    - [Paper](https://arxiv.org/abs/2502.13970v2)
    - [Data](https://data.neonscience.org/data-products/DP4.00200.001)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
