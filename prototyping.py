import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import fluxnet_shuttle as shuttle
    import pandas as pd
    import geopandas as gpd

    from matplotlib import pyplot as plt
    from matplotlib_scalebar.scalebar import ScaleBar
    import altair as alt
    import folium

    from shapely.geometry import LineString, MultiPoint, Point, box

    import os
    import zipfile
    import marimo as mo
    import numpy as np
    import json
    import datetime
    import requests
    import sys
    import time
    from tqdm import tqdm

    from icoscp_core.icos import meta, ECO_STATION
    from icoscp_core.queries.dataobjlist import SamplingHeightFilter

    return ECO_STATION, gpd, meta, mo, np, os, pd, plt


@app.cell
def _():
    import eddy

    return (eddy,)


@app.cell
def _(mo):
    mo.md(r"""
    # Data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Natural Earth - Borders & Coastlines
    """)
    return


@app.cell
def _(mo):
    from eddy.data import GDF_COUNTRIES, GDF_STATES
    _ax_coast = GDF_COUNTRIES.plot(linewidth = 0.25, facecolor = 'None', edgecolor = 'k')
    _ax_coast.set_axis_off()
    _ax_coast.set_title('GDF_COUNTRIES: Country Borders')
    _ax_states = GDF_STATES.plot(facecolor = 'None', linewidth = 0.25)
    _ax_states.set_axis_off()
    _ax_states.set_title('GDF_STATES: Country Borders + Internal State Borders')
    mo.vstack([
        mo.hstack([
            _ax_coast,
            _ax_states
        ], gap = 1, justify = 'space-around'),
        mo.accordion({
            '`GDF_COUNTRIES`': GDF_COUNTRIES,
            '`GDF_STATES`': GDF_STATES
        })
    ])
    return GDF_COUNTRIES, GDF_STATES


@app.cell
def _(mo):
    mo.md(r"""
    ## Flux data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Fluxnet
    """)
    return


@app.cell
async def _(eddy):
    gdf_fluxnet = await eddy.data.load_fluxnet_snapshot()
    gdf_fluxnet_path = await eddy.data.load_fluxnet_snapshot(return_type = 'path')
    gdf_fluxnet
    return gdf_fluxnet, gdf_fluxnet_path


@app.cell
def _(GDF_COUNTRIES, gdf_fluxnet, gdf_wind, plt):
    _uk_bbox = (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085)
    _fig, _ax = plt.subplots(figsize=(10, 10))
    gdf_fluxnet.loc[gdf_fluxnet['location_lat'].between(_uk_bbox[1], _uk_bbox[3]) & gdf_fluxnet['location_long'].between(_uk_bbox[0], _uk_bbox[2])].plot(
        marker='o', color='red', markersize=10, figsize=(10, 10), alpha=0.7, ax=_ax
    )
    GDF_COUNTRIES.plot(ax=_ax, linewidth=0.25, facecolor='None')
    gdf_wind.plot(ax=_ax, markersize=2, color='green', alpha = 0.5)
    _ax.set_xlim(_uk_bbox[0], _uk_bbox[2])
    _ax.set_ylim(_uk_bbox[1], _uk_bbox[3])
    _ax.set_axis_off()
    _ax
    return


@app.cell
def _(gdf_fluxnet):
    gdf_fluxnet.loc[gdf_fluxnet['site_id'].str.startswith('UK-')]
    return


@app.cell
def _(GDF_COUNTRIES, gdf_fluxnet, plt):
    _fig, _ax = plt.subplots(figsize = (10,20))
    GDF_COUNTRIES.plot(ax=_ax, facecolor = 'None', linewidth = 0.25)
    _ax.scatter(gdf_fluxnet['location_long'], gdf_fluxnet['location_lat'], color = 'red', s = 0.25, alpha = 0.5)
    _ax.set_axis_off()
    _ax
    return


@app.cell
async def _(eddy, gdf_fluxnet_path):
    sites = ['UK-HpF']
    fluxnet_site_folders = await eddy.data.load_fluxnet_sites(
        sites, snapshot_filepath = gdf_fluxnet_path, force_download = False, 
        extract_zip = True
    )
    fluxnet_site_folders
    return (fluxnet_site_folders,)


@app.cell
def _(fluxnet_site_folders, mo, os):
    _site_dir = fluxnet_site_folders[0]
    mo.accordion({
        'README': mo.plain_text(open(os.path.join(_site_dir, 'README.txt')).read()),
        'Dir contents': [_f.name for _f in sorted(_site_dir.iterdir())]
    }, multiple = True, lazy = True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### AmeriFlux
    [BADM Standards](https://ameriflux.lbl.gov/data/badm/badm-standards/) - for variable names/definitions
    """)
    return


@app.cell
def _(gpd, pd):
    _cols = ['FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END', 'LOCATION_DATE_START', 'LOCATION_COMMENT', 'LOCATION_LAT', 'LOCATION_LONG', 'LOCATION_ELEV']
    # read only the values we want from the flat-structured excel file
    df_ameriflux_badm = pd.read_excel(
        'data/AmeriFlux/AMF_AA-Flx_BIF_CCBY4_20260527.xlsx', 
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
def _(gdf_ameriflux_badm, gpd, pd):
    #df_ameriflux = pd.read_excel('data/AmeriFlux/AMF_AA-Flx_BIF_CCBY4_20260527.xlsx')
    df_ameriflux = pd.read_csv('data/AmeriFlux/BASE_MeasurementHeight_20260527.csv')
    df_ameriflux['var_base'] = df_ameriflux['Variable'].str.split('(_\d)+', regex=True).apply(lambda x: x[0])

    df_ameriflux = df_ameriflux.join(
        pd.read_csv('data/AmeriFlux/flux-met_processing_variables_20260618.csv', index_col = 1), on = 'var_base'
    ).sort_values(['Site_ID', 'Variable'])
    df_ameriflux = df_ameriflux.loc[df_ameriflux['Type'] == 'MET_WIND']
    gdf_ameriflux = gpd.GeoDataFrame(df_ameriflux.join(gdf_ameriflux_badm[['geometry', 'FLUX_MEASUREMENTS_DATE_START', 'FLUX_MEASUREMENTS_DATE_END']], on = 'Site_ID').sort_values(['Site_ID', 'Variable'])).reset_index(drop = True)
    gdf_ameriflux.set_crs('EPSG:4326', inplace = True)
    del df_ameriflux
    gdf_ameriflux
    return (gdf_ameriflux,)


@app.cell
def _(GDF_COUNTRIES, gdf_ameriflux, plt):
    _fig, _ax = plt.subplots(figsize = (10,20))
    GDF_COUNTRIES.plot(ax=_ax, facecolor = 'None', linewidth = 0.5)
    gdf_ameriflux.dissolve(by = 'Site_ID', aggfunc = {'Height':'max'}).plot(
        'Height',
        color = 'red', markersize = 0.5, alpha = 0.5,
        ax = _ax
    )
    _ax.set_axis_off()
    _ax
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ICOS
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - measurement height metadata isn't available via the `icoscp_core` package.

    Email response:
    ```email
    ​Hi Hugo,

    Within our data portal, the EC sensor height is available as part of the Archive file (example).

    In the example, there are two files with variable information:
    - ICOSETC_NL-Loo_VARINFO_METEO_L2.csv (which has the sonic anemometer (WS / Gill HS-50 at 38.2 m))
    - ICOSETC_NL-Loo_VARINFO_FLUXES_L2.csv (which has the gas analyser (CO2, H2O, etc. / LI-7200RS at 38.2 m))

    It would theoretically be possible to script through the various files, if you had them all in the same directory, e.g., but that is obviously not particularly convenient.

    I am unsure if there is a simpler/better way to get this information, but the Ecosystem Thematic Centre might be able to help you get the data in a more convenient way. I would suggest contacting them directly at info@icos-etc.eu

    Best,
    Andrew
    ```
    """)
    return


@app.cell
def _(ECO_STATION, meta):
    eco_stations = meta.list_stations(ECO_STATION)
    # for st in eco_stations:
    #     detailed = meta.get_station_meta(st.uri)
    #     print(st.id, st.name, detailed)  # inspect `detailed` for height fields
    eco_stations
    return (eco_stations,)


@app.cell
def _(eco_stations, meta):
    stn_batch = eco_stations[400:]
    dobjs = meta.list_data_objects(station=stn_batch, limit = 10000, order_by = None)
    return (dobjs,)


@app.cell
def _(dobjs, eco_stations):
    dobjs_by_station = {}
    for d in dobjs:
        dobjs_by_station.setdefault(d.station_uri, []).append(d)

    all_rows = []
    all_variable_labels = set()

    stations_with_dobjs = [s for s in eco_stations if s.uri in dobjs_by_station]
    print(f"\n{len(stations_with_dobjs)} stations have at least one data object.")
    print(f"Pulling full metadata for up to {2} "
            f"object(s) per station...\n")

    # for i, st in enumerate(stations_with_dobjs, start=1):
    #     candidates = dobjs_by_station[st.uri][:DOBJS_PER_STATION]
    #     print(f"[{i}/{len(stations_with_dobjs)}] {st.id} - "
    #             f"{len(candidates)} object(s) to inspect")

    #     for d in candidates:
    #         try:
    #             full = meta.get_dobj_meta(d)
    #         except Exception as exc:
    #             print(f"    ! get_dobj_meta failed for {d.uri}: {exc}")
    #             continue
    return (dobjs_by_station,)


@app.cell
def _(dobjs_by_station, meta, np):
    _site_uri = list(dobjs_by_station.keys())[2]
    _i = np.random.randint(0, len(dobjs_by_station[_site_uri]))
    _d = dobjs_by_station[_site_uri][_i]
    try:
        full = meta.get_dobj_meta(_d)
    except Exception as exc:
        print(f"    ! get_dobj_meta failed for {_d.uri}: {exc}")

    full.specificInfo.columns  # None :(
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Wind
    """)
    return


@app.cell
def _(GDF_COUNTRIES, gpd, mo, pd):
    df_wind = pd.read_excel('data/wind/Global-Wind-Power-Tracker-February-2026.xlsx', sheet_name = 'Data')
    gdf_wind = gpd.GeoDataFrame(df_wind, geometry=gpd.points_from_xy(df_wind['Longitude'], df_wind['Latitude']))
    gdf_wind.set_crs('EPSG:4326', inplace = True)
    gdf_wind['project'] = gdf_wind['Project Name'].astype(str) + ', phase ' + gdf_wind['Phase Name'].astype(str)
    del df_wind

    ax = gdf_wind.plot(column = 'Capacity (MW)', figsize = (10,20), legend = False, markersize = 0.1, alpha = 0.1)
    GDF_COUNTRIES.plot(ax=ax, facecolor = 'None', linewidth = 0.25)
    ax.set_axis_off()
    mo.vstack([
        ax,
        gdf_wind
    ])
    return (gdf_wind,)


@app.cell
def _(GDF_STATES, gdf_wind, mo):
    gdf_wind_us = gdf_wind.loc[gdf_wind['Country/Area'] == 'United States']
    _ax = gdf_wind_us.to_crs('ESRI:102003').plot(column = 'Capacity (MW)', figsize = (10,20), legend = False, markersize = 0.5, alpha = 0.5)
    GDF_STATES.clip_by_rect(*gdf_wind_us.total_bounds).to_crs('ESRI:102003').plot(ax = _ax, facecolor = 'None', linewidth = 0.5)
    _ax.set_axis_off()
    _ax.set_title('Wind Farms in the US')

    mo.vstack([
        _ax,
        gdf_wind_us
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Raw data
    [bico](https://github.com/holukas/bico/tree/master/src) - SwissFluxNet tool for reading raw EC data. Doesn't run on M-series macs, but could virtualise? Not yet set up.
    """)
    return


@app.cell
def _(np):
    np.fromfile('data/downloader/buoy.ecflux.z01.00.20231227.160000.10hz.dat', dtype=np.float32)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Weather stations
    """)
    return


@app.cell
def _():
    # response = requests.get('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt')
    # with open('data/ghcnd-stations.txt', 'wb') as _f:
    #     _f.write(response.content)
    return


if __name__ == "__main__":
    app.run()
