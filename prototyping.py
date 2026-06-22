import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
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
    import altair as alt

    from shapely.geometry import LineString, MultiPoint, Point

    import os
    import zipfile
    import marimo as mo
    import numpy as np
    import json
    import datetime
    import requests
    import sys
    import time

    from icoscp_core.icos import meta, ECO_STATION
    from icoscp_core.queries.dataobjlist import SamplingHeightFilter

    return (
        ECO_STATION,
        LineString,
        datetime,
        gpd,
        meta,
        mo,
        np,
        os,
        pd,
        plt,
        requests,
        shuttle,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Natural Earth
    """)
    return


@app.cell
def _(gpd, mo):
    gdf_coastline = gpd.read_file('/Users/hugoneely/Documents/4-work/1-current/PhD/repos/eddy-cov/data/NaturalEarth/ne_110m_coastline')
    gdf_states = gpd.read_file('/Users/hugoneely/Documents/4-work/1-current/PhD/repos/eddy-cov/data/NaturalEarth/ne_110m_admin_1_states_provinces')
    _ax_coast = gdf_coastline.plot(linewidth = 0.5, color = 'k')
    _ax_coast.set_axis_off()
    _ax_coast.set_title('gdf_coastline: Global Coastline')
    _ax_states = gdf_states.plot(facecolor = 'None', linewidth = 0.5)
    _ax_states.set_axis_off()
    _ax_states.set_title('gdf_states: US States')
    mo.vstack([
        mo.hstack([
            _ax_coast,
            _ax_states
        ], gap = 0.1),
        mo.accordion({
            '`gdf_coastline`': gdf_coastline,
            '`gdf_states`': gdf_states
        })
    ])
    return gdf_coastline, gdf_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fluxnet
    """)
    return


@app.cell
async def _(datetime, gpd, os, pd, shuttle):
    #fname = await shuttle.listall(output_dir = 'data')
    def get_fname(search_dir = 'data'):
        files = os.listdir(search_dir)
        files = [f for f in files if f.endswith('.csv') and f.startswith('fluxnet_shuttle_snapshot_')]
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV files found in directory {dir}.")
        file_dt = [datetime.datetime.strptime(f.split('_')[-1].replace('.csv', ''), '%Y%m%dT%H%M%S') for f in files]
        files = [f for _, f in sorted(zip(file_dt, files), reverse=True)]
        return os.path.join(search_dir, files[0])


    try:
        fname = get_fname()
        print(f'loaded latest file with fname="{fname}" from disk')
    except FileNotFoundError as e:
        fname = await shuttle.listall(output_dir = 'data')
        print(f'downloaded file with fname="{fname}" from shuttle')
    gdf_locs = pd.read_csv(fname)
    gdf_locs = gpd.GeoDataFrame(gdf_locs, geometry=gpd.points_from_xy(gdf_locs['location_long'], gdf_locs['location_lat']))
    gdf_locs
    return fname, gdf_locs


@app.cell
def _(gdf_coastline, gdf_locs, plt):
    _fig, _ax = plt.subplots(figsize = (10,20))
    gdf_coastline.plot(ax=_ax, color = 'k', linewidth = 0.5)
    _ax.scatter(gdf_locs['location_long'], gdf_locs['location_lat'], color = 'red', s = 0.5, alpha = 0.5)
    _ax.set_axis_off()
    _ax
    return


@app.cell
async def _(fname, os, retur, shuttle):
    sites = ['UK-HpF']
    def get_fnames(search_dir = os.path.join('data','sites')):
        if not os.path.isdir(search_dir):
            raise NotADirectoryError(f"Directory {search_dir} does not exist.")
        retur

    fnames = await shuttle.download(sites, snapshot_file=fname, output_dir = os.path.join('data', 'sites'))
    fnames
    return (fnames,)


@app.cell
def _(fnames, os, zipfile):
    site_dir = os.path.join('data', 'sites', 'UK-HpF')
    with zipfile.ZipFile(fnames[0], 'r') as zip_ref:
        zip_ref.extractall(site_dir)
    return (site_dir,)


@app.cell
def _(mo, os, site_dir):
    mo.md(open(os.path.join(site_dir, 'README.txt')).read())
    return


@app.cell
def _(os, site_dir):
    csvs = [os.path.join(site_dir, _f) for _f in os.listdir(site_dir) if _f.endswith('.csv')]
    csvs
    return (csvs,)


@app.cell
def _(csvs, pd):
    pd.read_csv(csvs[1])
    return


@app.cell
def _(csvs, pd):
    _df_info = pd.read_csv(csvs[14])
    _var_to_search = 'TA_F_MDS'
    _grp = _df_info.loc[_df_info['DATAVALUE'] == _var_to_search, 'GROUP_ID'].values[0]
    _df_info.loc[_df_info['GROUP_ID'] == _grp, ['VARIABLE', 'DATAVALUE']]
    return


@app.cell
def _(csvs, pd):
    pd.read_csv(csvs[14])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Raw data
    [bico](https://github.com/holukas/bico/tree/master/src) - SwissFluxNet tool for reading raw EC data. Doesn't run on M-series macs, but could virtualise? Not yet set up.
    """)
    return


@app.cell
def _(np):
    np.fromfile('data/downloader/buoy.ecflux.z01.00.20231227.160000.10hz.dat', dtype=np.float32)
    return


@app.cell
def _(pd):
    pd.read_csv('downloader/buoy.ecflux.z01.00.20231227.160000.10hz.dat',)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weather stations
    """)
    return


@app.cell
def _(requests):
    response = requests.get('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt')
    with open('data/ghcnd-stations.txt', 'wb') as _f:
        _f.write(response.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AmeriFlux
    [BADM Standards](https://ameriflux.lbl.gov/data/badm/badm-standards/) - for variable names/definitions
    """)
    return


@app.cell
def _(gpd, pd):
    df_af_badm = pd.read_excel('data/AmeriFlux/AMF_AA-Flx_BIF_CCBY4_20260527.xlsx')
    df_af_badm = df_af_badm.loc[df_af_badm['VARIABLE_GROUP'] == 'GRP_LOCATION'].pivot_table(index = 'GROUP_ID', columns = 'VARIABLE', values = 'DATAVALUE', aggfunc = 'first')
    df_af_badm = df_af_badm.join(
        pd.read_excel('data/AmeriFlux/AMF_AA-Flx_BIF_CCBY4_20260527.xlsx', usecols = ['SITE_ID', 'GROUP_ID']).groupby('GROUP_ID').agg('first')
    ).set_index('SITE_ID')

    gdf_af_badm = gpd.GeoDataFrame(df_af_badm, geometry = gpd.points_from_xy(df_af_badm['LOCATION_LONG'], df_af_badm['LOCATION_LAT']))
    gdf_af_badm.set_crs('EPSG:4326', inplace = True)
    del df_af_badm
    gdf_af_badm
    return (gdf_af_badm,)


@app.cell
def _(gdf_af_badm, gpd, pd):
    #df_af = pd.read_excel('data/AmeriFlux/AMF_AA-Flx_BIF_CCBY4_20260527.xlsx')
    df_af = pd.read_csv('data/AmeriFlux/BASE_MeasurementHeight_20260527.csv')
    df_af['var_base'] = df_af['Variable'].str.split('(_\d)+', regex=True).apply(lambda x: x[0])

    df_af = df_af.join(
        pd.read_csv('data/AmeriFlux/flux-met_processing_variables_20260618.csv', index_col = 1), on = 'var_base'
    ).sort_values(['Site_ID', 'Variable'])
    df_af = df_af.loc[df_af['Type'] == 'MET_WIND']
    gdf_af = gpd.GeoDataFrame(df_af.join(gdf_af_badm['geometry'], on = 'Site_ID').sort_values(['Site_ID', 'Variable'])).reset_index(drop = True)
    gdf_af.set_crs('EPSG:4326', inplace = True)
    del df_af
    gdf_af
    return (gdf_af,)


@app.cell
def _(gdf_af, gdf_coastline, plt):
    _fig, _ax = plt.subplots(figsize = (10,20))
    gdf_coastline.plot(ax=_ax, color = 'k', linewidth = 0.5)
    gdf_af.dissolve(by = 'Site_ID', aggfunc = {'Height':'max'}).plot(
        'Height',
        color = 'red', markersize = 0.5, alpha = 0.5,
        ax = _ax
    )
    _ax.set_axis_off()
    _ax
    return


@app.cell
def _(gdf_af):
    gdf_af.plot.box(column = 'Height', by = 'var_base', figsize = (8,15), vert = False, grid = True).values[0]
    return


@app.cell
def _(gdf_af):
    gdf_af.loc[gdf_af['Height'] > 75]
    return


@app.cell
def _(gdf_af):
    gdf_af.loc[gdf_af['geometry'].isna(),'Site_ID'].value_counts().sort_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ICOS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Seems measurement height metadata isn't available via the `icoscp_core` package. Will have to look into BADM. Might be available by bulk? Otherwise, will have to find individually for each site.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wind
    """)
    return


@app.cell
def _(gpd, mo, pd):
    df_wind = pd.read_excel('data/wind/Global-Wind-Power-Tracker-February-2026.xlsx', sheet_name = 'Data')
    gdf_wind = gpd.GeoDataFrame(df_wind, geometry=gpd.points_from_xy(df_wind['Longitude'], df_wind['Latitude']))
    gdf_wind.set_crs('EPSG:4326', inplace = True)
    del df_wind

    ax = gdf_wind.plot(column = 'Capacity (MW)', figsize = (10,20), legend = False, markersize = 0.5, alpha = 0.5)
    ax.set_axis_off()
    mo.vstack([
        ax,
        gdf_wind
    ])
    return (gdf_wind,)


@app.cell
def _(gdf_states, gdf_wind):
    gdf_wind_us = gdf_wind.loc[gdf_wind['Country/Area'] == 'United States']
    _ax = gdf_wind_us.to_crs('ESRI:102003').plot(column = 'Capacity (MW)', figsize = (10,20), legend = False, markersize = 0.5, alpha = 0.5)
    gdf_states.clip_by_rect(*gdf_wind_us.total_bounds).to_crs('ESRI:102003').plot(ax = _ax, facecolor = 'None', linewidth = 0.5)
    _ax.set_axis_off()
    _ax
    return (gdf_wind_us,)


@app.cell
def _(gdf_states):
    _bnds = [-1997481.6625468, -150098.36507867, -88060.94924008, 1229139.94296026]
    #for _i, _val in enumerate(gdf_states.to_crs('ESRI:102003').clip_by_rect(*_bnds)):
    #    print(_i, _val.centroid)
    #    print(gdf_states.iloc[_i]['postal'])

    gdf_states.to_crs('ESRI:102003').clip_by_rect(*_bnds)
    return


@app.cell
def _(LineString, gdf_af, gdf_states, gdf_wind_us, mo, pd, plt):
    PAD = 10000
    MIN_HEIGHT = 50

    MARKER_WIND = '1'
    MARKER_SIZE_WIND = 100
    MARKER_WIDTH_WIND = 0.75
    COLOUR_WIND = 'blue'

    MARKER_SIZE_EC = 25
    MARKER_EC = '^'
    COLOUR_EC = 'green'

    ALPHA = 0.5

    gdf_af_temp = gdf_af.dissolve(by = 'Site_ID', aggfunc = {'Height': 'max'}).to_crs('ESRI:102003')
    gdf_af_temp['geometry_af'] = gdf_af_temp['geometry']
    gdf_af_temp.to_crs('ESRI:102003', inplace = True)
    gdf_nbrs = gdf_wind_us[['Project Name', 'Capacity (MW)', 'geometry']].to_crs('ESRI:102003').sjoin_nearest(
        gdf_af_temp[['Height', 'geometry_af', 'geometry']],
        how = 'left', distance_col = 'dist_m',
        max_distance = 100_000,
    )
    del gdf_af_temp

    height_ind = gdf_nbrs['Height'] >= MIN_HEIGHT

    _fig, _ax = plt.subplots(1,1, figsize = (7, 5))

    # zoom in to the area around the wind turbines with height >= MIN_HEIGHT
    _bounds = gdf_nbrs.loc[height_ind].total_bounds
    _bounds[0:2] = _bounds[0:2] - PAD
    _bounds[2:4] = _bounds[2:4] + PAD

    gdf_nbrs.loc[height_ind].plot(  # wind
        color = COLOUR_WIND,
        ax = _ax, 
        markersize = MARKER_SIZE_WIND, alpha = ALPHA, 
        marker = MARKER_WIND, linewidth = MARKER_WIDTH_WIND
    )
    gdf_nbrs.set_geometry('geometry_af').loc[height_ind].plot(  # eddy covariance
        color = COLOUR_EC, 
        ax = _ax, 
        legend = False, 
        markersize = MARKER_SIZE_EC, alpha = ALPHA,
        marker = MARKER_EC
    )

    gdf_states.to_crs('ESRI:102003').plot(ax=_ax, facecolor = 'None', linewidth = 0.5, autolim = False, zorder = 0)

    # lines
    gdf_nbrs.loc[height_ind].apply(lambda row: LineString([row['geometry'].centroid, row['geometry_af'].centroid]) if not pd.isnull(row[['geometry_af', 'geometry']].values).any() else None, axis = 1).plot(
        linewidth = 0.1, color = 'grey',
        ax = _ax
    )
    _legend = _ax.legend(
        ['Wind Turbine', 'Eddy Covariance Tower'],
        loc = 'upper right',
        markerscale = 1,
        fontsize = 10,
        frameon = True,
        fancybox = True,
        framealpha = 0.5
    )

    _ax.set_axis_off()


    mo.vstack([
        #mo.ui.matplotlib(_ax),
        _ax,
        gdf_nbrs.loc[height_ind],
        _bounds
    ], align = 'center')
    return gdf_nbrs, height_ind


@app.cell
def _(gdf_nbrs, height_ind):
    gdf_nbrs[height_ind]['dist_m'].describe().astype(int)
    return


if __name__ == "__main__":
    app.run()
