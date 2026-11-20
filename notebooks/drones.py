import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eddy - drones
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
    from sklearn.neighbors import BallTree
    from shapely.geometry import Point
    import numpy as np

    return BallTree, Point, np


@app.cell
def _():
    import eddy

    return (eddy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is the state of the art?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. How does UAV eddy covariance work?
    2. How does UAV EC compare to regular?
        - Drawbacks, tradeoffs, etc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How could this extend the wind project?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Are UAVs better equipped to measure wind wakes?
        - Could these provide measurements in lieu of a tower, if there are no towers located close enough to the points we're interested in?
    2. Can UAVs supplement EC tower measurements?
    2. Are the sites Yves has data for close to any wind turbines?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. UAVs measuring wind wakes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Claude convo](https://claude.ai/chat/81910502-e1ac-4c8a-b958-7429b63fdf65)
    - UAVs likely can only get 2Hz samples at best - so half the rate of CWEX EC data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. UVAs supplementing EC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Claude convo](https://claude.ai/chat/81910502-e1ac-4c8a-b958-7429b63fdf65)
    - Can provide spatially-varying roughness lengths - see [here](https://meetingorganizer.copernicus.org/EGU24/EGU24-9398.html)
        - Produces more refined footprint estimations, which vary with time
    - More specific plant type data
        - Height as a function of time (and space?)
        - Other variables maybe more useful: plant area density profile, leaf area index proxies, etc? Data helps better parameterise plants.
            - Do we need new measurements for this??
    - Access to direct measurements of vertical profiles - less reliance on models for this (e.g. LogLL) - free myself from the shackles of MOST
    - If equipped with thermal cameras, could get direct measurements of energy exchange and determine thermal defecit/change at night.

    - Big q is how long these measurements would be - how long is a data run? I'd assume max an hour or two?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Yves data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Winery
    > Rathfinny wine estate @ lat = 50.797955, lon = 0.131614
    """)
    return


@app.cell
def _(eddy):
    gdf_wind = eddy.turbines.data.load_locs()
    return (gdf_wind,)


@app.cell
def _(BallTree, Point, gdf_wind, np):
    _winery = Point(0.131614, 50.797955)
    ball_tree = BallTree(
        np.deg2rad(
            np.reshape(gdf_wind.to_crs('EPSG:4326').geometry.apply(lambda x: (x.y, x.x)).to_list(), (-1, 2))
        ),
        metric = 'haversine'
    )

    dist, indices = ball_tree.query(
        np.radians([[_winery.y, _winery.x]]), 
        k = 5
    )

    gdf_wind.iloc[indices[0]].assign(dist_m = dist[0] * 6371000).sort_values('dist_m')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Closest to winery is 28km away, so likely not relevant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
