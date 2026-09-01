import marimo

__generated_with = "0.24.0"
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
    import eddy

    return


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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##
    """)
    return


if __name__ == "__main__":
    app.run()
