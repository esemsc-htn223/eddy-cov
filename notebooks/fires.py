import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eddy - Fires
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports & Data
    """)
    return


@app.cell
def _():
    import eddy.data as d

    d.DATA_DIR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fires
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Data assimilation with current fire-spread algorithms:
        Fire-EnSF/WRF-SFIRE/FARSITE
    """)
    return


@app.cell
def _(gdf_fluxnet):
    gdf_fluxnet.loc[gdf_fluxnet['site_name'].str.contains('warra', case = False)]
    return


@app.cell
def _(pd):
    # load flux and meterological data
    df_warra = pd.read_csv('data/TERN/TERN_AU-Wrr_FLUXNET_2013-2021_v1.3_r1/TERN_AU-Wrr_FLUXNET_FLUXMET_HH_2013-2021_v1.3_r1.csv')

    # collapse start and end cols into a single period col
    df_warra['TIMESTAMP_START'] = pd.to_datetime(df_warra['TIMESTAMP_START'], format = '%Y%m%d%H%M')
    df_warra['TIMESTAMP_END'] = pd.to_datetime(df_warra['TIMESTAMP_END'], format = '%Y%m%d%H%M')
    df_warra = df_warra.set_index('TIMESTAMP_START')

    # get variable info
    df_warra_info = pd.read_csv('data/TERN/TERN_AU-Wrr_FLUXNET_2013-2021_v1.3_r1/TERN_AU-Wrr_FLUXNET_BIFVARINFO_HH_2013-2021_v1.3_r1.csv')
    df_warra_info = df_warra_info.loc[df_warra_info['VARIABLE_GROUP'] == 'GRP_VAR_INFO'].pivot(
        index = 'GROUP_ID', 
        columns = 'VARIABLE', 
        values = 'DATAVALUE'
    )[['VAR_INFO_VARNAME', 'VAR_INFO_DEFINITION', 'VAR_INFO_UNIT', 'VAR_INFO_HEIGHT', 'VAR_INFO_DATE', 'VAR_INFO_MODEL']].rename(
        columns = {
            'VAR_INFO_DEFINITION': 'description', 
            'VAR_INFO_UNIT': 'units',
            'VAR_INFO_HEIGHT': 'height',
            'VAR_INFO_DATE': 'date', 
            'VAR_INFO_MODEL': 'model',
            'VAR_INFO_VARNAME': 'variable'
        }
    ).set_index(
        'variable', 
        drop = True
    ).sort_index()
    df_warra_info['date'] = pd.to_datetime(df_warra_info['date'], format = 'mixed', yearfirst= True)

    df_warra_info = df_warra_info.loc[df_warra.columns.intersection(df_warra_info.index)]



    df_warra
    return df_warra, df_warra_info


@app.cell
def _(df_warra):
    df_warra.nunique().sort_values(ascending = True)
    return


@app.cell
def _(df_warra):
    cat_cols = [col for col in df_warra.columns if '_QC' in col and df_warra[col].dtype == 'int64']
    df_warra[cat_cols].nunique().sort_values(ascending = False)
    return


@app.cell
def _(df_warra_info):
    df_warra_info
    return


@app.cell
def _(df_warra, pd):
    from dataclasses import dataclass
    from typing import Literal

    @dataclass
    class FireInfo:
        name: str
        url: str|None = None
        relates_to: Literal['fire', 'site', 'both'] = 'fire'

        @classmethod
        def from_dict(cls, data: dict):
            return cls(
                name = data.get('name'),
                url = data.get('url'),
                relates_to = data.get('relates_to', 'fire')
            )

        @classmethod
        def list_from_dict(cls, data: list[dict]):
            return [cls.from_dict(item) for item in data]

    @dataclass
    class TowerFire:
        name: str
        site_id: str
        start_date: pd.Timestamp
        end_date: pd.Timestamp
        sources: list[FireInfo]|None = None
        df: pd.DataFrame|None = None

        def __post_init__(self):
            if self.start_date > self.end_date:
                raise ValueError(f"start_date {self.start_date} cannot be after end_date {self.end_date}.")
            if not isinstance(self.start_date, pd.Timestamp):
                try:
                    self.start_date = pd.Timestamp(self.start_date)
                except Exception as e:
                    raise ValueError(f"start_date {self.start_date} is not a valid date. Error: {e}")
            if not isinstance(self.end_date, pd.Timestamp):
                try:
                    self.end_date = pd.Timestamp(self.end_date)
                except Exception as e:
                    raise ValueError(f"end_date {self.end_date} is not a valid date. Error: {e}")


        @classmethod
        def from_dict(cls, data: dict):
            return cls(
                name = data.get('name'),
                site_id = data.get('site_id'),
                start_date = pd.Timestamp(data.get('start_date')),
                end_date = pd.Timestamp(data.get('end_date')),
                sources = FireInfo.list_from_dict(data.get('sources', []))
            )

        def __repr__(self):
            return f'TowerFire(name={self.name!r}, site_id={self.site_id!r}, start_date={self.start_date!r}, end_date={self.end_date!r})'

    fire_warra = TowerFire(
        name = 'Warra',
        site_id = 'AU-Wrr',
        start_date = pd.Timestamp('2019-01-15'),
        end_date = pd.Timestamp('2019-03-31'),
        sources = FireInfo.list_from_dict([
            {
                'name': 'Tasmanian Bushfires 2018-19',
                'url': 'https://knowledge.aidr.org.au/resources/2018-19-bushfire-tas-tasmanian-bushfires/',
                'relates_to': 'fire'
            },
            {
                'name': 'Paper',
                'url': 'https://www.mdpi.com/2571-6255/4/2/15',
                'relates_to': 'site'
            }
        ]),
        df = df_warra
    )

    df_warra['fire_flag'] = ((df_warra.index >= fire_warra.start_date) & (df_warra.index <= fire_warra.end_date)).astype(int)
    return Literal, fire_warra


@app.cell
def _(fire_warra, pd, plt):
    def plot_site_data(ax, col, *, df:pd.DataFrame, 
                       start_date: str|pd.Timestamp|None = None, end_date: str|pd.Timestamp|None = None, 
                       highlight_period: tuple[str|pd.Timestamp, str|pd.Timestamp]|None = None,
                       clip_min_vals: bool = True, **kwargs):
        """
        Plot Wind Direction (WD), Wind Speed (WS), CO2 Flux (CO2_F_MDS), or Sensible Heat Flux (H_F_MDS) for the specified date range.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the data.
        col : str
            The column name to plot (e.g., 'WD', 'WS', 'CO2_F_MDS', 'H_F_MDS').
        df : pd.DataFrame
            DataFrame containing the flux data.
        start_date : str | pd.Timestamp | None, optional
            Start date for the plot in 'YYYY-MM-DD' format (default is None - plots all data).
        end_date : str | pd.Timestamp | None, optional
            End date for the plot in 'YYYY-MM-DD' format (default is None - plots all data).
        highlight_period : tuple[str | pd.Timestamp, str | pd.Timestamp] | None, optional
            Tuple containing the start and end dates of the period to highlight in the plot (default is None - no highlighting).
        clip_min_vals : bool, optional
            If True, clip the data to exclude values equal to the minimum value of the column (default is True).
        **kwargs : dict
            Additional keyword arguments to pass to df.plot().

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if start_date is not None and not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
        if end_date is not None and not isinstance(end_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date, format = '%Y-%m-%d')

        ind = df.index.notna()
        if start_date is not None:
            ind &= (df.index >= start_date)
        if end_date is not None:
            ind &= (df.index <= end_date)
        if clip_min_vals:
            ind &= (df[col] > df[col].min())

        df.loc[ind, col].plot(
            ax = ax, 
            **kwargs
        )
        if highlight_period is not None:
            highlight_start, highlight_end = highlight_period
            if not isinstance(highlight_start, pd.Timestamp):
                highlight_start = pd.to_datetime(highlight_start, format = '%Y-%m-%d')
            if not isinstance(highlight_end, pd.Timestamp):
                highlight_end = pd.to_datetime(highlight_end, format = '%Y-%m-%d')
            ax.axvspan(
                highlight_start, highlight_end, 
                color = 'red', alpha = 0.2
            )
        return ax

    _start = '2019-01-01'
    _end = '2019-01-30'
    _highlight = (fire_warra.start_date, fire_warra.end_date)

    fig, axs = plt.subplots(4,1, figsize = (15, 10), sharex = True)
    vars = {
        'WD': {
            'axes_ind': 0,
            'colour': 'mediumturquoise',
            'title': 'Wind Direction (WD)'
        },
        'WS': {
            'axes_ind': 1,
            'colour': 'royalblue',
            'title': 'Wind Speed (WS)'
        },
        'CO2_F_MDS': {
            'axes_ind': 2,
            'colour': 'lightcoral',
            'title': 'CO2 Flux (CO2_F_MDS)'
        },
        'H_F_MDS': {
            'axes_ind': 3,
            'colour': 'goldenrod',
            'title': 'Heat Flux',
            'label': 'Sensible Heat Flux (H_F_MDS)'
        },
        'G_F_MDS': {
            'axes_ind': 3,
            'colour': 'darkkhaki',
            'title': 'Heat Flux',
            'label': 'Ground Heat Flux (G_F_MDS)'
        }
    }
    last_ax_ind = None
    for var in vars.keys():
        if last_ax_ind is not None and vars.get(var).get('axes_ind') == last_ax_ind:
            _highlight = None
        else:
            _highlight = (fire_warra.start_date, fire_warra.end_date)
        plot_site_data(
            axs[vars.get(var).get('axes_ind')], var, df = fire_warra.df, start_date = _start, end_date = _end,
            highlight_period = _highlight,
            linewidth = 0.5, color = vars.get(var).get('colour', 'black'),
            title = f'{vars.get(var).get("title", var)}',
            label = vars.get(var).get('label', var),
            clip_min_vals = True
        )
        last_ax_ind = vars.get(var).get('axes_ind', last_ax_ind)
        axs[vars.get(var).get('axes_ind')].axvline('2019-01-28-16:00:00', linestyle = '--', color = 'grey', linewidth = 0.5)


    axs[3].legend(loc = 'best', fontsize = 10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Identifying fires
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Signals of fire measurement:
    - Power loss - some variables suddently hit 0/a flagged value (e.g. -9999)
        - Not all variables are lost. This is worth investigating - is this just because they can be extrapolated easily?
    - CO2 spike?
        - Seen in the Warra data, would be useful to know if there are other variables that spike.
    """)
    return


@app.cell
def _(df_warra):
    flag_cols = [col for col in df_warra if '_QC' in col]
    flag_cols += [col for col in df_warra if 'METHOD' in col]
    flag_cols += ['NIGHT']

    time_cols = ['TIMESTAMP_END']

    var_cols = df_warra.drop(columns = flag_cols + time_cols).columns

    df_warra[var_cols]
    return


@app.cell
def _(pd):
    df_fluxnet_vars = pd.read_csv('data/fluxnet_variables.csv').dropna().rename(columns = {
        'Variable': 'variable',
        'Units': 'units',
        'Description': 'description'
    }).set_index('variable', drop = True)

    df_fluxnet_vars
    return (df_fluxnet_vars,)


@app.cell
def _(pd):
    qualifiers = pd.DataFrame({
        'qualifier': ['_PI', '_QC', '_F', '_UI'],
        'description': ['Provided by PI/tower team', 'Quality Control Flag', 'Gapfilled variable', 'Instrument units'],
    })
    qualifiers
    return


@app.cell
def _(Literal, df_fluxnet_vars, np, pd, plt):
    def plot_vars(
        df: pd.DataFrame,
        variables: list[str],
        *,
        qualifiers: list[str]|None = None,
        start_date: str|pd.Timestamp|None = None, end_date: str|pd.Timestamp|None = None,
        highlight_period: tuple[str|pd.Timestamp, str|pd.Timestamp]|None = None,
        highlight_kwds: dict|None = None,
        min_vals_treatment: Literal['drop', 'none'] = 'drop',
        min_vals_highlight_kwds: dict|None = None,
        axvline_time: str|None = None,
        axvline_kwds: dict|None = None,
        figsize: tuple[int, int]|None = None, 
        sharex: bool = True, 
        **plot_kwargs
    ):
        '''
        Plot specified variables from a DataFrame with options for highlighting periods and handling minimum values.

        Parameters
        ----------
        df : pd.DataFrame
                DataFrame containing the data to plot.
        variables : list[str]
                List of variable names to plot. Any column starting with the variable name
                and including the qualifiers specified in `qualifiers` will be included in the plot.
        qualifiers : list[str] | None, optional
                List of qualifier columns corresponding to the variables (default is None). To include the base variable,
                include an empty string (''). Passing None will include all qualifiers.
        start_date : str | pd.Timestamp | None, optional
                Start date for the plot in 'YYYY-MM-DD' format (default is None - plots all data).
        end_date : str | pd.Timestamp | None, optional
                End date for the plot in 'YYYY-MM-DD' format (default is None - plots all data).
        highlight_period : tuple[str | pd.Timestamp, str | pd.Timestamp] | None, optional
                Tuple containing the start and end dates of the period to highlight in the plot (default is None - no highlighting).
        highlight_kwds : dict, optional
                Additional keyword arguments for highlighting (default is None).
        min_vals_treatment : {'drop', 'none'}, optional
                How to handle minimum values: 'drop' to exclude them (replace with NaN), or 'none' to leave them as is (default is 'drop').
        figsize : tuple[int, int]|None, optional
                Size of the figure. Defaults to (7, 2*len(variable_groups)) if not specified.
        sharex : bool, optional
                Whether to share the x-axis across subplots (default is True).
        **plot_kwargs : dict
                Additional keyword arguments to pass to df.plot().

        Returns
        -------
        fig : matplotlib.figure.Figure
                The figure object containing the plots.
        axs : numpy.ndarray
                Array of axes objects for the subplots.
        '''
        if len(variables) == 0:
            return None, None

        variable_groups = {var for var in variables if any([col == var or col.startswith(var + '_') for col in df.columns])}

        if min_vals_treatment.lower() == 'drop':
            replace_dict = {-9999: np.nan, -9999.0: np.nan}
        else:
            replace_dict = {}

        index_filter = df.index.notna()
        if start_date is not None:
            if not isinstance(start_date, pd.Timestamp):
                start_date = pd.to_datetime(start_date, format = '%Y-%m-%d')
            index_filter &= (df.index >= start_date)
        if end_date is not None:
            if not isinstance(end_date, pd.Timestamp):
                end_date = pd.to_datetime(end_date, format = '%Y-%m-%d')
            index_filter &= (df.index <= end_date)

        df_wkg = df.loc[index_filter].replace(replace_dict).copy()

        group_cols = {}
        for i, group in enumerate(variable_groups):
            if qualifiers:
                if qualifiers == ['']:
                    columns = [col for col in df.columns if col == group]
                else:
                    columns = [col for col in df.columns if (col == group or col.startswith(group + '_')) and any([qualifier in col for qualifier in qualifiers])]
            else:
                columns = [col for col in df.columns if col == group or col.startswith(group + '_')]
            if not columns:
                continue
            group_cols[group] = columns

        if len(group_cols) == 0:
            return None, None

        fig, axs = plt.subplots(
            len(group_cols), 1, 
            figsize = (7, 2*len(group_cols)) if not figsize else figsize, 
            sharex=sharex
        )
        for i, (group, columns) in enumerate(group_cols.items()):
            df_wkg[columns].plot(
                **plot_kwargs,
                title = f'{group}: {df_fluxnet_vars.loc[group, "description"]} - {len(columns)} col{"s" if len(columns) != 1 else ""}', 
                ax = axs[i]
            )
            if highlight_period:
                if not highlight_kwds: 
                    highlight_kwds = {}
                axs[i].axvspan(*highlight_period, **highlight_kwds)
            if axvline_time:
                if not axvline_kwds:
                    axvline_kwds = {}
                axs[i].axvline(axvline_time, **axvline_kwds)

        return fig, axs


    return (plot_vars,)


@app.cell
def _(df_fluxnet_vars, df_warra, fire_warra, mo, pd, plot_vars):
    figs = []
    for qual in ['', '_F', '_QC', '_PI']:
        _fig, _ = plot_vars(
            df_warra, df_fluxnet_vars.drop(index = ['TIMESTAMP', 'TIME']).index, 
            qualifiers = [qual],
            start_date = pd.to_datetime('2018-12-15'), end_date = pd.to_datetime('2019-02-28'),
            legend = False, 
            min_vals_treatment = 'drop',
            linewidth = 0.5,
            highlight_period = (fire_warra.start_date, fire_warra.end_date),
            highlight_kwds = {'color': 'red', 'alpha': 0.2},
            axvline_time=pd.to_datetime('2019-01-28-16:00:00'), axvline_kwds = {'linestyle': '--', 'color': 'grey', 'linewidth': 0.5}
        )

        figs.append(
            mo.vstack(
                [mo.md(f'**Qualifier: {qual if qual else "Base"}**'), _fig],
                align = 'center'
            )
        )
    mo.hstack(figs)
    return


@app.cell
def _(df_warra_corr, df_warra_info):
    df_warra_info.loc[df_warra_corr.iloc[:40].index]
    return


@app.cell
def _(df_warra, drop_cols, plt):
    windows = [3, 6, 12, 24, 48]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    _fig, _ax = plt.subplots(figsize = (len(windows) + 2, 7), sharex = True, sharey = True)
    for i, window in enumerate(windows):
        df_std = df_warra.drop(columns = ['TIMESTAMP_END']).rolling(window = window).std()
        df_std['fire_flag'] = df_warra['fire_flag']
        if i == 0:
            _ser_corr = df_std.corr()['fire_flag'].sort_values(ascending=False).drop(drop_cols, errors='ignore').iloc[:40]
            _ser_corr.plot.barh(
                ax = _ax, 
                color = colors[i],
                alpha = 0.25
            )
            corr_index = _ser_corr.index
            del _ser_corr
        else:
            df_std.corr()['fire_flag'].loc[corr_index].plot.barh(
                ax = _ax, 
                color = colors[i],
                alpha = 0.25
            )

        #_axs[i].set_title(f'Rolling Window: {window} periods', fontsize = 12)
    _fig
    return


@app.cell
def _(df_warra):
    df_warra
    return


@app.cell
def _(df_warra, pd, plt):
    ind_has_power = df_warra.index.to_series() <= pd.to_datetime('2019-01-28-16:00:00')
    drop_cols = ['fire_flag', 'TIMESTAMP_END']

    df_warra_corr = df_warra.loc[ind_has_power].corr()['fire_flag'].drop(drop_cols).sort_values(ascending=False)

    df_warra_corr.iloc[:40].plot.bar(figsize = (20, 3), color = 'tab:orange')
    plt.title('Correlation of Variables with Fire Flag (Before Power Loss)', fontsize = 14)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 12)
    plt.box(False)
    plt.show()
    return df_warra_corr, drop_cols


@app.cell
def _(fire_warra):
    fire_warra.start_date, fire_warra.end_date
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Normal behaviour modelling?
    Siying has been doing this
    - Anomaly detection
    - [Changepoint detection/analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464762/)
        - If there are changes in the
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
