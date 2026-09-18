__all__ = ['CWEX_DIR', 'CWEX']

import numpy as np
import pandas as pd
import geopandas as gpd

from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar

import os
import pathlib
import warnings
from typing import Literal, Iterator

from eddy.util import DATA_DIR
from eddy.data.common import smallest_file, var_metadata, average_over_time, standardise_df

FLUX_DIR = DATA_DIR / 'flux'
NCAR_DIR = FLUX_DIR / 'NCAR'
CWEX_DIR = NCAR_DIR / 'CWEX'

_dirs = [FLUX_DIR, NCAR_DIR, CWEX_DIR]
for _dir in _dirs:
    if not _dir.exists():
        os.makedirs(_dir)


class CWEX:
    '''
    Object to handle and process data from the Crop Wind Energy Experiment (CWEX) 2011.

    Attributes
    ----------
    dir_path : pathlib.Path
        The directory containing the CWEX netCDF files.
    files : tuple of pathlib.Path
        The netCDF files in time order.
    file_start_times : np.ndarray
        The first timestamp covered by each file.
    file_end_times : np.ndarray
        The end of each file's coverage.
    file_duration : np.timedelta64
        The nominal length of one file.
    stations : tuple of str
        The station names.
    n_stations : int
        The number of stations.
    n_samples : int
        Number of samples per record. Note this only applies to variables with multiple samples (those taken by the sonic anemometers).
    variables : tuple of str
        The names of the data variables (excluding dimensions).
    start_time, end_time : np.datetime64
        Time bounds of the data.
    metadata : dict
        Dimensions, shape, dtype and attributes for all variables.
    time : np.ndarray
        The full time axis.
    towers : gpd.GeoDataFrame
        The four tower locations (latitude, longitude, altitude).
    turbines : gpd.GeoDataFrame
        The wind turbine locations.
    locations : gpd.GeoDataFrame
        Towers and turbines together, with a `type` column distinguishing them.
    plotting_crs : pyproj.CRS
        The CRS used for site maps - the site's UTM zone, so distances are true metres.

    Methods
    -------
    timeseries(var_name, ...)
        Read a timeseries variable from the files - either the full time series or a window.
    time_axis(...)
        The time values for a window.
    tke(...)
        Estimated turbulent kinetic energy from the sonic anemometer components.
    wind(...)
        Horizontal wind direction and speed for a measurement set.
    plot_windrose(station, ...)
        Plot a windrose for a given station.
    plot_tke(station, ...)
        Plot the turbulent kinetic energy for a given station.
    plot_site(time, ...)
        Map the towers and turbines, with a downwind arrow from each tower.
    downwind_components(direction, ...)
        Convert a meteorological wind direction into downwind map components.
    speed_colourmap(...)
        The discrete colourmap used for the wind roses.
    '''

    _FILE_GLOB = 'cwex_*.nc'
    _FILENAME_TIME_FORMAT = '%Y%m%d%H'
    _TURBINE_KML = 'cwex_turbines.kml'
    _DEFAULT_FILE_DURATION = np.timedelta64(4, 'h')

    def __init__(self, dir_path: str|pathlib.Path = CWEX_DIR):
        '''
        Create a CWEX object.

        Parameters
        ----------
        dir_path : str or pathlib.Path, optional
            The directory holding the CWEX netCDF files. Defaults to `CWEX_DIR`.
        '''
        self.dir_path = pathlib.Path(dir_path)
        if not self.dir_path.is_dir():
            raise FileNotFoundError(f"CWEX directory does not exist: {self.dir_path}")

        self.files = tuple(sorted(self.dir_path.glob(self._FILE_GLOB)))
        if not self.files:
            raise FileNotFoundError(f"No files matching '{self._FILE_GLOB}' found in {self.dir_path}.")

        self.file_start_times = np.array(
            [self._filename_start_time(f) for f in self.files], dtype='datetime64[ns]'
        )

        # nominal file length, from the most common spacing between files
        if len(self.file_start_times) > 1:
            gaps = np.diff(self.file_start_times)
            values, counts = np.unique(gaps, return_counts=True)
            self.file_duration = values[np.argmax(counts)]
        else:
            self.file_duration = self._DEFAULT_FILE_DURATION

        # a file runs until the next one starts; the last runs for one nominal duration
        self.file_end_times = np.append(
            self.file_start_times[1:], self.file_start_times[-1] + self.file_duration
        )

        with Dataset(smallest_file(self.dir_path, self._FILE_GLOB)) as ds:
            self.stations = tuple(
                b''.join(row.compressed()).decode().upper() for row in ds['station'][:]
            )
            self.n_stations = ds.dimensions['station'].size
            self.n_samples = ds.dimensions['sample'].size
            self.variables = tuple(v for v in ds.variables if v not in ds.dimensions)
            self._var_dims = {v: ds.variables[v].dimensions for v in ds.variables}

        self._time = None  # will store cached time values
        self._plotting_crs = None  # will store the cached site CRS

    @classmethod
    def from_dir(cls, dir_path: str|pathlib.Path) -> 'CWEX':
        '''
        Create a CWEX object from a directory of netCDF files.

        Parameters
        ----------
        dir_path : str or pathlib.Path
            The directory holding the CWEX netCDF files.

        Returns
        -------
        CWEX
        '''
        return cls(dir_path)

    def refresh(self) -> 'CWEX':
        '''Re-scan the directory and clear the cached time axis. Returns self.'''
        self.__init__(self.dir_path)
        return self

    def _filename_start_time(self, file_path: pathlib.Path) -> np.datetime64:
        '''The first timestamp covered by a file.'''
        date_str, hr_str = file_path.stem.split('_')[1], file_path.stem.split('_')[2]
        return pd.to_datetime(date_str + hr_str, format=self._FILENAME_TIME_FORMAT).to_datetime64()

    @property
    def n_files(self) -> int:
        '''The number of netCDF files in the directory.'''
        return len(self.files)

    @property
    def start_time(self) -> np.datetime64:
        '''Minimum measurement time in the dataset.'''
        return self.file_start_times[0]

    @property
    def end_time(self) -> np.datetime64:
        '''Maximum measurement time in the dataset.'''
        return self.file_end_times[-1]

    @property
    def metadata(self) -> dict:
        '''Dimensions, shape, dtype and attributes for every variable in the dataset.'''
        with Dataset(smallest_file(self.dir_path, self._FILE_GLOB)) as ds:
            return var_metadata(ds, var_name=None)

    @property
    def time(self) -> np.ndarray:
        '''
        The full time axis. Cached after first call to reduce number of file reads - use `refresh()` to remove from memory.
        '''
        if self._time is None:
            self._time = self.time_axis()
        return self._time

    @property
    def plotting_crs(self):
        '''
        The CRS used for site maps - the local UTM zone, so that distances, arrow lengths and the
        scale bar are all in true metres.

        Neither of the package-wide CRSs is suitable here. `BASE_CRS` (Web Mercator), which every
        `@standardise_df` loader reprojects to, overstates distance by a factor of 1.34 at this
        site's latitude. `PLOTTING_CRS` is geographic, in degrees, and exists for folium.

        Cached after the first call, and cleared by `refresh()`.
        '''
        if self._plotting_crs is None:
            self._plotting_crs = self.locations.estimate_utm_crs()
        return self._plotting_crs

    @plotting_crs.setter
    def plotting_crs(self, crs):
        self._plotting_crs = crs

    # --- station handling -------------------------------------------------------------------

    def _station_index(self, station: int|str) -> int|slice:
        '''
        Resolve a station name, 1-based index or 'all' to an index into the station dimension.

        Parameters
        ----------
        station : int or str
            A station name (e.g. 'NCAR1', case-insensitive), a 1-based index, or 'all'.

        Returns
        -------
        int or slice
        '''
        if isinstance(station, str):
            if station.lower() == 'all':
                return slice(None)
            if station.upper() not in self.stations:
                raise ValueError(f"Invalid station '{station}'. Must be one of {self.stations}, or 'all'.")
            return self.stations.index(station.upper())

        if isinstance(station, (int, np.integer)):
            if not 1 <= station <= self.n_stations:
                raise IndexError(f"Station index {station} is out of bounds - CWEX11 has {self.n_stations} stations (valid range: 1-{self.n_stations}).")
            return int(station) - 1  # station indices are 1-based

        raise TypeError(f"station must be an int or str, got {type(station)}.")

    def _station_label(self, station: int|str) -> str:
        '''The name of a single station, for titles and labels.'''
        index = self._station_index(station)
        if isinstance(index, slice):
            raise ValueError("A single station is required here, not 'all'.")
        return self.stations[index]

    def _sample_axis(
            self, var_name: str, *,
            sample: int|Literal['all'] = 'all',
            station: int|str = 'all'
    ) -> int|None:
        '''
        The position of the sample axis in the array `timeseries` would return, or None if there
        is no sample axis. Selecting a single sample or station drops that axis, so the position
        cannot be inferred from the number of dimensions alone.

        Parameters
        ----------
        var_name : str
            The variable that would be read.
        sample : int or 'all', optional
            The sample selection that would be passed to `timeseries`.
        station : int or str, optional
            The station selection that would be passed to `timeseries`.

        Returns
        -------
        int or None
        '''
        keeps_sample = isinstance(sample, str) and sample.lower() == 'all'
        keeps_station = isinstance(station, str) and station.lower() == 'all'

        axes = [
            dim for dim in self._var_dims[var_name]
            if dim == 'time'
            or (dim == 'sample' and keeps_sample)
            or (dim == 'station' and keeps_station)
        ]
        return axes.index('sample') if 'sample' in axes else None

    @staticmethod
    def _flatten_samples(values: np.ndarray, sample_axis: int) -> np.ndarray:
        '''
        Fold the sample axis into the time axis, so each sample becomes its own observation.

        Parameters
        ----------
        values : np.ndarray
            An array whose sample axis directly follows its time axis.
        sample_axis : int
            The position of the sample axis.

        Returns
        -------
        np.ndarray
            The array with its time and sample axes merged into one leading axis.
        '''
        if sample_axis != 1:
            raise ValueError(f"Expected the sample axis to follow the time axis, but found it at axis {sample_axis}.")
        return values.reshape(-1, *values.shape[2:])

    # --- reading ----------------------------------------------------------------------------

    def file_timeseries(
            self,
            ds: Dataset,
            var_name: str,
            *,
            sample: int|Literal['all'] = 'all',
            station: int|str = 'all',
            return_type: Literal['masked_array', 'data_with_nan', 'mask'] = 'data_with_nan'
    ) -> np.ndarray:
        '''
        Read a variable from an open netCDF file.

        Parameters
        ----------
        ds : netCDF4.Dataset
            The open dataset to read from.
        var_name : str
            The variable to extract. Must have a 'time' dimension.
        sample : int or 'all', optional
            The sample index to extract, or 'all' (default) for every sample.
        station : int or str, optional
            A station name, a 1-based index, or 'all' (default).
        return_type : {'masked_array', 'data_with_nan', 'mask'}, optional
            Whether to return the masked array, the data with missing values as NaN (default),
            or the missing-value mask itself.

        Returns
        -------
        np.ndarray
        '''
        if var_name in ds.dimensions:
            raise ValueError(f"Variable name '{var_name}' is a dimension in the dataset, not a variable.")
        if var_name not in ds.variables:
            raise ValueError(f"Variable name '{var_name}' not found in dataset variables: {list(ds.variables.keys() - ds.dimensions.keys())}.")
        if 'time' not in ds.variables[var_name].dimensions:
            raise ValueError(f"Cannot make a timeseries for variable '{var_name}' because it does not have a 'time' dimension. Its dimensions are: {ds.variables[var_name].dimensions}.")

        return_type = return_type.lower()
        if return_type not in ['masked_array', 'data_with_nan', 'mask']:
            raise ValueError(f"Invalid return_type '{return_type}'. Must be one of 'masked_array', 'data_with_nan', or 'mask'.")

        station_index = self._station_index(station)

        if isinstance(sample, str):
            if sample.lower() != 'all':
                raise ValueError(f"Invalid sample '{sample}'. Must be an int or 'all'.")
            sample = slice(None)
        elif not 0 <= sample < ds.dimensions['sample'].size:
            raise IndexError(f"Sample index {sample} is out of bounds for dimension 'sample' with size {ds.dimensions['sample'].size}.")

        # build the indexer in the variable's own dimension order
        indexers = []
        for dim in ds.variables[var_name].dimensions:
            if dim == 'time':
                indexers.append(slice(None))
            elif dim == 'station':
                indexers.append(station_index)
            elif dim == 'sample':
                indexers.append(sample)

        values = ds[var_name][tuple(indexers)]

        if return_type == 'masked_array':
            return values
        if return_type == 'mask':
            # getmaskarray, not .mask, which collapses to a scalar False when nothing is masked
            return np.ma.getmaskarray(values)
        return values.filled(np.nan)

    def _relevant_files(
            self, t_min: np.datetime64|None, t_max: np.datetime64|None
    ) -> Iterator[tuple[pathlib.Path, np.datetime64, bool]]:
        '''
        Walk the files overlapping a time window.

        Yields
        ------
        tuple of (pathlib.Path, np.datetime64, bool)
            Each relevant file, its start time, and whether it needs slicing because the window
            cuts through it.
        '''
        for file_path, file_start, file_end in zip(self.files, self.file_start_times, self.file_end_times):
            if t_min is not None and t_min > file_end:
                continue  # window starts after this file ends - not there yet
            if t_max is not None and t_max < file_start:
                break  # window ended before this file starts - done
            whole_file = ((t_min is None or t_min <= file_start)
                          and (t_max is None or file_end <= t_max))
            yield file_path, file_start, not whole_file

    @staticmethod
    def _file_times(ds: Dataset, file_start: np.datetime64) -> np.ndarray:
        '''Reconstruct a file's timestamps from its offsets. [ms] because some are half-seconds.'''
        return file_start + (ds['time'][:] * 1000).astype('timedelta64[ms]')

    @staticmethod
    def _window_mask(times: np.ndarray, t_min: np.datetime64|None, t_max: np.datetime64|None) -> np.ndarray:
        '''A boolean mask selecting the timestamps inside a window.'''
        mask = np.ones(times.shape, dtype=bool)
        if t_min is not None:
            mask &= times >= t_min
        if t_max is not None:
            mask &= times <= t_max
        return mask

    def _iter_chunks(
            self,
            var_name: str|None,
            *,
            sample: int|Literal['all'] = 'all',
            station: int|str = 'all',
            return_type: Literal['masked_array', 'data_with_nan', 'mask'] = 'data_with_nan',
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            need_time: bool = False
    ) -> Iterator[tuple[np.ndarray|None, np.ndarray|None]]:
        '''Walk the relevant files, yielding one (time, values) chunk per file. `var_name=None` gives a time-only pass.'''
        for file_path, file_start, needs_slicing in self._relevant_files(t_min, t_max):
            with Dataset(file_path) as ds:
                times = self._file_times(ds, file_start) if (need_time or needs_slicing) else None
                values = None
                if var_name is not None:
                    values = self.file_timeseries(
                        ds, var_name, sample=sample, station=station, return_type=return_type
                    )

                if needs_slicing:
                    mask = self._window_mask(times, t_min, t_max)
                    times = times[mask]
                    if values is not None:
                        values = values[mask]

                yield (times if need_time else None), values

    @staticmethod
    def _concat(chunks: list, return_type: str) -> np.ndarray:
        '''Join per-file chunks. Masked arrays need np.ma.concatenate to keep the mask.'''
        if not chunks:
            return np.array([], dtype=np.float32)
        concatenate = np.ma.concatenate if return_type == 'masked_array' else np.concatenate
        return concatenate(chunks, axis=0)

    def timeseries(
            self,
            var_name: str,
            *,
            sample: int|Literal['all'] = 'all',
            station: int|str = 'all',
            return_type: Literal['masked_array', 'data_with_nan', 'mask'] = 'data_with_nan',
            return_time: bool = False,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = None
    ) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
        '''
        Read one a time series from all relevant files. Can be optionally indexed by sample and station, and sliced to a time window. 
        Can also average over fixed-length time bins.

        Parameters
        ----------
        var_name : str
            The variable to extract.
        sample : int or 'all', optional
            The sample index to extract, or 'all' (default).
        station : int or str, optional
            A station name, a 1-based index, or 'all' (default).
        return_type : {'masked_array', 'data_with_nan', 'mask'}, optional
            Whether to return the masked array, the data with missing values as NaN (default),
            or the missing-value mask itself.
        return_time : bool, optional
            If True, also return the timestamps. Defaults to False.
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include. If None (default), no bound is applied.
        time_averaging : str or np.timedelta64 or None, optional
            If given, average over fixed-length time bins, e.g. '30m'. Bins are anchored on the
            Unix epoch, so '30m' bins start on the hour and half hour. Only the time axis is
            reduced; sample and station axes are kept. Not supported with return_type='mask'.

        Returns
        -------
        np.ndarray, or (np.ndarray, np.ndarray)
            The values, preceded by the timestamps if `return_time` is True.
        '''
        if t_min is not None and not isinstance(t_min, np.datetime64):
            raise TypeError(f"t_min must be a numpy.datetime64 object or None, got {type(t_min)}.")
        if t_max is not None and not isinstance(t_max, np.datetime64):
            raise TypeError(f"t_max must be a numpy.datetime64 object or None, got {type(t_max)}.")
        if t_min is not None and t_max is not None and t_min >= t_max:
            raise ValueError(f"t_min ({t_min}) must be less than t_max ({t_max}).")
        if time_averaging is not None and return_type.lower() == 'mask':
            raise ValueError("time_averaging is not supported with return_type='mask'.")

        # times are needed to return them, to slice a partial file, or to build averaging bins
        need_time = return_time or time_averaging is not None

        time_chunks, var_chunks = [], []
        for times, values in self._iter_chunks(
            var_name, sample=sample, station=station, return_type=return_type,
            t_min=t_min, t_max=t_max, need_time=need_time
        ):
            if need_time:
                time_chunks.append(times)
            var_chunks.append(values)

        out_var = self._concat(var_chunks, return_type.lower())
        out_time = np.ma.getdata(np.concatenate(time_chunks, axis=0)) if need_time and time_chunks else None

        if time_averaging is not None and out_var.size:
            out_time, out_var = average_over_time(out_time, out_var, time_averaging)

        if return_time:
            return out_time, out_var
        return out_var

    def time_axis(
            self, *,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = None
    ) -> np.ndarray:
        '''
        The time axis for a window, without reading a data variable.

        Parameters
        ----------
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include.
        time_averaging : str or np.timedelta64 or None, optional
            If given, return the start of each averaging bin rather than every timestamp.

        Returns
        -------
        np.ndarray
        '''
        chunks = [times for times, _ in self._iter_chunks(None, t_min=t_min, t_max=t_max, need_time=True)]
        if not chunks:
            return np.array([], dtype='datetime64[ns]')

        times = np.ma.getdata(np.concatenate(chunks, axis=0))
        if time_averaging is not None:
            times, _ = average_over_time(times, times.astype('int64').astype(np.float64), time_averaging)
        return times

    # --- derived quantities -----------------------------------------------------------------

    def tke(
            self, *,
            station: int|str = 'all',
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            return_time: bool = False
    ) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
        '''
        Timeseries of turbulent kinetic energy, 0.5 * (var(u) + var(v) + var(w)).
        Variances are taken across the per-second samples, so each record is 1s apart.

        Parameters
        ----------
        station : int or str, optional
            A station name, a 1-based index, or 'all' (default).
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include.
        return_time : bool, optional
            If True, also return the timestamps. Defaults to False.

        Returns
        -------
        np.ndarray, or (np.ndarray, np.ndarray)
            TKE in m^2 s^-2, preceded by the timestamps if `return_time` is True.
        '''
        time_chunks, tke_chunks = [], []

        for file_path, file_start, needs_slicing in self._relevant_files(t_min, t_max):
            with Dataset(file_path) as ds:
                times = self._file_times(ds, file_start)
                total = None
                for var_name in ('u_4_5m', 'v_4_5m', 'w_4_5m'):
                    component = self.file_timeseries(
                        ds, var_name, sample='all', station=station, return_type='data_with_nan'
                    )
                    variance = np.nanvar(component, axis=1, ddof=1)  # across samples
                    total = variance if total is None else total + variance

                if needs_slicing:
                    mask = self._window_mask(times, t_min, t_max)
                    times, total = times[mask], total[mask]

                time_chunks.append(times)
                tke_chunks.append(0.5 * total)

        out_tke = np.concatenate(tke_chunks, axis=0) if tke_chunks else np.array([], dtype=np.float32)

        if return_time:
            out_time = np.ma.getdata(np.concatenate(time_chunks, axis=0)) if time_chunks else np.array([], dtype='datetime64[ns]')
            return out_time, out_tke
        return out_tke

    def wind(
            self, *,
            station: int|str = 'all',
            instrument: Literal['sonic-anemometer', 'weather-vane'] = 'sonic',
            samples: Literal['mean', 'all'] = 'mean',
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m',
            units: Literal['deg', 'rad'] = 'deg',
            return_time: bool = False
    ) -> tuple[np.ndarray, ...]:
        '''
        Wind direction and speed from a pair of velocity components.

        Direction follows the meteorological convention (the direction the wind blows from, 
        clockwise from true north).

        Parameters
        ----------
        station : int or str, optional
            A station name, a 1-based index, or 'all' (default).
        instrument : {'sonic-anemometer', 'weather-vane'}, optional
            Which instrument to use. The sonic anemometers are at 4.5 m, the weather vanes at 10 m. 
            Defaults to 'sonic-anemometer'. Will accept 'sonic' or 'vane' as shorthand.
        samples : {'mean', 'all'}, optional
            How to treat the sample axis of the sonic anemometers. 'mean' (default) vector-averages
            the samples of each record, giving one wind vector per record. 'all' keeps every sample
            as its own observation, flattening them into the time axis, so a record contributes
            `n_samples` observations rather than one. The weather vanes have no sample axis, so
            this has no effect on them.
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include.
        time_averaging : str or np.timedelta64 or None, optional
            Averaging bin length, e.g. '30m' (the default). None disables averaging. Applied 
            before the samples are handled, so with samples='all' the samples that survive are 
            the per-sample bin means.
        units : {'deg', 'rad'}, optional
            Units for the returned direction. Defaults to degrees, on 0-360.
        return_time : bool, optional
            If True, also return the timestamps. Defaults to False.

        Returns
        -------
        tuple of np.ndarray
            (direction, speed), preceded by the timestamps if `return_time` is True. With 
            samples='all' the timestamps repeat, as the samples within a record do not carry 
            individual timestamps.
        '''
        units = units.lower()
        if units not in ['deg', 'rad']:
            raise ValueError(f"Invalid units '{units}'. Must be one of 'deg' or 'rad'.")

        samples = samples.lower()
        if samples not in ['mean', 'all']:
            raise ValueError(f"Invalid samples '{samples}'. Must be one of 'mean' or 'all'.")

        instrument = instrument.lower().replace('_', '-').replace(' ', '-')
        if instrument == 'sonic-anemometer' or instrument == 'sonic':
            u_var = 'u_4_5m'
            v_var = 'v_4_5m'
        elif instrument == 'weather-vane' or instrument == 'vane':
            u_var = 'U_10m'
            v_var = 'V_10m'
        else:
            raise ValueError(f"Invalid instrument '{instrument}'. Must be one of 'sonic-anemometer', 'weather-vane', 'sonic', or 'vane'.")

        read = dict(sample='all', station=station, return_type='data_with_nan',
                    t_min=t_min, t_max=t_max, time_averaging=time_averaging)
        time, u = self.timeseries(u_var, return_time=True, **read)
        v = self.timeseries(v_var, **read)

        # 4.5 m anemometers have a sample axis, but 10 m vanes do not. Selecting a single station
        # drops an axis, so find the sample axis by name rather than by counting dimensions.
        sample_axis = self._sample_axis(u_var, sample='all', station=station)
        if sample_axis is not None:
            if samples == 'mean':
                u = np.nanmean(u, axis=sample_axis)
                v = np.nanmean(v, axis=sample_axis)
            else:
                # flatten the samples into the time axis, so each is its own observation
                u = self._flatten_samples(u, sample_axis)
                v = self._flatten_samples(v, sample_axis)
                time = np.repeat(time, self.n_samples)

        direction = np.arctan2(-u, -v)  # negative to get direction the wind comes from
        speed = np.hypot(u, v)

        if units == 'deg':
            direction = np.degrees(direction) % 360

        if return_time:
            return time, direction, speed
        return direction, speed

    @staticmethod
    def downwind_components(
            direction: np.ndarray, speed: np.ndarray|None = None, *,
            units: Literal['deg', 'rad'] = 'deg'
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Convert a meteorological wind direction into map components pointing downwind.

        `wind()` reports the direction the wind blows *from*, so the downwind vector is the
        negated pair - which recovers the sign of the original u and v components.

        Parameters
        ----------
        direction : np.ndarray
            Wind direction, as returned by `wind()`.
        speed : np.ndarray or None, optional
            If given, the components are scaled by the speed rather than left as unit vectors.
        units : {'deg', 'rad'}, optional
            The units of `direction`. Defaults to degrees.

        Returns
        -------
        np.ndarray, np.ndarray
            The eastward and northward components of the downwind direction.
        '''
        units = units.lower()
        if units not in ['deg', 'rad']:
            raise ValueError(f"Invalid units '{units}'. Must be one of 'deg' or 'rad'.")

        theta = np.radians(direction) if units == 'deg' else direction
        east, north = -np.sin(theta), -np.cos(theta)

        if speed is not None:
            east, north = east * speed, north * speed
        return east, north

    @staticmethod
    def _vector_mean(
            direction: np.ndarray, speed: np.ndarray, *, axis: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Reduce direction and speed along an axis, averaging the wind as a vector.

        Directions are averaged through their components, never as bearings, which would wrap
        incorrectly through north. The speed is a scalar mean - the conventional mean wind speed.

        Parameters
        ----------
        direction : np.ndarray
            Wind direction in degrees.
        speed : np.ndarray
            Wind speed.
        axis : int, optional
            The axis to reduce. Defaults to 0, the time axis.

        Returns
        -------
        np.ndarray, np.ndarray
            The mean direction in degrees, and the mean speed.
        '''
        east, north = CWEX.downwind_components(direction, speed)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slices are expected
            east, north = np.nanmean(east, axis=axis), np.nanmean(north, axis=axis)
            speed = np.nanmean(speed, axis=axis)

        return np.degrees(np.arctan2(-east, -north)) % 360, speed

    def _resolve_window(
            self,
            time: np.datetime64|str|None = None, *,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m'
    ) -> tuple[np.datetime64, np.datetime64, str]:
        '''
        Work out which time window to read, from either a timestamp or an explicit window.

        Parameters
        ----------
        time : np.datetime64 or str or None, optional
            A timestamp. The window becomes the averaging bin containing it, or the record
            nearest it when `time_averaging` is None.
        t_min, t_max : np.datetime64 or None, optional
            An explicit window. Cannot be combined with `time`; a missing bound falls back to the
            start or end of the campaign.
        time_averaging : str or np.timedelta64 or None, optional
            The averaging period, used to size the bin around `time`.

        Returns
        -------
        np.datetime64, np.datetime64, str
            The window bounds, and a label describing it for a plot title.
        '''
        has_window = t_min is not None or t_max is not None

        if time is not None and has_window:
            raise ValueError("Give either `time` or `t_min`/`t_max`, not both.")
        if time is None and not has_window:
            raise ValueError("Give either `time`, for a single averaging bin, or `t_min`/`t_max`, for a window.")

        if has_window:
            t_min = self.start_time if t_min is None else np.datetime64(t_min)
            t_max = self.end_time if t_max is None else np.datetime64(t_max)
            return t_min, t_max, f'{str(t_min)[:16]} to {str(t_max)[:16]}'

        time = np.datetime64(time, 'ns')
        if time_averaging is None:
            # records sit on half-second offsets, so an exact timestamp matches nothing - widen
            # to the records either side of the instant asked for
            margin = np.timedelta64(1, 's')
            return time - margin, time + margin, str(time)[:19]

        period = np.timedelta64(pd.Timedelta(time_averaging)).astype('timedelta64[ns]')
        period_ns = period.astype('int64')
        start = ((time.astype('int64') // period_ns) * period_ns).astype('datetime64[ns]')
        # one ns short of the next bin, as the window mask includes t_max
        return start, start + period - np.timedelta64(1, 'ns'), f'{str(start)[:16]} + {time_averaging}'

    def _station_snapshot(
            self, var_name: str, *,
            t_min: np.datetime64, t_max: np.datetime64,
            time_averaging: str|np.timedelta64|None = '30m'
    ) -> np.ndarray:
        '''
        Reduce a variable to a single value per station over a time window.

        Parameters
        ----------
        var_name : str
            The variable to read.
        t_min, t_max : np.datetime64
            The window to average over.
        time_averaging : str or np.timedelta64 or None, optional
            Passed through to `timeseries`.

        Returns
        -------
        np.ndarray
            One value per station, in `self.stations` order. Stations with no valid data are NaN.
        '''
        values = self.timeseries(
            var_name, sample='all', station='all', return_type='data_with_nan',
            t_min=t_min, t_max=t_max, time_averaging=time_averaging
        )

        sample_axis = self._sample_axis(var_name, sample='all', station='all')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slices are expected
            if sample_axis is not None:
                values = np.nanmean(values, axis=sample_axis)
            values = np.nanmean(values, axis=0)  # collapse the time axis

        return np.atleast_1d(values)

    def _variable_label(self, var_name: str) -> str:
        '''
        A display label for a variable, as `long_name / units` where those attributes exist.

        Parameters
        ----------
        var_name : str
            The variable to label.

        Returns
        -------
        str
        '''
        attributes = self.metadata.get(var_name, {}).get('attributes', {})
        label = attributes.get('long_name', var_name)
        units = attributes.get('units')
        return f'{label} / {units}' if units else label

    # --- locations --------------------------------------------------------------------------

    @property
    @standardise_df
    def towers(self) -> gpd.GeoDataFrame:
        '''
        The tower locations, from latitude/longitude/altitude variables stored in the smallest file.

        Returns
        -------
        gpd.GeoDataFrame
            Indexed by station name, with an `altitude` column in metres.
        '''
        with Dataset(smallest_file(self.dir_path, self._FILE_GLOB)) as ds:
            lat = np.ma.getdata(ds['latitude'][:])
            lon = np.ma.getdata(ds['longitude'][:])
            alt = np.ma.getdata(ds['altitude'][:])

        gdf = gpd.GeoDataFrame(
            {'id': [s.lower() for s in self.stations], 'altitude': alt},
            geometry=gpd.points_from_xy(lon, lat, crs='EPSG:4326')
        ).set_index('id')
        gdf['type'] = 'eddy tower'
        gdf.index = gdf.index.str.upper()
        return gdf

    @property
    @standardise_df
    def turbines(self) -> gpd.GeoDataFrame:
        '''
        The wind turbine locations, from `cwex_turbines.kml`.

        Returns
        -------
        gpd.GeoDataFrame
            Indexed by turbine id.
        '''
        kml_path = self.dir_path / self._TURBINE_KML
        if not kml_path.exists():
            raise FileNotFoundError(f"No turbine locations found at {kml_path}.")

        gdf = gpd.read_file(kml_path)[['Name', 'geometry']].rename(columns={'Name': 'id'})
        gdf['id'] = gdf['id'].str.upper()
        gdf = gdf.set_index('id').sort_index()
        gdf['type'] = 'wind turbine'
        return gdf


    @property
    @standardise_df
    def locations(self) -> gpd.GeoDataFrame:
        '''
        The towers and turbines together.

        Returns
        -------
        gpd.GeoDataFrame
            Indexed by id, with a `type` category column of 'eddy tower' or 'wind turbine'.
        '''
        gdf = pd.concat([self.towers, self.turbines]).sort_index()
        gdf['type'] = gdf['type'].astype('category')
        return gdf

    # --- plotting ---------------------------------------------------------------------------

    @staticmethod
    def speed_colourmap(
            speed_bins: np.ndarray|None = None, cmap: str = 'turbo'
    ) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        '''
        The discrete colourmap and norm for wind-speed bins.

        Both `plot_windrose` and any shared colourbar should call this, so the bars and the bar
        cannot drift apart.

        Parameters
        ----------
        speed_bins : np.ndarray or None, optional
            The bin edges in m/s. Defaults to 2 m/s bins from 0 to 18.
        cmap : str, optional
            The name of the continuous colourmap to sample. Defaults to 'turbo'.

        Returns
        -------
        matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm
        '''
        if speed_bins is None:
            speed_bins = np.arange(0, 20, 2)

        speed_bins = np.asarray(speed_bins)
        n_bins = len(speed_bins) - 1
        discrete = mcolors.ListedColormap(plt.get_cmap(cmap)(np.linspace(0, 1, n_bins)))
        return discrete, mcolors.BoundaryNorm(speed_bins, discrete.N)

    def plot_windrose(
            self,
            station: int|str,
            *,
            instrument: Literal['sonic-anemometer', 'weather-vane'] = 'sonic-anemometer',
            samples: Literal['mean', 'all'] = 'mean',
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m',
            n_sectors: int = 36,
            speed_bins: np.ndarray|None = None,
            cmap: str = 'turbo',
            r_max: float|None = 8,
            r_ticks: list|None = None,
            title: str|None = None,
            ax: plt.Axes|None = None,
            return_fig: bool = False
    ) -> plt.Axes|tuple[plt.Figure, plt.Axes]:
        '''
        Draw one station's wind rose: a polar histogram of direction, stacked by speed bin.

        Plots a single station onto a single axes - composing a multi-panel figure is the
        caller's job.

        Parameters
        ----------
        station : int or str
            The station to plot, as a name or 1-based index. 'all' is not accepted.
        instrument : {'sonic-anemometer', 'weather-vane'}, optional
            Which instrument to use. The sonic anemometers are at 4.5 m, the weather vanes at 10 m. 
            Defaults to 'sonic-anemometer'. Will accept 'sonic' or 'vane' as shorthand.
        samples : {'mean', 'all'}, optional
            How to treat the sonic anemometers' sample axis. 'mean' (default) counts one wind
            vector per record; 'all' counts every sample separately, so each record contributes
            `n_samples` observations to the histogram. Frequencies remain a percentage of the
            observations counted, so the two are directly comparable. No effect on the vanes.
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include.
        time_averaging : str or np.timedelta64 or None, optional
            Averaging bin length, e.g. '30m' (the default).
        n_sectors : int, optional
            The number of direction sectors. Defaults to 36 (10 degrees each).
        speed_bins : np.ndarray or None, optional
            Speed bin edges in m/s. Defaults to 2 m/s bins from 0 to 18. Records outside the
            outermost edge are excluded from the bars.
        cmap : str, optional
            The colourmap sampled for the speed bins. Defaults to 'turbo'.
        r_max : float or None, optional
            The radial limit, as a percentage. Defaults to 8; None lets matplotlib choose.
        r_ticks : list or None, optional
            Radial tick positions, as percentages. Defaults to every 2% up to `r_max`.
        title : str or None, optional
            The axes title. Defaults to the station name.
        ax : matplotlib.axes.Axes or None, optional
            A polar axes to draw into. If None, one is created.
        return_fig : bool, optional
            If True, return the Figure and Axes. Defaults to False.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
        '''
        label = self._station_label(station)

        if speed_bins is None:
            speed_bins = np.arange(0, 20, 2)
        speed_bins = np.asarray(speed_bins)
        speed_cmap, _ = self.speed_colourmap(speed_bins, cmap)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # clockwise - meteorological convention

        direction, speed = self.wind(
            station=station, instrument=instrument, samples=samples,
            t_min=t_min, t_max=t_max, time_averaging=time_averaging, units='deg'
        )

        valid = np.isfinite(direction) & np.isfinite(speed)
        direction, speed = direction[valid], speed[valid]

        if direction.size == 0:
            ax.set_title(f'{label} — no data', pad=15)
            return ax.figure if return_fig else ax

        sector_width = 360 / n_sectors
        dir_edges = np.linspace(-sector_width / 2, 360 - sector_width / 2, n_sectors + 1)
        dir_centres = np.radians(dir_edges[:-1] + sector_width / 2)

        # wrap the final half-sector back into the first, so north is centred on 0
        direction = np.where(direction >= 360 - sector_width / 2, direction - 360, direction)

        counts, _, _ = np.histogram2d(direction, speed, bins=[dir_edges, speed_bins])
        freq = 100 * counts / direction.size  # % of valid records

        bottom = np.zeros(n_sectors) + 0.25  # small hole in the middle, so the bars read clearly
        for j, colour in enumerate(speed_cmap.colors):
            ax.bar(
                dir_centres, freq[:, j],
                width=np.radians(sector_width) * 0.9,
                bottom=bottom,
                color=colour,
                label=f'{speed_bins[j]}-{speed_bins[j + 1]}'
            )
            bottom += freq[:, j]

        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.set_rlabel_position(280)
        ax.set_title(title if title is not None else label, pad=15)

        if r_max is not None:
            ax.set_ylim(0, r_max)
            if r_ticks is None:
                r_ticks = list(np.arange(0, r_max + 1, 2))
        if r_ticks is not None:
            ax.set_yticks(r_ticks)
            ax.set_yticklabels(
                ['' if t == 0 else f'{t:g}%' for t in r_ticks],
                verticalalignment='bottom', horizontalalignment='right'
            )

        ax.grid(color='black', linewidth=0.5, axis='y')
        ax.xaxis.grid(False)

        return (ax.figure, ax) if return_fig else ax

    def plot_tke(
            self,
            station: int|str,
            *,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            colour: str = 'tab:blue',
            gap_shading: bool = True,
            title: str|None = None,
            ax: plt.Axes|None = None,
            return_fig: bool = False
    ) -> plt.Axes|tuple[plt.Figure, plt.Axes]:
        '''
        Draw one station's TKE timeseries, shading the gaps where data is missing.

        Plots a single station onto a single axes - composing a multi-panel figure is the
        caller's job.

        Parameters
        ----------
        station : int or str
            The station to plot, as a name or 1-based index. 'all' is not accepted.
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include.
        colour : str, optional
            The marker colour. Defaults to 'tab:blue'.
        gap_shading : bool, optional
            If True (default), shade the periods with no data in grey.
        title : str or None, optional
            The axes title. Defaults to the station name.
        ax : matplotlib.axes.Axes or None, optional
            The axes to draw into. If None, one is created.
        return_fig : bool, optional
            If True, return the Figure and Axes. Defaults to False.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.figure.Figure
        '''
        label = self._station_label(station)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        time, tke = self.tke(
            station=station,
            t_min=t_min, t_max=t_max, return_time=True
        )

        ax.scatter(time, tke, marker='.', s=1, alpha=0.5, linewidth=0.25, color=colour)

        if gap_shading:
            ax.fill_between(
                time, 0, 1, where=np.isnan(tke),
                color='lightgrey', alpha=0.9, transform=ax.get_xaxis_transform()
            )

        ax.set_ylabel('TKE / m$^2$ s$^{-2}$')
        ax.set_title(title if title is not None else label)
        if time.size:
            ax.set_xlim(time[0], time[-1])

        return (ax.figure, ax) if return_fig else ax

    def plot_site(
            self,
            time: np.datetime64|str|None = None,
            *,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m',
            instrument: Literal['sonic-anemometer', 'weather-vane'] = 'weather-vane',
            colour_by: str|None = None,
            cmap: str = 'viridis',
            colour_limits: tuple[float, float]|None = None,
            colour_towers: str = 'tab:green',
            colour_missing: str = 'lightgrey',
            colour_turbines: str = 'tab:blue',
            speed_colours: bool = True,
            speed_bins: np.ndarray|None = None,
            speed_cmap: str = 'turbo',
            colour_arrows: str = 'black',
            arrow_length: float = 150,
            length_by_speed: bool = False,
            reference_speed: float = 8,
            annotate: bool = False,
            scale_bar: bool = True,
            colourbar: bool = True,
            legend: bool = True,
            crs: str|None = None,
            pad: float = 300,
            figsize: tuple[float, float] = (7, 8),
            title: str|None = None,
            ax: plt.Axes|None = None,
            return_fig: bool = False
    ) -> plt.Axes|tuple[plt.Figure, plt.Axes]:
        '''
        Map the towers and turbines, with an arrow from each tower showing the wind at a moment.
        Arrows point the direction the air is travelling.

        Parameters
        ----------
        time : np.datetime64 or str or None, optional
            The time at which to plot wind direction. The averaging bin containing it is used, or the records either
            side of it when `time_averaging` is None. Cannot be combined with `t_min`/`t_max`.
        t_min, t_max : np.datetime64 or None, optional
            An explicit window over which to average the wind data. Cannot be combined with `time`.
        time_averaging : str or np.timedelta64 or None, optional
            The averaging period, e.g. '30m' (the default).
        instrument : {'sonic-anemometer', 'weather-vane'}, optional
            Which instrument the wind comes from. Defaults to the 10 m weather vane.
        colour_by : str or None, optional
            A variable to colour the tower markers by, e.g. 'co2_4_5m'. If None (default), the
            markers are a uniform `colour_towers`.
        cmap : str, optional
            The colourmap for `colour_by`. Defaults to 'viridis'.
        colour_limits : tuple of float or None, optional
            (vmin, vmax) for `colour_by`. If None (default), the values' own range is used.
        colour_towers : str, optional
            The tower marker colour when `colour_by` is None.
        colour_missing : str, optional
            The marker colour for towers whose `colour_by` value is missing.
        colour_turbines : str, optional
            The turbine marker colour.
        speed_colours : bool, optional
            If True (default), colour the arrows by wind speed, sharing the wind roses' discrete speed
            scale. If False, the arrows are a uniform `colour_arrows`.
        speed_bins : np.ndarray or None, optional
            Speed bin edges for `speed_colours`. Defaults to 2 m/s bins from 0 to 18.
        speed_cmap : str, optional
            The colourmap sampled for the speed bins. Defaults to 'turbo'.
        colour_arrows : str, optional
            The arrow colour when `speed_colours` is False.
        arrow_length : float, optional
            Arrow length in metres - fixed, or the length drawn at `reference_speed` when
            `length_by_speed` is True. Defaults to 150m.
        length_by_speed : bool, optional
            If True, scale the arrow length with wind speed. Defaults to False.
        reference_speed : float, optional
            The speed drawn at `arrow_length` when `length_by_speed` is True, in m/s.
        annotate : bool, optional
            If True, label each tower with its station name. Defaults to False.
        scale_bar : bool, optional
            If True (default), add a scale bar.
        colourbar : bool, optional
            If True (default), add a colourbar for whichever colour mappings are in use.
        legend : bool, optional
            If True (default), label the tower and turbine markers.
        crs : str or None, optional
            The CRS to draw in. Must be projected in metres, as the arrows and `pad` are metre
            quantities. Defaults to `self.plotting_crs`, the site's UTM zone.
        pad : float, optional
            Metres of margin around the locations. Defaults to 300.
        figsize : tuple of float, optional
            Figure size, used only when `ax` is None.
        title : str or None, optional
            The axes title. Defaults to the instrument and the time window.
        ax : matplotlib.axes.Axes or None, optional
            The axes to draw into. If None, one is created.
        return_fig : bool, optional
            If True, return the Figure and Axes. Defaults to False.

        Returns
        -------
        matplotlib.axes.Axes or tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        '''
        t_min, t_max, window_label = self._resolve_window(
            time, t_min=t_min, t_max=t_max, time_averaging=time_averaging
        )

        crs = self.plotting_crs if crs is None else crs
        crs_info = gpd.GeoSeries([], crs=crs).crs
        if not crs_info.is_projected:
            raise ValueError(f"plot_site needs a projected CRS in metres, but got '{crs}', which is geographic.")

        towers = self.towers.reindex(list(self.stations)).to_crs(crs)
        turbines = self.turbines.to_crs(crs)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        turbines.plot(ax=ax, color=colour_turbines, marker='1', markersize=100, linewidth=0.75, zorder=2)

        x, y = towers.geometry.x.to_numpy(), towers.geometry.y.to_numpy()

        direction, speed = self.wind(
            station='all', instrument=instrument, samples='mean',
            t_min=t_min, t_max=t_max, time_averaging=time_averaging, units='deg'
        )
        direction = np.atleast_2d(direction)
        speed = np.atleast_2d(speed)

        if direction.size:
            direction, speed = self._vector_mean(direction, speed, axis=0)
        else:  # the window falls outside the campaign's coverage
            direction = speed = np.full(self.n_stations, np.nan)

        has_wind = np.isfinite(direction) & np.isfinite(speed)

        # tower markers, optionally carrying a variable
        values = self._station_snapshot(
            colour_by, t_min=t_min, t_max=t_max, time_averaging=time_averaging
        ) if colour_by is not None else None

        marker_mappable = None
        if values is None:
            ax.scatter(x, y, color=colour_towers, marker='^', s=60,
                       edgecolor='black', linewidth=0.5, zorder=4)
        else:
            known = np.isfinite(values)
            limits = colour_limits if colour_limits is not None else (
                (np.nanmin(values), np.nanmax(values)) if known.any() else (0, 1)
            )
            marker_mappable = ax.scatter(
                x[known], y[known], c=values[known], cmap=cmap,
                norm=mcolors.Normalize(*limits), marker='^', s=60,
                edgecolor='black', linewidth=0.5, zorder=4
            )
            if (~known).any():
                ax.scatter(x[~known], y[~known], color=colour_missing, marker='^', s=60,
                           edgecolor='black', linewidth=0.5, zorder=4)

        # downwind arrows, one per tower with valid wind
        arrows = None
        if has_wind.any():
            east, north = self.downwind_components(direction[has_wind])
            lengths = (arrow_length * speed[has_wind] / reference_speed) if length_by_speed else arrow_length
            quiver_kwargs = dict(
                angles='xy', scale_units='xy', scale=1, pivot='tail',
                width=0.004, headwidth=4, headlength=5, headaxislength=4.5, zorder=3
            )
            if speed_colours:
                speed_map, speed_norm = self.speed_colourmap(speed_bins, speed_cmap)
                arrows = ax.quiver(x[has_wind], y[has_wind], east * lengths, north * lengths,
                                   speed[has_wind], cmap=speed_map, norm=speed_norm, **quiver_kwargs)
            else:
                arrows = ax.quiver(x[has_wind], y[has_wind], east * lengths, north * lengths,
                                   color=colour_arrows, **quiver_kwargs)

            if length_by_speed:
                ax.quiverkey(arrows, 0.88, 0.04, arrow_length, f'{reference_speed:g} m s$^{{-1}}$',
                             labelpos='W', coordinates='axes')

        if annotate:
            for station, x_i, y_i in zip(towers.index, x, y):
                # above the marker, clear of the arrow leaving it
                ax.annotate(station, (x_i, y_i), textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=8, zorder=5)

        if legend:
            ax.legend(
                handles=[
                    plt.Line2D([], [], linestyle='none', marker='^', markersize=7,
                               markerfacecolor=colour_towers if values is None else 'none',
                               markeredgecolor='black', label='Eddy tower'),
                    plt.Line2D([], [], linestyle='none', marker='1', markersize=10,
                               color=colour_turbines, label='Wind turbine')
                ],
                loc='best', fontsize=9, frameon=True, fancybox=True, framealpha=0.5
            )

        if colourbar and marker_mappable is not None:
            ax.figure.colorbar(marker_mappable, ax=ax, location='right', shrink=0.7,
                               label=self._variable_label(colour_by))
        if colourbar and arrows is not None and speed_colours:
            ax.figure.colorbar(arrows, ax=ax, location='bottom', shrink=0.7,
                               spacing='proportional', label='Wind speed / m s$^{-1}$')

        bounds = pd.concat([towers, turbines]).total_bounds
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
        ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
        ax.set_aspect('equal')
        ax.set_axis_off()

        if scale_bar:
            # lower right, to stay clear of the legend; valid because the axes are in true metres
            ax.add_artist(ScaleBar(1, location='lower right'))

        if title is None:
            title = f'{instrument} — {window_label}'
            if not has_wind.any():
                title = f'{title} — no wind data'
        ax.set_title(title)

        return (ax.figure, ax) if return_fig else ax

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"CWEX(dir_path='{self.dir_path}', n_files={self.n_files}, stations={self.stations})"
