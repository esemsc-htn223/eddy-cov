__all__ = [
    'REANALYSIS_DIR', 'Coverage', 'Reanalysis', 'ERA5', 'WindToolkit',
    'DATASETS', 'get_dataset', 'datasets_for_site'
]

import numpy as np
import pandas as pd

import requests

from netCDF4 import Dataset

import os
import time
import pathlib
import threading
from dataclasses import dataclass
from typing import Literal

from eddy.util import DATA_DIR, logger_data
from eddy.util import average_over_time

REANALYSIS_DIR = DATA_DIR / 'reanalysis'
if not REANALYSIS_DIR.exists():
    REANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Coverage:
    '''
    Where and when a reanalysis product is valid, and at what resolution.

    Should be held by every Reanalysis subclass, and used to check a site before reading the data.
    Provides the `contains` and `covers` methods for checking a site and window, 
    and a `describe` method summarising the product.

    Attributes
    ----------
    name : str
        The dataset key, as used by `get_dataset`.
    region : str
        A human-readable description of the domain, e.g. 'global' or 'contiguous United States'.
    bounds : tuple of float or None
        (lat_min, lat_max, lon_min, lon_max) in degrees, or None if global. 
        Note this is only approximate (models are not expected to lie on a lat/lon rectangle).
        Only used as a pre-filter for `contains`.
    start, end : np.datetime64
        The time period covered.
    grid_spacing_km : float
        Nominal horizontal grid spacing.
    time_step : np.timedelta64
        Native temporal resolution.
    heights_m : tuple of float
        Heights above ground at which wind is available, in metres.
    notes : str
        Caveats worth knowing before trusting the numbers at a new site.
    '''

    name: str
    region: str
    bounds: tuple[float, float, float, float]|None
    start: np.datetime64
    end: np.datetime64
    grid_spacing_km: float
    time_step: np.timedelta64
    heights_m: tuple[float, ...]
    notes: str = ''

    @property
    def is_global(self) -> bool:
        '''Whether the product covers the whole globe.'''
        return self.bounds is None     

    def covers_time(self, t_min: np.datetime64|None = None, t_max: np.datetime64|None = None) -> bool:
        '''Whether a time window falls within the product's coverage.'''
        if t_min is None and t_max is None:
            raise ValueError("Provide one or both of t_min and t_max to check temporal coverage.")
        if t_min is not None and np.datetime64(t_min) < self.start:
            return False
        if t_max is not None and np.datetime64(t_max) > self.end:
            return False
        return True

    def covers_space(self, lat: float|None = None, lon: float|None = None) -> bool:
        '''Whether a site falls within the product's coverage.'''
        if lat is None or lon is None:
            raise ValueError("Provide both lat and lon to check spatial coverage.")
        if self.is_global:
            return True
        lat_min, lat_max, lon_min, lon_max = self.bounds
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max 

    def covers(self, *, 
               t_min: np.datetime64|None = None, t_max: np.datetime64|None = None,
               lat: float|None = None, lon: float|None = None) -> bool:
        '''
        Check if a time window and/or a site fall within the product's coverage.

        Parameters
        ----------
        t_min, t_max : np.datetime64 or None, optional
            The window. If both are None, the check is only spatial.
        lat, lon : float or None, optional
            The site. If both are None, the check is only temporal. 
            Both must be provided to check spatial coverage.

        Returns
        -------
        bool
        '''

        in_space = True  # default to True if no lat/lon is provided - answer depends only on time
        if lat is not None and lon is not None:
            in_space = self.covers_space(lat, lon)
        else:
            if t_min is None and t_max is None:
                raise ValueError("Provide one or all of a t_min, t_max, or a lat/lon pair to check coverage.")

        in_time = self.covers_time(t_min=t_min, t_max=t_max)
        return in_space and in_time

    def describe(self) -> str:
        '''A one-line summary of the domain and resolution.'''
        extent = 'global' if self.is_global else (
            f'{self.bounds[0]:g} to {self.bounds[1]:g} N, {self.bounds[2]:g} to {self.bounds[3]:g} E'
        )
        return (f'{self.name}: {self.region} ({extent}), '
                f'{str(self.start)[:10]} to {str(self.end)[:10]}, '
                f'{self.grid_spacing_km:g} km, {self.time_step / np.timedelta64(1, "m"):g} min, '
                f'heights {", ".join(f"{h:g}" for h in self.heights_m)} m')


class OutOfCoverageError(ValueError):
    '''Raised when a site or time window falls outside a reanalysis product's domain.'''


class Reanalysis:
    '''
    Base class for a gridded reanalysis or model-reanalysis wind product.

    Attributes
    ----------
    coverage : Coverage
        The product's spatial and temporal domain and resolution.
    dir_path : pathlib.Path
        Where files for this product live.

    Methods
    -------
    check_site(lat, lon, ...)
        Verify a site and window fall inside the domain.
    wind(lat, lon, ...)
        Wind direction and speed at a location and height.
    download(...)
        Download the data for a window. Requires credentials.
    '''

    coverage: Coverage

    def __init__(self, dir_path: str|pathlib.Path|None = None):
        '''
        Parameters
        ----------
        dir_path : str or pathlib.Path or None, optional
            Where to store files. Defaults to `REANALYSIS_DIR / <dataset name>`.
        '''
        self.dir_path = pathlib.Path(dir_path) if dir_path is not None else REANALYSIS_DIR / self.coverage.name
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True, exist_ok=True)

    def check_site(
            self, lat: float, lon: float, *,
            t_min: np.datetime64|None = None, t_max: np.datetime64|None = None,
            raise_on_fail: bool = True
    ) -> bool:
        '''
        Verify that a site and time window fall inside this product's domain.

        Should be used to check validity before using at a new site, as some
        products will return the nearest grid cell to a site well outside their 
        domain if asked.

        Parameters
        ----------
        lat, lon : float
            Site latitude and longitude in degrees.
        t_min, t_max : np.datetime64 or None, optional
            The window of interest.
        raise_on_fail : bool, optional
            Whether to raise rather than return False. Defaults to True.

        Returns
        -------
        bool

        Raises
        ------
        OutOfCoverageError
            If the site or window is outside the domain and `raise_on_fail` is True.
        '''
        in_space = self.coverage.covers_space(lat=lat, lon=lon)
        in_time = self.coverage.covers_time(t_min=t_min, t_max=t_max)
        if in_time and in_space:
            return True

        if raise_on_fail:
            problem = []
            if not in_space:
                problem.append(f'({lat:.4f}, {lon:.4f}) is outside {self.coverage.region}')
            if not in_time:
                problem.append(f'the window falls outside {str(self.coverage.start)[:10]} to {str(self.coverage.end)[:10]}')
            alternatives = [d for d in datasets_for_site(lat, lon, t_min=t_min, t_max=t_max)
                            if d != self.coverage.name]
            suggestion = f" Valid here: {', '.join(alternatives)}." if alternatives else ''
            raise OutOfCoverageError(
                f"{self.coverage.name} does not cover this site: {', and '.join(problem)}.{suggestion}"
            )
        return False

    def wind(
            self, lat: float, lon: float, *,
            height: float = 80.0,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m',
            units: Literal['deg', 'rad'] = 'deg',
            return_time: bool = False
    ) -> tuple[np.ndarray, ...]:
        raise NotImplementedError(f"{type(self).__name__} does not implement wind().")

    def download(self, *, t_min: np.datetime64, t_max: np.datetime64, **kwargs) -> pathlib.Path:
        raise NotImplementedError(f"{type(self).__name__} does not implement download().")

    @staticmethod
    def _interpolate_height(
            height: float, levels: np.ndarray, east: np.ndarray, north: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Interpolate wind components between the two levels bracketing `height`.

        Values are held at the nearest level if `height` is outside the available levels.
        For heights within the range, a logarithmic interpolation between the two nearest levels
        is used.

        Parameters
        ----------
        height : float
            Target height in metres.
        levels : np.ndarray
            Available heights, ascending.
        east, north : np.ndarray
            Components at each level, with the level axis last.

        Returns
        -------
        np.ndarray, np.ndarray
        '''
        if height <= levels[0]:
            return east[..., 0], north[..., 0]
        if height >= levels[-1]:
            return east[..., -1], north[..., -1]

        upper = int(np.searchsorted(levels, height))
        lower = upper - 1
        weight = np.log(height / levels[lower]) / np.log(levels[upper] / levels[lower])

        return (east[..., lower] + weight * (east[..., upper] - east[..., lower]),
                north[..., lower] + weight * (north[..., upper] - north[..., lower]))

    @staticmethod
    def _to_bins(
            time: np.ndarray, east: np.ndarray, north: np.ndarray,
            time_averaging: str|np.timedelta64|None,
            method: Literal['average', 'interpolate']
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Move the given components into a regular time grid, either by averaging or interpolating.

        Parameters
        ----------
        time : np.ndarray
            Native timestamps.
        east, north : np.ndarray
            Wind components.
        time_averaging : str or np.timedelta64 or None
            The bin length. None returns the native timestamps unchanged.
        method : {'average', 'interpolate'}
            How to move onto the bins.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray
            Bin start times, and the components on those bins.
        '''
        if time_averaging is None:
            return time, east, north

        if method == 'average':
            binned_time, stacked = average_over_time(
                time, np.stack([east, north], axis=-1), time_averaging
            )
            return binned_time, stacked[..., 0], stacked[..., 1]

        period = pd.Timedelta(time_averaging).to_timedelta64().astype('timedelta64[ns]')
        period_ns = period.astype('int64')
        native_ns = time.astype('datetime64[ns]').astype('int64')
        first_bin = (native_ns[0] // period_ns) * period_ns
        last_bin = (native_ns[-1] // period_ns) * period_ns
        binned_ns = np.arange(first_bin, last_bin + period_ns, period_ns)

        # bin centres, so a 30 min mean is represented by the value in the middle of the bin
        targets = binned_ns + period_ns // 2
        return (binned_ns.astype('datetime64[ns]'),
                np.interp(targets, native_ns, east),
                np.interp(targets, native_ns, north))

    def _missing_data(self, path: pathlib.Path, t_min, t_max) -> OSError:
        '''The error raised when a data file is absent.'''
        return FileNotFoundError(
            f"No {self.coverage.name} data at {path}. Fetch it with "
            f"{type(self).__name__}().download(t_min=np.datetime64('{t_min}'), "
            f"t_max=np.datetime64('{t_max}'), lat=..., lon=...). "
            f"This requires credentials - see the class docstring."
        )

    def __repr__(self):
        return f'{type(self).__name__}(dir_path={str(self.dir_path)!r})'

    def __str__(self):
        return self.coverage.describe()


class ERA5(Reanalysis):
    '''
    ECMWF ERA5 single-level winds, at 10 m and 100 m.

    **Coverage:** 
        *region:* global
        *time:* 1940 to present. 
        *time step:* 1 h
        *heights:* 10 and 100 m
        *grid spacing:* 0.25 degrees (~31 km)

    Access needs a free Copernicus Climate Data Store account and a personal access token. 
    See https://cds.climate.copernicus.eu/how-to-api. The dataset's terms of use have to be accepted
    once on the web form before using the API.
    '''

    coverage = Coverage(
        name='era5',
        region='global',
        bounds=None,
        start=np.datetime64('1940-01-01'),
        end=np.datetime64('2100-01-01'),
        grid_spacing_km=31.0,
        time_step=np.timedelta64(1, 'h'),
        heights_m=(10.0, 100.0),
        notes=('Global, but very coarse (0.25 deg grid spacing). Wind over land is smoothed by the low resolution of the model surface.')
    )

    _CDS_DATASET = 'reanalysis-era5-single-levels'
    _VARIABLES = ('10m_u_component_of_wind', '10m_v_component_of_wind',
                  '100m_u_component_of_wind', '100m_v_component_of_wind')

    def _data_path(self, lat: float, lon: float, t_min: np.datetime64, t_max: np.datetime64) -> pathlib.Path:
        '''The data file for a site and window.'''
        return self.dir_path / (
            f'era5_{lat:.3f}_{lon:.3f}_{str(np.datetime64(t_min))[:10]}_{str(np.datetime64(t_max))[:10]}.nc'
        )

    def download(
            self, *, lat: float, lon: float,
            t_min: np.datetime64, t_max: np.datetime64,
            margin_deg: float = 0.5, force: bool = False
    ) -> pathlib.Path:
        '''
        Download ERA5 winds for a small box around a site.

        Parameters
        ----------
        lat, lon : float
            Site coordinates in degrees.
        t_min, t_max : np.datetime64
            The window to fetch.
        margin_deg : float, optional
            Half-width of the requested box, in degrees. Defaults to 0.5, which is two grid cells
            either side - enough to see the neighbours.
        force : bool, optional
            Re-download even if the data exists. Defaults to False.

        Returns
        -------
        pathlib.Path
        '''
        import cdsapi

        self.check_site(lat, lon, t_min=t_min, t_max=t_max)
        path = self._data_path(lat, lon, t_min, t_max)
        if path.exists() and not force:
            return path

        times = pd.date_range(pd.Timestamp(t_min), pd.Timestamp(t_max), freq='h')
        request = {
            'product_type': ['reanalysis'],
            'variable': list(self._VARIABLES),
            'year': sorted({f'{t.year:04d}' for t in times}),
            'month': sorted({f'{t.month:02d}' for t in times}),
            'day': sorted({f'{t.day:02d}' for t in times}),
            'time': [f'{h:02d}:00' for h in range(24)],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            'area': [lat + margin_deg, lon - margin_deg, lat - margin_deg, lon + margin_deg],
        }

        logger_data.info(f'DOWNLOADING ERA5 to {path} for ({lat}, {lon}) {t_min} to {t_max}')
        cdsapi.Client().retrieve(self._CDS_DATASET, request).download(str(path))
        return path

    def wind(
            self, lat: float, lon: float, *,
            height: float = 80.0,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m',
            units: Literal['deg', 'rad'] = 'deg',
            return_time: bool = False
    ) -> tuple[np.ndarray, ...]:
        '''
        Wind direction and speed at a site, from the downloaded ERA5 box.

        Parameters
        ----------
        lat, lon : float
            Site coordinates in degrees.
        height : float, optional
            Height above ground in metres. Defaults to 80. 
            Interpolated logarithmically between 10 m and 100 m. 
            Heights above 100 m are held at 100 m, and same for heights below 10 m.
        t_min, t_max : np.datetime64 or None, optional
            The window, also used to find the data file.
        time_averaging : str or np.timedelta64 or None, optional
            Bin length for the output. ERA5 is hourly, so bins shorter than an hour are
            interpolated rather than averaged. Defaults to '30m'.
        units : {'deg', 'rad'}, optional
            Units for the returned direction. Defaults to degrees.
        return_time : bool, optional
            If True, also return the bin start times. Defaults to False.

        Returns
        -------
        tuple of np.ndarray
            (direction, speed), preceded by the timestamps if `return_time` is True.
        '''
        self.check_site(lat, lon, t_min=t_min, t_max=t_max)

        path = self._data_path(lat, lon, t_min, t_max)
        if not path.exists():
            raise self._missing_data(path, t_min, t_max)

        with Dataset(path) as ds:
            time_var = ds['valid_time'] if 'valid_time' in ds.variables else ds['time']
            time = (np.datetime64('1970-01-01T00:00:00', 's')
                    + np.ma.getdata(time_var[:]).astype('timedelta64[s]')).astype('datetime64[ns]')
            lats = np.ma.getdata(ds['latitude'][:])
            lons = np.ma.getdata(ds['longitude'][:])
            i = int(np.argmin(np.abs(lats - lat)))
            j = int(np.argmin(np.abs(lons - lon)))
            east = np.stack([np.ma.filled(ds['u10'][:, i, j], np.nan),
                             np.ma.filled(ds['u100'][:, i, j], np.nan)], axis=-1)
            north = np.stack([np.ma.filled(ds['v10'][:, i, j], np.nan),
                              np.ma.filled(ds['v100'][:, i, j], np.nan)], axis=-1)

        window = np.ones(time.shape, dtype=bool)
        if t_min is not None:
            window &= time >= np.datetime64(t_min, 'ns')
        if t_max is not None:
            window &= time <= np.datetime64(t_max, 'ns')
        time, east, north = time[window], east[window], north[window]

        east, north = self._interpolate_height(height, np.asarray(self.coverage.heights_m), east, north)
        time, east, north = self._to_bins(time, east, north, time_averaging, 'interpolate')

        direction = np.arctan2(-east, -north)
        speed = np.hypot(east, north)
        if units.lower() == 'deg':
            direction = np.degrees(direction) % 360

        if return_time:
            return time, direction, speed
        return direction, speed


class WindToolkit(Reanalysis):
    '''
    The NLR WIND Toolkit, a WRF downscaling produced for wind-energy applications.

    **Coverage:** 
        *region:* one model domain at a time, selected by `region` from `REGIONS`. 
        Domains cover parts of North America, Asia, South America and Europe, 
        but there is no data outside them.
        *time:* varies by region; 2007 to 2014 for the default 'conus'.
        *time step:* 5, 15, 30 or 60 min, selected by `interval`.
        *heights:* 10, 40, 60, 80, 100, 120, 140, 160 and 200 m
        *grid spacing:* 2 km

    Note this is a model output rather than an observational analysis. The model does not 
    include wind turbines, so speeds are unaffected by wakes.

    Access needs a free API key from https://developer.nlr.gov/signup/ in the `NLR_API_KEY` 
    environment variable, and a contact email address in `NLR_API_EMAIL`. 
    One request gives one point and one year. Requests are held to one per second, and both the 
    daily limit and the maximum request weight are checked before sending. 
    See https://developer.nlr.gov/docs/wind/wind-toolkit/.
    '''

    _BASE_URL = 'https://developer.nlr.gov/api/wind-toolkit/v2/wind/'
    _STANDARD_HEIGHTS = (10.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0)

    # API rate limits. The daily limit falls to 2000 for formats other than csv.
    MAX_REQUEST_WEIGHT = 175000100
    REQUESTS_PER_DAY = 10000
    SECONDS_PER_REQUEST = 1.0
    MAX_CONCURRENT_REQUESTS = 20

    _request_lock = threading.Lock()
    _last_request_time = 0.0
    _requests_made = 0

    REGIONS = {
        'conus': {
            'endpoint': 'wtk-download',
            'region': 'contiguous United States',
            'bounds': (23.0, 51.0, -127.0, -65.0),
            'start': np.datetime64('2007-01-01'),
            'end': np.datetime64('2014-12-31T23:55'),
            'interval': 5,
        },
        'canada': {
            'endpoint': 'wtk-canada-5min-download',
            'region': 'Canada',
            'bounds': (41.0, 84.0, -142.0, -52.0),
            'start': np.datetime64('2007-01-01'),
            'end': np.datetime64('2014-12-31T23:55'),
            'interval': 5,
        },
        'mexico': {
            'endpoint': 'wtk-mexico-5min-download',
            'region': 'Mexico',
            'bounds': (14.0, 33.0, -118.0, -86.0),
            'start': np.datetime64('2007-01-01'),
            'end': np.datetime64('2014-12-31T23:55'),
            'interval': 5,
        },
        'hawaii': {
            'endpoint': 'wtk-hawaii-met-toolkit-v1-0-0-download',
            'region': 'Hawaii',
            'bounds': (18.0, 23.0, -161.0, -154.0),
            'start': np.datetime64('2000-01-01'),
            'end': np.datetime64('2025-12-31T23:55'),
            'interval': 5,
        },
        'indonesia': {
            'endpoint': 'wtk-indonesia-v1-0-0-download',
            'region': 'Indonesia',
            'bounds': (-11.0, 6.0, 95.0, 141.0),
            'start': np.datetime64('2007-01-01'),
            'end': np.datetime64('2021-12-31T23:45'),
            'interval': 15,
        },
        'vietnam': {
            'endpoint': 'vietnam-wtk-download',
            'region': 'Vietnam',
            'bounds': (8.0, 24.0, 102.0, 110.0),
            'start': np.datetime64('2015-01-01'),
            'end': np.datetime64('2015-12-31T23:45'),
            'interval': 15,
        },
        'philippines': {
            'endpoint': 'philippines-wtk-download',
            'region': 'the Philippines',
            'bounds': (4.0, 21.0, 116.0, 127.0),
            'start': np.datetime64('2015-01-01'),
            'end': np.datetime64('2015-12-31T23:45'),
            'interval': 15,
        },
        'bangladesh': {
            'endpoint': 'wtk-bangladesh-download',
            'region': 'Bangladesh',
            'bounds': (20.0, 27.0, 88.0, 93.0),
            'start': np.datetime64('2014-01-01'),
            'end': np.datetime64('2017-12-31T23:45'),
            'interval': 15,
        },
        'kazakhstan': {
            'endpoint': 'wtk-kazakhstan-download',
            'region': 'Kazakhstan',
            'bounds': (40.0, 56.0, 46.0, 88.0),
            'start': np.datetime64('2015-01-01'),
            'end': np.datetime64('2015-12-31T23:45'),
            'interval': 15,
        },
        'south-america': {
            'endpoint': 'wtk-sup3rwind-south-america-v1-0-0-5min-download',
            'region': 'South America',
            'bounds': (-56.0, 13.0, -82.0, -34.0),
            'start': np.datetime64('2005-01-01'),
            'end': np.datetime64('2024-12-31T23:55'),
            'interval': 5,
        },
    }

    def __init__(
            self, dir_path: str|pathlib.Path|None = None, *,
            region: str = 'conus', interval: int|None = None
    ):
        '''
        Parameters
        ----------
        dir_path : str or pathlib.Path or None, optional
            Where to store downloaded timeseries.
        region : str, optional
            A key of `REGIONS`. Defaults to 'conus'.
        interval : int or None, optional
            The time step in minutes: 5, 15, 30 or 60. Defaults to the region's finest.
        '''
        region = region.lower().replace('_', '-').replace(' ', '-')
        if region not in self.REGIONS:
            raise ValueError(f"Invalid region '{region}'. Must be one of {sorted(self.REGIONS)}.")

        if interval is not None and interval not in (5, 15, 30, 60):
            raise ValueError(f"Invalid interval '{interval}'. Must be one of 5, 15, 30 or 60 minutes.")

        self.region = region
        self.endpoint = self.REGIONS[region]['endpoint']
        self.interval = int(self.REGIONS[region]['interval'] if interval is None else interval)
        self.coverage = self.coverage_for(region, interval=self.interval)
        super().__init__(dir_path)

    @classmethod
    def coverage_for(cls, region: str, *, interval: int|None = None) -> Coverage:
        '''
        The `Coverage` of one region, without building a `WindToolkit` or touching the filesystem.

        Parameters
        ----------
        region : str
            A key of `REGIONS`.
        interval : int or None, optional
            The time step in minutes. Defaults to the region's finest.

        Returns
        -------
        Coverage
        '''
        spec = cls.REGIONS[region]
        return Coverage(
            name=f'wtk-{region}',
            region=spec['region'],
            bounds=spec['bounds'],
            start=spec['start'],
            end=spec['end'],
            grid_spacing_km=2.0,
            time_step=np.timedelta64(int(spec['interval'] if interval is None else interval), 'm'),
            heights_m=cls._STANDARD_HEIGHTS,
            notes=('A regional model domain, so there is no data outside it. Model output rather '
                   'than an observational analysis. No turbines modelled, so winds are freestream.')
        )

    @staticmethod
    def _credentials() -> tuple[str, str]:
        '''
        The API key and contact email address required by the API, read from the environment.

        Returns
        -------
        str, str
        '''
        api_key = os.getenv('NLR_API_KEY')
        email = os.getenv('NLR_API_EMAIL')

        missing = [name for name, value in (('NLR_API_KEY', api_key), ('NLR_API_EMAIL', email))
                   if not value]
        if missing:
            raise RuntimeError(
                f"{' and '.join(missing)} must be set to download WIND Toolkit data. "
                "Register for a free key at https://developer.nlr.gov/signup/."
            )
        return api_key, email

    @classmethod
    def request_weight(
            cls, *, n_attributes: int, n_years: int, interval: int, n_sites: int = 1
    ) -> int:
        '''
        The weight the API assigns to a request, which may not exceed `MAX_REQUEST_WEIGHT`.

        Parameters
        ----------
        n_attributes : int
            The number of attributes requested.
        n_years : int
            The number of years requested.
        interval : int
            The time step in minutes.
        n_sites : int, optional
            The number of sites covered by the WKT geometry. Defaults to 1, the value for a
            POINT. For an area, the API's `site_count` endpoint gives it.

        Returns
        -------
        int
        '''
        intervals_per_year = (60 / interval) * 24 * 365
        return int(n_sites * n_attributes * n_years * intervals_per_year)

    @classmethod
    def _get(cls, url: str, params: dict, *, timeout: float, retries: int = 2):
        '''
        Make a request, holding to one per second and retrying if the API asks us to wait.

        Requests are serialised, which also keeps within the limit on requests in process at
        once.

        Parameters
        ----------
        url : str
            The URL to request.
        params : dict
            The query parameters.
        timeout : float
            The request timeout, in seconds.
        retries : int, optional
            How many times to retry after a 429. Defaults to 2.

        Returns
        -------
        requests.Response
        '''
        for attempt in range(retries + 1):
            with cls._request_lock:
                wait = cls.SECONDS_PER_REQUEST - (time.monotonic() - cls._last_request_time)
                if wait > 0:
                    time.sleep(wait)

                response = requests.get(url, params=params, timeout=timeout)

                cls._last_request_time = time.monotonic()
                cls._requests_made += 1

            if response.status_code != 429:
                return response

            if attempt < retries:
                delay = float(response.headers.get('Retry-After', 2 ** attempt))
                logger_data.info(f'WIND Toolkit rate limited, waiting {delay}s before retrying')
                time.sleep(delay)

        raise RuntimeError(
            f"WIND Toolkit refused the request after {retries + 1} attempts, having hit a rate "
            f"limit. The limits are {cls.REQUESTS_PER_DAY} requests a day and one a second; "
            f"{cls._requests_made} requests have been made in this session."
        )

    def heights_needed(self, height: float) -> list[float]:
        '''
        The model heights needed to produce a wind speed at a given height.

        Parameters
        ----------
        height : float
            The height above ground, in metres.

        Returns
        -------
        list of float
            One height if `height` is a model height or outside the available range, otherwise
            the two either side of it.
        '''
        levels = np.asarray(self._STANDARD_HEIGHTS)
        if height in levels or height <= levels[0] or height >= levels[-1]:
            return [float(levels[np.argmin(np.abs(levels - height))])]

        upper = int(np.searchsorted(levels, height))
        return [float(levels[upper - 1]), float(levels[upper])]

    def _data_path(self, lat: float, lon: float, height: float, year: int) -> pathlib.Path:
        '''The data file for a site, height and year.'''
        return self.dir_path / (
            f'wtk_{self.region}_{lat:.3f}_{lon:.3f}_{height:g}m_{self.interval}min_{year}.csv'
        )

    def download(
            self, *, lat: float, lon: float,
            t_min: np.datetime64, t_max: np.datetime64,
            height: float = 80.0, force: bool = False, timeout: float = 300
    ) -> list[pathlib.Path]:
        '''
        Download and store the wind timeseries for a site and height.

        One file is written per calendar year in the window, as the API serves a single year
        per request.

        Parameters
        ----------
        lat, lon : float
            The site coordinates, in degrees.
        t_min, t_max : np.datetime64
            The window to fetch.
        height : float, optional
            The height in metres. Defaults to 80. A height between two model heights fetches both.
        force : bool, optional
            Whether to download again when the data is already stored. Defaults to False.
        timeout : float, optional
            The request timeout, in seconds. Defaults to 300.

        Returns
        -------
        list of pathlib.Path
        '''
        self.check_site(lat, lon, t_min=t_min, t_max=t_max)
        api_key, email = self._credentials()

        attributes = [f'{quantity}_{level:g}m'
                      for level in self.heights_needed(height)
                      for quantity in ('windspeed', 'winddirection')]

        weight = self.request_weight(
            n_attributes=len(attributes), n_years=1, interval=self.interval
        )
        if weight > self.MAX_REQUEST_WEIGHT:
            raise ValueError(
                f"A request for {len(attributes)} attributes at {self.interval} min has weight "
                f"{weight}, above the API's maximum of {self.MAX_REQUEST_WEIGHT}. Request fewer "
                "attributes or use a longer interval."
            )

        years = [year for year in range(pd.Timestamp(t_min).year, pd.Timestamp(t_max).year + 1)
                 if force or not self._data_path(lat, lon, height, year).exists()]
        if self._requests_made + len(years) > self.REQUESTS_PER_DAY:
            raise RuntimeError(
                f"Fetching {len(years)} years would take this session past the API's limit of "
                f"{self.REQUESTS_PER_DAY} requests a day ({self._requests_made} already made)."
            )

        written = []
        for year in range(pd.Timestamp(t_min).year, pd.Timestamp(t_max).year + 1):
            path = self._data_path(lat, lon, height, year)
            if path.exists() and not force:
                written.append(path)
                continue

            logger_data.info(
                f'DOWNLOADING WIND Toolkit {self.region} {year} to {path} for ({lat}, {lon})'
            )
            response = self._get(
                f'{self._BASE_URL}{self.endpoint}.csv',
                {
                    'api_key': api_key,
                    'wkt': f'POINT({lon:.6f} {lat:.6f})',
                    'names': str(year),
                    'attributes': ','.join(attributes),
                    'interval': str(self.interval),
                    'utc': 'true',
                    'leap_day': 'true',
                    'email': email,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            path.write_bytes(response.content)
            written.append(path)

        return written

    def site_metadata(
            self, lat: float, lon: float, *, height: float = 80.0, year: int|None = None
    ) -> dict:
        '''
        The grid point matched to a site, read from a stored file's header.

        The distance to that point is the authoritative coverage test, as `Coverage.bounds` is
        only an approximate pre-filter.

        Parameters
        ----------
        lat, lon : float
            The requested site coordinates, in degrees.
        height : float, optional
            The height whose file to read. Defaults to 80.
        year : int or None, optional
            The year whose file to read. Defaults to the earliest stored year.

        Returns
        -------
        dict
            The `site_id`, `latitude`, `longitude` and `distance_km` of the matched grid point.
        '''
        if year is None:
            stored = sorted(self.dir_path.glob(
                f'wtk_{self.region}_{lat:.3f}_{lon:.3f}_{height:g}m_{self.interval}min_*.csv'
            ))
            if not stored:
                raise self._missing_data(self._data_path(lat, lon, height, 0), None, None)
            path = stored[0]
        else:
            path = self._data_path(lat, lon, height, year)
            if not path.exists():
                raise self._missing_data(path, None, None)

        header = pd.read_csv(path, nrows=1)
        matched_lat = float(header['Latitude'].iloc[0])
        matched_lon = float(header['Longitude'].iloc[0])

        dlat = matched_lat - lat
        dlon = (matched_lon - lon) * np.cos(np.radians(lat))
        distance_km = float(np.hypot(dlat, dlon) * 111.32)

        if distance_km > 2 * self.coverage.grid_spacing_km:
            raise OutOfCoverageError(
                f"({lat:.4f}, {lon:.4f}) matched a {self.coverage.name} grid point "
                f"{distance_km:.1f} km away, so it lies outside the model domain "
                f"({self.coverage.region})."
            )

        return {'site_id': header.get('SiteID', pd.Series([None])).iloc[0],
                'latitude': matched_lat, 'longitude': matched_lon, 'distance_km': distance_km}

    @staticmethod
    def _read_stored(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        The timestamps and wind components in a stored file.

        Parameters
        ----------
        path : pathlib.Path
            A file written by `download`.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            The timestamps, the eastward and northward components with the height axis last, and
            the heights themselves.
        '''
        frame = pd.read_csv(path, skiprows=2)
        time = pd.to_datetime(frame[['Year', 'Month', 'Day', 'Hour', 'Minute']]).to_numpy()

        levels = sorted(
            float(column.split('_')[1].removesuffix('m'))
            for column in frame.columns if column.startswith('windspeed_')
        )
        east, north = [], []
        for level in levels:
            speed = frame[f'windspeed_{level:g}m'].to_numpy(dtype=np.float64)
            theta = np.radians(frame[f'winddirection_{level:g}m'].to_numpy(dtype=np.float64))
            east.append(-speed * np.sin(theta))
            north.append(-speed * np.cos(theta))

        return (time.astype('datetime64[ns]'), np.stack(east, axis=-1),
                np.stack(north, axis=-1), np.asarray(levels))

    def wind(
            self, lat: float, lon: float, *,
            height: float = 80.0,
            t_min: np.datetime64|None = None,
            t_max: np.datetime64|None = None,
            time_averaging: str|np.timedelta64|None = '30m',
            units: Literal['deg', 'rad'] = 'deg',
            return_time: bool = False
    ) -> tuple[np.ndarray, ...]:
        '''
        The wind direction and speed at a site.

        Parameters
        ----------
        lat, lon : float
            The site coordinates, in degrees.
        height : float, optional
            The height above ground, in metres. Defaults to 80, a model height. Other heights are
            interpolated logarithmically between the model heights either side.
        t_min, t_max : np.datetime64 or None, optional
            Bounds on the timestamps to include.
        time_averaging : str or np.timedelta64 or None, optional
            The averaging interval, e.g. '30m' (the default). Intervals longer than the time step
            are vector-averaged. None disables averaging.
        units : {'deg', 'rad'}, optional
            The units for the wind direction. Defaults to 'deg'.
        return_time : bool, optional
            If True, also return the timestamps. Defaults to False.

        Returns
        -------
        tuple of np.ndarray
            The wind directions and speeds, preceded by the times if `return_time` is True.
        '''
        self.check_site(lat, lon, t_min=t_min, t_max=t_max)

        start = pd.Timestamp(t_min if t_min is not None else self.coverage.start).year
        stop = pd.Timestamp(t_max if t_max is not None else self.coverage.end).year

        times, easts, norths, levels = [], [], [], None
        for year in range(start, stop + 1):
            path = self._data_path(lat, lon, height, year)
            if not path.exists():
                raise self._missing_data(path, t_min, t_max)
            year_time, year_east, year_north, levels = self._read_stored(path)
            times.append(year_time)
            easts.append(year_east)
            norths.append(year_north)

        time = np.concatenate(times)
        east, north = np.concatenate(easts), np.concatenate(norths)

        window = np.ones(time.shape, dtype=bool)
        if t_min is not None:
            window &= time >= np.datetime64(t_min, 'ns')
        if t_max is not None:
            window &= time <= np.datetime64(t_max, 'ns')
        time, east, north = time[window], east[window], north[window]

        east, north = self._interpolate_height(height, levels, east, north)
        time, east, north = self._to_bins(time, east, north, time_averaging, 'average')

        direction = np.arctan2(-east, -north)
        speed = np.hypot(east, north)
        if units.lower() == 'deg':
            direction = np.degrees(direction) % 360

        if return_time:
            return time, direction, speed
        return direction, speed


DATASETS: dict[str, type[Reanalysis]] = {
    'era5': ERA5,
    'wtk': WindToolkit,
    'wind-toolkit': WindToolkit,
}


def get_dataset(spec: str|Reanalysis|type[Reanalysis], **kwargs) -> Reanalysis:
    '''
    Resolve a reanalysis product given as a name, a class or an instance.

    Parameters
    ----------
    spec : str or Reanalysis or type
        A key of `DATASETS` (case-insensitive), a `Reanalysis` subclass, or a configured instance,
        which is returned unchanged. A WIND Toolkit region may be given as 'wtk-canada'.
    **kwargs
        Passed to the constructor.

    Returns
    -------
    Reanalysis
    '''
    if isinstance(spec, Reanalysis):
        return spec

    if isinstance(spec, type) and issubclass(spec, Reanalysis):
        return spec(**kwargs)

    key = str(spec).lower().replace('_', '-').replace(' ', '-')
    if key.startswith('wtk-') and key[4:] in WindToolkit.REGIONS:
        return WindToolkit(region=key[4:], **kwargs)
    if key not in DATASETS:
        raise ValueError(
            f"Invalid reanalysis dataset '{spec}'. Must be one of {sorted(set(DATASETS))}, "
            f"a WIND Toolkit region ({', '.join('wtk-' + r for r in sorted(WindToolkit.REGIONS))}), "
            "or a Reanalysis subclass or instance."
        )
    return DATASETS[key](**kwargs)


def datasets_for_site(
        lat: float, lon: float, *,
        t_min: np.datetime64|None = None, t_max: np.datetime64|None = None
) -> list[str]:
    '''
    Which reanalysis products cover a given site and time window.

    Note that the WIND Toolkit test here is the declared bounding box only, which is only
    an approximate rectangle covering the actual model domain. A site may be inside the 
    bounding box but outside the domain, and some bounding boxes overlap causing a site to be
    inside multiple boxes at once. Use `WindToolkit.grid_index` to get an authoritative answer
    by measuring the distance to the nearest grid point (though note this requires network access
    credentials).

    Parameters
    ----------
    lat, lon : float
        Site coordinates in degrees.
    t_min, t_max : np.datetime64 or None, optional
        The time window of interest.

    Returns
    -------
    list of str
        Dataset names accepted by `get_dataset`.
    '''
    candidates = [ERA5.coverage] + [
        WindToolkit.coverage_for(region) for region in WindToolkit.REGIONS
    ]
    return [c.name for c in candidates if c.covers(t_min = t_min, t_max = t_max, lat = lat, lon = lon)]