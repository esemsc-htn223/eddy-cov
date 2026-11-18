__all__ = [
    'VON_KARMAN', 'GRAVITY',
    'psi_m', 'CovarianceAccumulator',
    'WindProfile', 'LogLawProfile', 'PowerLawProfile', 'PROFILES', 'get_profile'
]

import numpy as np

VON_KARMAN = 0.4
GRAVITY = 9.81


def psi_m(zeta: np.ndarray, *, zeta_limits: tuple[float, float] = (-50.0, 10.0)) -> np.ndarray:
    '''
    Integrated stability correction for momentum, in the Businger-Dyer form.

    Unstable conditions use the Paulson (1970) expression, stable conditions the linear
    -5*zeta of Businger et al. (1971). Both are only established over roughly |zeta| <= 2;
    beyond that the result is an extrapolation, so `zeta` is clipped rather than allowed to
    overflow.

    Parameters
    ----------
    zeta : np.ndarray
        Stability parameter (z - d) / L.
    zeta_limits : tuple of float, optional
        Bounds applied to `zeta` before evaluation. Defaults to (-50, 10).

    Returns
    -------
    np.ndarray
    '''
    zeta = np.clip(np.asarray(zeta, dtype=np.float64), *zeta_limits)
    out = np.zeros_like(zeta)

    unstable = zeta < 0
    x = (1 - 16 * zeta[unstable]) ** 0.25
    out[unstable] = (2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2)
                     - 2 * np.arctan(x) + np.pi / 2)

    stable = zeta >= 0
    out[stable] = -5 * zeta[stable]

    return out


class CovarianceAccumulator:
    '''
    Running sums of eddy-covariance moments over time bins, from which fluxes can be computed.

    Samples are added in chunks with `add` and reduced to fluxes with `fluxes`. 
    Only stores sums.

    Parameters
    ----------
    shape : tuple of int
        The shape of one accumulator, e.g. (n_bins, n_stations).

    Attributes
    ----------
    sums : dict of str to np.ndarray
        The running sums, keyed by `SUM_KEYS`. Each has the shape given in the `shape` parameter.

    Methods
    -------
    add(bin_index, u, v, w, t)
        Add a chunk of samples to the bins they belong to.
    fluxes(...)
        Reduce the sums to friction velocity, heat flux and Obukhov length.
    '''

    # sums needed to reconstruct the means and covariance matrix of (u, v, w, t)
    SUM_KEYS = ('n', 'u', 'v', 'w', 't',
                'uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'ut', 'vt', 'wt')

    def __init__(self, shape: tuple[int, ...]):
        '''Create a new accumulator with the given shape.'''
        self.shape = tuple(shape)
        self.sums = {key: np.zeros(self.shape, dtype=np.float64) for key in self.SUM_KEYS}

    def add(
            self, bin_index: np.ndarray,
            u: np.ndarray, v: np.ndarray, w: np.ndarray, t: np.ndarray
    ) -> 'CovarianceAccumulator':
        '''
        Add a chunk of high-rate samples, in place.

        Samples with any component missing are excluded from every sum together.

        Parameters
        ----------
        bin_index : np.ndarray
            The index of the bin each sample belongs to. Should have the same shape as the
            leading axes of the components `u`, `v`, `w` and `t`.
        u, v, w : np.ndarray
            The wind components. `u` is eastward, `v` northward and `w` upward.
        t : np.ndarray
            The virtual temperature.

        Returns
        -------
        CovarianceAccumulator
            This accumulator, with the new samples added.
        '''
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & np.isfinite(t)
        u = np.where(valid, u, 0.0)
        v = np.where(valid, v, 0.0)
        w = np.where(valid, w, 0.0)
        t = np.where(valid, t, 0.0)

        terms = {
            'n': valid.astype(np.float64),
            'u': u, 'v': v, 'w': w, 't': t,
            'uu': u * u, 'uv': u * v, 'uw': u * w,
            'vv': v * v, 'vw': v * w, 'ww': w * w,
            'ut': u * t, 'vt': v * t, 'wt': w * t,
        }
        for key, term in terms.items():
            np.add.at(self.sums[key], bin_index, term)

        return self

    @staticmethod
    def _rotation_matrices(u_bar: np.ndarray, v_bar: np.ndarray, w_bar: np.ndarray) -> np.ndarray:
        '''
        Compute the double-rotation matrices aligning each bin's axes with its mean wind.

        The first rotation turns towards the mean horizontal wind (yaw), to account for the cross-wind component.
        The second rotation turns upward (pitch) to account for the vertical wind component.

        Parameters
        ----------
        u_bar, v_bar, w_bar : np.ndarray
            The mean wind components in each bin. `u_bar` is eastward, `v_bar` northward and `w_bar` upward.

        Returns
        -------
        np.ndarray
            Rotation matrices with shape (`input_shape`, 3, 3), where `input_shape` 
            is the shape of the input arrays.
        '''
        theta = np.arctan2(v_bar, u_bar)
        horizontal = np.hypot(u_bar, v_bar)
        phi = np.arctan2(w_bar, horizontal)

        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        zero, one = np.zeros_like(ct), np.ones_like(ct)

        yaw = np.stack([np.stack([ct, st, zero], axis=-1),
                        np.stack([-st, ct, zero], axis=-1),
                        np.stack([zero, zero, one], axis=-1)], axis=-2)
        pitch = np.stack([np.stack([cp, zero, sp], axis=-1),
                          np.stack([zero, one, zero], axis=-1),
                          np.stack([-sp, zero, cp], axis=-1)], axis=-2)

        return pitch @ yaw

    def fluxes(self, *, min_samples: int = 2) -> dict[str, np.ndarray]:
        '''
        Compute friction velocity (u*), kinematic heat flux (w*theta) and Obukhov length (L) in each bin.

        Fluxes follow the eddy-covariance convention of using a double-rotated frame, 
        where the mean wind is aligned with the x-axis and the mean vertical wind is zero.

        Parameters
        ----------
        min_samples : int, optional
            The fewest valid samples a bin may hold. Bins with fewer are returned as NaN.
            Defaults to 2.

        Returns
        -------
        dict of str to np.ndarray
            A dictionary mapping variable names to their corresponding arrays. 
            Each has the same shape as the accumulator. The keys are:
            - `u_star`: friction velocity
            - `w_theta`: kinematic heat flux
            - `obukhov_length`: Obukhov length
            - `mean_speed`: mean wind speed
            - `mean_direction`: mean wind direction (meteorological, in degrees)
            - `uw` and `vw`: rotated covariances
            - `n_samples`: valid sample count
            
        '''
        n = self.sums['n']
        enough = n >= min_samples
        safe_n = np.where(enough, n, np.nan)

        u_bar, v_bar, w_bar, t_bar = (self.sums[k] / safe_n for k in ('u', 'v', 'w', 't'))

        def cov(key, a, b):
            return self.sums[key] / safe_n - a * b

        covariance = np.stack([
            np.stack([cov('uu', u_bar, u_bar), cov('uv', u_bar, v_bar), cov('uw', u_bar, w_bar)], axis=-1),
            np.stack([cov('uv', u_bar, v_bar), cov('vv', v_bar, v_bar), cov('vw', v_bar, w_bar)], axis=-1),
            np.stack([cov('uw', u_bar, w_bar), cov('vw', v_bar, w_bar), cov('ww', w_bar, w_bar)], axis=-1),
        ], axis=-2)
        flux_t = np.stack([cov('ut', u_bar, t_bar), cov('vt', v_bar, t_bar), cov('wt', w_bar, t_bar)], axis=-1)

        rotation = self._rotation_matrices(u_bar, v_bar, w_bar)
        rotated = rotation @ covariance @ np.swapaxes(rotation, -1, -2)
        rotated_t = np.einsum('...ij,...j->...i', rotation, flux_t)  # matrix-vector product, equivalent to `@` but with broadcasting - required due to the shape of `flux_t` being (..., 3) rather than (..., 3, 1)

        uw, vw = rotated[..., 0, 2], rotated[..., 1, 2]
        w_theta = rotated_t[..., 2]

        u_star = (uw ** 2 + vw ** 2) ** 0.25

        # L is undefined in exactly neutral conditions, where the heat flux vanishes
        with np.errstate(divide='ignore', invalid='ignore'):
            obukhov = -(u_star ** 3) * (t_bar + 273.15) / (VON_KARMAN * GRAVITY * w_theta)
            obukhov = np.where(w_theta == 0, np.inf, obukhov)

        return {
            'u_star': np.where(enough, u_star, np.nan),
            'w_theta': np.where(enough, w_theta, np.nan),
            'obukhov_length': np.where(enough, obukhov, np.nan),
            'mean_speed': np.where(enough, np.hypot(u_bar, v_bar), np.nan),
            'mean_direction': np.where(enough, np.degrees(np.arctan2(-u_bar, -v_bar)) % 360, np.nan),
            'uw': np.where(enough, uw, np.nan),
            'vw': np.where(enough, vw, np.nan),
            'n_samples': n,
        }

    def __repr__(self):
        return f'{type(self).__name__}(shape={self.shape}, samples={self.sums["n"].sum():.0f})'


class WindProfile:
    '''
    Base class for vertical wind-speed extrapolation.

    A profile raises a wind speed measured at one height to another, given whatever state that
    method needs (friction velocity and Obukhov length for the log law, a shear exponent for the
    power law). Subclasses are registered in `PROFILES` and resolved in other classes/fns 
    by `get_profile`.

    Methods
    -------
    speed_at(height, ...)
        The wind speed at `height`.
    validity(height, ...)
        Per-period diagnostics and a boolean flag for whether the method's assumptions hold.
    '''

    name = 'profile'

    def speed_at(
            self, height: float, *,
            reference_height: float, reference_speed: np.ndarray, **state
    ) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not implement speed_at().")

    def validity(self, height: float, **state) -> dict[str, np.ndarray]:
        return {}

    def __repr__(self):
        return f'{type(self).__name__}(name={self.name!r})'


class LogLawProfile(WindProfile):
    '''
    Monin-Obukhov similarity as a difference from the measurement height. Uses the differenced form of the LogLL 
    profile, allowing for the calculation of wind speed at a target height without use of the roughness length z_0:

        U(z) = U_r + (u* / k) * [ ln((z - d) / (z_r - d)) - psi_m((z - d) / L) + psi_m((z_r - d) / L) ]

    The roughness length is still available from `roughness_length`, and can be used as a diagnostic - 
    if the roughness length is not well-defined or is implausible (e.g., a negative value), it may 
    indicate that the data is unreliable.

    The underlying similarity theory is only valid for |zeta| <= 2, where zeta = (z - d) / L.
    This condition is checked by the `validity` method. The upper/lower bound can be changed
    by adjusting the `zeta_max` parameter.
    Note that the log law is still applied even when the period is outside the valid range.


    Parameters
    ----------
    displacement_height : float, optional
        Zero-plane displacement d, in metres. Defaults to 0.
    zeta_max : float, optional
        The largest |zeta| considered valid. Defaults to 2.
    '''

    name = 'log-law'

    def __init__(self, *, displacement_height: float = 0.0, zeta_max: float = 2.0):
        self.displacement_height = float(displacement_height)
        self.zeta_max = float(zeta_max)

    def _zeta(self, height: float, obukhov_length: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return (height - self.displacement_height) / np.asarray(obukhov_length, dtype=np.float64)

    def speed_at(
            self, height: float, *,
            reference_height: float, reference_speed: np.ndarray,
            u_star: np.ndarray, obukhov_length: np.ndarray, **state
    ) -> np.ndarray:
        '''
        The wind speed at `height`.

        Parameters
        ----------
        height : float
            Target height above ground, in metres.
        reference_height : float
            The height at which `reference_speed` was measured, in metres.
        reference_speed : np.ndarray
            The measured wind speed.
        u_star : np.ndarray
            Friction velocity.
        obukhov_length : np.ndarray
            Obukhov length, in metres.

        Returns
        -------
        np.ndarray
        '''
        d = self.displacement_height
        if height <= d or reference_height <= d:
            raise ValueError(
                f"Both heights must sit above the displacement height ({d} m), "
                f"got height={height} and reference_height={reference_height}."
            )

        shear = np.log((height - d) / (reference_height - d))
        correction = (psi_m(self._zeta(reference_height, obukhov_length))
                      - psi_m(self._zeta(height, obukhov_length)))

        return reference_speed + np.asarray(u_star) / VON_KARMAN * (shear + correction)

    def roughness_length(
            self, *, reference_height: float, reference_speed: np.ndarray,
            u_star: np.ndarray, obukhov_length: np.ndarray
    ) -> np.ndarray:
        '''
        The roughness length implied by the measured wind.

        The psi_m(z0 / L) term is dropped, as it is of order 1e-3 for a crop canopy.

        Returns
        -------
        np.ndarray
        '''
        d = self.displacement_height
        with np.errstate(divide='ignore', invalid='ignore'):
            exponent = (-VON_KARMAN * np.asarray(reference_speed) / np.asarray(u_star)
                        + psi_m(self._zeta(reference_height, obukhov_length)))
        return (reference_height - d) * np.exp(exponent)

    def validity(self, height: float, *, obukhov_length: np.ndarray) -> dict[str, np.ndarray]:
        '''
        The stability parameter zeta at the given height, 
        and whether similarity theory holds there.

        Parameters
        ----------
        height : float
            The height at which to evaluate the stability parameter, in metres.
        obukhov_length : np.ndarray
            The Obukhov length values, in metres.

        Returns
        -------
        dict of str to np.ndarray
            Outputs have the same shape as `obukhov_length`. Keys are:
            - `zeta`: the stability parameter at the given height, in metres.
            - `most_valid`: a boolean array indicating whether the Monin-Obukhov Similarity Theory (MOST) is 
            valid at the given height, based on the `zeta_max` threshold.
        '''
        zeta = self._zeta(height, obukhov_length)
        return {'zeta': zeta, 'most_valid': np.abs(zeta) <= self.zeta_max}


class PowerLawProfile(WindProfile):
    '''
    The engineering power law, U(z) = U_r * ((z - d) / (z_r - d)) ** alpha.

    Carries no validity information. 
    , so it is offered as an alternative rather than a default.
    Fitting `alpha` from the 4.5 m sonic and 10 m vane of a CWEX tower does not work - the lever
    arm is far too short, and the fitted exponent reaches 0.9 at the 95th percentile, which sends
    hub-height speeds to implausible values. A fixed exponent is bounded, if crude.

    Parameters
    ----------
    alpha : float or np.ndarray, optional
        The shear exponent, either fixed or one value per period. Defaults to 1/7, the
        conventional neutral value over open land.
    displacement_height : float, optional
        Zero-plane displacement d, in metres. Defaults to 0.
    '''

    name = 'power-law'

    def __init__(self, *, alpha: float|np.ndarray = 1 / 7, displacement_height: float = 0.0):
        self.alpha = alpha
        self.displacement_height = float(displacement_height)

    def speed_at(
            self, height: float, *,
            reference_height: float, reference_speed: np.ndarray, alpha: np.ndarray|None = None
    ) -> np.ndarray:
        '''
        The wind speed at the given height.

        Parameters
        ----------
        height : float
            Target height above ground, in metres.
        reference_height : float
            The height at which `reference_speed` was measured, in metres.
        reference_speed : np.ndarray
            The measured wind speed.
        alpha : np.ndarray or None, optional
            A per-period shear exponent, overriding the one given to the constructor.

        Returns
        -------
        np.ndarray
        '''
        d = self.displacement_height
        if height <= d or reference_height <= d:
            raise ValueError(
                f"Both heights must sit above the displacement height ({d} m), "
                f"got height={height} and reference_height={reference_height}."
            )

        exponent = self.alpha if alpha is None else alpha
        return reference_speed * ((height - d) / (reference_height - d)) ** exponent


PROFILES: dict[str, type[WindProfile]] = {
    'log-law': LogLawProfile,
    'logll': LogLawProfile,
    'log': LogLawProfile,
    'power-law': PowerLawProfile,
    'power': PowerLawProfile,
}


def get_profile(spec: str|WindProfile|type[WindProfile], **kwargs) -> WindProfile:
    '''
    Resolve a profile given as a name, a class or an instance.

    Parameters
    ----------
    spec : str or WindProfile or type
        A key of `PROFILES` (case-insensitive, underscores and spaces accepted for hyphens), a
        `WindProfile` subclass, or a configured instance, which is returned unchanged.
    **kwargs
        Passed to the constructor when `spec` names a class. Arguments the class does not accept
        are dropped, so a caller can offer the union of what every profile might want.

    Returns
    -------
    WindProfile
    '''
    if isinstance(spec, WindProfile) and issubclass(type(spec), WindProfile):
        return spec

    if isinstance(spec, type) and issubclass(spec, WindProfile):
        cls = spec
    else:
        key = str(spec).lower().replace('_', '-').replace(' ', '-')
        if key not in PROFILES:
            raise ValueError(
                f"Invalid profile '{spec}'. Must be one of {sorted(set(PROFILES))}, "
                "or a WindProfile subclass or instance."
            )
        cls = PROFILES[key]

    accepted = cls.__init__.__code__.co_varnames[:cls.__init__.__code__.co_argcount + cls.__init__.__code__.co_kwonlyargcount]
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})
