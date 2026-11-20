__all__ = [
    'psi_m',
    'WindProfile', 'LogLawProfile', 'PowerLawProfile', 'PROFILES', 'get_profile'
]

import numpy as np

from eddy.util import VON_KARMAN


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
