__all__ = ['CovarianceAccumulator']

import numpy as np

from eddy.util import VON_KARMAN, GRAVITY


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
        n_trailing = int(np.prod(self.shape[1:]))
        flat_index = (np.asarray(bin_index).reshape(-1, 1) * n_trailing + np.arange(n_trailing)).ravel()
        term_shape = np.shape(bin_index) + self.shape[1:]
        for key, term in terms.items():
            self.sums[key] += np.bincount(
                flat_index, weights=np.broadcast_to(term, term_shape).ravel(), minlength=self.sums[key].size
            ).reshape(self.shape)

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
