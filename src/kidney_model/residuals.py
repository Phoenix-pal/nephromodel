"""Implicit residual assembled from the model conservation equations."""

import numpy as np

from .constants import COMPARTMENTS, KA, KC, KD, K0, SALT, UREA
from .parameters import ModelParameters
from .state import geometric_state_size, pack_residual_state, unpack_state
from .transport import DCT_junction, JunctionSolveError, p_junction, solve_DCT_junction, solute_res, water_res


def pressure_res(k, alpha, pressure, p: ModelParameters):
    if k == K0:
        result = alpha[K0] / p.alpha_star - 1.0
        for j in (KD, KA, KC):
            result += p.alpha_bar[j] * (1.0 + p.nu[j] * (pressure[j] - pressure[K0])) / p.alpha_star
        return result
    return pressure[k] - pressure[K0] - (alpha[k] / p.alpha_bar[k] - 1.0) / p.nu[k]


def implicit_residual(y_1, y_0, p: ModelParameters, PCT_flow=None, p_vas=None, p_pap=None):
    alpha_1, c_1, pressure = unpack_state(y_1, p)
    alpha_0, c_0, _ = unpack_state(y_0, p)
    if (
        np.any(~np.isfinite(y_1))
        or np.min(alpha_1) <= 0
        or np.min(c_1) <= 0
    ):
        return np.ones(geometric_state_size(p)) * 1e6

    try:
        dct_junc = solve_DCT_junction(alpha_1, c_1, pressure, p)
    except JunctionSolveError:
        return np.ones(geometric_state_size(p)) * 1e6
    r_alpha = np.zeros((4, p.N))
    r_c = np.zeros((2, 4, p.N))
    r_pressure = np.zeros((4, p.N))
    for k in COMPARTMENTS:
        r_alpha[k] = water_res(k, alpha_1, alpha_0, c_1, pressure, dct_junc[2], p, PCT_flow, p_vas, p_pap)
        for i in (SALT, UREA):
            r_c[i, k] = solute_res(i, k, alpha_1, alpha_0, c_1, c_0, pressure, dct_junc, p, PCT_flow, p_vas, p_pap)
        r_pressure[k] = pressure_res(k, alpha_1, pressure, p)

    # One stored concentration represents the shared D-A tip.
    r_c[:, KD, -1] += r_c[:, KA, -1]
    return pack_residual_state(r_alpha, r_c, r_pressure, p)
