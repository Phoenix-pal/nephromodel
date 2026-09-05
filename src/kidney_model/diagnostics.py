"""Model diagnostics that are useful in scripts and tests."""

import numpy as np

from .constants import KA, KD, SALT, UREA
from .parameters import ModelParameters
from .state import unpack_state
from .transport import DCT_junction, solve_DCT_junction, solute_flow, water_flow


def continuity_and_mass(y_state, p: ModelParameters):
    alpha, c, pressure = unpack_state(y_state, p)
    dct = solve_DCT_junction(alpha, c, pressure, p)
    return {
        "tip_concentration_error_salt": float(c[SALT, KA, -1] - c[SALT, KD, -1]),
        "tip_concentration_error_urea": float(c[UREA, KA, -1] - c[UREA, KD, -1]),
        "tip_water_flux_error": float(water_flow(KD, alpha, c, pressure, dct[2], p)[-1] - water_flow(KA, alpha, c, pressure, dct[2], p)[-1]),
        "tip_salt_flux_error": float(solute_flow(SALT, KD, c, alpha, pressure, dct, p)[-1] - solute_flow(SALT, KA, c, alpha, pressure, dct, p)[-1]),
        "tip_urea_flux_error": float(solute_flow(UREA, KD, c, alpha, pressure, dct, p)[-1] - solute_flow(UREA, KA, c, alpha, pressure, dct, p)[-1]),
        "junction_residual_Linf": float(np.max(np.abs(DCT_junction(dct, alpha, c, pressure, p)))),
        "salt_mass": float(np.sum(alpha * c[SALT]) * p.dx),
        "urea_mass": float(np.sum(alpha * c[UREA]) * p.dx),
    }


def state_summary(y_state, p: ModelParameters):
    alpha, c, pressure = unpack_state(y_state, p)
    return {
        "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
        "salt_min": float(c[SALT].min()), "salt_max": float(c[SALT].max()),
        "urea_min": float(c[UREA].min()), "urea_max": float(c[UREA].max()),
        "pressure_min": float(pressure.min()), "pressure_max": float(pressure.max()),
        "osmolarity_max": float((2 * c[SALT] + c[UREA]).max()),
    }
