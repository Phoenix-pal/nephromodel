"""Model diagnostics that are useful in scripts and tests."""

import numpy as np

from .constants import KA, KC, KD, K0, SALT, UREA
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


def physiology_metrics(y_state, p: ModelParameters, previous_state=None, dt=None):
    """Return urine output, tip continuity, and boundary-accounted balances."""
    alpha, c, pressure = unpack_state(y_state, p)
    dct = solve_DCT_junction(alpha, c, pressure, p)
    water = np.array([water_flow(k, alpha, c, pressure, dct[2], p) for k in range(4)])
    solute = np.array([
        [solute_flow(i, k, c, alpha, pressure, dct, p) for k in range(4)]
        for i in (SALT, UREA)
    ])
    urine_flow = float(water[KC, -1])
    urine_osm = float(2 * c[SALT, KC, -1] + c[UREA, KC, -1])
    result = {
        "urine_flow": urine_flow,
        "urine_osmolarity": urine_osm,
        "urine_plasma_ratio": urine_osm / p.c_cortex,
        "urine_flow_positive": urine_flow > 0,
        "tip_water_flux_error": float(water[KD, -1] - water[KA, -1]),
        "tip_salt_flux_error": float(solute[SALT, KD, -1] - solute[SALT, KA, -1]),
        "tip_urea_flux_error": float(solute[UREA, KD, -1] - solute[UREA, KA, -1]),
        "alpha_sum_error": float(np.max(np.abs(alpha.sum(axis=0) - 1.0))),
        "pressure_min": float(pressure.min()),
        "pressure_max": float(pressure.max()),
    }
    if previous_state is not None:
        if dt is None or dt <= 0:
            raise ValueError("positive dt is required when previous_state is supplied")
        old_alpha, old_c, _ = unpack_state(previous_state, p)
        old_mass = np.sum(old_alpha[None] * old_c, axis=(1, 2)) * p.dx
        new_mass = np.sum(alpha[None] * c, axis=(1, 2)) * p.dx
        water_boundary_net_in = water[K0, 0] + water[KD, 0] - water[KA, 0] + water[KC, 0] - water[KC, -1]
        solute_boundary_net_in = solute[:, K0, 0] + solute[:, KD, 0] - solute[:, KA, 0] + solute[:, KC, 0] - solute[:, KC, -1]
        result["water_balance_error"] = float((alpha.sum() - old_alpha.sum()) * p.dx / dt - water_boundary_net_in)
        result["solute_balance_error"] = ((new_mass - old_mass) / dt - solute_boundary_net_in).tolist()
    return result
