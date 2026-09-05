"""Finite-difference operators and constitutive relations."""

import numpy as np

from .constants import KA, KC, KD, K0, SALT, UREA
from .parameters import ModelParameters


def diff_x(y: np.ndarray, p: ModelParameters) -> np.ndarray:
    return (y[1:] - y[:-1]) / p.dx


def diff_D(y: np.ndarray, p: ModelParameters) -> np.ndarray:
    return diff_x(y, p)


def diff_0(y: np.ndarray, p: ModelParameters) -> np.ndarray:
    return diff_x(y, p)


def diff_C(y: np.ndarray, p: ModelParameters) -> np.ndarray:
    return diff_x(y, p)


def diff_A(y: np.ndarray, p: ModelParameters) -> np.ndarray:
    """Ascending-limb gradient; its physical direction is tip to cortex."""
    return (y[:-1] - y[1:]) / p.dx


def grad(k: int, y: np.ndarray, p: ModelParameters) -> np.ndarray:
    if k == KA:
        return diff_A(y, p)
    if k in (K0, KD, KC):
        return diff_x(y, p)
    raise ValueError(f"Unknown compartment k={k}")


def div(k: int, f: np.ndarray, p: ModelParameters) -> np.ndarray:
    if k == KA:
        return (f[:-1] - f[1:]) / p.dx
    return (f[1:] - f[:-1]) / p.dx


def diff_t(y_1: np.ndarray, y_0: np.ndarray, p: ModelParameters) -> np.ndarray:
    return (y_1 - y_0) / p.dt


def average(y: np.ndarray) -> np.ndarray:
    return 0.5 * (y[1:] + y[:-1])


def average_geom(k: int, y: np.ndarray) -> np.ndarray:
    if k == KA:
        return average(y[::-1])[::-1]
    return average(y)


def osmotic_pressure(k: int, c: np.ndarray, p: ModelParameters) -> np.ndarray:
    value = 2.0 * c[SALT, k, :] + c[UREA, k, :]
    return value + p.colloid if k == K0 else value


def psi(k: int, c: np.ndarray, pressure: np.ndarray, p: ModelParameters) -> np.ndarray:
    return pressure[k, :] - osmotic_pressure(k, c, p)


def chemical_potential(i: int, k: int, c: np.ndarray) -> np.ndarray:
    return c[i, k, :]


def AHL_pump(c_salt: np.ndarray, p: ModelParameters) -> np.ndarray:
    pump = np.zeros(p.N)
    mm = 0.15 / p.c_star
    for l in range(round(0.4 * p.N)):
        pump[l] = p.pump_strength / (1.0 + mm / max(c_salt[l], 1e-12))
    return pump
