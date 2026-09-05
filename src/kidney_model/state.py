"""Packing, unpacking, and construction of the geometric state vector."""

import numpy as np
from scipy.optimize import fsolve

from .constants import COMPARTMENTS, KA, KC, KD, K0, SALT, UREA
from .parameters import ModelParameters


def geometric_state_size(p: ModelParameters) -> int:
    return 16 * p.N - 2


def _pack_c_geometric(c_full, p, enforce_tip=True):
    c_work = np.array(c_full, copy=True)
    if enforce_tip:
        c_work[:, KA, -1] = c_work[:, KD, -1]
    parts = []
    for i in (SALT, UREA):
        parts.extend((c_work[i, K0], c_work[i, KD], c_work[i, KA, : p.N - 1], c_work[i, KC]))
    return np.concatenate(parts)


def _unpack_c_geometric(vec, p):
    c = np.zeros((2, 4, p.N))
    pos = 0
    for i in (SALT, UREA):
        c[i, K0] = vec[pos:pos + p.N]; pos += p.N
        c[i, KD] = vec[pos:pos + p.N]; pos += p.N
        c[i, KA, : p.N - 1] = vec[pos:pos + p.N - 1]; pos += p.N - 1
        c[i, KC] = vec[pos:pos + p.N]; pos += p.N
        c[i, KA, -1] = c[i, KD, -1]
    if pos != len(vec):
        raise ValueError(f"Unused concentration entries: pos={pos}, len={len(vec)}")
    return c


def unpack_state(y, p: ModelParameters):
    y = np.asarray(y)
    geom_len = geometric_state_size(p)
    if len(y) == geom_len:
        alpha = y[:4 * p.N].reshape(4, p.N)
        c_end = 4 * p.N + (8 * p.N - 2)
        c = _unpack_c_geometric(y[4 * p.N:c_end], p)
        pressure = y[c_end:].reshape(4, p.N)
        return alpha, c, pressure
    if len(y) == 16 * p.N:
        alpha = y[:4 * p.N].reshape(4, p.N)
        c = y[4 * p.N:12 * p.N].reshape(2, 4, p.N).copy()
        c[:, KA, -1] = c[:, KD, -1]
        pressure = y[12 * p.N:].reshape(4, p.N)
        return alpha, c, pressure
    raise ValueError(f"State length {len(y)} incompatible with N={p.N}")


def pack_state(alpha, c, pressure, p: ModelParameters):
    return np.concatenate([alpha.ravel(), _pack_c_geometric(c, p), pressure.ravel()])


def pack_residual_state(r_alpha, r_c, r_pressure, p: ModelParameters):
    return np.concatenate([r_alpha.ravel(), _pack_c_geometric(r_c, p, enforce_tip=False), r_pressure.ravel()])


def make_initial_state(p: ModelParameters):
    """Create the native baseline state and solve its algebraic volume profile."""
    from .residuals import pressure_res

    alpha = np.zeros((4, p.N))
    c = np.zeros((2, 4, p.N))
    pressure = np.zeros((4, p.N))
    for i in (SALT, UREA):
        c[i] = p.filtrates[i]
    c[:, KA, -1] = c[:, KD, -1]
    for k in COMPARTMENTS:
        alpha[k] = p.alpha_bar[k]
        pressure[k] = p.P_v if k == K0 else p.P_p

    def solve_alpha(alpha_vec):
        a = alpha_vec.reshape(4, p.N)
        return np.concatenate([pressure_res(k, a, pressure, p) for k in COMPARTMENTS])

    alpha = fsolve(solve_alpha, alpha.ravel()) .reshape(4, p.N)
    return pack_state(alpha, c, pressure, p)
