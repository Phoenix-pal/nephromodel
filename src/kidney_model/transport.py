"""Water and solute fluxes, junction equations, and conservation residuals."""

import numpy as np
from scipy.optimize import fsolve

from .constants import KA, KC, KD, K0, SALT, UREA, COMPARTMENTS, SOLUTES
from .operators import AHL_pump, average_geom, chemical_potential, diff_t, div, grad, psi
from .parameters import ModelParameters


def water_trans(k, c, pressure, p):
    return p.zeta[k - 1] * (psi(k, c, pressure, p) - psi(K0, c, pressure, p))


def p_junction(k, alpha, c, pressure, p):
    if k == KD:
        n = -1
        return (
            alpha[k, n] ** 2 * pressure[k, n] / p.rho[k]
            + alpha[k + 1, n] ** 2 * pressure[k + 1, n] / p.rho[k + 1]
        ) / (alpha[k, n] ** 2 / p.rho[k] + alpha[k + 1, n] ** 2 / p.rho[k + 1])
    if k == KA:
        n = 0
        weight_a = 2 * p.q * c[SALT, k, n] + c[UREA, k, n]
        return (
            alpha[k, n] ** 2 * weight_a * pressure[k, n] / p.rho[k]
            + alpha[k + 1, n] ** 2 * p.c_cortex * pressure[k + 1, n] / p.rho[k + 1]
        ) / (
            alpha[k, n] ** 2 * weight_a / p.rho[k]
            + alpha[k + 1, n] ** 2 * p.c_cortex / p.rho[k + 1]
        )
    raise ValueError("k must be KD or KA for p_junction")


def DCT_junction(y, alpha, c, pressure, p):
    c_s, c_u, p_ac = y
    res = np.zeros(3)
    res[0] = (
        p.q * (alpha[KA, 0] ** 2 / p.rho[KA]) * (p_ac - pressure[KA, 0]) * c[SALT, KA, 0]
        + (alpha[KC, 0] ** 2 / p.rho[KC]) * (p_ac - pressure[KC, 0]) * c_s
        + alpha[KC, 0] * p.D[SALT, KC] * (c_s - c[SALT, KC, 0])
    )
    res[1] = (
        (alpha[KA, 0] ** 2 / p.rho[KA]) * (p_ac - pressure[KA, 0]) * c[UREA, KA, 0]
        + (alpha[KC, 0] ** 2 / p.rho[KC]) * (p_ac - pressure[KC, 0]) * c_u
        + alpha[KC, 0] * p.D[UREA, KC] * (c_u - c[UREA, KC, 0])
    )
    res[2] = 2.0 * c_s + c_u - p.c_cortex
    return res


def solve_DCT_junction(alpha, c, pressure, p):
    guess = np.array([c[SALT, KC, 0], c[UREA, KC, 0], p_junction(KA, alpha, c, pressure, p)])
    return fsolve(DCT_junction, guess, args=(alpha, c, pressure, p))


def water_flow(k, alpha, c, pressure, p_ac, p: ModelParameters,
               PCT_flow=None, p_vas=None, p_pap=None):
    PCT_flow = p.F_PCT if PCT_flow is None else PCT_flow
    p_vas = p.P_v if p_vas is None else p_vas
    p_pap = p.P_p if p_pap is None else p_pap
    au = np.zeros(p.N + 1)
    au[1:-1] = -(average_geom(k, alpha[k]) ** 2) * grad(k, pressure[k], p) / p.rho[k]
    if k == K0:
        au[0] = 2.0 * (alpha[k, 0] ** 2 / p.rho[k]) * (p_vas - pressure[k, 0]) / p.dx
    elif k == KD:
        au[0] = PCT_flow
        p_da = p_junction(KD, alpha, c, pressure, p)
        au[-1] = 2.0 * (alpha[k, -1] ** 2 / p.rho[k]) * (pressure[k, -1] - p_da) / p.dx
    elif k == KA:
        p_da = p_junction(KD, alpha, c, pressure, p)
        au[0] = 2.0 * (alpha[k, 0] ** 2 / p.rho[k]) * (pressure[k, 0] - p_ac) / p.dx
        au[-1] = 2.0 * (alpha[k, -1] ** 2 / p.rho[k]) * (p_da - pressure[k, -1]) / p.dx
    elif k == KC:
        au[0] = 2.0 * (alpha[k, 0] ** 2 / p.rho[k]) * (p_ac - pressure[k, 0]) / p.dx
        au[-1] = 2.0 * (alpha[k, -1] ** 2 / p.rho[k]) * (pressure[k, -1] - p_pap) / p.dx
    else:
        raise ValueError(f"Unknown compartment k={k}")
    if k == K0:
        au[-1] = 0.0
    return au


def water_res(k, alpha_1, alpha_0, c, pressure, p_ac, p: ModelParameters,
              PCT_flow=None, p_vas=None, p_pap=None):
    w = sum((water_trans(j, c, pressure, p) for j in (KD, KA, KC)), np.zeros(p.N)) if k == K0 else -water_trans(k, c, pressure, p)
    return diff_t(alpha_1[k], alpha_0[k], p) + div(k, water_flow(k, alpha_1, c, pressure, p_ac, p, PCT_flow, p_vas, p_pap), p) - w


def solute_trans(i, k, c, p: ModelParameters):
    flux = p.gamma[i, k - 1] * (chemical_potential(i, k, c) - chemical_potential(i, K0, c))
    pump = AHL_pump(c[SALT, KA], p) if k == KA and i == SALT else 0.0
    return flux + pump


def _upwind_concentration_faces(i, k, c, flow, dct_junc, p):
    face = np.zeros(p.N + 1)
    ck = c[i, k]
    for j in range(1, p.N):
        face[j] = (ck[j] if flow[j] >= 0 else ck[j - 1]) if k == KA else (ck[j - 1] if flow[j] >= 0 else ck[j])
    if k == K0:
        face[0] = p.filtrates[i] if flow[0] >= 0 else c[i, K0, 0]
        face[-1] = c[i, K0, -1]
    elif k == KD:
        face[0] = p.filtrates[i] if flow[0] >= 0 else c[i, KD, 0]
        face[-1] = c[i, KD, -1] if flow[-1] >= 0 else c[i, KD, -1]
    elif k == KA:
        face[0] = c[i, KA, 0] if flow[0] >= 0 else dct_junc[i]
        face[-1] = c[i, KD, -1] if flow[-1] >= 0 else c[i, KA, -1]
    elif k == KC:
        face[0] = dct_junc[i] if flow[0] >= 0 else c[i, KC, 0]
        face[-1] = c[i, KC, -1]
    else:
        raise ValueError(f"Unknown compartment k={k}")
    return face


def solute_flow(i, k, c, alpha, pressure, dct_junc, p: ModelParameters,
                PCT_flow=None, p_vas=None, p_pap=None):
    flow = water_flow(k, alpha, c, pressure, dct_junc[2], p, PCT_flow, p_vas, p_pap)
    conc_face = _upwind_concentration_faces(i, k, c, flow, dct_junc, p)
    diffusion = np.zeros(p.N + 1)
    diffusion[1:-1] = -p.D[i, k] * average_geom(k, alpha[k]) * grad(k, c[i, k], p)
    if k == K0:
        diffusion[0] = 2.0 * p.D[i, k] * alpha[k, 0] * (p.filtrates[i] - c[i, k, 0]) / p.dx
    elif k == KD:
        diffusion[0] = 2.0 * p.D[i, k] * alpha[k, 0] * (p.filtrates[i] - c[i, k, 0]) / p.dx
        diffusion[-1] = 0.0
    elif k == KA:
        diffusion[0] = 0.0
        diffusion[-1] = 2.0 * p.D[i, k] * alpha[k, -1] * (c[i, KD, -1] - c[i, KA, -1]) / p.dx
    elif k == KC:
        diffusion[0] = 2.0 * p.D[i, k] * alpha[k, 0] * (dct_junc[i] - c[i, k, 0]) / p.dx
    return diffusion + flow * conc_face


def solute_res(i, k, alpha_1, alpha_0, c_1, c_0, pressure, dct_junc, p: ModelParameters,
               PCT_flow=None, p_vas=None, p_pap=None):
    storage = diff_t(alpha_1[k] * c_1[i, k], alpha_0[k] * c_0[i, k], p)
    flux_div = div(k, solute_flow(i, k, c_1, alpha_1, pressure, dct_junc, p, PCT_flow, p_vas, p_pap), p)
    exchange = sum((solute_trans(i, j, c_1, p) for j in (KD, KA, KC)), np.zeros(p.N)) if k == K0 else -solute_trans(i, k, c_1, p)
    return storage + flux_div - exchange
