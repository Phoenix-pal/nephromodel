"""Construction of dimensional and nondimensional model parameters."""

from dataclasses import dataclass

import numpy as np

from .constants import KA, KC, KD, K0, SALT, UREA


@dataclass
class ModelParameters:
    """All numerical inputs and derived coefficients for one model instance."""

    N: int
    dt: float
    dx: float
    L: float
    c_star: float
    RT: float
    pressure: float
    tau: float
    area_tot: float
    circ_D: float
    circ_A: float
    circ_C: float
    rho: np.ndarray
    D: np.ndarray
    pump_strength: float
    nu: np.ndarray
    alpha_bar: np.ndarray
    alpha_star: float
    colloid: float
    q: float
    F_PCT: float
    P_v: float
    P_p: float
    filtrates: np.ndarray
    c_cortex: float
    zeta: np.ndarray
    gamma: np.ndarray
    collecting_water_permeable: bool = True


def build_parameters(
    N: int = 70,
    dt: float = 10.0,
    collecting_water_permeable: bool = True,
) -> ModelParameters:
    """Build the scaled parameter set used internally by the solver."""
    N = int(N)
    if N < 2:
        raise ValueError("N must be at least 2")

    dx = 1.0 / N
    L = 1.0
    c_star = 0.001
    RT = 19344.0

    rad_0, rad_D, rad_A, rad_C = 0.0025, 0.0008, 0.001, 0.0012
    area_0 = np.pi * rad_0**2
    area_D = np.pi * rad_D**2
    area_A = np.pi * rad_A**2
    area_C = np.pi * rad_C**2
    area_tot = area_0 + area_D + area_A + area_C
    circ_D, circ_A, circ_C = 2 * np.pi * rad_D, 2 * np.pi * rad_A, 2 * np.pi * rad_C

    pressure = c_star * RT
    hydr_resist = 8 * 0.6915 * np.pi**2 * (0.0075 * 10**-3)
    rho_interst = 8 * 3.5 * np.pi**2 * (0.0075 * 10**-3)
    tau = hydr_resist * L**2 / (area_tot * pressure)

    F_PCT_dim = 0.167e-6
    P_v_dim, P_p_dim = 0.0, 6.4
    D_dim = np.zeros((2, 4))
    D_dim[SALT, K0], D_dim[UREA, K0] = 0.00025, 0.0002
    D_dim[SALT, 1:], D_dim[UREA, 1:] = 1.5e-5, 1.5e-5
    pump_strength_dim = 14.2e-6

    rho = np.ones(4)
    rho[K0] = rho_interst / hydr_resist
    D = D_dim * tau / L**2
    pump_strength = pump_strength_dim * circ_A * tau / (area_tot * c_star)
    nu = np.ones(4) / 10.0
    alpha_bar = np.array([area_0, area_D, area_A, area_C]) / area_tot

    alpha_star = 1.0
    colloid = 0.01
    q = 1.0 / 3.0
    F_PCT = F_PCT_dim * tau / (area_tot * L)
    P_v, P_p = P_v_dim / pressure, P_p_dim / pressure
    filtrates = np.array([145.0, 5.0])
    c_cortex = 2.0 * filtrates[SALT] + filtrates[UREA]

    zeta = np.zeros((3, N))
    gamma = np.zeros((2, 3, N))
    for l in range(N):
        zeta[2, l] = 3e-5 / 760.0 if collecting_water_permeable else 0.0
        gamma[SALT, 0, l] = 1.61e-5
        gamma[SALT, 2, l] = 0.04e-5
        gamma[UREA, 0, l] = 1.5e-5

        if l <= round(0.4 * N):
            zeta[0, l] = 17.1e-5 / 760.0
            gamma[SALT, 1, l], gamma[UREA, 1, l] = 6.27e-5, 0.86e-5
        elif l <= round(0.6 * N):
            zeta[0, l] = 25.7e-5 / 760.0
            gamma[SALT, 1, l], gamma[UREA, 1, l] = 26e-5, 6.7e-5
        else:
            zeta[0, l] = 20.4e-5 / 760.0
            gamma[SALT, 1, l], gamma[UREA, 1, l] = 26e-5, 6.7e-5
            gamma[UREA, 2, l] = 1.5e-5

    zeta[0] *= circ_D * pressure * tau / area_tot
    zeta[1] *= circ_A * pressure * tau / area_tot
    zeta[2] *= circ_C * pressure * tau / area_tot
    gamma[:, 0] *= circ_D * tau / area_tot
    gamma[:, 1] *= circ_A * tau / area_tot
    gamma[:, 2] *= circ_C * tau / area_tot

    return ModelParameters(
        N=N, dt=float(dt), dx=dx, L=L, c_star=c_star, RT=RT,
        pressure=pressure, tau=tau, area_tot=area_tot,
        circ_D=circ_D, circ_A=circ_A, circ_C=circ_C, rho=rho, D=D,
        pump_strength=pump_strength, nu=nu, alpha_bar=alpha_bar,
        alpha_star=alpha_star, colloid=colloid, q=q, F_PCT=F_PCT,
        P_v=P_v, P_p=P_p, filtrates=filtrates, c_cortex=c_cortex,
        zeta=zeta, gamma=gamma,
        collecting_water_permeable=collecting_water_permeable,
    )
