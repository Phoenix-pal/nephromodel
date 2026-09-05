"""Construction of dimensional and nondimensional model parameters."""

from dataclasses import dataclass

import numpy as np

from .constants import KA, KC, KD, K0, SALT, UREA
from .nondimensional import ReferenceScales


@dataclass
class ModelParameters:
    """Nondimensional numerical inputs for one model instance.

    ``dt`` and ``L`` are nondimensional time and length.  Concentrations,
    pressures, axial fluxes, and all transport coefficients stored here are
    also nondimensional.  ``scales`` is retained solely for explicit
    conversion at I/O boundaries and is never used by the model equations.
    """

    N: int
    dt: float
    dx: float
    L: float
    scales: ReferenceScales
    length_scale_dimensional: float
    tau_dimensional: float
    pressure_scale_dimensional: float
    flow_scale_dimensional: float
    rho: np.ndarray
    D: np.ndarray
    pump_strength: float
    pump_half_saturation: float
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
    plasma_salt: float
    plasma_urea: float
    plasma_osmolarity: float
    legacy_concentration_scale: float
    collecting_water_permeable: bool = True


def build_parameters(
    N: int = 70,
    dt: float = 10.0,
    collecting_water_permeable: bool = True,
    scales: ReferenceScales | None = None,
) -> ModelParameters:
    """Build the scaled parameter set used internally by the solver."""
    N = int(N)
    if N < 2:
        raise ValueError("N must be at least 2")

    scales = ReferenceScales() if scales is None else scales
    dx = 1.0 / N
    L = 1.0
    length_scale = scales.length

    rad_0, rad_D, rad_A, rad_C = 0.0025, 0.0008, 0.001, 0.0012
    area_0 = np.pi * rad_0**2
    area_D = np.pi * rad_D**2
    area_A = np.pi * rad_A**2
    area_C = np.pi * rad_C**2
    area_tot = area_0 + area_D + area_A + area_C
    circ_D, circ_A, circ_C = 2 * np.pi * rad_D, 2 * np.pi * rad_A, 2 * np.pi * rad_C

    pressure_scale = scales.pressure
    hydr_resist = 8 * 0.6915 * np.pi**2 * (0.0075 * 10**-3)
    rho_interst = 8 * 3.5 * np.pi**2 * (0.0075 * 10**-3)
    time_scale = hydr_resist * length_scale**2 / (area_tot * pressure_scale)

    F_PCT_dim = 0.167e-6
    P_v_dim, P_p_dim = 0.0, 6.4
    D_dim = np.zeros((2, 4))
    D_dim[SALT, K0], D_dim[UREA, K0] = 0.00025, 0.0002
    D_dim[SALT, 1:], D_dim[UREA, 1:] = 1.5e-5, 1.5e-5
    pump_strength_dim = 14.2e-6

    rho = np.ones(4)
    rho[K0] = rho_interst / hydr_resist
    D = D_dim * time_scale / length_scale**2
    pump_strength = pump_strength_dim * circ_A * time_scale / (area_tot * scales.concentration)
    nu = np.ones(4) / 10.0
    alpha_bar = np.array([area_0, area_D, area_A, area_C]) / area_tot

    alpha_star = 1.0
    colloid = 0.01
    q = 1.0 / 3.0
    F_PCT = F_PCT_dim * time_scale / (area_tot * length_scale)
    P_v, P_p = scales.pressure_hat(P_v_dim), scales.pressure_hat(P_p_dim)
    plasma_salt = float(scales.concentration_hat(scales.plasma_salt_mmol_L))
    plasma_urea = float(scales.concentration_hat(scales.plasma_urea_mmol_L))
    filtrates = np.array([plasma_salt, plasma_urea])
    c_cortex = 2.0 * filtrates[SALT] + filtrates[UREA]
    legacy_concentration_scale = float(
        scales.concentration_hat(scales.legacy_source_concentration_mmol_L)
    )
    pump_half_saturation = float(scales.concentration_hat(150.0))

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

    zeta[0] *= circ_D * pressure_scale * time_scale / area_tot
    zeta[1] *= circ_A * pressure_scale * time_scale / area_tot
    zeta[2] *= circ_C * pressure_scale * time_scale / area_tot
    gamma[:, 0] *= circ_D * time_scale / area_tot
    gamma[:, 1] *= circ_A * time_scale / area_tot
    gamma[:, 2] *= circ_C * time_scale / area_tot

    return ModelParameters(
        N=N, dt=float(dt), dx=dx, L=L, scales=scales,
        length_scale_dimensional=length_scale,
        tau_dimensional=time_scale,
        pressure_scale_dimensional=pressure_scale,
        flow_scale_dimensional=area_tot * length_scale / time_scale,
        rho=rho, D=D, pump_strength=pump_strength,
        pump_half_saturation=pump_half_saturation,
        nu=nu, alpha_bar=alpha_bar,
        alpha_star=alpha_star, colloid=colloid, q=q, F_PCT=F_PCT,
        P_v=P_v, P_p=P_p, filtrates=filtrates, c_cortex=c_cortex,
        zeta=zeta, gamma=gamma,
        plasma_salt=plasma_salt, plasma_urea=plasma_urea,
        plasma_osmolarity=c_cortex,
        legacy_concentration_scale=legacy_concentration_scale,
        collecting_water_permeable=collecting_water_permeable,
    )
