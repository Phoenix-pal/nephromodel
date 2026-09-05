"""Optional matplotlib visualizations for model results."""

import numpy as np

from .constants import COMPARTMENTS, COMP_NAMES, SALT, UREA
from .parameters import ModelParameters
from .state import unpack_state


def plot_state(y_state, p: ModelParameters, normalize_osm_au=True):
    """Plot pressure, volume, solutes, osmolarity, and water flux profiles."""
    import matplotlib.pyplot as plt
    from .transport import solve_DCT_junction, water_flow

    alpha, c, pressure = unpack_state(y_state, p)
    x = np.linspace(p.dx / 2, 1 - p.dx / 2, p.N)
    x_face = np.linspace(0, 1, p.N + 1)
    dct = solve_DCT_junction(alpha, c, pressure, p)
    figures = []
    for title, values, ylabel in (
        ("Pressure profiles", pressure, "scaled pressure"),
        ("Volume density alpha", alpha, "alpha"),
        ("NaCl salt concentration", c[SALT], "salt"),
        ("Urea concentration", c[UREA], "urea"),
        ("Mobile osmolarity", 2 * c[SALT] + c[UREA], "2*salt + urea"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in COMPARTMENTS:
            ax.plot(x, values[k], label=COMP_NAMES[k])
        ax.set(title=title, xlabel="x: cortex to papilla", ylabel=ylabel)
        ax.grid(True); ax.legend(); figures.append(fig)
    if normalize_osm_au:
        fig, ax = plt.subplots(figsize=(8, 4))
        osm = (2 * c[SALT] + c[UREA]) / p.legacy_concentration_scale
        for k in COMPARTMENTS:
            ax.plot(x, osm[k], label=COMP_NAMES[k])
        ax.axhline(7, linestyle="--", color="gray", label="7 a.u.")
        ax.set(title="Mobile osmolarity, dynamic-passive-style a.u.", xlabel="x", ylabel="osmolarity (a.u.)")
        ax.grid(True); ax.legend(); figures.append(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in COMPARTMENTS:
        ax.plot(x_face, water_flow(k, alpha, c, pressure, dct[2], p), label=COMP_NAMES[k])
    ax.set(title="Raw axial water flux", xlabel="x face", ylabel="water flux")
    ax.grid(True); ax.legend(); figures.append(fig)
    return figures
