"""Plots and animations corresponding to the original notebook pipeline."""

from pathlib import Path

import numpy as np

from .constants import COMPARTMENTS, COMP_NAMES, KA, KC, KD, K0, SALT, UREA
from .parameters import ModelParameters
from .state import unpack_state
from .transport import solve_DCT_junction, water_flow


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_dynamic_source(dynamic_state, show=False):
    """Plot osmolarity profiles from the dynamic-passive source file."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dynamic_state["x_face_dyn"], dynamic_state["osm_D"], "r", label="dynamic D")
    ax.plot(dynamic_state["x_face_dyn"], dynamic_state["osm_A"], "g", label="dynamic A")
    ax.plot(dynamic_state["x_face_dyn"], dynamic_state["osm_C"], "b", label="dynamic C")
    ax.plot(dynamic_state["x_cell_dyn"], dynamic_state["osm_0"], "m", label="dynamic 0")
    ax.axhline(7.0, linestyle="--", color="gray", label="7 a.u.")
    ax.set(xlabel="x", ylabel="dynamic passive osmolarity (a.u.)", title="conv_to_third_ss source state")
    ax.grid(True); ax.legend()
    if show:
        plt.show()
    return fig


def plot_initial_condition(y_state, p: ModelParameters, show=False):
    """Plot the file-derived initial condition mapped to the full model."""
    plt = _plt()
    _, c, _ = unpack_state(y_state, p)
    x = np.linspace(p.dx / 2.0, 1.0 - p.dx / 2.0, p.N)
    osm_au = (2.0 * c[SALT] + c[UREA]) / p.legacy_concentration_scale
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, color in ((KD, "r"), (KA, "g"), (KC, "b"), (K0, "m")):
        ax.plot(x, osm_au[k], color, label=f"full IC {COMP_NAMES[k].split('_')[0]}")
    ax.axhline(7.0, linestyle="--", color="gray", label="7 a.u.")
    ax.set(xlabel="x", ylabel="full IC osmolarity (a.u.)", title="Direct file-derived initial condition")
    ax.grid(True); ax.legend()
    if show:
        plt.show()
    return fig


def _water_fluxes(y_state, p):
    alpha, c, pressure = unpack_state(y_state, p)
    dct = solve_DCT_junction(alpha, c, pressure, p)
    flux = np.zeros((4, p.N + 1))
    for k in COMPARTMENTS:
        flux[k] = water_flow(k, alpha, c, pressure, dct[2], p)
    return alpha, c, pressure, flux


def trajectory_data(result, p: ModelParameters):
    """Return time-series diagnostics for the solver history."""
    history = result.get("history", [])
    reports = result.get("reports", [])
    times = np.array([r.get("time", (i + 1) * p.dt) for i, r in enumerate(reports)], dtype=float)
    residual = np.array([r.get("residual_Linf", np.nan) for r in reports], dtype=float)
    series = {"time": times, "residual": residual}
    for key in ("salt_min", "urea_min", "salt_max", "urea_max"):
        series[key] = np.array([r.get(key, np.nan) for r in reports], dtype=float)

    osm = {"D_tip": [], "A_tip": [], "C_tip": [], "I_tip": [], "global_max": []}
    for y_state in history[1:]:
        _, c, _ = unpack_state(y_state, p)
        value = (2.0 * c[SALT] + c[UREA]) / p.legacy_concentration_scale
        osm["D_tip"].append(value[KD, -1])
        osm["A_tip"].append(value[KA, -1])
        osm["C_tip"].append(value[KC, -1])
        osm["I_tip"].append(value[K0, -1])
        osm["global_max"].append(np.max(value))
    for key, values in osm.items():
        series[key] = np.asarray(values, dtype=float)
    return series


def plot_result(result, p: ModelParameters, state_index=-1, include_time=True):
    """Create the full static diagnostic plot set from the original workflow."""
    plt = _plt()
    history = result.get("history", [])
    if not history:
        raise ValueError("result['history'] is empty")
    y_state = history[state_index]
    alpha, c, pressure, flux = _water_fluxes(y_state, p)
    x = np.linspace(p.dx / 2.0, 1.0 - p.dx / 2.0, p.N)
    x_face = np.linspace(0.0, 1.0, p.N + 1)
    mobile_osm = 2.0 * c[SALT] + c[UREA]
    mobile_osm_au = mobile_osm / p.legacy_concentration_scale
    figures = []

    for title, ylabel, values in (
        ("Pressure profiles", "scaled pressure", pressure),
        ("Volume density alpha", "alpha", alpha),
        ("NaCl salt concentration", "salt", c[SALT]),
        ("Urea concentration", "urea", c[UREA]),
        ("Mobile osmolarity = 2*salt + urea", "mobile osmolarity", mobile_osm),
        ("Mobile osmolarity, dynamic-passive-style a.u.", "osmolarity (a.u.)", mobile_osm_au),
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in COMPARTMENTS:
            ax.plot(x, values[k], label=COMP_NAMES[k])
        if values is mobile_osm_au:
            ax.axhline(7.0, linestyle="--", color="gray", label="7 a.u.")
        ax.set(title=title, xlabel="x: cortex to papilla", ylabel=ylabel)
        ax.grid(True); ax.legend(); figures.append(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in COMPARTMENTS:
        ax.plot(x_face, flux[k], label=COMP_NAMES[k])
    ax.set(title="Raw axial water flux v = alpha*u", xlabel="x face: cortex to papilla", ylabel="water flux")
    ax.grid(True); ax.legend(); figures.append(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_face, flux[KD], label="D path: cortex to papilla")
    ax.plot(1.0 + (1.0 - x_face[::-1]), flux[KA, ::-1], label="A path: papilla to cortex")
    ax.plot(2.2 + x_face, flux[KC], label="C path: cortex to papilla")
    ax.axvline(1.0, linestyle="--", linewidth=0.8, label="D-A loop tip")
    ax.set(title="Water flux along physical nephron path", xlabel="physical path coordinate", ylabel="flux along physical direction")
    ax.grid(True); ax.legend(); figures.append(fig)

    if include_time and result.get("reports"):
        series = trajectory_data(result, p)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(series["time"], series["residual"])
        ax.set_yscale("log"); ax.set(title="fsolve residual over time", xlabel="time", ylabel="residual Linf")
        ax.grid(True); figures.append(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for key in ("salt_min", "urea_min", "salt_max", "urea_max"):
            ax.plot(series["time"], series[key], label=key.replace("_", " "))
        ax.set(title="Concentration diagnostics", xlabel="time", ylabel="concentration")
        ax.grid(True); ax.legend(); figures.append(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label in (("D_tip", "D tip"), ("A_tip", "A tip"), ("C_tip", "C tip"), ("I_tip", "0 tip"), ("global_max", "global max")):
            ax.plot(series["time"][:len(series[key])], series[key], label=label, linewidth=2.5 if key == "global_max" else 1.5)
        ax.axhline(7.0, linestyle="--", color="gray", label="7 a.u.")
        ax.set(title="Osmolarity diagnostics over time", xlabel="time", ylabel="osmolarity (a.u.)")
        ax.grid(True); ax.legend(); figures.append(fig)
    return figures


def animate_result(result, p: ModelParameters, frame_stride=10, gamma_label=1.3):
    """Create the osmolarity animation used in the original notebook."""
    import matplotlib.animation as animation

    history = result.get("history", [])
    if not history:
        raise ValueError("result['history'] is empty")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    plt = _plt()
    x = np.linspace(p.dx / 2.0, 1.0 - p.dx / 2.0, p.N)
    reports = result.get("reports", [])
    osm = []
    times = []
    for index, y_state in enumerate(history):
        _, c, _ = unpack_state(y_state, p)
        value = (2.0 * c[SALT] + c[UREA]) / p.legacy_concentration_scale
        osm.append(value)
        times.append(0.0 if index == 0 else reports[index - 1].get("time", index * p.dt) if index - 1 < len(reports) else index * p.dt)
    osm = np.asarray(osm)
    times = np.asarray(times)
    indices = np.arange(0, len(history), frame_stride)
    if indices[-1] != len(history) - 1:
        indices = np.append(indices, len(history) - 1)
    values = osm[indices]
    pad = 0.05 * (float(np.nanmax(values)) - float(np.nanmin(values)) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 6))
    lines = [
        ax.plot(x, values[0, k], color=color, linewidth=2.5, label=label)[0]
        for k, color, label in ((KD, "r", r"$k=D$"), (KA, "g", r"$k=A$"), (KC, "b", r"$k=C$"), (K0, "m", r"$k=0$"))
    ]
    ax.axhline(7.0, linestyle="--", color="gray", label="7 a.u.")
    ax.set(xlabel="x", ylabel="Osmolarity (a.u.)", ylim=(float(np.nanmin(values)) - pad, float(np.nanmax(values)) + pad))
    ax.grid(True); ax.legend()

    def update(frame_number):
        index = indices[frame_number]
        for line, k in zip(lines, (KD, KA, KC, K0)):
            line.set_data(x, osm[index, k])
        ax.set_title(f"gamma = {gamma_label}, time={times[index]:.3f}")
        return lines

    return animation.FuncAnimation(fig, update, frames=len(indices), interval=40, blit=True), fig


def save_animation(ani, path):
    """Save animation as HTML, GIF, or a video based on the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".html":
        path.write_text(ani.to_jshtml(), encoding="utf-8")
    elif suffix == ".gif":
        from matplotlib.animation import PillowWriter

        ani.save(path, writer=PillowWriter(fps=25))
    else:
        ani.save(path, fps=25)
