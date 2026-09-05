"""Initial-state loading and interpolation from dynamic-passive trajectories."""

from pathlib import Path

import numpy as np

from .constants import KA, KC, KD, K0, SALT, UREA
from .parameters import ModelParameters
from .state import make_initial_state, pack_state, unpack_state


def resolve_file(path_string: str | Path) -> Path:
    path = Path(path_string)
    candidates = [path, Path("/mnt/data") / path, Path("/content") / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find dynamic state file: {path_string}")


def load_dynamic_state(path_string: str | Path, row_index: int = 0):
    """Load one row of the legacy ``8*N_dyn + 7`` trajectory format."""
    path = resolve_file(path_string)
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D dynamic trajectory, got shape {data.shape}")
    if not -data.shape[0] <= row_index < data.shape[0]:
        raise IndexError(f"row_index={row_index} outside [-{data.shape[0]}, {data.shape[0]})")
    row_index = row_index % data.shape[0]
    if (data.shape[1] - 7) % 8 != 0:
        raise ValueError(f"Expected columns = 8*N_dyn + 7, got {data.shape[1]}")
    n_dyn = (data.shape[1] - 7) // 8
    length = n_dyn + 1
    row = data[row_index].copy()
    q_D, q_C, q_0 = row[:length], row[length:2 * length], row[2 * length:3 * length]
    s_D, s_A, u_C = row[3 * length:6 * length].reshape(3, length)
    s_0 = row[6 * length:7 * n_dyn + 6]
    u_0 = row[7 * n_dyn + 6:8 * n_dyn + 6]
    q_A_raw = row[8 * n_dyn + 6:]
    if len(q_A_raw) == 1:
        q_A = np.full(length, q_A_raw[0])
    elif len(q_A_raw) == length:
        q_A = q_A_raw.copy()
    else:
        raise ValueError(f"Unexpected q_A length={len(q_A_raw)}")
    return {
        "path": str(path), "Y_shape": data.shape, "row_index": row_index,
        "N_dyn": n_dyn, "L": length,
        "x_face_dyn": np.linspace(0.0, 1.0, length),
        "x_cell_dyn": np.linspace(1 / (2 * n_dyn), 1 - 1 / (2 * n_dyn), n_dyn),
        "q_D": q_D, "q_C": q_C, "q_0": q_0, "q_A": q_A,
        "s_D": s_D, "s_A": s_A, "u_C": u_C, "s_0": s_0, "u_0": u_0,
        "osm_D": 2 * s_D, "osm_A": 2 * s_A, "osm_C": u_C,
        "osm_0": 2 * s_0 + u_0,
    }


def load_conv_to_third_state(path_string: str | Path = "conv_to_third_ss.npy", row_index: int = -1):
    """Load the final row of ``conv_to_third_ss.npy`` by default.

    The file uses the same legacy trajectory layout as ``load_dynamic_state``;
    this named wrapper documents the intended third-steady-state workflow.
    """
    return load_dynamic_state(path_string, row_index=row_index)


def select_best_dynamic_row(path_string: str | Path, criterion: str = "collecting_osm"):
    """Select an admissible high-osmolarity legacy source row.

    A selected source row must be finite, have non-negative stored solutes,
    and have positive collecting-duct outlet flow.  It is still only a seed:
    the full-model mapper regenerates volume fractions and pressures and does
    not transfer the legacy flow profiles.
    """
    path = resolve_file(path_string)
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2 or (data.shape[1] - 7) % 8 != 0:
        raise ValueError(f"Expected a 2D trajectory with 8*N_dyn + 7 columns, got {data.shape}")
    n_dyn = (data.shape[1] - 7) // 8
    if criterion != "collecting_osm":
        raise ValueError(f"Unknown seed criterion: {criterion}")
    length = n_dyn + 1
    collecting_osm_outlet_column = 6 * length - 1
    collecting_flow_outlet_column = 2 * length - 1
    solute_columns = np.r_[
        np.arange(3 * length, 6 * length),
        np.arange(6 * length, 8 * n_dyn + 6),
    ]
    column = np.asarray(data[:, collecting_osm_outlet_column], dtype=float)
    valid = (
        np.isfinite(data).all(axis=1)
        & (np.min(data[:, solute_columns], axis=1) >= 0.0)
        & (data[:, collecting_flow_outlet_column] > 0.0)
    )
    if not np.any(valid):
        raise ValueError("No finite, non-negative source row with positive collecting outlet flow")
    valid_indices = np.flatnonzero(valid)
    return int(valid_indices[np.argmax(column[valid])])


def make_initial_condition_from_file(dyn, p: ModelParameters):
    """Map dynamic-passive profiles onto the full model without blending."""
    x_cell = np.linspace(p.dx / 2, 1 - p.dx / 2, p.N)
    face_to_cell = lambda values: np.interp(x_cell, dyn["x_face_dyn"], values)
    cell_to_cell = lambda values: np.interp(x_cell, dyn["x_cell_dyn"], values)
    alpha, c, pressure = unpack_state(make_initial_state(p), p)
    # The legacy trajectory stores concentrations normalized by 147.5 mmol/L.
    # Convert that source convention explicitly into the solver's c/c_star.
    scale = p.legacy_concentration_scale
    c[SALT, KD] = np.maximum(face_to_cell(dyn["s_D"]) * scale, 1e-8)
    c[SALT, KA] = np.maximum(face_to_cell(dyn["s_A"]) * scale, 1e-8)
    c[UREA, KC] = np.maximum(face_to_cell(dyn["u_C"]) * scale, 1e-8)
    c[SALT, K0] = np.maximum(cell_to_cell(dyn["s_0"]) * scale, 1e-8)
    c[UREA, K0] = np.maximum(cell_to_cell(dyn["u_0"]) * scale, 1e-8)
    return pack_state(alpha, c, pressure, p)
