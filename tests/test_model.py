import numpy as np

from kidney_model import build_parameters
from kidney_model.constants import KA, KC, KD, SALT, UREA
from kidney_model.residuals import implicit_residual
from kidney_model.state import geometric_state_size, make_initial_state, pack_state, unpack_state
from kidney_model.diagnostics import physiology_metrics
from kidney_model.initialization import select_best_dynamic_row
from kidney_model.transport import DCT_junction, solve_DCT_junction, solute_flow


def test_state_round_trip_and_shared_tip():
    p = build_parameters(N=6, dt=0.01)
    y = make_initial_state(p)
    alpha, c, pressure = unpack_state(y, p)
    assert len(y) == geometric_state_size(p)
    assert np.allclose(pack_state(alpha, c, pressure, p), y)
    assert c[SALT, KA, -1] == c[SALT, KD, -1]
    assert c[UREA, KA, -1] == c[UREA, KD, -1]


def test_initial_state_is_finite_and_positive():
    p = build_parameters(N=6, dt=0.01)
    y = make_initial_state(p)
    alpha, c, pressure = unpack_state(y, p)
    assert np.isfinite(y).all()
    assert alpha.min() > 0
    assert c.min() > 0
    assert np.isfinite(pressure).all()


def test_implicit_residual_shape():
    p = build_parameters(N=6, dt=0.01)
    y = make_initial_state(p)
    residual = implicit_residual(y, y, p)
    assert residual.shape == y.shape
    assert np.isfinite(residual).all()


def test_dct_equations_match_actual_boundary_fluxes():
    p = build_parameters(N=6, dt=0.01)
    alpha, c, pressure = unpack_state(make_initial_state(p), p)
    c[:, KA, :] = np.array([[100.0], [20.0]])
    c[:, KC, :] = np.array([[80.0], [135.0]])
    # Deliberately choose a junction state with reverse A water flow.
    pressure[KA, :] = 1.0
    pressure[KC, :] = 0.5
    dct = solve_DCT_junction(alpha, c, pressure, p)
    assert np.max(np.abs(DCT_junction(dct, alpha, c, pressure, p))) < 1e-7
    salt_a = solute_flow(SALT, KA, c, alpha, pressure, dct, p)[0]
    salt_c = solute_flow(SALT, KC, c, alpha, pressure, dct, p)[0]
    urea_a = solute_flow(UREA, KA, c, alpha, pressure, dct, p)[0]
    urea_c = solute_flow(UREA, KC, c, alpha, pressure, dct, p)[0]
    assert abs(salt_c - p.q * salt_a) < 1e-7
    assert abs(urea_c - urea_a) < 1e-7


def test_physiology_metrics_report_native_baseline():
    p = build_parameters(N=6, dt=0.01)
    metrics = physiology_metrics(make_initial_state(p), p)
    assert metrics["urine_flow"] == 0.0
    assert metrics["urine_plasma_ratio"] == 1.0


def test_best_seed_selector_returns_known_dynamic_row():
    assert select_best_dynamic_row("dynamic_stable_v2.npy") == 0
