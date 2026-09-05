import numpy as np

from kidney_model import build_parameters
from kidney_model.constants import KA, KC, KD, SALT, UREA
from kidney_model.residuals import implicit_residual
from kidney_model.state import geometric_state_size, make_initial_state, pack_state, unpack_state
from kidney_model.diagnostics import physiology_metrics
from kidney_model.initialization import select_best_dynamic_row
from kidney_model.initialization import load_dynamic_state, make_initial_condition_from_file
from kidney_model.nondimensional import ReferenceScales
from kidney_model.solver import run_model_fsolve
from kidney_model.transport import DCT_junction, dct_junction_mode, solve_DCT_junction, solute_flow, water_flow


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


def test_forward_dct_equations_match_actual_boundary_fluxes():
    p = build_parameters(N=6, dt=0.01)
    alpha, c, pressure = unpack_state(make_initial_state(p), p)
    c[:, KA, :] = np.array([[100.0], [20.0]])
    c[:, KC, :] = np.array([[80.0], [135.0]])
    # Forward A-to-C cortical flow uses fractional salt reabsorption.
    pressure[KA, :] = 1.0
    pressure[KC, :] = 0.5
    dct = solve_DCT_junction(alpha, c, pressure, p)
    assert dct_junction_mode(dct, alpha, c, pressure, p) == "forward_cortical"
    assert np.max(np.abs(DCT_junction(dct, alpha, c, pressure, p))) < 1e-7
    salt_a = solute_flow(SALT, KA, c, alpha, pressure, dct, p)[0]
    salt_c = solute_flow(SALT, KC, c, alpha, pressure, dct, p)[0]
    urea_a = solute_flow(UREA, KA, c, alpha, pressure, dct, p)[0]
    urea_c = solute_flow(UREA, KC, c, alpha, pressure, dct, p)[0]
    assert abs(salt_c - p.q * salt_a) < 1e-7
    assert abs(urea_c - urea_a) < 1e-7


def test_reverse_dct_flow_is_closed_and_cannot_supply_cortical_salt_or_water():
    p = build_parameters(N=6, dt=0.01)
    alpha, c, pressure = unpack_state(make_initial_state(p), p)
    c[:, KA, :] = np.array([[100.0], [20.0]])
    c[:, KC, :] = np.array([[80.0], [135.0]])
    pressure[KA, :] = 0.5
    pressure[KC, :] = 1.0
    dct = solve_DCT_junction(alpha, c, pressure, p)
    assert dct_junction_mode(dct, alpha, c, pressure, p) == "closed_reverse"
    water_a = water_flow(KA, alpha, c, pressure, dct[2], p)[0]
    water_c = water_flow(KC, alpha, c, pressure, dct[2], p)[0]
    assert water_a < 0.0 and water_c < 0.0
    assert abs(water_c - water_a) < 1e-8
    for solute in (SALT, UREA):
        flux_a = solute_flow(solute, KA, c, alpha, pressure, dct, p)[0]
        flux_c = solute_flow(solute, KC, c, alpha, pressure, dct, p)[0]
        assert abs(flux_c - flux_a) < 1e-7


def test_physiology_metrics_report_native_baseline():
    p = build_parameters(N=6, dt=0.01)
    metrics = physiology_metrics(make_initial_state(p), p)
    assert metrics["urine_flow"] == 0.0
    assert metrics["urine_plasma_ratio"] == 1.0


def test_best_seed_selector_returns_known_dynamic_row():
    assert select_best_dynamic_row("dynamic_stable_v2.npy") == 0


def test_best_seed_selector_excludes_reverse_collecting_outlet_flow():
    selected = select_best_dynamic_row("conv_to_third_ss.npy")
    source = load_dynamic_state("conv_to_third_ss.npy", selected)
    assert source["q_C"][-1] > 0.0


def test_reference_scales_are_explicit_and_consistent():
    scales = ReferenceScales()
    assert np.isclose(scales.concentration_hat(145.0), 145.0)
    assert np.isclose(scales.concentration_hat(5.0), 5.0)
    assert np.isclose(scales.pressure_hat(6.4), 6.4 / 19.344)
    assert np.isclose(scales.concentration_mmol_L(145.0), 145.0)


def test_parameters_expose_only_explicit_nondimensional_conventions():
    p = build_parameters(N=6, dt=0.01)
    assert np.isclose(p.plasma_osmolarity, 295.0)
    assert np.isclose(p.legacy_concentration_scale, 147.5)
    assert np.isclose(p.pump_half_saturation, 150.0)
    assert np.isclose(p.P_p, 6.4 / 19.344)


def test_fixed_solver_refines_n70_first_step():
    p = build_parameters(N=70, dt=0.01)
    seed = make_initial_condition_from_file(load_dynamic_state("conv_to_third_ss.npy", -1), p)
    result = run_model_fsolve(
        p, y_start=seed, steps=1, print_every=1,
        xtol=1e-8, maxfev=8000, residual_tol=1e-6,
    )
    report = result["reports"][0]
    assert report["success"] is True
    assert report["solver_attempts"] >= 1
    assert report["residual_Linf"] <= 1e-6
