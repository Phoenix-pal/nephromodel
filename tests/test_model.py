import numpy as np

from kidney_model import build_parameters
from kidney_model.constants import KA, KD, SALT, UREA
from kidney_model.residuals import implicit_residual
from kidney_model.state import geometric_state_size, make_initial_state, pack_state, unpack_state


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
