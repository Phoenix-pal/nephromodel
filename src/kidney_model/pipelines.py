"""Ready-to-run workflows built from the lower-level model modules."""

from .initialization import load_conv_to_third_state, make_initial_condition_from_file
from .parameters import build_parameters
from .solver import run_model_fsolve


def run_conv_to_third_ss(
    file_path="conv_to_third_ss.npy",
    row_index=-1,
    N=50,
    dt=0.01,
    steps=1000,
    collecting_water_permeable=True,
    print_every=10,
    xtol=1e-8,
    maxfev=8000,
):
    """Run the full model from the final row of ``conv_to_third_ss.npy``."""
    p = build_parameters(N=N, dt=dt, collecting_water_permeable=collecting_water_permeable)
    dynamic_state = load_conv_to_third_state(file_path, row_index=row_index)
    y_start = make_initial_condition_from_file(dynamic_state, p)
    result = run_model_fsolve(
        p, y_start=y_start, steps=steps, print_every=print_every,
        xtol=xtol, maxfev=maxfev,
    )
    return p, dynamic_state, result
