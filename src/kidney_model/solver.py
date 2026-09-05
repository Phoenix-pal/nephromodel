"""Implicit time integration routines."""

import numpy as np
from scipy.optimize import fsolve
from dataclasses import replace

from .constants import SALT, UREA
from .parameters import ModelParameters
from .residuals import implicit_residual
from .state import make_initial_state, unpack_state


def _summary(y, p):
    alpha, c, pressure = unpack_state(y, p)
    return {
        "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
        "salt_min": float(c[SALT].min()), "salt_max": float(c[SALT].max()),
        "urea_min": float(c[UREA].min()), "urea_max": float(c[UREA].max()),
        "c_min": float(c.min()), "c_max": float(c.max()),
        "pressure_min": float(pressure.min()), "pressure_max": float(pressure.max()),
        "finite": bool(np.all(np.isfinite(y))),
    }


def run_model_fsolve(
    p: ModelParameters,
    y_start=None,
    steps: int = 1000,
    print_every: int = 10,
    xtol: float = 1e-8,
    maxfev: int = 8000,
    residual_tol: float = 1e-6,
):
    """Run fixed-step implicit integration with ``scipy.optimize.fsolve``."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    y = make_initial_state(p) if y_start is None else np.asarray(y_start).copy()
    history, reports = [y.copy()], []
    for step in range(1, steps + 1):
        old = y.copy()
        y_new, info, ier, message = fsolve(
            lambda candidate: implicit_residual(candidate, old, p),
            old,
            full_output=True,
            xtol=xtol,
            maxfev=maxfev,
        )
        residual = implicit_residual(y_new, old, p)
        report = {
            "step": step, "time": step * p.dt, "success": ier == 1,
            "solver_success": ier == 1,
            "ier": ier, "message": message, "nfev": info["nfev"],
            "residual_Linf": float(np.max(np.abs(residual))),
        }
        report.update(_summary(y_new, p))
        report["success"] = bool(
            ier == 1
            and report["residual_Linf"] <= residual_tol
            and report["alpha_min"] > 0
            and report["salt_min"] > 0
            and report["urea_min"] > 0
        )
        reports.append(report)
        if step == 1 or step % print_every == 0 or step == steps or ier != 1:
            print(
                f"step={step:05d}/{steps} "
                f"t={step * p.dt:.5g} "
                f"success={report['success']} "
                f"Linf={report['residual_Linf']:.3e} "
                f"salt_min={report['salt_min']:.5g} "
                f"urea_min={report['urea_min']:.5g} "
                f"salt_max={report['salt_max']:.5g} "
                f"urea_max={report['urea_max']:.5g} "
                f"nfev={report['nfev']}"
            )
        if not report["success"]:
            break
        y = y_new.copy()
        history.append(y.copy())
    return {"final_state": y, "history": history, "reports": reports}


def solve_one_step_positive(y_old, local_dt, p: ModelParameters, xtol=1e-10, maxfev=8000):
    """Solve one implicit step, rejecting non-positive trial states."""
    local_p = replace(p, dt=float(local_dt))

    def residual(candidate):
        try:
            alpha, c, _ = unpack_state(candidate, local_p)
            if np.any(~np.isfinite(candidate)) or np.min(alpha) <= 0 or np.min(c) <= 0:
                return np.ones_like(candidate) * 1e6
        except (ValueError, FloatingPointError):
            return np.ones_like(candidate) * 1e6
        return implicit_residual(candidate, y_old, local_p)

    y_new, info, ier, message = fsolve(residual, y_old.copy(), full_output=True, xtol=xtol, maxfev=maxfev)
    result = residual(y_new)
    report = {"success": ier == 1, "ier": ier, "message": message,
              "nfev": info.get("nfev", np.nan), "dt": local_dt,
              "residual_Linf": float(np.max(np.abs(result)))}
    report.update(_summary(y_new, local_p))
    return y_new, report


def run_model_positive_adaptive(
    p: ModelParameters,
    y_start,
    t_final=100.0,
    dt_initial=0.1,
    dt_min=1e-5,
    dt_max=None,
    residual_tol=1e-7,
    xtol=1e-10,
    maxfev=8000,
    print_every_accept=1,
):
    """Adaptive implicit integration with a strict positivity acceptance rule."""
    if dt_max is None:
        dt_max = dt_initial
    y = np.asarray(y_start).copy()
    history, reports = [y.copy()], []
    time, accepted, attempted, local_dt = 0.0, 0, 0, float(dt_initial)
    while time < t_final - 1e-14:
        attempted += 1
        local_dt = min(local_dt, t_final - time)
        y_try, report = solve_one_step_positive(y, local_dt, p, xtol, maxfev)
        physical = report["finite"] and report["alpha_min"] > 1e-10 and report["c_min"] > 1e-10
        if report["success"] and physical and report["residual_Linf"] <= residual_tol:
            time += local_dt
            accepted += 1
            report.update({"time": time, "accepted_step": accepted, "attempted_step": attempted})
            reports.append(report)
            history.append(y_try.copy())
            y = y_try.copy()
            if accepted == 1 or accepted % print_every_accept == 0:
                print(f"accept={accepted:05d} t={time:.6g} dt={local_dt:.3g} Linf={report['residual_Linf']:.3e}")
            local_dt = min(dt_max, local_dt * 1.25)
        else:
            print(f"reject at t={time:.6g} dt={local_dt:.3g} success={report['success']} Linf={report['residual_Linf']:.3e}")
            local_dt *= 0.5
            if local_dt < dt_min:
                break
    return {"final_state": y, "history": history, "reports": reports,
            "t_final_reached": time, "accepted_steps": accepted, "attempted_steps": attempted}
