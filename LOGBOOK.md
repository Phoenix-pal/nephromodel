# Kidney Model Logbook

This is the single continuing development and validation record for the
project. Add new entries at the top of **Current work**, always stating what
changed, why it changed, and the verification result.

## Current work

### 2026-09-05 - Main CLI default initial condition

**Change**

The main `python -m kidney_model` command now uses the final row of
`conv_to_third_ss.npy` as its default initial condition. The uniform native
plasma seed remains available through `--native-initial-condition`.

**Reason**

The intended main workflow is to start the full model from the supplied
third-steady-state legacy trajectory rather than from an unrelated uniform
plasma state.

**Verification**

The default command is checked with a zero-step run to confirm that it loads
the legacy file and maps its final row successfully.

### 2026-09-05 - Bistability animation labels

**Change**

Updated every bistability-animation legend entry to include its source file,
compartment, and physical flow direction. The horizontal axis is now labelled
as cortex (0) to papilla (1).

**Direction convention**

- Descending limb and collecting duct: cortex -> papilla (downward).
- Ascending limb: papilla -> cortex (upward).
- Interstitium: no single tubular axial flow direction.

**Verification**

Generated a labelled GIF using the revised standalone animation script.

### 2026-09-05 - Standalone legacy-animation script

**Change**

Rewrote `src/kidney_model/test.py` as a standalone trajectory-comparison
animation tool. It now loads two distinct legacy files, validates their
layout, aligns animation frames safely by trajectory fraction, and saves a
GIF without requiring IPython or LaTex.

**Reason**

The prior notebook-export script loaded the same file twice, depended on an
undefined IPython `display`, and used hard-coded frame indices.

**Verification**

The script is run from the project root and writes the requested GIF.

### 2026-09-05 — Physiological recheck: cortical reverse-flow correction

**Problem**

`results/physiology_recheck.md` found that the previous cortical A-C/DCT rule
continued to apply `F_C,salt = q F_A,salt` (`q=1/3`) when cortical flow
reversed. This allowed the omitted cortex to become an unmodelled source of
salt and water.

**Changes**

- Forward A-to-C flow retains the fractional salt-reabsorption rule.
- Reverse flow is now a closed A-C connection: A and C water, salt, and urea
  fluxes must match.
- Corrected the sign in the ascending-limb physical-path plot.
- Added forward- and reverse-flow junction tests.
- `--best-seed` now requires finite source data, non-negative stored solutes,
  and positive collecting-duct outlet flow.
- Added `cortical_junction_mode` and cortical net fluxes to physiology output.

**Result**

- `pytest -q`: **11 passed**.
- N=20 adaptive transient simulations reached nondimensional time 1.0 for
  both source seeds.
- In reverse mode, cortical water/salt/urea net fluxes were approximately zero
  (`~1e-14`).
- `conv_to_third_ss.npy --best-seed` now selects row `1279`, rather than the
  previous high-concentration row with inward collecting-duct flow.

Detailed results: `results/physiology_fix_recheck.md`.

## Earlier work

### Modularization from the original notebook

Converted `simp_kidney_rebase_nondim_A_full_A_geometry.ipynb` into the Python
package in `src/kidney_model`:

- `parameters.py` — parameters and scales
- `state.py` — state packing/unpacking and shared D-A tip
- `operators.py` — finite-difference operators
- `transport.py` — water/solute transport and junction equations
- `residuals.py` — implicit residual assembly
- `solver.py` — fixed and adaptive solvers
- `initialization.py` — legacy `.npy` seed import
- `diagnostics.py` — continuity, mass, and physiology metrics
- `visualization.py`, `plotting.py` — plots and animation
- `cli.py` — `python -m kidney_model` command-line interface

### Solver convergence and status reporting

**Problem**

`fsolve` can report convergence while the actual residual is still too high.
The CLI originally used less strict settings than the notebook, causing some
runs to stop as early as step 2.

**Changes**

- Set default `xtol=1e-8` and `maxfev=8000`.
- `success=True` now requires solver convergence, an acceptable residual, and
  positive state variables.
- A candidate that stops on relative-step tolerance but fails the residual
  check is automatically refined with a tighter tolerance.
- Corrected the DCT junction to use the same upwind boundary-flux law as the
  transport solver.

**Example result**

- N=50, final `conv_to_third_ss` row: steps 1–100 passed.
- N=70, step 1 previously had `Linf=1.199e-6` and was rejected.
- After refinement: `Linf=1.476e-11`, `success=True`.

### Explicit nondimensionalization

Added `nondimensional.py` to define scales and conversions explicitly:

```text
x_hat = x / L
t_hat = t / tau
c_hat = c / c_star
p_hat = p / (R*T*c_star)
```

- `c_star = 1 mmol/L`.
- Pressure scale = `19.344` reference pressure units.
- Internal values 145 (salt) and 5 (urea) are concentrations normalized by
  `c_star`, not undocumented raw units.
- The legacy source file uses a declared concentration reference of
  `147.5 mmol/L`, replacing an unexplained scaling factor.
- Equation coefficients are nondimensional; dimensional scales are retained
  only for conversion and reporting.

### Plot and animation support

Use these flags to save output:

```powershell
--save-plot-dir ".\results\plots"
--save-animation ".\results\conv_to_third_animation.html"
```

Use `--plot` or `--animation` to open interactive windows.

## Current interpretation

The model now has improved numerical continuity and a physiologically safer
cortical reverse-flow closure. It is still not a complete physiological kidney
model, and its concentration results must not yet be interpreted as direct
predictions for a real animal.

## Remaining work

- Investigate negative local pressures and the valid range of the compliance
  law.
- Perform grid-convergence and time-step-convergence studies.
- Establish long-time steady state separately from nonlinear step convergence.
- Resolve or document the hybrid legacy initial condition: source flows are
  not imported and some solutes retain native baseline profiles.
- Define species-specific segment and transport parameters.
- Add separate vascular compartments if countercurrent blood exchange and
  desert-rodent physiology are research objectives.
