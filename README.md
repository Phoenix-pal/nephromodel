# Kidney concentrating model

This project is a modular Python version of
`simp_kidney_rebase_nondim_A_full_A_geometry.ipynb`. It simulates water and
solute transport through the interstitium, descending limb, ascending limb,
and collecting duct using the notebook's nondimensional equations.

## Layout

`src/kidney_model/parameters.py` builds an explicit parameter object;
`operators.py` contains the full A-geometry finite-difference operators;
`transport.py` contains fluxes and junction equations; `state.py` owns the
shared D-A tip state packing; `residuals.py` assembles the implicit system;
`solver.py` provides fixed-step and positivity-adaptive integration;
`initialization.py` maps legacy dynamic-passive `.npy` files; and
`plotting.py` contains optional matplotlib plots.

The package has no import-time simulation, plotting, or file loading.

For fixed-step runs, `success=True` means both that `fsolve` converged and
that the final residual is at most `1e-6`; a solver status of `ier=1` alone is
not accepted. The DCT/A-C junction is checked against the same upwind
boundary fluxes used by the transport equations.

The cortical DCT/A-C closure is direction-aware. Fractional salt
reabsorption is used only for forward A-to-C flow. If either cortical boundary
flow reverses, the omitted cortical segment becomes a closed connection: water,
salt, and urea fluxes are continuous, so it cannot act as an unmodelled source.
The CLI physiology report exposes this mode and the cortical net fluxes.

## Install and run

```text
python -m pip install -e ".[plot,dev]"
python -m kidney_model --N 50 --steps 10 --dt 0.01
```

The main CLI uses the final row of `conv_to_third_ss.npy` as its default
initial condition. Use `--native-initial-condition` to run from the uniform
plasma seed instead.

Another dynamic-passive file can be used as an initial condition:

```text
python -m kidney_model.cli --dynamic-file path/to/dynamic_stable_v2.npy --N 50 --steps 100
```

For the supplied third-steady-state seed, use the final trajectory row:

```text
python -m kidney_model --dynamic-file conv_to_third_ss.npy --row-index -1 --N 50 --steps 1000 --dt 0.01
```

To screen a trajectory file and use its highest-osmolarity admissible legacy
seed (finite/non-negative solutes and positive source collecting outlet flow):

```text
python -m kidney_model --dynamic-file dynamic_stable_v2.npy --best-seed --N 50 --steps 1000 --dt 0.01
```

The seed selector is optional. For `conv_to_third_ss.npy`, the original
workflow remains available with `--row-index -1`.

Add `--plot` to display final-state plots, or save them without opening
windows:

```text
python -m kidney_model --dynamic-file conv_to_third_ss.npy --row-index -1 --N 50 --steps 1000 --dt 0.01 --plot
python -m kidney_model --dynamic-file conv_to_third_ss.npy --row-index -1 --N 50 --steps 1000 --dt 0.01 --save-plot-dir results/plots
```

The full notebook-style visualization workflow is enabled by `--plot` when a
dynamic file is supplied: it includes the source-state plot, mapped initial
condition, final profiles, water-flux plots, and time diagnostics. The
osmolarity animation can be displayed or exported separately:

```text
python -m kidney_model --dynamic-file conv_to_third_ss.npy --row-index -1 --N 50 --steps 1000 --dt 0.01 --animation --frame-stride 10
python -m kidney_model --dynamic-file conv_to_third_ss.npy --row-index -1 --N 50 --steps 1000 --dt 0.01 --save-animation results/conv_to_third.html --frame-stride 10
```

The legacy notebook is intentionally kept as the scientific reference. The
Python modules accept a `ModelParameters` object instead of relying on mutable
globals, making multiple model configurations possible in one process.

## Nondimensional convention

The solver now uses one explicit nondimensional convention throughout:

```text
x_hat = x / L
t_hat = t / tau
c_hat = c / c_star
p_hat = p / (R*T*c_star)
```

The reference values are `c_star = 0.001 mmol/mL` (1 mmol/L), `L = 1` model
length unit, and `p_star = c_star*R*T = 19.344` reference pressure units.
Thus 145 mmol/L salt is stored as `c_hat=145`, and 6.4 reference pressure
units is stored as `p_hat=0.330...`. All coefficients used by the equations
are derived from these scales in `parameters.py`. The legacy `.npy` source
uses an explicitly declared concentration reference of 147.5 mmol/L; its
values are converted by `ReferenceScales`, not by an unexplained numerical
factor. Use `p.scales` to convert solver output back to dimensional units.

## Verification

```text
pytest
```
