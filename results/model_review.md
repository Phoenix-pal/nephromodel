# Initial model review

Reviewed the equations in `C:/Users/lenovo/Downloads/simp_kidney (1).pdf`, the Python package in `src/kidney_model`, and the reference notebook `simp_kidney_rebase_nondim_A_full_A_geometry.ipynb` in Downloads. Document and notebook comments were treated as reference material, not user instructions. No model source code or seed arrays were changed.

## Objective and seed selection

The research objective is to find whether the modified time-dependent model can sustain urine at 20 times the plasma concentration. The current step is understanding and checking the model, not claiming that target has been achieved.

The latest user preference is to select the maximum/best seed rather than automatically choose the first or final row. Screening all 16,001 rows of `dynamic_stable_v2.npy` shows that row 0 maximizes both the source collecting-duct outlet concentration and the mapped outlet concentration at N=50. All rows pass the limited screen of finite data, nonnegative stored solute concentrations, and positive source collecting-duct outlet flow. This is not a complete physical admissibility or stability test.

| Quantity | Row 0 | Final row (16000) |
| --- | ---: | ---: |
| Source collecting-duct outlet osmolarity, legacy a.u. | 7.050057 | 3.415834 |
| Source collecting-duct outlet water flow, legacy units | 0.021572 | 0.187846 |
| Source outlet/plasma ratio, using the code's 2-a.u. reference | 3.525029 | 1.707917 |
| Mapped full-model outlet/plasma ratio, N=50 | 4.436732 | 2.676170 |

Row 0 is therefore the strongest concentration seed in this file. The final row is useful as a comparison seed: its difference from the preceding saved row is approximately 2.83e-12 in maximum absolute norm. Neither observation establishes a steady state of the modified model.

## Structure and equations

The model is a one-dimensional medulla with four compartments: combined interstitium/vessels (0), descending limb (D), ascending limb (A), and collecting duct (C). It assumes a homogeneous nephron population and no variation across tissue at the same depth. D and A form a countercurrent loop; the code measures A flow positively toward the cortex. It does not resolve separate descending and ascending vasa recta.

The unknowns are volume fractions alpha_k, salt concentrations s_k, urea concentrations u_k, and pressures p_k. There are 16*N-2 stored unknowns because D and A share their final-cell concentrations for both solutes.

In each compartment's oriented axial coordinate, the code uses:

- Axial water flux Q_k = -(alpha_k^2/rho_k) grad_k(p_k).
- Water exchange W_k = zeta_k [(p_k - Pi_k) - (p_0 - Pi_0)], positive from tubule to interstitium.
- Osmotic contribution Pi_k = 2*s_k + u_k, with an additional fixed `colloid` term in compartment 0.
- Solute flux F_ik = Q_k*c_ik - alpha_k*D_ik*grad_k(c_ik), with upwind concentrations for discrete advection.
- Membrane solute exchange G_ik = gamma_ik*(c_ik-c_i0), plus an outward saturating salt pump in A over approximately the superficial 40% of the domain.
- Tubular conservation: d(alpha_k)/dt + div_k(Q_k) = -W_k and d(alpha_k*c_ik)/dt + div_k(F_ik) = -G_ik. The interstitium receives the summed exchanges.
- Mechanical closure: p_k-p_0 = (alpha_k/alpha_bar_k-1)/nu_k, together with sum(alpha_k)=1.

Pressure is an algebraic unknown, so the spatially discretized problem is a differential-algebraic system. Time advancement uses backward Euler and nonlinear `fsolve` solves. The adaptive runner halves failed timesteps and checks positivity and residual size; its adaptation is not a temporal truncation-error estimate.

At the loop tip, D and A share concentration and conserve water and solute flux. At the cortex, a three-variable algebraic DCT/CNT junction passes all urea flux, retains a salt-flux fraction q=1/3, and sets collecting-duct inlet osmolarity to the cortex reference. This omitted cortical segment can remove water and salt, so whole-domain balances must account for those boundary transfers. The collecting-duct outlet has prescribed papillary pressure and advective solute outflow. The interstitial tip is sealed.

## Nondimensionalization and the target

The notebook identifies L=1 cm, c_star=0.001 mmol/mL (1 mmol/L), and pressure scale c_star*RT=19.344 mmHg. Code concentrations 145 and 5 therefore represent dimensionless concentrations relative to a 1 mmol/L reference; they are not normalized so that plasma is one. The cortex/plasma osmolarity reference is 2*145+5=295. The computed timescale is approximately 0.722213 in the dimensional time unit (seconds under the notebook's stated units).

Use the collecting-duct outlet metric R(t) = [2*s_C(t,outlet)+u_C(t,outlet)]/295, with positive outlet water flux. The goal is R>=20, equivalent to 5900 in the code's concentration scale. Existing plots divide by 147.5 instead, so their value is 2*R: the target is 40 a.u., and the plotted 7-a.u. reference is only 3.5 times plasma.

This model computes ideal osmolarity from concentrations. Experimental osmolality per kg water is a related but distinct quantity, particularly at very high concentrations; a precise comparison needs an explicit conversion/physical assumption.

## Differences from the supplied PDF

- The PDF defines passive transport through logarithmic chemical potentials. The code's `chemical_potential` returns concentration directly. This changes the transport law and the meaning/scaling of its coefficients; it is not just nondimensionalization.
- The PDF's illustrative salt pump is linear in A salt concentration. The code uses saturation with half-saturation concentration 150 and a different spatial extent.
- The general PDF has immobile osmolyte amount divided by volume. The code instead adds a constant interstitial osmotic contribution of 0.01.
- The code uses unequal resting areas, spatially varying permeability, nonzero tubular axial diffusion, and upwind advection. These differ from the PDF's illustrative numerical setup.
- The modified discretization reverses A gradient/divergence orientation and merges the final D/A solute storage equations to enforce a shared tip concentration.

These differences should be treated as explicit modeling choices to assess, rather than automatically classified as errors or silently reverted.

## What loading the seed actually does

Each legacy row has 407 entries corresponding to N_dyn=50: four flow profiles (A stored as one scalar) and five solute profiles. The importer interpolates D salt, A salt, C urea, interstitial salt, and interstitial urea, multiplying them by 147.5. It leaves D/A urea at 5 and C salt at 145, and regenerates native pressures and volume fractions. Saved flows are not transferred. Packing also forces A's final-cell concentration to equal D's.

Consequently, the mapped seed is a hybrid initial condition. Its C salt baseline adds 290/295 to the urine/plasma ratio relative to the interpolated legacy urea-only C profile. This explains why the mapped concentration is higher than the source's ratio; it is not newly generated concentrating power. The importer floors transferred concentrations at 1e-8, although the inspected source concentrations are already positive.

At N=50, the steady residual of mapped row 0 is about 81.78 in the unscaled maximum norm; for mapped final row it is about 80.56. These seeds therefore require transient adjustment in the modified model.

## Verification and remaining work

The three existing tests passed. They cover state packing/shared-tip equality, positive finite native initialization, and residual shape/finiteness; they do not establish scientific accuracy or stability.

Short adaptive integrations from both row 0 and the final row at N=12 reached nondimensional time 0.03 in three accepted steps of 0.01. The largest accepted residual was below 1.7e-9. Row 0 ended at outlet/plasma ratio about 4.146; the final-row seed ended at about 2.611. These are coarse-grid smoke checks, not steady results, and should not be directly compared with N=50 initial values to infer a time trend.

For the final-row smoke check, loop-tip water and solute mismatches were near floating-point precision and the DCT residual was approximately 5.7e-14. Whole-domain balance, sustained concentration, perturbation stability, and grid/time refinement remain untested.

Before seeking a 20-fold solution, establish a baseline using row 0 and track outlet/plasma ratio, positive urine flow, boundary-accounted water/solute balances, and steady residuals separately from timestep solve residuals. Compare against the final-row seed to investigate dependence on initialization. A high transient peak or a successful nonlinear timestep solve does not by itself establish a stable concentrating solution.

For biological context, experimental work reports kangaroo rats capable of urine osmolality above 6000 mosmol/kg water: https://journals.physiology.org/doi/full/10.1152/ajpregu.00289.2017 . This motivates the approximate target but does not validate the present model or establish species-specific parameters.
