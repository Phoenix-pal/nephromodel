The numerical fixes improve the model substantially, but the present code is not yet physiologically validated for finding a desert-rodent maximum urine concentration. This review covers the current source after commit `0da5fd4` and the working changes to nondimensionalization and fixed-step solving. It supersedes the earlier junction-mismatch finding: that specific mismatch is now fixed. Earlier audit figures describe the earlier code, not this revision.

**What is now verified**

All nine existing tests passed. Two additional transient checks used N=20, initial dt=0.02, maximum dt=0.05, and the positivity-adaptive runner to nondimensional time 1.0. Both reached the requested time with 22 accepted steps. These checks are short transients, not steady-state or mesh-convergence studies.

| Check | dynamic_stable_v2, row 0 | conv_to_third_ss, final row |
| --- | ---: | ---: |
| Maximum accepted timestep residual | 1.26e-8 | 1.26e-8 |
| Maximum actual DCT salt/urea flux mismatch | 4.64e-13 | 4.23e-14 |
| Maximum whole-domain solute balance error | 1.47e-10 | 1.78e-10 |
| Final urine/plasma ratio | 3.3217 | 4.0899 |
| Final urine flow / prescribed descending inlet flow | 4.0544 | 3.4813 |
| Final tubular faces with negative oriented flow | 28 | 29 |
| Final steady residual, maximum norm | 30.4820 | 30.0099 |

The DCT equations now use the actual upwind boundary fluxes. The tip and boundary-accounted whole-domain balances also pass these checks. Fixed-step acceptance now checks residual magnitude and can refine a candidate with tighter nonlinear tolerance. `ReferenceScales` makes concentration and pressure conversions explicit and preserves the original default numerical values.

**1. Remaining major physiological issue: reversed flow changes cortical salt reabsorption into salt supply.**

In `transport.py:60`, the equation remains F_C,s = q F_A,s, with q=1/3, regardless of flow direction. F_A is signed from A into the cortical junction; F_C is signed from the junction into C. The net cortical salt delivery to the represented medulla is therefore F_C,s - F_A,s = (q-1)F_A,s. For forward flow this is negative, representing reabsorption outside the modeled medulla. When F_A,s is negative, it is positive: the omitted cortical segment supplies salt.

This is not a numerical conservation mismatch anymore. It is the physical meaning of the equations that needs resolution. In the row-0 recheck at time 1, cortical salt supply was +0.143319 in scaled flux units; cortical water removal was -0.00140064, meaning net water supply by this junction. Reverse tubular flows were present at all accepted steps of that run.

A transient flow reversal is not automatically impossible. However, a rule justified as fixed fractional salt reabsorption cannot be extended to reverse flow without specifying the salt source and transport mechanism. Either constrain the reduced cortical model to its intended forward-flow operating regime, with a compatible initialization, or provide a direction-aware cortical transport/balance model. Reducing dt alone is not a physiological correction.

The high urine flow relative to descending inflow is not, by itself, a mass-conservation failure: the interstitial cortical boundary can supply water and the cortical junction also exchanges water. It does demonstrate that this transient is not simply concentrating a fixed descending inflow by water extraction.

**2. The test described as a reverse-flow test actually uses forward flow.**

`tests/test_model.py:43` sets A pressure to 1.0 and C pressure to 0.5. The solved junction pressure is approximately 0.562048, producing Q_A=+0.064434 and Q_C=+0.018930 for N=6. Thus its comment claiming reverse A flow is incorrect, and the asserted test does not cover that branch. Swapping those pressures gives genuinely negative boundary flows; an independent check confirmed that the new numerical flux identities hold there too. Add explicit flow-sign assertions and cover both regimes, including the intended cortical source/sink behavior.

**3. The ascending physical-path plot still reverses the sign a second time.**

`visualization.py:123` plots `-flux[KA, ::-1]` and labels it flow along the papilla-to-cortex path. The model's A gradient and divergence already make positive Q_A correspond to that direction. Reverse the indexing for path order, but do not negate this quantity when labeling it along physical A direction. A separate common cortex-to-papilla signed-coordinate plot would require a different sign convention and label.

**4. The best-seed selector can select reverse urine flow.**

`initialization.py:67` checks only whether the outlet concentration is finite. For `conv_to_third_ss.npy`, it selects row 12653, whose legacy collecting-duct outlet water flow is -0.00374962. This is a maximum concentration value but not an outward urine stream. The chosen row for `dynamic_stable_v2.npy` remains row 0 and its source outlet flow is positive.

Screen the complete selected row for finite, admissible concentration data and positive source outlet flow when describing it as a physiological seed. Recheck the mapped state separately because saved flows are discarded. Also, source endpoint concentration does not generally order mapped outlet concentration: interpolation uses neighboring source points. The selector's claim of monotonic ranking is only true under additional profile assumptions, not for arbitrary trajectories.

**5. The importer still creates a hybrid initial condition.**

`initialization.py` transfers five solute profiles while `state.py` supplies native volumes and pressures. D/A urea remains at the baseline and collecting-duct salt remains at 145. The shared-tip packing overwrites the A tip concentration with the D value. Saved axial flows are not transferred, and interpolation is not inventory-preserving.

These choices are permissible as initial guesses but cannot establish that the full model starts from the source model's steady state. The current scale refactor does not remove the extra collecting-duct salt inventory. Previously, switching off collecting-duct water permeability still left a high initial urine concentration. That illustrates seed retention; it does not show that water impermeability improves physiological concentration.

**6. The anatomical reduction still limits a desert-rodent interpretation.**

`constants.py`, `state.py`, and `parameters.py` represent one D, one A, one C, and a combined interstitium/vessel compartment. There are no independent descending/ascending vasa recta, vascular membrane exchanges, blood perfusion, or oxygen transport. All loops reach the same depth; no loop-length distribution or collecting-duct coalescence is represented. The approximately 67% resting area assigned to compartment 0 is an effective model compartment, not a measured interstitial volume fraction.

This is a legitimate reduced model, but it cannot directly test how vascular countercurrent exchange or nephron populations contribute to a species' concentrating ability. Kangaroo-rat anatomical reconstructions show distinct descending and ascending vessels and organized interactions with collecting ducts. A shared core can only approximate their combined effects. [Primary vascular anatomy study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3469668/).

**7. Segment transport needs species-specific justification.**

`parameters.py` makes D water-permeable all the way to the tip and imposes abrupt permeability changes by grid index. A is water-impermeable, and its superficial region actively transports salt. The latter two are useful physiological features. However, the model does not explicitly identify proximal straight, descending thin, prebend, ascending thin, and thick ascending segments or give sources for their boundaries and parameter values.

Primary kangaroo-rat anatomy found a longer AQP1-positive descending segment relative to loop length than in Munich-Wistar rat, with a distinct prebend region. That supports testing segment lengths rather than assuming uniform water permeability. [Kangaroo-rat thin-limb anatomy](https://pmc.ncbi.nlm.nih.gov/articles/PMC3774486/).

Measured Munich-Wistar thin-limb urea permeabilities differ substantially by segment: approximately 40 to 265 x 10^-5 cm/s in the cited experiments. The code uses 1.5 x 10^-5 throughout D and 0.86 or 6.7 x 10^-5 in A. These are material differences to investigate, not a reason to substitute laboratory-rat measurements into a desert-rodent model without qualification. [Primary perfused-tubule study](https://pubmed.ncbi.nlm.nih.gov/24197065/).

The encoded water coefficients are not obviously orders of magnitude implausible. Interpreting them as hydraulic permeabilities in cm/s/mmHg, Pf = Lp*RT/Vwater with Vwater approximately 0.018 mL/mmol gives D values around 2,418–3,634 micrometers/s and a C value around 424 micrometers/s. Their original units and sources should nevertheless be documented.

**8. Passive flux and a saturating pump are defensible reduced laws.**

`operators.py` calls concentration a chemical potential, although the PDF defines a logarithmic potential. The implemented concentration-difference flux is a different constitutive law, but it is also used in renal transport modeling with permeability coefficients in velocity units. It should be named and documented honestly rather than replaced automatically by a logarithm. A saturating salt transport law is likewise defensible; transporter kinetics and membrane area must have compatible units. [Primary renal modeling study with these transport laws](https://pmc.ncbi.nlm.nih.gov/articles/PMC2877498/).

The salt pump acts without an explicit ATP, oxygen, or energy budget. Raising its magnitude may therefore produce a mathematical result without demonstrating energetic feasibility. Kangaroo-rat TAL Na/K-ATPase measurements motivate investigating stronger transport, but do not identify a unique multiplier for this model. [Primary comparative transport study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5966814/).

**9. ADH and urea recycling remain simplified.**

`collecting_water_permeable` switches C water permeability between zero and one constant profile. There is no vasopressin variable or timed response, and C urea permeability is unaffected by the switch. The cortical inlet is still forced to plasma osmolarity even when medullary C water permeability is zero. Thus the switch is a water-permeability experiment, not a full simulation of ADH withdrawal.

Vasopressin-dependent AQP2 trafficking has been measured directly, and experimental restoration of UT-A1 in transporter-deficient mice changes concentrating ability. A physiological antidiuresis protocol should distinguish water permeability, terminal C urea transport, cortical delivery, and elapsed adaptation time. [AQP2 trafficking experiment](https://pubmed.ncbi.nlm.nih.gov/7573395/), [urea-transporter restoration experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC4849813/).

The absence of an explicit reflection coefficient is not automatically an error: unity reflection for urea is supported by rat terminal collecting-duct experiments despite appreciable urea permeability. Avoid assuming that high permeability necessarily requires a small reflection coefficient. [Primary reflection-coefficient experiment](https://pubmed.ncbi.nlm.nih.gov/2705534/).

**10. Mechanical and dimensional assumptions need an audit before changing geometry.**

`residuals.py` enforces constant total volume and a linear pressure–area compliance. These equations are coherent within the reduced model. Their applicable pressure range and the fixed membrane perimeter during volume changes are not validated. The single effective core resistance is not a substitute for calibrated tissue and vascular resistances.

`parameters.py` uses 8*pi^2*viscosity in hydraulic resistance. For a single circular tube with cross-sectional area A=pi*r^2, Poiseuille flow gives Q = -A^2/(8*pi*viscosity) dp/dx. The extra pi needs an explicit effective-geometry justification if the radii are literal cylindrical radii; it should not be silently corrected without determining how the original coefficient was calibrated. [Primary hemodynamic modeling paper stating the cylindrical relation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7725998/).

The new `ReferenceScales.length` is currently unused by `build_parameters`, which independently sets L=1 and uses it in dimensional scaling. Custom scales cannot be passed to the builder. The defaults work, but this is not yet a general geometry/rescaling interface. Several coefficients, including compliance and colloid, remain directly specified as dimensionless constants. A species/length sweep should rebuild every derived coefficient from a documented dimensional parameter set.

**11. The osmotic target still requires the correct metric.**

Use collecting-duct outlet R = (2*salt_C + urea_C)/plasma_osmolarity with positive outlet flow. The plasma reference is 295; the legacy plot reference is 147.5. Thus R=20 corresponds to 5900 in the default internal osmolarity scale and 40 legacy a.u. The plotting refactor preserves this factor of two; it does not normalize plasma to one.

The two-solute model computes ideal osmolarity. It does not explicitly model potassium, other urinary solutes, concentration-dependent osmotic activities, or the conversion to experimental osmolality per kg water. Lumping salt is reasonable for a reduced concentration model, but the result is not a full urine composition prediction.

**12. Numerical success still does not mean physiological stability.**

`solver.py` now validates the nonlinear timestep solve more carefully. It still does not establish a steady state, physically acceptable flow regime, or grid/time convergence. `diagnostics.py` has useful new boundary-accounted balances, but the CLI supplies no previous state when printing them, so that invocation omits the balance errors. The solver also does not use these diagnostics as acceptance conditions or record cortical source terms.

The forced shared D/A final concentration merges two full cells into one solute storage node. This conserves their combined solute inventory, but it is a finite mixing region, not merely an equality at a zero-volume boundary. Its effective axial extent shrinks with grid refinement. Index-based segment changes likewise move with N: at N=50 the pump occupies 40% of the domain, while the first A permeability region occupies 42%. These effects should be included in a convergence study before interpreting a maximum.

`plotting.py` and `visualization.py` retain legacy units and the arbitrary 7-a.u. reference. The animation's `gamma_label` is a display argument, not a simulation parameter. `pipelines.py`, CLI defaults, and file names describing a third steady state do not establish that an imported profile is a steady state of these equations. `__init__.py` and `__main__.py` are routing/export modules with no additional physiological laws.

**Recommended order**

First resolve the cortical source/sink law during reverse flow, fix the A-flow plot, and correct the reverse-flow test and seed screening. Then choose and document a species and dimensional parameter set, including segment boundaries and the intended vascular reduction. Establish a consistent baseline and evaluate long-time urine/plasma ratio together with urine flow, salt/urea excretion, cortical and vascular exchange, inventories, steady residuals, and grid/time refinement. A maximum should be searched under those constraints, rather than by optimizing an unconstrained peak concentration. Primary optimization work has likewise constrained physiological urine flow while maximizing concentration. [Constrained urine-concentration optimization](https://pubmed.ncbi.nlm.nih.gov/19915926/).

The recheck data are saved in `results/physiology_recheck_results.json`. No model source files were modified during this recheck.
