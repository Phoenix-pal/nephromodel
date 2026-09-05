# Physiology fix recheck

This recheck follows the issues recorded in `physiology_recheck.md`.

## Implemented corrections

- The cortical A-C/DCT junction now uses the fractional salt-reabsorption and
  fixed cortical-osmolality closure only for forward A-to-C flow.
- When either cortical boundary flow reverses, the omitted cortical segment is
  a closed connection: A and C water, salt, and urea fluxes must match. It can
  no longer create a salt or water source.
- The ascending-limb physical-path plot no longer negates a flux that is
  already positive in the papilla-to-cortex direction.
- The legacy best-seed selector now requires finite data, non-negative stored
  solutes, and a positive collecting-duct outlet flow. For
  `conv_to_third_ss.npy` it selects row 1279 rather than the prior
  high-concentration inward-flow row.
- The audit script has been updated for explicit nondimensional scales and the
  direction-aware junction law.

## Verification

`pytest -q` passed 11 tests, including independent forward and reverse
junction tests.

Two N=20 positivity-adaptive transients reached nondimensional time 1.0 in 22
accepted steps each. At their final states the cortical junction was in
`closed_reverse` mode and its net fluxes were numerically zero:

| Seed | Urine/plasma ratio | Urine flow | Water net into model | Salt net into model |
| --- | ---: | ---: | ---: | ---: |
| `dynamic_stable_v2.npy`, row 0 | 3.36657 | 0.0151011 | 2.73e-17 | -9.95e-14 |
| `conv_to_third_ss.npy`, final row | 4.16332 | 0.0130348 | -4.47e-17 | 6.92e-14 |

These checks establish numerical closure of the corrected reduced junction.
They do not validate the hybrid legacy initialization, the reduced vascular
anatomy, segment parameters, negative local pressures, or long-time/grid-time
physiological stability.
