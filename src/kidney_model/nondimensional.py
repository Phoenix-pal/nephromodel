"""Reference scales and conversions for the nondimensional model.

All arrays used by the solver are dimensionless.  This module is the single
place where the reference units are documented and where external values are
converted into those solver variables.

The legacy notebook uses ``c_star = 0.001 mmol/mL`` (1 mmol/L) and
``p_star = c_star * R*T = 19.344`` in its pressure units.  Consequently a
plasma salt concentration of 145 mmol/L is stored internally as 145.0, not
as 0.145 or 1.0.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceScales:
    """Dimensional reference values used only at input/output boundaries."""

    length: float = 1.0  # cm in the notebook's dimensional convention
    concentration: float = 0.001  # mmol/mL == 1 mmol/L
    RT: float = 19344.0
    plasma_salt_mmol_L: float = 145.0
    plasma_urea_mmol_L: float = 5.0
    legacy_source_concentration_mmol_L: float = 147.5

    @property
    def pressure(self) -> float:
        """Pressure scale ``c_star * R*T``."""
        return self.concentration * self.RT

    @property
    def plasma_osmolarity_mmol_L(self) -> float:
        return 2.0 * self.plasma_salt_mmol_L + self.plasma_urea_mmol_L

    def concentration_hat(self, value_mmol_L):
        """Convert concentration in mmol/L to ``c_hat = c/c_star``."""
        return np.asarray(value_mmol_L, dtype=float) / (1000.0 * self.concentration)

    def pressure_hat(self, value):
        """Convert pressure in the reference pressure units to ``p_hat``."""
        return np.asarray(value, dtype=float) / self.pressure

    def concentration_mmol_L(self, value_hat):
        """Convert solver concentration ``c_hat`` back to mmol/L."""
        return np.asarray(value_hat, dtype=float) * (1000.0 * self.concentration)

    def pressure_dimensional(self, value_hat):
        """Convert solver pressure ``p_hat`` back to the reference units."""
        return np.asarray(value_hat, dtype=float) * self.pressure
