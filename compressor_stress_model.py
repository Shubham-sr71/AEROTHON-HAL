"""
compressor_stress_model.py

Nominal centrifugal stress estimation for a compressor blade.

Simplified relative to stress_model.py (turbine): no Modified Neuber
local-stress-concentration step here, since we don't have the cyclic
material properties (Kt, K', n') that step requires for the compressor
blade -- only the nominal (equivalent mean) centrifugal stress is used
downstream by creep_model.py.

Assumptions
-----------
• Homogeneous material
• Equivalent mean stress (no spanwise stress distribution)
"""

import numpy as np


class CompressorStressModel:

    def __init__(
        self,
        blade_density,
        r_tip,
        r_root,
    ):

        self.rho = blade_density

        self.r_tip = r_tip

        self.r_root = r_root

    # -------------------------------------------------
    # Equivalent mean centrifugal stress
    # -------------------------------------------------

    def nominal_stress(
        self,
        omega,
    ):

        return (

            0.5

            * self.rho

            * omega**2

            * (self.r_tip**2 - self.r_root**2)

        )

    # -------------------------------------------------

    def update(
        self,
        rpm,
    ):

        omega = 2 * np.pi * rpm / 60.0

        S = self.nominal_stress(
            omega
        )

        return S
