"""
compressor_measurement_model.py

Physics-Based Compressor Measurement Model

Inputs
------
Pamb, Tamb : Ambient conditions [Pa, K]
Mach       : Flight Mach number
P2, T2     : Compressor exit conditions [Pa, K]  (measured)

Outputs
-------
P1, T1  : Ram-corrected compressor inlet stagnation conditions [Pa, K]
eta_c   : Measured (diagnostic) compression efficiency

Mirrors "Turbine Exit Temperature.py"'s MeasurementModel structure, but
runs the INVERSE direction: the turbine model predicts exit temperature
forward from an assumed eta_t; this model instead backs out eta_c from
directly measured station-2 (compressor exit) sensor data, since that's
what the dataset provides.

STATION MAPPING (do not revert):
    Pamb, Tamb, Mach -> station 1 (compressor inlet, ram-corrected, derived)
    P2, T2           -> station 2 (compressor exit, given directly)
P3, T3 (combustor exit / turbine inlet) are NOT used here -- an earlier
version of this model incorrectly used them as the compressor exit,
which combined a compression-shaped pressure term with a combustion-
shaped temperature term and produced physically invalid (negative)
efficiencies.
"""

import numpy as np


class MeasurementModel:

    def __init__(self,
                 gas_cp_function,
                 gas_constant=287.05):

        self.cp = gas_cp_function
        self.R = gas_constant

    # --------------------------------------------------

    def gamma(self, T):

        cp = self.cp(T)

        return cp / (cp - self.R)

    # --------------------------------------------------

    def inlet_stagnation_conditions(self,
                                    Pamb,
                                    Tamb,
                                    Mach):

        gamma = self.gamma(Tamb)

        ram_factor = 1.0 + (gamma - 1.0) / 2.0 * Mach**2

        T1 = Tamb * ram_factor

        P1 = Pamb * ram_factor ** (gamma / (gamma - 1.0))

        return P1, T1

    # --------------------------------------------------

    def compression_efficiency(self,
                               P1,
                               T1,
                               P2,
                               T2):

        gamma = self.gamma(T1)

        exponent = (gamma - 1.0) / gamma

        pressure_term = (P2 / P1) ** exponent - 1.0

        temp_term = (T2 / T1) - 1.0

        eta_c = pressure_term / temp_term

        return eta_c

    # --------------------------------------------------

    def update(self,
               Pamb,
               Tamb,
               Mach,
               P2,
               T2):

        P1, T1 = self.inlet_stagnation_conditions(
            Pamb,
            Tamb,
            Mach,
        )

        eta_c = self.compression_efficiency(
            P1,
            T1,
            P2,
            T2,
        )

        return eta_c, P1, T1
