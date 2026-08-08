"""
====================================================================
measurement_model.py

Physics-Based Turbine Measurement Model

Outputs
-------
Predicted Turbine Exit Temperature (T5)

====================================================================
"""

import torch


class MeasurementModel:

    def __init__(

        self,

        gas_cp_function,

        gas_constant=287.0,

    ):

        self.cp = gas_cp_function

        self.R = gas_constant

    # ==========================================================

    def gamma(

        self,

        T,

    ):

        cp = self.cp(T)

        gamma = cp / torch.clamp(cp - self.R, min=1e-6)

        return gamma

    # ==========================================================

    def isentropic_exit_temperature(

        self,

        T4,

        P4,

        P5,

    ):

        gamma = self.gamma(T4)

        exponent = torch.clamp((gamma - 1.0) / torch.clamp(gamma, min=1e-6), min=-10.0, max=10.0)

        pressure_ratio = torch.clamp(P5 / torch.clamp(P4, min=1e-6), min=1e-6, max=10.0)

        T5s = T4 * torch.pow(pressure_ratio, exponent)

        return T5s

    # ==========================================================

    def turbine_exit_temperature(

        self,

        T4,

        P4,

        P5,

        eta_t,

    ):

        T5s = self.isentropic_exit_temperature(

            T4,

            P4,

            P5,

        )

        T5 = T4 - eta_t * (T4 - T5s)

        return T5, T5s

    # ==========================================================

    def update(

        self,

        T4,

        P4,

        P5,

        eta_t,

    ):

        T5, T5s = self.turbine_exit_temperature(

            T4,

            P4,

            P5,

            eta_t,

        )

        return T5, T5s