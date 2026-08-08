"""
====================================================================
oxidation_model.py

Physics-Based Thermally Grown Oxide (TGO) Model

Hidden State
------------
TGO Thickness (x_tgo)

Physics
-------
Parabolic Oxidation Law

dx/dt = kp / (2x)

where

kp = k0 exp(-Q/RT)

The oxidation_factor is provided by the Transformer through
the mapping layer to account for unmodelled oxidation effects.

====================================================================
"""

import torch


class OxidationModel:

    def __init__(

        self,

        k0,

        activation_energy,

        gas_constant=8.314,

    ):

        self.k0 = k0

        self.Q = activation_energy

        self.R = gas_constant

    # ==========================================================
    # Parabolic oxidation constant
    # ==========================================================

    def oxidation_rate(

        self,

        blade_temperature,

        oxidation_factor=1.0,

    ):

        kp = (

            self.k0

            * oxidation_factor

            * torch.exp(

                torch.clamp(
                    -self.Q / (self.R * torch.clamp(blade_temperature, min=1e-3)),
                    min=-50.0,
                    max=0.0,
                )

            )

        )

        return kp

    # ==========================================================
    # One integration step
    # ==========================================================

    def update(
        self,
        x_tgo,
        blade_temperature,
        dt,
        oxidation_factor=1.0,
    ):

        kp = self.oxidation_rate(
            blade_temperature,
            oxidation_factor,
        )

        x_tgo_safe = torch.clamp(x_tgo, min=1e-9)

        dxdt = kp / (2.0 * x_tgo_safe)

        x_new = x_tgo + dxdt * dt

        return x_new