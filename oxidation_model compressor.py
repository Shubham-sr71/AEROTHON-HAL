"""
oxidation_model.py

Thermally Grown Oxide (TGO) growth model

Hidden State
------------
x_tgo

Physics
-------
Parabolic oxidation kinetics
"""

import numpy as np


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

    # ------------------------------------------

    def oxidation_rate(

        self,

        blade_temperature,

    ):

        kp = (

            self.k0

            * np.exp(

                -self.Q

                /(self.R*blade_temperature)

            )

        )

        return kp

    # ------------------------------------------

    def update(

        self,

        x_tgo,

        blade_temperature,

        dt,

    ):

        kp = self.oxidation_rate(

            blade_temperature

        )

        dxdt = kp / (2 * max(x_tgo, 1e-9))

        x_new = x_tgo + dxdt*dt

        return x_new
    