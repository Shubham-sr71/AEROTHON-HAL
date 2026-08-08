"""
creep_model.py

Primary hidden state

Accumulated creep strain
"""

import numpy as np


class CreepModel:

    def __init__(

        self,

        A,

        n,

        activation_energy,

        gas_constant=8.314,

    ):

        self.A = A

        self.n = n

        self.Q = activation_energy

        self.R = gas_constant

    # ------------------------------------------

    def creep_rate(

        self,

        blade_temperature,

        stress,

    ):

        rate = (

            self.A

            * stress**self.n

            * np.exp(

                -self.Q

                /(self.R*blade_temperature)

            )

        )

        return rate

    # ------------------------------------------

    def update(

        self,

        creep_strain,

        blade_temperature,

        stress,

        dt,

    ):

        rate = self.creep_rate(

            blade_temperature,

            stress,

        )

        creep_new = (

            creep_strain

            +

            rate*dt

        )

        return creep_new