"""
====================================================================
creep_model.py

Physics-Based Secondary Creep Model

Hidden State
------------
Accumulated Creep Strain

Model
-----
Norton-Bailey Law

ε̇ = A σⁿ exp(-Q / RT)

where

ε̇ : creep strain rate
σ  : von Mises stress
T  : blade metal temperature

The creep_factor is learned by the Transformer through the
mapping layer to account for unmodelled degradation effects.
====================================================================
"""

import torch


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

    # ==========================================================
    # Creep strain rate
    # ==========================================================

    def creep_rate(

        self,

        blade_temperature,
        stress,

        creep_factor=1.0,

    ):

        stress_safe = torch.clamp(stress, min=1e-6, max=1e8)
        temp_safe = torch.clamp(blade_temperature, min=300.0, max=2000.0)

        rate = (

            1e-24

            * creep_factor

            * torch.pow(stress_safe, 0.5)

            * torch.exp(

                torch.clamp(
                    -self.Q / (self.R * temp_safe),
                    min=-20.0,
                    max=0.0,
                )

            )

        )

        return rate

    # ==========================================================
    # One integration step
    # ==========================================================

    def update(

        self,

        creep_strain,

        blade_temperature,

        stress,

        dt,

        creep_factor=1.0,

    ):

        rate = self.creep_rate(

            blade_temperature=blade_temperature,

            stress=stress,

            creep_factor=creep_factor,

        )

        creep_new = (

            creep_strain

            + torch.clamp(rate * dt, max=1e-4)

        )

        return creep_new