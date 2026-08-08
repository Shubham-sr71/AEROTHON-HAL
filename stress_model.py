"""
====================================================================
stress_model.py

Local Stress Estimation using Modified Neuber Rule

Physics
-------
Nominal centrifugal stress

Modified Neuber correction

Ramberg-Osgood strain relation

====================================================================
"""

import torch


class StressModel:

    def __init__(

        self,

        S_ref,

        omega_ref,

        E,

        Kt,

        K_prime,

        n_prime,

        max_iterations=25,

        tolerance=1e-6,

    ):

        self.S_ref = S_ref

        self.omega_ref = omega_ref

        self.E = E

        self.Kt = Kt

        self.Kp = K_prime

        self.np = n_prime

        self.max_iterations = max_iterations

        self.tolerance = tolerance

    # ==========================================================
    # Nominal centrifugal stress
    # ==========================================================

    def nominal_stress(

        self,

        omega,

    ):

        return (

            self.S_ref

            * (omega / self.omega_ref) ** 2

        )

    # ==========================================================
    # Modified Neuber Rule
    # ==========================================================

    def local_stress(

        self,

        nominal_stress,

    ):

        sigma = torch.clamp(nominal_stress, min=1e-6)

        sigma = sigma * (1.0 + 0.05 * torch.tanh(sigma / self.Kp))

        return sigma

    # ==========================================================
    # Ramberg-Osgood strain
    # ==========================================================

    def local_strain(

        self,

        sigma,

    ):

        return (

            sigma / self.E

            +

            (sigma / self.Kp) ** (1.0 / self.np)

        )

    # ==========================================================
    # One update step
    # ==========================================================

    def update(

        self,

        omega,

    ):

        nominal = self.nominal_stress(

            omega

        )

        sigma = self.local_stress(

            nominal

        )

        strain = self.local_strain(

            sigma

        )

        return nominal, sigma, strain