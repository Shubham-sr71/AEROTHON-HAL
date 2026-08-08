"""
====================================================================
tmf_model.py

Thermo-Mechanical Fatigue Model

Hidden State
------------
Accumulated TMF Damage

Physics
-------
Coffin-Manson + Basquin
Ramberg-Osgood
Miner Damage Rule

====================================================================
"""

import torch


class TMFModel:

    def __init__(

        self,

        E,

        K_prime,

        n_prime,

        sigma_f,

        epsilon_f,

        b,

        c,

        max_iterations=30,

        tolerance=1e-6,

    ):

        self.E = E

        self.Kp = K_prime

        self.np = n_prime

        self.sigma_f = sigma_f

        self.epsilon_f = epsilon_f

        self.b = b

        self.c = c

        self.max_iterations = max_iterations

        self.tolerance = tolerance

    # ==========================================================
    # Ramberg-Osgood strain range
    # ==========================================================

    def strain_range(

        self,

        delta_sigma,

    ):

        return (

            delta_sigma / self.E

            +

            torch.pow(

                delta_sigma / (2.0 * self.Kp),

                1.0 / self.np,

            )

        )

    # ==========================================================
    # Solve Coffin-Manson-Basquin equation
    # ==========================================================

    def cycles_to_failure(

        self,

        delta_sigma,

        sigma_mean,

    ):

        delta_eps = self.strain_range(

            delta_sigma

        )

        Nf = torch.tensor(

            1.0e4,

            dtype=delta_sigma.dtype,

            device=delta_sigma.device,

        )

        for _ in range(self.max_iterations):

            term1 = (

                ((self.sigma_f - sigma_mean) / self.E)

                *

                torch.pow(

                    2.0 * Nf,

                    self.b,

                )

            )

            term2 = (

                self.epsilon_f

                *

                torch.pow(

                    2.0 * Nf,

                    self.c,

                )

            )

            f = (

                delta_eps / 2.0

                -

                term1

                -

                term2

            )

            df = (

                -((self.sigma_f - sigma_mean) / self.E)

                * self.b

                * 2.0

                * torch.pow(

                    2.0 * Nf,

                    self.b - 1.0,

                )

                -

                self.epsilon_f

                * self.c

                * 2.0

                * torch.pow(

                    2.0 * Nf,

                    self.c - 1.0,

                )

            )

            Nf_new = Nf - f / df

            Nf_new = torch.clamp(

                Nf_new,

                min=1.0,

            )

            if torch.abs(Nf_new - Nf) < self.tolerance:

                Nf = Nf_new

                break

            Nf = Nf_new

        return Nf

    # ==========================================================
    # Update damage
    # ==========================================================

    def update(

        self,

        damage_old,

        delta_sigma,

        sigma_mean,

        cycles=1.0,

    ):

        Nf = self.cycles_to_failure(

            delta_sigma,

            sigma_mean,

        )

        dD = cycles / Nf

        damage_new = damage_old + dD

        return damage_new, dD, Nf