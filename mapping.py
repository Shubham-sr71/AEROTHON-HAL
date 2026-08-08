"""
====================================================================
mapping.py

Physics Mapping Layer

Maps the latent degradation state predicted by the Transformer
to physically meaningful correction factors.

The governing physics equations are NEVER modified.

Only uncertain parameters are corrected.

====================================================================
"""

from dataclasses import dataclass

import torch


# ==============================================================
# Physics Corrections
# ==============================================================

@dataclass
class PhysicsCorrections:

    gas_htc_factor: torch.Tensor

    oxidation_factor: torch.Tensor

    creep_factor: torch.Tensor

    clearance_offset: torch.Tensor

    efficiency_offset: torch.Tensor

    latent_delta: torch.Tensor


# ==============================================================
# Physics Mapping
# ==============================================================

class PhysicsMapping:

    def __init__(self):

        ##########################################################
        # Maximum allowable corrections
        ##########################################################

        self.max_htc_loss = 0.25

        self.max_kp_increase = 0.50

        self.max_creep_increase = 0.40

        self.max_clearance = 2.0e-4

        self.max_eta_loss = 0.03

    # ==========================================================

    def map(

        self,

        z,

    ):

        """
        Parameters
        ----------
        z : torch.Tensor (...,4)

        z[...,0] Thermal

        z[...,1] Material

        z[...,2] Geometry

        z[...,3] Aerodynamics

        Returns
        -------
        PhysicsCorrections
        """

        ##########################################################
        # Keep latent state bounded
        ##########################################################

        z = torch.clamp(

            z,

            min=0.0,

            max=1.0,

        )

        ##########################################################
        # Thermal
        ##########################################################

        htc_factor = (

            1.0

            - self.max_htc_loss * z[..., 0]

        )

        ##########################################################
        # Material
        ##########################################################

        oxidation_factor = (

            1.0

            + self.max_kp_increase * z[..., 1]

        )

        creep_factor = (

            1.0

            + self.max_creep_increase * z[..., 1]

        )

        ##########################################################
        # Geometry
        ##########################################################

        clearance_offset = (

            self.max_clearance * z[..., 2]

        )

        ##########################################################
        # Aerodynamics
        ##########################################################

        efficiency_offset = (

            -self.max_eta_loss * z[..., 3]

        )

        ##########################################################

        return PhysicsCorrections(

            gas_htc_factor=htc_factor,

            oxidation_factor=oxidation_factor,

            creep_factor=creep_factor,

            clearance_offset=clearance_offset,

            efficiency_offset=efficiency_offset,

            latent_delta=z,

        )