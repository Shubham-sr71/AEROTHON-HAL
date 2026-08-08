"""
====================================================================
turbine_efficiency.py

Physics-Based Turbine Efficiency Model

Hidden State
------------
Accumulated Creep Strain

Outputs
-------
Tip Clearance
Turbine Efficiency

====================================================================
"""

import torch


class TurbineEfficiencyModel:

    def __init__(

        self,

        blade_height,

        initial_clearance,

        clean_efficiency,

        K1,

        K2,

    ):

        self.h = blade_height

        self.c0 = initial_clearance

        self.eta0 = clean_efficiency

        self.K1 = K1

        self.K2 = K2

    # ==========================================================

    def blade_growth(

        self,

        creep_strain,

    ):

        return self.h * creep_strain

    # ==========================================================

    def tip_clearance(

        self,

        creep_strain,

        clearance_offset=0.0,

    ):

        creep_safe = torch.clamp(creep_strain, min=0.0, max=1e-3)
        clearance = (

            self.c0

            -

            self.blade_growth(creep_safe)

            +

            clearance_offset

        )

        return torch.clamp(clearance, min=1e-6)

    # ==========================================================

    def efficiency(

        self,

        clearance,

        efficiency_offset=0.0,

    ):

        delta = clearance / self.h

        eta = (

            self.eta0

            -

            0.5 * self.K1 * torch.tanh(delta)

            -

            0.25 * self.K2 * delta**2

            +

            efficiency_offset

        )

        eta = torch.clamp(eta, min=0.5, max=1.0)
        return eta
        return eta

    # ==========================================================

    def update(

        self,

        creep_strain,

        clearance_offset=0.0,

        efficiency_offset=0.0,

    ):

        clearance = self.tip_clearance(

            creep_strain,

            clearance_offset,

        )

        eta = self.efficiency(

            clearance,

            efficiency_offset,

        )

        return clearance, eta