"""
====================================================================
compressor_thermal_model.py

Physics-Based Thermal Model for HPC Blade

State Variable
--------------
Blade Metal Temperature (Tm)

Physics
-------
Lumped-capacitance energy balance on an UNCOOLED compressor blade
(mirrors thermal_model.py's structure, but no internal cooling term
and no TBC layer -- compressor blades in this model are bare metal,
uncooled).

m_b Cp_b(Tm) dTm/dt = Heat from compressed air (hot-side convection only)

Assumptions
-----------
• Lumped thermal capacitance
• Uniform blade metal temperature
• Uncooled blade (no internal cooling flow, no TBC)
• Temperature-dependent gas/blade properties
• Empirical Nusselt correlation, constant blade geometry
====================================================================
"""

import numpy as np

from compressor_thermophysical_properties import (
    blade_cp,
    gas_cp,
    gas_density,
    gas_viscosity,
    gas_conductivity,
)


class CompressorBladeThermalModel:

    def __init__(
        self,
        blade_mass,
        hot_surface_area,
        characteristic_length,
        flow_area,
    ):

        self.m = blade_mass

        self.A_hot = hot_surface_area

        self.L = characteristic_length

        self.flow_area = flow_area

    # ==========================================================
    # Convection (compressed-air side)
    # ==========================================================

    def hot_side_convection(
        self,
        Tg,
        Pg,
        mdot_core,
    ):

        rho = gas_density(Pg, Tg)

        mu = gas_viscosity(Tg)

        k = gas_conductivity(Tg)

        cp = gas_cp(Tg)

        V = mdot_core / (rho * self.flow_area)

        Re = rho * V * self.L / mu

        Pr = cp * mu / k

        #
        # Empirical Nu correlation (Dittus-Boelter form)
        #
        Nu = 0.023 * Re**0.8 * Pr**0.4

        hg = Nu * k / self.L

        return hg

    # ==========================================================
    # One integration step
    # ==========================================================

    def update(
        self,
        Tm,
        Tg,
        Pg,
        mdot_core,
        dt,
    ):

        # ----------------------------------------
        # Blade properties
        # ----------------------------------------

        cp_blade = blade_cp(Tm)

        # ----------------------------------------
        # Heat transfer coefficient
        # ----------------------------------------

        hg = self.hot_side_convection(
            Tg,
            Pg,
            mdot_core,
        )

        # ----------------------------------------
        # Heat entering blade (uncooled -- no TBC/internal-cooling term)
        # ----------------------------------------

        q_hot = (
            hg
            * self.A_hot
            * (Tg - Tm)
        )

        # ----------------------------------------
        # Temperature derivative
        # ----------------------------------------

        dTdt = q_hot / (
            self.m
            * cp_blade
        )

        # ----------------------------------------
        # Forward Euler
        # ----------------------------------------

        Tm_new = Tm + dTdt * dt

        return Tm_new
