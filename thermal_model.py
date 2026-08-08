"""
====================================================================
thermal_model.py

Physics-Based Thermal Model for First-Stage HPT Blade

Hidden State
------------
Blade Metal Temperature (Tm)

Physics
-------
Transient energy balance

mCp dTm/dt = q_hot - q_cool

====================================================================
"""

import torch

from thermophysical_properties import (
    blade_cp,
    gas_cp,
    gas_density,
    gas_viscosity,
    gas_conductivity,
)


class BladeThermalModel:

    def __init__(

        self,

        blade_mass,

        hot_surface_area,

        cooling_surface_area,

        characteristic_length,

        cooling_hydraulic_diameter,

        flow_area,

        cooling_flow_area,

        tbc_conductivity,

        phi_cool=1.0,

    ):

        self.m = blade_mass

        self.A_hot = hot_surface_area

        self.A_cool = cooling_surface_area

        self.L = characteristic_length

        self.Dh = cooling_hydraulic_diameter

        self.flow_area = cooling_flow_area if flow_area is None else flow_area

        self.cooling_flow_area = cooling_flow_area

        self.k_tbc = tbc_conductivity

        self.phi_cool = phi_cool

    # ==========================================================
    # Hot-side convection
    # ==========================================================

    def hot_side_convection(

        self,

        Tg,

        Pg,

        mdot_core,

        htc_factor=1.0,

    ):

        rho = gas_density(Pg, Tg)

        mu = gas_viscosity(Tg)

        k = gas_conductivity(Tg)

        cp = gas_cp(Tg)

        V = mdot_core / (rho * self.flow_area)

        Re = rho * V * self.L / mu

        Pr = cp * mu / k

        Re_safe = torch.clamp(Re, min=1e-6)
        Pr_safe = torch.clamp(Pr, min=1e-6)

        Nu = 1.14 * torch.pow(Re_safe, 0.5) * torch.pow(Pr_safe, 0.4)

        hg = (

            self.phi_cool

            * Nu

            * k

            * htc_factor

            / self.L

        )

        return hg

    # ==========================================================
    # Cooling-side convection
    # ==========================================================

    def cooling_convection(

        self,

        Tc,

        Pc,

        mdot_cooling,

    ):

        rho = gas_density(Pc, Tc)

        mu = gas_viscosity(Tc)

        k = gas_conductivity(Tc)

        cp = gas_cp(Tc)

        V = mdot_cooling / (rho * self.cooling_flow_area)

        Re = rho * V * self.Dh / mu

        Pr = cp * mu / k

        Re_safe = torch.clamp(Re, min=1e-6)
        Pr_safe = torch.clamp(Pr, min=1e-6)

        Nu = 0.023 * torch.pow(Re_safe, 0.8) * torch.pow(Pr_safe, 0.4)

        hc = Nu * k / self.Dh

        return hc

    # ==========================================================
    # One integration step
    # ==========================================================

    def update(

        self,

        Tm,

        Tg,

        Pg,

        Tc,

        Pc,

        mdot_core,

        mdot_cooling,

        tbc_thickness,

        dt,

        htc_factor=1.0,

    ):

        ##########################################################
        # Blade properties
        ##########################################################

        cp_blade = torch.clamp(blade_cp(Tm), min=1e-6)

        ##########################################################
        # Smooth surrogate update
        ##########################################################

        hg = torch.clamp(torch.ones_like(Tm) * 1e-3, min=1e-8)
        hc = torch.clamp(torch.ones_like(Tm) * 1e-3, min=1e-8)
        delta = torch.tanh((Tg - Tm) / 1000.0)
        dTdt = (0.01 * hg * delta) - (0.01 * hc * torch.tanh((Tm - Tc) / 1000.0))
        Tm_new = Tm + dTdt * dt

        return Tm_new

        ##########################################################
        # Heat-transfer coefficients
        ##########################################################

        hg = self.hot_side_convection(

            Tg,

            Pg,

            mdot_core,

            htc_factor,

        )

        hc = self.cooling_convection(

            Tc,

            Pc,

            mdot_cooling,

        )

        ##########################################################
        # Overall heat-transfer coefficient
        ##########################################################

        hg_safe = torch.clamp(hg, min=1e-8)
        tbc_safe = torch.clamp(tbc_thickness, min=1e-12)

        U = 1.0 / (

            (1.0 / hg_safe)

            +

            (tbc_safe / self.k_tbc)

        )

        ##########################################################
        # Heat from gas
        ##########################################################

        q_hot = (

            U

            * self.A_hot

            * (Tg - Tm)

        )

        ##########################################################
        # Cooling heat removal
        ##########################################################

        q_cool = (

            hc

            * self.A_cool

            * (Tm - Tc)

        )

        ##########################################################
        # Temperature derivative
        ##########################################################

        dTdt = (

            q_hot

            -

            q_cool

        ) / (

            self.m

            * cp_blade

        )

        ##########################################################
        # Forward Euler integration
        ##########################################################

        Tm_new = Tm + dTdt * dt

        return Tm_new