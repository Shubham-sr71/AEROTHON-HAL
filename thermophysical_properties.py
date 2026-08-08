"""
====================================================================
thermophysical_properties.py

Temperature-dependent thermophysical properties for
the HPT Digital Twin.

All functions operate on torch tensors.

====================================================================
"""

import torch


# ==========================================================
# Blade Specific Heat (Inconel / Ni-based Superalloy)
# ==========================================================

def blade_cp(T):

    return (

        314.2

        + 0.406 * T

        - 1.65e-4 * T**2

        + 3.04e-8 * T**3

    )


# ==========================================================
# Combustion Gas Specific Heat
# ==========================================================

def gas_cp(T):

    return (

        1003.8

        + 0.172 * T

        + 3.4e-5 * T**2

        - 1.9e-8 * T**3

    )


# ==========================================================
# Gas Density
# ==========================================================

def gas_density(

    P,

    T,

    Z=1.0,

):

    R = 287.0

    T_safe = torch.clamp(T, min=1e-6)

    return P / (Z * R * T_safe)


# ==========================================================
# Dynamic Viscosity
# Sutherland's Law
# ==========================================================

def gas_viscosity(T):

    mu0 = 1.716e-5

    T0 = 273.15

    S = 110.4

    T_safe = torch.clamp(T, min=1e-3)

    return (

        mu0

        * torch.pow(T_safe / T0, 1.5)

        * (T0 + S)

        / (T_safe + S)

    )


# ==========================================================
# Thermal Conductivity
# Modified Sutherland Law
# ==========================================================

def gas_conductivity(T):

    k0 = 0.0241

    T0 = 273.15

    Sk = 194.0

    T_safe = torch.clamp(T, min=1e-3)

    return (

        k0

        * torch.pow(T_safe / T0, 1.5)

        * (T0 + Sk)

        / (T_safe + Sk)

    )