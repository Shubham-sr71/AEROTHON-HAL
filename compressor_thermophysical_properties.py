"""
compressor_thermophysical_properties.py

Air and Ti-alloy blade property correlations for the compressor side.
Mirrors thermophysical_properties.py (turbine, Ni-superalloy + combustion
gas) but swapped for air (pre-combustion) and IMI 834 (near-alpha Ti,
real ~600C HPC blade alloy).
"""

import numpy as np

# -----------------------------------------------------
# Blade Specific Heat (IMI 834, near-alpha Ti alloy)
# -----------------------------------------------------

def blade_cp(T):

    return (
        520.0
        +0.20*T
    )


# -----------------------------------------------------
# Air Specific Heat (compressor working fluid, pre-combustion)
# -----------------------------------------------------

def gas_cp(T):

    return (
        1005.0
        +0.05*T
        +1.5e-5*T**2
    )


# -----------------------------------------------------
# Density
# -----------------------------------------------------

def gas_density(
    P,
    T,
    Z=1.0,
):

    R = 287.05

    return P / (Z * R * T)


# -----------------------------------------------------
# Dynamic Viscosity
# Sutherland
# -----------------------------------------------------

def gas_viscosity(T):

    mu0 = 1.716e-5

    T0 = 273.15

    S = 110.4

    return (
        mu0
        * (T / T0)**1.5
        * (T0 + S)
        / (T + S)
    )


# -----------------------------------------------------
# Thermal Conductivity
# Modified Sutherland
# -----------------------------------------------------

def gas_conductivity(T):

    k0 = 0.0241

    T0 = 273.15

    Sk = 194.0

    return (
        k0
        * (T / T0)**1.5
        * (T0 + Sk)
        / (T + Sk)
    )
