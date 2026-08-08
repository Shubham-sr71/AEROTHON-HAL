"""
physics.py
==========
Physics core of the Combustor Digital Twin.

Implements a first-principles energy-balance model of a gas-turbine combustor,
augmented with the Lefebvre combustion-loading parameter and a data-calibrated
health-degradation term. This module contains NO machine learning — it is the
"white box" backbone that the hybrid model (model.py) corrects with a learned
residual.

Station numbering (matches the provided CSVs):
    2 -> combustor inlet   (compressor discharge):  P2_Pa, T2_K
    3 -> combustor exit / turbine inlet (TARGET):    P3_Pa, T3_K
    4 -> turbine exit (consumed by the turbine twin): P4_Pa, T4_K

Governing energy balance (per unit mass of air, adiabatic combustor):
    m_dot_fuel * LHV * eta_b  =  (m_dot_air + m_dot_fuel) * cp_gas * (T3 - T2)

Rearranged with FAR = m_dot_fuel / m_dot_air:
    T3 = T2 + FAR * eta_b * LHV / (cp_gas * (1 + FAR))

Combustion efficiency eta_b is decomposed into two independent, physically
distinct effects (standard practice in gas-turbine performance/PHM modelling):
    eta_b(theta, H) = eta_max * H * eta_lefebvre(theta)
        H              -> slow health degradation (fouling, liner wear,
                           injector coking) that develops over engine life
        eta_lefebvre() -> fast, reversible aerothermal effect of the
                           instantaneous operating point (low pressure /
                           high velocity -> weak-extinction / lean blow-out
                           risk), captured by Lefebvre's loading parameter.

Lefebvre loading parameter (Lefebvre & Ballal, "Gas Turbine Combustion"):
    theta = m_dot_air * exp(T2 / b) / (P3 ** n)
Combustion efficiency rises steeply with theta and saturates near 1 for a
healthy, well-loaded combustor:
    eta_lefebvre(theta) = 1 - exp(-c * theta ** k)

Because the dataset gives no combustor reference volume/geometry, `theta`
here is a dimensionally-reduced PROXY (V_ref folded into the calibrated
constant `c`) — standard practice when only sensor data (no CAD) is
available. This is documented explicitly so it is never mistaken for a
geometry-exact figure.

Air mass flow is not measured directly in the data, so it is estimated with
a calibrated corrected-flow power law:
    m_dot_air = C * P2^a * RPM^b * T2^c
whose exponents are fit once (see calibrate()) against the energy balance
solved on the labelled training set, then frozen for deployment/inference —
exactly how a real engine model's flow function would be calibrated against
rig/fleet data.
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---- Fixed thermodynamic constants (jet fuel / combustion gas) ----------
LHV = 43.0e6        # J/kg, lower heating value of Jet-A / kerosene
CP_GAS = 1150.0      # J/(kg.K), mean cp of combustion products (hot side)
ETA_MAX = 0.999      # asymptotic max combustion efficiency, healthy combustor
B_CONST = 300.0      # K, empirical temperature-scaling in loading parameter


@dataclass
class CombustorPhysicsParams:
    """Calibrated physics constants for one instance of the twin."""
    mair_a: float       # exponent on P2 in air-flow proxy
    mair_b: float       # exponent on RPM in air-flow proxy
    mair_c: float       # exponent on T2 in air-flow proxy
    mair_logC: float    # ln(C) prefactor in air-flow proxy
    lef_c: float         # Lefebvre efficiency curve coefficient
    lef_k: float         # Lefebvre efficiency curve exponent
    health_slope: float  # population-level dHealth/dCycle (prior)
    health_intercept: float  # population-level health at Cycle=0 (prior)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CombustorPhysicsParams":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


def air_mass_flow(P2, RPM, T2, params: CombustorPhysicsParams):
    """Calibrated corrected-flow proxy for combustor (compressor discharge) air flow [kg/s-scale]."""
    return np.exp(params.mair_logC) * (P2 ** params.mair_a) * (RPM ** params.mair_b) * (T2 ** params.mair_c)


def loading_parameter(m_air, T2, P3, n=1.75):
    """Lefebvre combustion loading parameter proxy (see module docstring)."""
    return m_air * np.exp(T2 / B_CONST) / (P3 ** n)


def eta_lefebvre(theta, params: CombustorPhysicsParams):
    """Saturating efficiency-vs-loading curve: eta = 1 - exp(-c * theta^k)."""
    theta = np.clip(theta, 1e-30, None)
    return 1.0 - np.exp(-params.lef_c * theta ** params.lef_k)


def population_health_prior(cycle, params: CombustorPhysicsParams):
    """Fleet-level linear degradation prior: CombustorHealth ~ a + b*Cycle, clipped to physical range."""
    h = params.health_intercept + params.health_slope * cycle
    return np.clip(h, 0.5, 1.0)


def combustion_efficiency(theta, health, params: CombustorPhysicsParams):
    """eta_b = eta_max * Health * eta_lefebvre(theta)."""
    return ETA_MAX * health * eta_lefebvre(theta, params)


def t3_energy_balance(T2, fuel_flow, m_air, eta_b):
    """Solve the combustor energy balance for T3 given a combustion efficiency."""
    FAR = fuel_flow / np.clip(m_air, 1e-9, None)
    return T2 + FAR * eta_b * LHV / (CP_GAS * (1.0 + FAR)), FAR


def predict_t3_physics(P2, RPM, T2, P3, fuel_flow, cycle, params: CombustorPhysicsParams,
                        health_override=None):
    """
    Full white-box forward model: operating conditions -> T3.
    Returns a dict of intermediate physical quantities plus T3_phys.
    """
    m_air = air_mass_flow(P2, RPM, T2, params)
    theta = loading_parameter(m_air, T2, P3)
    health = population_health_prior(cycle, params) if health_override is None else health_override
    eta_b = combustion_efficiency(theta, health, params)
    T3_phys, FAR = t3_energy_balance(T2, fuel_flow, m_air, eta_b)
    return {
        "m_air_proxy": m_air,
        "theta_loading_param": theta,
        "eta_lefebvre": eta_lefebvre(theta, params),
        "health_used": health,
        "eta_combustion": eta_b,
        "FAR": FAR,
        "T3_phys": T3_phys,
    }


def calibrate(df_train_with_health) -> CombustorPhysicsParams:
    """
    Calibrate all physics constants from a labelled training set that has
    columns: P2_Pa, T2_K, P3_Pa, T3_K, RPM_rev_min, FuelFlow_kg_s, Cycle,
    CombustorHealth.

    This is a one-time "twin calibration" step, analogous to tuning a
    physics model against rig/fleet instrumentation.
    """
    df = df_train_with_health

    # 1) Implied air mass flow from the energy balance, using the KNOWN
    #    (labelled) CombustorHealth to back out eta_b = ETA_MAX * Health,
    #    then solving continuity for m_dot_air.
    X = (df.T3_K - df.T2_K) * CP_GAS / LHV
    eta_b_implied = ETA_MAX * df.CombustorHealth
    FAR_implied = X / (eta_b_implied - X)
    valid = (FAR_implied > 0) & np.isfinite(FAR_implied)
    m_air_implied = df.FuelFlow_kg_s[valid] / FAR_implied[valid]

    # 2) Fit the corrected-flow proxy m_air = C * P2^a * RPM^b * T2^c in log-space
    d = df[valid]
    X_log = np.column_stack([np.log(d.P2_Pa), np.log(d.RPM_rev_min), np.log(d.T2_K)])
    y_log = np.log(m_air_implied)
    lr = LinearRegression().fit(X_log, y_log)
    mair_a, mair_b, mair_c = lr.coef_
    mair_logC = lr.intercept_
    flow_r2 = r2_score(y_log, lr.predict(X_log))

    # 3) With m_air now computable from sensors alone, recompute FAR, and the
    #    *implied* combustion efficiency, isolate the loading-parameter effect:
    #    ratio = eta_b_implied_from_energy_balance / CombustorHealth  should
    #    be explained by eta_lefebvre(theta).
    m_air_proxy_all = np.exp(mair_logC) * (df.P2_Pa ** mair_a) * (df.RPM_rev_min ** mair_b) * (df.T2_K ** mair_c)
    FAR_proxy = df.FuelFlow_kg_s / m_air_proxy_all
    X_all = (df.T3_K - df.T2_K) * CP_GAS / LHV
    eta_b_from_data = X_all * (1 + FAR_proxy) / FAR_proxy
    ratio = np.clip(eta_b_from_data / df.CombustorHealth, 1e-4, 1.2)
    theta_all = loading_parameter(m_air_proxy_all, df.T2_K, df.P3_Pa)

    def curve(theta, c, k):
        return 1.0 - np.exp(-c * np.clip(theta, 1e-30, None) ** k)

    try:
        popt, _ = curve_fit(curve, theta_all, ratio, p0=[1e6, 0.3], maxfev=20000)
        lef_c, lef_k = float(popt[0]), float(popt[1])
    except Exception:
        # Fall back to an (almost) always-saturated efficiency curve if the
        # fit doesn't converge (i.e. theta has negligible explanatory power
        # in this dataset) — eta_lefebvre(theta) ~= 1 everywhere.
        lef_c, lef_k = 1e12, 1.0

    # 4) Population-level linear health-vs-cycle prior
    hp = np.polyfit(df.Cycle, df.CombustorHealth, 1)
    health_slope, health_intercept = float(hp[0]), float(hp[1])

    params = CombustorPhysicsParams(
        mair_a=float(mair_a), mair_b=float(mair_b), mair_c=float(mair_c), mair_logC=float(mair_logC),
        lef_c=lef_c, lef_k=lef_k,
        health_slope=health_slope, health_intercept=health_intercept,
    )
    diagnostics = {
        "air_flow_proxy_fit_R2": float(flow_r2),
        "eta_health_ratio_mean": float(ratio.mean()),
        "eta_health_ratio_std": float(ratio.std()),
        "theta_ratio_corr": float(np.corrcoef(theta_all, ratio)[0, 1]),
    }
    return params, diagnostics
