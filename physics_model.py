"""
=========================================================
physics_model.py

Physics-Based Digital Twin for High Pressure Turbine

Orchestrates all turbine physics modules and maintains the
persistent hidden turbine state.

Persistent State
----------------
TurbineState

Outputs
-------
Updated state
Physics outputs for current cycle

=========================================================
"""

import torch

from turbine_state import TurbineState

from thermal_model import BladeThermalModel
from oxidation_model import OxidationModel
from stress_model import StressModel
from creep_model import CreepModel
from tmf_model import TMFModel
from turbine_efficiency import TurbineEfficiencyModel
from measurenment_model import MeasurementModel

from thermophysical_properties import gas_cp


class TurbinePhysicsModel(torch.nn.Module):

    def __init__(

        self,

        thermal_model,

        stress_model,

        oxidation_model,

        creep_model,

        tmf_model,

        efficiency_model,

        measurement_model,

    ):

        super().__init__()

        self.state = TurbineState()

        self.thermal = thermal_model

        self.stress = stress_model

        self.oxidation = oxidation_model

        self.creep = creep_model

        self.tmf = tmf_model

        self.efficiency = efficiency_model

        self.measurement = measurement_model

    # =====================================================

    def reset(self):

        self.state.reset()

    # =====================================================

    def step(

        self,

        inputs,

        corrections=None,

    ):

        """
        Parameters
        ----------
        inputs

            T3
            P3
            T4
            P4
            P5
            RPM
            FuelFlow
            mdot_core
            mdot_cooling
            Tcool
            Pcool
            flight_hours

        corrections

            PhysicsCorrections
            (optional)

        Returns
        -------
        state
        outputs
        """

        ####################################################
        # Time Step
        ####################################################

        dt = inputs["flight_hours"] * 3600.0

        ####################################################
        # Mapping Corrections
        ####################################################

        if corrections is None:

            htc_factor = 1.0

            oxidation_factor = 1.0

            creep_factor = 1.0

            clearance_offset = 0.0

            efficiency_offset = 0.0

        else:

            htc_factor = corrections.gas_htc_factor

            oxidation_factor = corrections.oxidation_factor

            creep_factor = corrections.creep_factor

            clearance_offset = corrections.clearance_offset

            efficiency_offset = corrections.efficiency_offset

        ####################################################
        # Thermal Model
        ####################################################

        blade_temperature = self.thermal.update(

            Tm=self.state.blade_temperature,

            Tg=inputs["T4"],

            Pg=inputs["P4"],

            Tc=inputs["Tcool"],

            Pc=inputs["Pcool"],

            mdot_core=inputs["mdot_core"],

            mdot_cooling=inputs["mdot_cooling"],

            tbc_thickness=self.state.tbc_thickness,

            dt=dt,

            htc_factor=htc_factor,

        )

        ####################################################
        # Stress
        ####################################################

        nominal_stress, local_stress, local_strain = (

            self.stress.update(

                inputs["RPM"]

            )

        )

        ####################################################
        # Oxidation
        ####################################################

        tgo = self.oxidation.update(

            x_tgo=self.state.tgo_thickness,

            blade_temperature=blade_temperature,

            dt=dt,

            oxidation_factor=oxidation_factor,

        )

        ####################################################
        # Simple TBC Consumption
        ####################################################

        initial_tbc = 300e-6
        oxide_to_tbc_ratio = 0.2
        tbc_loss = oxide_to_tbc_ratio * tgo
    

        tbc = initial_tbc - tbc_loss


        ####################################################
        # Creep
        ####################################################

        creep_strain = self.creep.update(

            creep_strain=self.state.creep_strain,

            blade_temperature=blade_temperature,

            stress=local_stress,

            dt=dt,

            creep_factor=creep_factor,

        )

        ####################################################
        # TMF
        ####################################################

        delta_sigma = torch.abs(

            local_stress

            -

            self.state.previous_stress

        )

        sigma_mean = (

            local_stress

            +

            self.state.previous_stress

        ) / 2.0

        tmf_damage, dD, Nf = self.tmf.update(

            damage_old=self.state.tmf_damage,

            delta_sigma=delta_sigma,

            sigma_mean=sigma_mean,

            cycles=1,

        )

        ####################################################
        # Clearance / Efficiency
        ####################################################

        self.efficiency.clearance_offset = clearance_offset

        clearance, eta = self.efficiency.update(

            creep_strain

        )

        eta = eta + efficiency_offset

        eta = torch.clamp(

            eta,

            min=0.5,
            max=1.0,

        )

        ####################################################
        # Turbine Exit Temperature
        ####################################################

        T5, T5s = self.measurement.update(

            T4=inputs["T4"],

            P4=inputs["P4"],

            P5=inputs["P5"],

            eta_t=eta,

        )

        ####################################################
        # Update Persistent State
        ####################################################

        self.state.blade_temperature = torch.nan_to_num(blade_temperature, nan=800.0, posinf=800.0, neginf=800.0)

        self.state.tgo_thickness = torch.nan_to_num(tgo, nan=0.0, posinf=0.0, neginf=0.0)

        self.state.tbc_thickness = torch.nan_to_num(tbc, nan=300e-6, posinf=300e-6, neginf=300e-6)

        self.state.creep_strain = torch.nan_to_num(creep_strain, nan=0.0, posinf=0.0, neginf=0.0)

        self.state.tmf_damage = torch.nan_to_num(tmf_damage, nan=0.0, posinf=0.0, neginf=0.0)

        self.state.tip_clearance = torch.nan_to_num(clearance, nan=5e-4, posinf=5e-4, neginf=5e-4)

        self.state.previous_stress = torch.nan_to_num(local_stress, nan=0.0, posinf=0.0, neginf=0.0)

        self.state.increment_cycle()

        self.state.add_flight_hours(

            inputs["flight_hours"]

        )

        ####################################################
        # Outputs
        ####################################################

        outputs = {

            "blade_temperature": blade_temperature,

            "stress": local_stress,

            "strain": local_strain,

            "tgo_thickness": tgo,

            "tbc_thickness": tbc,

            "creep_strain": creep_strain,

            "tmf_damage": tmf_damage,

            "tip_clearance": clearance,

            "turbine_efficiency": eta,

            "predicted_T5": T5,

            "predicted_T5s": T5s,

        }

        return self.state.copy(), outputs


########################################################################
# Factory Function
########################################################################

def build_physics_model():

    thermal = BladeThermalModel(
        blade_mass=0.35,
        hot_surface_area=0.004,
        cooling_surface_area=0.003,
        characteristic_length=0.025,
        cooling_hydraulic_diameter=0.0015,
        flow_area=0.004,
        cooling_flow_area=8e-5,
        tbc_conductivity=1.5,
    )

    stress = StressModel(
        S_ref=220e6,
        omega_ref=12000,
        E=190e9,
        Kt=2.0,
        K_prime=1500e6,
        n_prime=0.15,
    )

    oxidation = OxidationModel(
        k0=1.0e-13,
        activation_energy=2.6e5,
    )

    creep = CreepModel(
        A=1.0e-25,
        n=5.0,
        activation_energy=4.2e5,
    )

    tmf = TMFModel(
        E=190e9,
        K_prime=1500e6,
        n_prime=0.15,
        sigma_f=1500e6,
        epsilon_f=0.35,
        b=-0.08,
        c=-0.6,
    )

    efficiency = TurbineEfficiencyModel(
        blade_height=0.05,
        initial_clearance=5e-4,
        clean_efficiency=0.92,
        K1=2.2,
        K2=5.0,
    )

    measurement = MeasurementModel(
        gas_cp_function=gas_cp
    )

    return TurbinePhysicsModel(
        thermal_model=thermal,
        stress_model=stress,
        oxidation_model=oxidation,
        creep_model=creep,
        tmf_model=tmf,
        efficiency_model=efficiency,
        measurement_model=measurement,
    )