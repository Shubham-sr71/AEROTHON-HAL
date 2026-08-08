"""
====================================================================
turbine_state.py

Persistent Hidden State of the Turbine

====================================================================
"""

from dataclasses import dataclass, field

import copy

import torch

blade_temperature: float = 800.0
@dataclass
class TurbineState:

    ############################################################
    # Coating State
    ############################################################

    blade_temperature: torch.Tensor = field(

        default_factory=lambda: torch.tensor(800.0)

    )

    tgo_thickness: torch.Tensor = field(

        default_factory=lambda: torch.tensor(0.0)

    )

    tbc_thickness: torch.Tensor = field(

        default_factory=lambda: torch.tensor(300e-6)

    )

    ############################################################
    # Structural Damage
    ############################################################

    creep_strain: torch.Tensor = field(

        default_factory=lambda: torch.tensor(0.0)

    )

    tmf_damage: torch.Tensor = field(

        default_factory=lambda: torch.tensor(0.0)

    )

    ############################################################
    # Geometry
    ############################################################

    tip_clearance: torch.Tensor = field(

        default_factory=lambda: torch.tensor(5e-4)

    )

    previous_stress: torch.Tensor = field(

        default_factory=lambda: torch.tensor(0.0)

    )

    ############################################################
    # Latent Degradation State
    ############################################################

    z: torch.Tensor = field(

        default_factory=lambda: torch.zeros(4)

    )

    ############################################################
    # Life Tracking
    ############################################################

    cycle_number: int = 0

    flight_hours: float = 0.0

    ############################################################

    def copy(

        self,

    ):

        return TurbineState(
            blade_temperature=self.blade_temperature.detach().clone(),
            tgo_thickness=self.tgo_thickness.detach().clone(),
            tbc_thickness=self.tbc_thickness.detach().clone(),
            creep_strain=self.creep_strain.detach().clone(),
            tmf_damage=self.tmf_damage.detach().clone(),
            tip_clearance=self.tip_clearance.detach().clone(),
            previous_stress=self.previous_stress.detach().clone(),
            z=self.z.detach().clone(),
            cycle_number=self.cycle_number,
            flight_hours=self.flight_hours,
        )

    ############################################################

    def increment_cycle(

        self,

    ):

        self.cycle_number += 1

    ############################################################

    def add_flight_hours(

        self,

        hours,

    ):

        self.flight_hours += hours

    ############################################################

    def update_latent(

        self,

        delta_z,

    ):

        self.z = torch.clamp(

            self.z + delta_z,

            min=0.0,

            max=1.0,

        )

    ############################################################

    def reset(

        self,

    ):

        self.blade_temperature = torch.tensor(800.0)

        self.tgo_thickness = torch.tensor(0.0)

        self.tbc_thickness = torch.tensor(300e-6)

        self.creep_strain = torch.tensor(0.0)

        self.tmf_damage = torch.tensor(0.0)

        self.tip_clearance = torch.tensor(5e-4)

        self.previous_stress = torch.tensor(0.0)

        self.z = torch.zeros(4)

        self.cycle_number = 0

        self.flight_hours = 0.0

    ############################################################

    def as_dict(

        self,

    ):

        return {

            "cycle": self.cycle_number,

            "flight_hours": self.flight_hours,

            "tgo_thickness": self.tgo_thickness,

            "tbc_thickness": self.tbc_thickness,

            "creep_strain": self.creep_strain,

            "tmf_damage": self.tmf_damage,

            "tip_clearance": self.tip_clearance,

            "z_thermal": self.z[0],

            "z_material": self.z[1],

            "z_geometry": self.z[2],

            "z_aerodynamic": self.z[3],

        }