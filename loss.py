"""
====================================================================
loss.py

Loss Function for the Physics-Informed Turbine Digital Twin

Loss Components
---------------
1. Physics Loss
2. Turbine Health Loss
3. Temporal Smoothness Loss

====================================================================
"""

import torch
import torch.nn as nn


class TurbineLoss(nn.Module):

    def __init__(

        self,

        lambda_physics=1.0,

        lambda_health=0.3,

        lambda_smooth=0.05,

    ):

        super().__init__()

        self.lambda_physics = lambda_physics
        self.lambda_health = lambda_health
        self.lambda_smooth = lambda_smooth

        self.mse = nn.MSELoss()

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(

        self,

        ensemble_output,

        predicted_T4,

        measured_T4,

        target_health,

        previous_delta_z,

    ):

        ############################################################
        # Physics Loss
        ############################################################

        physics_loss = self.mse(

            predicted_T4,

            measured_T4,

        )

        ############################################################
        # Turbine Health Loss
        ############################################################

        health_loss = self.mse(

            ensemble_output["turbine_health"],

            target_health,

        )

        ############################################################
        # Temporal Smoothness Loss
        ############################################################

        current_delta = ensemble_output["delta_z"]

        smooth_loss = self.mse(

            current_delta,

            previous_delta_z,

        )

        ############################################################
        # Total Loss
        ############################################################

        total_loss = (

            self.lambda_physics * physics_loss

            +

            self.lambda_health * health_loss

            +

            self.lambda_smooth * smooth_loss

        )

        ############################################################

        return {

            "total_loss": total_loss,

            "physics_loss": physics_loss,

            "health_loss": health_loss,

            "smooth_loss": smooth_loss,

        }