"""
====================================================================
ensemble.py

Deep Ensemble of Physics-Informed Transformers

Each transformer is initialized independently.

Outputs
-------
Mean prediction
Epistemic uncertainty (standard deviation)

====================================================================
"""

import torch
import torch.nn as nn

from transformer import TurbineTransformer


class DeepEnsemble(nn.Module):

    def __init__(

        self,

        n_models=5,

        **transformer_kwargs,

    ):

        super().__init__()

        self.models = nn.ModuleList(

            [

                TurbineTransformer(

                    **transformer_kwargs

                )

                for _ in range(n_models)

            ]

        )

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(

        self,

        sequence,

    ):

        delta_predictions = []

        health_predictions = []

        latent_predictions = []

        ############################################################

        for model in self.models:

            output = model(sequence)

            delta_predictions.append(

                output["delta_z"]

            )

            health_predictions.append(

                output["turbine_health"]

            )

            latent_predictions.append(

                output["latent_feature"]

            )

        ############################################################
        # Stack
        ############################################################

        delta_predictions = torch.stack(

            delta_predictions,

            dim=0,

        )

        health_predictions = torch.stack(

            health_predictions,

            dim=0,

        )

        latent_predictions = torch.stack(

            latent_predictions,

            dim=0,

        )

        ############################################################
        # Mean
        ############################################################

        delta_mean = torch.mean(

            delta_predictions,

            dim=0,

        )

        health_mean = torch.mean(

            health_predictions,

            dim=0,

        )

        latent_mean = torch.mean(

            latent_predictions,

            dim=0,

        )

        ############################################################
        # Epistemic uncertainty
        ############################################################

        delta_std = torch.std(

            delta_predictions,

            dim=0,

        )

        health_std = torch.std(

            health_predictions,

            dim=0,

        )

        ############################################################

        return {

            "delta_z": delta_mean,

            "delta_uncertainty": delta_std,

            "turbine_health": health_mean,

            "health_uncertainty": health_std,

            "latent_feature": latent_mean,

        }

    