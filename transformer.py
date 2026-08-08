"""
=========================================================
transformer.py

Physics-Informed Temporal Transformer

Inputs
------
Sequence of turbine measurements and physics outputs

Shape:
(batch, sequence_length, input_dim)

Outputs
-------
Δz              : Latent degradation increments
Turbine Health  : Auxiliary health prediction
Latent Feature  : Shared feature representation

Author:
=========================================================
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# =========================================================
# Output Container
# =========================================================

@dataclass
class TransformerOutput:

    delta_z: torch.Tensor

    turbine_health: torch.Tensor

    latent_feature: torch.Tensor


# =========================================================
# Transformer
# =========================================================

class TurbineTransformer(nn.Module):

    def __init__(

        self,

        input_dim=19,

        sequence_length=50,

        embedding_dim=32,

        num_heads=4,

        num_layers=2,

        ff_dim=64,

        dropout=0.1,

        max_latent_step=0.02,

    ):

        super().__init__()

        self.max_latent_step = max_latent_step

        ####################################################
        # Input Embedding
        ####################################################

        self.embedding = nn.Linear(

            input_dim,

            embedding_dim

        )

        self.embedding_dropout = nn.Dropout(

            dropout

        )

        ####################################################
        # Learnable CLS Token
        ####################################################

        self.cls_token = nn.Parameter(

            torch.randn(

                1,

                1,

                embedding_dim,

            )

        )

        ####################################################
        # Learnable Position Embedding
        ####################################################

        self.position_embedding = nn.Parameter(

            torch.randn(

                1,

                sequence_length + 1,

                embedding_dim,

            )

        )

        ####################################################
        # Transformer Encoder
        ####################################################

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embedding_dim,

            nhead=num_heads,

            dim_feedforward=ff_dim,

            dropout=dropout,

            activation="gelu",

            batch_first=True,

            norm_first=True,

        )

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers,

        )

        ####################################################
        # Shared Projection
        ####################################################

        self.shared = nn.Sequential(

            nn.Linear(

                embedding_dim,

                32,

            ),

            nn.GELU(),

            nn.Dropout(

                dropout

            ),

            nn.Linear(

                32,

                16,

            ),

            nn.GELU(),

        )

        ####################################################
        # Latent Head
        ####################################################

        self.delta_head = nn.Linear(

            16,

            4,

        )

        ####################################################
        # Health Head
        ####################################################

        self.health_head = nn.Linear(

            16,

            1,

        )

        ####################################################
        # Weight Initialization
        ####################################################

        self.apply(

            self._init_weights

        )

    # =====================================================
    # Xavier Initialization
    # =====================================================

    def _init_weights(

        self,

        module,

    ):

        if isinstance(

            module,

            nn.Linear,

        ):

            nn.init.xavier_uniform_(

                module.weight

            )

            if module.bias is not None:

                nn.init.zeros_(

                    module.bias

                )

    # =====================================================
    # Forward
    # =====================================================

    def forward(

        self,

        x,

    ):

        """
        Parameters
        ----------
        x

        Shape

        (batch,50,input_dim)
        """

        batch_size = x.size(0)

        ####################################################
        # Input Embedding
        ####################################################

        x = self.embedding(

            x

        )

        x = self.embedding_dropout(

            x

        )

        ####################################################
        # CLS Token
        ####################################################

        cls = self.cls_token.expand(

            batch_size,

            -1,

            -1,

        )

        x = torch.cat(

            [

                cls,

                x,

            ],

            dim=1,

        )

        ####################################################
        # Position Encoding
        ####################################################

        x = x + self.position_embedding

        ####################################################
        # Transformer Encoder
        ####################################################

        x = self.encoder(

            x

        )

        ####################################################
        # CLS Feature
        ####################################################

        cls_feature = x[:, 0]

        ####################################################
        # Shared Projection
        ####################################################

        latent_feature = self.shared(

            cls_feature

        )

        ####################################################
        # Latent Degradation Increment
        ####################################################

        delta_z = (

            self.max_latent_step

            *

            torch.tanh(

                self.delta_head(

                    latent_feature

                )

            )

        )

        ####################################################
        # Auxiliary Health Prediction
        ####################################################

        turbine_health = torch.sigmoid(

            self.health_head(

                latent_feature

            )

        )

        ####################################################
        # Output
        ####################################################

        return TransformerOutput(

            delta_z=delta_z,

            turbine_health=turbine_health,

            latent_feature=latent_feature,

        )