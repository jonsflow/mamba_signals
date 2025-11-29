"""Mamba-based sequence model for price prediction."""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class MambaBlock(nn.Module):
    """Single Mamba layer block.

    Inspired by the Mamba architecture (Gu & Dao, 2023) adapted for OHLC prediction.
    Uses selective state space modeling for efficient sequence processing.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """Initialize Mamba block.

        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # State space parameters
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)
        self.D = nn.Parameter(torch.randn(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Project and split
        proj = self.in_proj(x)
        x_proj, gate = proj.chunk(2, dim=-1)

        # State space modeling simplified
        h = torch.zeros(batch_size, d_model, device=x.device)
        outputs = []

        for t in range(seq_len):
            # Get current step
            x_t = x_proj[:, t]  # (batch, d_model)

            # Simple SSM step: h_t = A * h_{t-1} + B * x_t
            A_scaled = torch.sigmoid(self.A)
            h = A_scaled.unsqueeze(0) * h + (1 - A_scaled.unsqueeze(0)) * self.B(x_t)

            # Output from state: y_t = C * h_t + D * x_t
            y_t = self.C(h) + self.D.unsqueeze(0) * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)

        # Gating
        y = y * torch.sigmoid(gate)
        y = self.dropout(y)

        # Output projection
        y = self.out_proj(y)

        return y


class MambaForecaster(nn.Module):
    """Mamba-based sequence forecaster for OHLC price prediction."""

    def __init__(self, config: ModelConfig):
        """Initialize forecaster.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Input embedding
        self.input_emb = nn.Linear(config.input_dim, config.hidden_dim)

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits tensor of shape (batch, output_dim)
        """
        # Embed input
        x = self.input_emb(x)  # (batch, seq_len, hidden_dim)

        # Mamba layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            x_residual = x
            x = mamba_layer(x)
            x = layer_norm(x + x_residual)

        # Pool over sequence dimension (take last state)
        x = x[:, -1, :]  # (batch, hidden_dim)

        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, output_dim)

        return logits

    def get_sequence_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get full sequence representation before classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Sequence representation of shape (batch, seq_len, hidden_dim)
        """
        x = self.input_emb(x)

        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            x_residual = x
            x = mamba_layer(x)
            x = layer_norm(x + x_residual)

        return x


class MambaLSTMHybrid(nn.Module):
    """Hybrid Mamba-LSTM model for improved temporal modeling."""

    def __init__(self, config: ModelConfig):
        """Initialize hybrid model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Input embedding
        self.input_emb = nn.Linear(config.input_dim, config.hidden_dim)

        # Mamba layer
        self.mamba = MambaBlock(config.hidden_dim, config.dropout)

        # LSTM layer for temporal context
        self.lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers - 1,
            dropout=config.dropout,
            batch_first=True
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits tensor of shape (batch, output_dim)
        """
        # Embed input
        x = self.input_emb(x)

        # Mamba processing
        x = self.mamba(x)

        # LSTM processing
        x, _ = self.lstm(x)
        x = self.layer_norm(x)

        # Take last timestep
        x = x[:, -1, :]

        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


def create_model(config: ModelConfig, model_type: str = "mamba") -> nn.Module:
    """Factory function to create model.

    Args:
        config: Model configuration
        model_type: Type of model ("mamba" or "hybrid")

    Returns:
        Initialized model
    """
    if model_type == "mamba":
        return MambaForecaster(config)
    elif model_type == "hybrid":
        return MambaLSTMHybrid(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
