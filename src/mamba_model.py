"""Mamba-based sequence encoder for state representation."""
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


class MambaEncoder(nn.Module):
    """Mamba-based sequence encoder for RL state representation."""

    def __init__(self, config: ModelConfig):
        """Initialize encoder.

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

        # Output projection to state dimension
        self.output_proj = nn.Linear(config.hidden_dim, config.state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            State vector of shape (batch, state_dim)
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

        # Project to state dimension
        state = self.output_proj(x)  # (batch, state_dim)

        return state

    def get_sequence_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get full sequence representation before final pooling.

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


class DQNHead(nn.Module):
    """DQN head that takes state vector and outputs Q-values for actions."""

    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 64):
        """Initialize DQN head.

        Args:
            state_dim: Input state dimension
            action_dim: Number of actions (3: HOLD, BUY, SELL)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State vector of shape (batch, state_dim)

        Returns:
            Q-values of shape (batch, action_dim)
        """
        return self.network(state)


class DQNAgent(nn.Module):
    """Complete DQN agent: Mamba encoder + DQN head."""

    def __init__(self, config: ModelConfig):
        """Initialize agent.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.encoder = MambaEncoder(config)
        self.dqn_head = DQNHead(
            state_dim=config.state_dim,
            action_dim=3,  # HOLD, BUY, SELL
            hidden_dim=config.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Q-values of shape (batch, 3)
        """
        state = self.encoder(x)
        q_values = self.dqn_head(state)
        return q_values
