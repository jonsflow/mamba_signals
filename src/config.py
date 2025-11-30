"""Configuration and hyperparameters for RL training."""
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    db_path: Path = Path("ohlc_data.db")
    symbol: str = "SPY"
    timeframe: str = "5m"  # '1m', '5m', or '1d' - 5m has best data coverage
    sequence_length: int = 64  # number of candles per window for Mamba encoder
    train_ratio: float = 0.7  # chronological split
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_samples: Optional[int] = 15000  # Limit data for quick iteration (None = use all, default 5000 for dev)


class ModelConfig(BaseModel):
    """Mamba encoder architecture configuration."""
    input_dim: int = 5  # OHLCV (Mamba model uses this, Baseline uses price differences)
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    state_dim: int = 64  # output state vector size


class RLConfig(BaseModel):
    """Reinforcement Learning training hyperparameters."""
    # Episode training
    episodes: int = 100  # for dev (use 100+ for big system)
    episode_length: int = 100  # candles per episode (for dev)

    # DQN parameters
    batch_size: int = 8
    learning_rate: float = 1e-3
    gamma: float = 0.99  # discount factor for future rewards

    # Exploration
    epsilon: float = 1.0  # initial exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99  # decay per episode (more aggressive for quick learning)

    # Experience replay
    replay_buffer_size: int = 1000  # for dev (use 10000+ for big system)
    min_buffer_size: int = 32  # minimum samples before training

    # Network updates
    target_update_freq: int = 100  # update target network every N steps

    # Device and misc
    device: str = "mps"  # 'mps' for M1/M2/M3, 'cuda' for NVIDIA, 'cpu' for CPU
    random_seed: int = 42
    save_every: int = 5  # save checkpoint every N episodes

    # Initial trading conditions
    initial_cash: float = 10000.0
    share_size: int = 1  # buy/sell 1 share at a time

    # Reward configuration preset
    reward_preset: str = "simple_delta"  # Simple reward: delta on SELL only


class Config(BaseModel):
    """Main configuration container."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    rl: RLConfig = RLConfig()
    project_root: Path = Path(__file__).parent.parent


# Default configuration instance
default_config = Config()
