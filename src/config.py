"""Configuration and hyperparameters for RL training."""
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    db_path: Path = Path("ohlc_data.db")
    symbol: str = "SPY"
    timeframe: str = "5m"  # '1m', '5m', or '1d' - 5m has best data coverage
    sequence_length: int = 128  # larger window for Mamba encoder
    train_ratio: float = 0.7  # chronological split
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_samples: Optional[int] = None  # Use all data available (None = no limit)


class ModelConfig(BaseModel):
    """Mamba encoder architecture configuration."""
    input_dim: int = 5  # OHLCV
    hidden_dim: int = 256  # larger hidden dimension for GPU
    num_layers: int = 2  # more layers for GPU
    dropout: float = 0.1
    state_dim: int = 256  # larger output state vector


class RLConfig(BaseModel):
    """Reinforcement Learning training hyperparameters."""
    # Episode training
    episodes: int = 100  # number of training episodes
    episode_length: int = 100  # candles per episode (not used, uses full data)

    # DQN parameters
    batch_size: int = 32  # reduced from 64 for faster training iterations
    learning_rate: float = 1e-3
    gamma: float = 0.99  # discount factor for future rewards

    # Exploration
    epsilon: float = 1.0  # initial exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.7  # decay per episode

    # Experience replay
    replay_buffer_size: int = 25000  # reduced from 50000 for faster training
    min_buffer_size: int = 32  # minimum samples before training

    # Network updates
    target_update_freq: int = 500  # update target network every N steps

    # Device and misc
    device: str = "cuda"  # 'cuda' for NVIDIA GPU, 'mps' for M1/M2/M3, 'cpu' for CPU
    random_seed: int = 42
    save_every: int = 10  # save checkpoint every N episodes

    # Initial trading conditions
    initial_cash: float = 10000.0
    share_size: int = 1  # buy/sell 1 share at a time

    # Reward configuration preset
    reward_preset: str = "trading_agent"  # simple P&L baseline - test generalization


class Config(BaseModel):
    """Main configuration container."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    rl: RLConfig = RLConfig()
    project_root: Path = Path(__file__).parent.parent


# Default configuration instance
default_config = Config()
