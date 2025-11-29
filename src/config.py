"""Configuration and hyperparameters for training."""
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    db_path: Path = Path("ohlc_data.db")
    symbols: list[str] = ["SPY"]  # Just SPY for simplicity
    timeframe: str = "5m"  # '1m', '5m', or '1d' - 5m has best data coverage
    sequence_length: int = 128  # number of candles per window
    test_split: float = 0.2
    val_split: float = 0.1


class LabelingConfig(BaseModel):
    """Configuration for generating labels."""
    future_bars: int = 5  # look-ahead window
    threshold_pct: float = 0.1  # % move to classify as up/down (lowered from 0.5)
    remove_neutral: bool = True  # skip samples in neutral zone


class ModelConfig(BaseModel):
    """Mamba model architecture configuration."""
    input_dim: int = 5  # OHLCV
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    output_dim: int = 2  # binary classification (up/down)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 4  # M1 optimized batch size (adjust if needed)
    learning_rate: float = 1e-3
    epochs: int = 50
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    device: str = "mps"  # 'mps' for M1/M2/M3, 'cuda' for NVIDIA, 'cpu' for CPU
    random_seed: int = 42
    save_every: int = 5  # save checkpoint every N epochs


class Config(BaseModel):
    """Main configuration container."""
    data: DataConfig = DataConfig()
    labeling: LabelingConfig = LabelingConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    project_root: Path = Path(__file__).parent.parent


# Default configuration instance
default_config = Config()
