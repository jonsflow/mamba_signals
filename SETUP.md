# Mamba Signals - Repository Initialization

## Overview
Complete initialization of a Mamba-based sequence model for OHLCV price prediction. The project trains on 1.3M+ historical price candles across 13 major stocks to predict directional movement.

## Project Structure

```
mamba_signals/
├── ohlc_data.db              # SQLite database with OHLCV data (200MB)
├── project.md                # Original project brief
├── train.py                  # Main training entry point
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── SETUP.md                 # This file
│
├── src/                     # Core modules
│   ├── __init__.py
│   ├── config.py           # Configuration and hyperparameters
│   ├── data_loader.py      # OHLCDataLoader for database access
│   ├── windowing.py        # SequenceWindower, LabelGenerator, train/val/test split
│   ├── mamba_model.py      # MambaBlock, MambaForecaster, MambaLSTMHybrid
│   └── training.py         # Trainer class for training loop & checkpointing
│
├── data/                   # Data directory (empty, ready for preprocessing)
├── models/                 # Model checkpoints (created during training)
├── notebooks/              # Jupyter notebooks for analysis
└── tests/                  # Unit tests (ready for implementation)
```

## Database Schema

### `ohlc_data` table
- **id**: Primary key
- **symbol**: Stock ticker (AAPL, AMD, AMZN, GOOGL, IBM, META, MSFT, MSTR, NFLX, NVDA, QQQ, SPY, TSLA)
- **timestamp**: Unix timestamp
- **open_price, high_price, low_price, close_price**: Price data
- **volume**: Trading volume
- **timeframe**: '1m', '5m', or '1d'
- **Indexes**: (symbol, timestamp), (symbol, timeframe, timestamp)

### Data Coverage
- **1,342,109** total candles
- **13 symbols** across 3 timeframes
- **1-minute data**: ~45k-74k candles (recent)
- **5-minute data**: ~48k-50k candles (recent)
- **Daily data**: 526-529 candles (~2 years)

### `collection_progress` table
- Tracks data collection status per symbol

## Configuration System

All settings in `src/config.py` via Pydantic models:

### DataConfig
```python
sequence_length: 128      # Candles per window
timeframe: '1m'          # '1m', '5m', or '1d'
test_split: 0.2          # 20% test data
val_split: 0.1           # 10% validation data
```

### LabelingConfig
```python
future_bars: 5           # Lookahead window
threshold_pct: 0.5       # % move to classify as up/down
remove_neutral: True     # Skip neutral samples
```

### ModelConfig
```python
input_dim: 5             # OHLCV features
hidden_dim: 256          # Model width
num_layers: 4            # Mamba layers
output_dim: 2            # Binary classification
```

### TrainingConfig
```python
batch_size: 32
learning_rate: 1e-3
epochs: 50
weight_decay: 1e-5
save_every: 5            # Checkpoint frequency
```

## Core Modules

### `data_loader.py` - OHLCDataLoader
Load and preprocess OHLCV data from SQLite:
```python
from src.data_loader import OHLCDataLoader
from src.config import default_config

loader = OHLCDataLoader("ohlc_data.db", default_config.data)
df = loader.get_ohlcv_data("AAPL")
df_norm, scaling = loader.normalize_ohlcv(df)
```

**Methods**:
- `get_ohlcv_data()`: Load symbol data with optional date range
- `get_all_symbols_data()`: Load all configured symbols
- `normalize_ohlcv()`: Min-max normalization with scaling parameters
- `get_data_summary()`: Statistics on available data

### `windowing.py` - Sequence Preparation
Create fixed-length windows and generate labels:

**SequenceWindower**:
```python
windower = SequenceWindower(sequence_length=128, future_bars=5)
windows, indices = windower.create_windows(df_normalized)
# windows: (num_samples, 128, 5) for OHLCV
```

**LabelGenerator**:
```python
label_gen = LabelGenerator(config.labeling)
labels, valid_mask = label_gen.generate_labels(df, windows, indices)
# labels: binary (0=down, 1=up)
# valid_mask: excludes neutral samples
```

**Dataset Splitting**:
```python
splits = split_dataset(windows, labels, valid_mask,
                       test_split=0.2, val_split=0.1)
X_train, y_train = splits['train']
X_val, y_val = splits['val']
X_test, y_test = splits['test']
```

### `mamba_model.py` - Model Architecture

**MambaBlock**:
- Selective state space modeling with simplified SSM
- Per-step state updates: `h_t = A * h_{t-1} + B * x_t`
- Gated output: `y_t = (C * h_t + D * x_t) * gate`

**MambaForecaster**:
- Input embedding (5 → hidden_dim)
- 4 Mamba layers with residual connections and layer norm
- Classification head: (hidden_dim → hidden_dim/2 → 2 classes)
- Pooling over last timestep

**MambaLSTMHybrid** (alternative):
- Mamba block + LSTM for improved temporal modeling
- Better for longer sequences

```python
from src.mamba_model import create_model
model = create_model(config.model, model_type="mamba")
```

### `training.py` - Trainer
Full training pipeline with checkpointing:

```python
trainer = Trainer(config, model_dir="models/")
trainer.fit((X_train, y_train), (X_val, y_val))
trainer.save_history()
```

**Features**:
- GPU/CPU automatic detection
- AdamW optimizer with gradient clipping
- CrossEntropyLoss for classification
- Best model and checkpoint saving
- Training history tracking

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Data Check
```bash
python -c "from src.data_loader import OHLCDataLoader; from src.config import default_config; loader = OHLCDataLoader('ohlc_data.db', default_config.data); print(loader.get_data_summary())"
```

### 3. Run Full Training
```bash
python train.py
```

Expected output:
- Loads all 13 symbols in configured timeframe
- Combines ~1M+ windows across all symbols
- Trains for 50 epochs with validation and checkpointing
- Saves best model to `models/best_model.pt`
- Evaluates on held-out test set

### 4. Training Progress
- Real-time progress bars with loss/accuracy
- Validation after each epoch
- Checkpoints every 5 epochs
- Best model saved automatically

## Key Design Decisions

1. **Mamba over Transformer**: Efficient O(N) sequence processing vs O(N²) attention
2. **Simplified SSM**: Tractable state space for market data (not full Mamba complexity)
3. **Per-symbol normalization**: Prevents data leakage across stocks
4. **5-bar lookahead**: Reasonable timeframe for intraday signals (5 minutes at 1m timeframe)
5. **0.5% threshold**: Filters noise while capturing meaningful moves
6. **Residual connections**: Improves gradient flow in deeper models

## Next Steps

1. **Run training**: `python train.py`
2. **Monitor results**: Check `models/training_history.json`
3. **Analyze predictions**: Create inference script in `notebooks/`
4. **Backtest signals**: Implement strategy evaluation
5. **Fine-tune hyperparameters**: Edit `src/config.py`

## Hyperparameter Tuning

Edit `src/config.py` to adjust:
- **Model capacity**: `hidden_dim`, `num_layers`
- **Training speed**: `learning_rate`, `batch_size`
- **Regularization**: `weight_decay`, `dropout`
- **Data windows**: `sequence_length`, `future_bars`, `threshold_pct`

## Notes

- Database file is ~200MB, all data is in local SQLite
- Training uses PyTorch with optional CUDA support
- Model checkpoints saved in `models/` directory
- No external API calls - fully offline training
- Reproducible with `random_seed=42`
