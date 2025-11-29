# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mamba Signals** is a PyTorch-based sequence model that predicts directional price movements (up/down) using OHLCV (Open, High, Low, Close, Volume) candle data. It implements a Mamba architecture (selective state space model) which provides O(N) sequence processing efficiency compared to O(N²) transformers.

The project is specifically optimized for **M1/M2/M3 MacBooks** using Metal Performance Shaders (MPS) GPU acceleration.

## Architecture Overview

The system has a clear pipeline structure:

```
SQLite Database (ohlc_data.db, 1.3M candles)
    ↓
OHLCDataLoader (src/data_loader.py)
    ↓
SequenceWindower + LabelGenerator (src/windowing.py)
    ↓
Train/Val/Test Split (src/windowing.py)
    ↓
MambaForecaster Model (src/mamba_model.py)
    ↓
Trainer Loop (src/training.py)
    ↓
Checkpoint & History (models/)
```

### Key Components

- **src/data_loader.py**: `OHLCDataLoader` class handles SQLite database connections, loads symbol data, and applies min-max normalization to OHLCV features and log-scale normalization to volume.

- **src/windowing.py**: `SequenceWindower` creates fixed-length (128-candle) sliding windows. `LabelGenerator` labels windows as UP/DOWN based on future price movement (5-bar lookahead, 0.1% threshold by default).

- **src/mamba_model.py**: Contains `MambaBlock` (selective state space layer) and `MambaForecaster` (full model with 4 layers + classification head). Also includes `MambaLSTMHybrid` as an alternative architecture.

- **src/training.py**: `Trainer` class orchestrates the training loop with device detection (MPS→CUDA→CPU), checkpointing, and loss/accuracy tracking.

- **src/config.py**: Pydantic-based configuration system. All hyperparameters (batch_size, hidden_dim, epochs, etc.) are defined here using `DataConfig`, `LabelingConfig`, `ModelConfig`, and `TrainingConfig` classes.

### Data

- **Source**: SQLite database with 1.3M+ OHLCV candles from 13 major stocks (AAPL, AMD, AMZN, GOOGL, IBM, META, MSFT, MSTR, NFLX, NVDA, QQQ, SPY, TSLA)
- **Timeframes**: 1m, 5m, 1d (config defaults to 5m for best data coverage)
- **File**: 191MB local file (no API calls, fully offline)

## Common Development Tasks

### Running Training
```bash
python train.py
```
This loads all symbols (default: SPY), creates sequences, trains the Mamba model, and saves checkpoints to `models/`.

### Modifying Configuration
Edit `src/config.py` to change:
- **Hyperparameters**: `batch_size`, `hidden_dim`, `num_layers`, `learning_rate`, `epochs`
- **Data settings**: `symbols`, `timeframe`, `sequence_length`
- **Label settings**: `future_bars`, `threshold_pct`

For memory issues on MacBook Air, reduce `batch_size` (16→8) and `hidden_dim` (256→128).

### Memory Optimization Tips
The project already includes memory-efficient features:
- MPS device support (Metal Performance Shaders for M-series Macs)
- Gradient clipping in training loop (src/training.py:91)
- Configurable batch size and model width

If training runs out of memory:
1. Reduce `batch_size` in src/config.py (line 35)
2. Reduce `hidden_dim` in src/config.py (line 27)
3. Reduce `num_layers` in src/config.py (line 28)
4. Reduce number of `symbols` in src/config.py (line 10)

### Model Selection
- **Default (MambaForecaster)**: Pure Mamba architecture, O(N) complexity
- **Hybrid (MambaLSTMHybrid)**: Mamba + LSTM layers for improved temporal modeling

Change in train.py by modifying the model creation call or in src/mamba_model.py:234 factory function.

### Training Artifacts
After training completes, check:
- `models/best_model.pt` - Best validation loss checkpoint
- `models/final_model.pt` - Final epoch model
- `models/training_history.json` - Loss/accuracy curves for analysis
- `models/checkpoint_epoch_*.pt` - Periodic checkpoints every 5 epochs

## Device Detection & MPS

The codebase auto-detects device in src/training.py:32-40:
1. Checks for MPS (M1/M2/M3 Macs)
2. Falls back to CUDA if available
3. Falls back to CPU as last resort

To verify MPS is working:
```bash
python3 -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

## Testing

A `tests/` directory exists but appears minimal. Key areas to test when modifying:
- Data loading and normalization in `OHLCDataLoader`
- Window and label generation in windowing.py
- Model forward passes with different batch sizes
- Training loop with different devices

## Important Notes

- **Batch Size Impact**: M1 Macs typically support batch_size=16 without issues. M1 Air might need batch_size=8.
- **Training Speed**: ~200-400 samples/sec on M1 Pro GPU; full 50 epochs takes 2-4 hours.
- **Sequence Length**: Fixed at 128 candles by default; this is embedded in the model architecture and data pipeline.
- **Labels**: Generated via future price comparison (future_bars=5 lookahead, threshold=0.1%). Neutral samples (price moves <0.1%) are filtered out by default.
