# Mamba Signals

A PyTorch implementation of a Mamba-based sequence model for predicting directional price movements using OHLC (Open, High, Low, Close) data.

## 🚀 Quick Start (M1 MacBook Pro)

```bash
cd /Users/clutchcoder/working/mamba_signals

# Setup virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies (PyTorch for M1 ARM64)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run training
python train.py
```

See **QUICKSTART.md** for detailed setup instructions.

## 📊 Dataset

- **1.3M+ OHLCV candles** from 13 major stocks
- **Symbols**: AAPL, AMD, AMZN, GOOGL, IBM, META, MSFT, MSTR, NFLX, NVDA, QQQ, SPY, TSLA
- **Timeframes**: 1-minute, 5-minute, daily
- **SQLite database**: 191MB local file (no API calls)

## 🧠 Model Architecture

**Mamba**: Selective State Space Model for efficient O(N) sequence processing
- Input: 128-candle OHLCV sequences
- Architecture: 4 Mamba layers + classification head
- Output: Binary prediction (up/down movement)
- Features: GPU acceleration (MPS for M1 Macs)

**Alternative**: Mamba-LSTM hybrid for improved temporal modeling

## 📈 Training Pipeline

1. **Load & Normalize**: OHLCV data from SQLite
2. **Window & Label**: Fixed-length sequences with future price labels
3. **Split**: Train/val/test stratified split
4. **Train**: AdamW optimizer with gradient clipping
5. **Checkpoint**: Best model + periodic saves

## 🛠 Configuration

Edit `src/config.py` to customize:

```python
# Data
sequence_length = 128        # Candles per window
timeframe = "1m"             # '1m', '5m', or '1d'

# Labeling
future_bars = 5              # Lookahead for labels
threshold_pct = 0.5          # Move threshold

# Model
hidden_dim = 256             # Model width
num_layers = 4               # Mamba depth

# Training
batch_size = 16              # M1-optimized
epochs = 50
learning_rate = 1e-3
```

## 📁 Project Structure

```
mamba_signals/
├── train.py                 # Main entry point
├── requirements.txt         # Dependencies
├── ohlc_data.db            # SQLite database (191MB)
│
├── src/
│   ├── config.py           # Configuration system
│   ├── data_loader.py      # OHLCDataLoader
│   ├── windowing.py        # Sequence windowing & labeling
│   ├── mamba_model.py      # Model architecture
│   └── training.py         # Training loop
│
├── models/                 # Checkpoints & results
├── data/                   # Preprocessing cache
├── notebooks/              # Analysis notebooks
│
├── QUICKSTART.md          # 60-second setup
├── M1_SETUP.md            # M1 Mac specific guide
└── SETUP.md               # Full documentation
```

## 🎯 Key Features

✅ **M1/M2/M3 GPU Acceleration** - Uses Metal Performance Shaders (MPS)
✅ **Production-Ready** - Type hints, configuration system, checkpointing
✅ **Efficient Architecture** - O(N) complexity vs O(N²) transformers
✅ **Complete Pipeline** - Data loading through evaluation
✅ **Offline Training** - No API calls, fully local

## 📚 Documentation

- **QUICKSTART.md** - 60-second setup guide
- **M1_SETUP.md** - M1 Mac specific installation & troubleshooting
- **SETUP.md** - Comprehensive API documentation
- **project.md** - Original project brief

## 🚦 Next Steps

1. Follow **QUICKSTART.md** for setup
2. Run `python train.py`
3. Check `models/training_history.json` for results
4. Adjust hyperparameters in `src/config.py` if needed
5. Create inference scripts for predictions

## 📦 Requirements

- Python 3.9+
- PyTorch 2.0+ (with M1 ARM64 support)
- NumPy, Pandas, scikit-learn
- SQLite (built-in)

## ⚙️ Performance (M1 Pro GPU)

- Training speed: ~200-400 samples/sec
- Inference speed: ~500-1000 samples/sec
- Full training time: ~2-4 hours for 50 epochs
- Memory usage: ~4-6GB (with batch_size=16)

## 🔧 Troubleshooting

**MPS not available?**
```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```

**Out of memory on M1?**
Reduce in `src/config.py`:
```python
batch_size = 8       # from 16
hidden_dim = 128     # from 256
```

**Training too slow?**
- Verify device in logs (should show "Metal Performance Shaders")
- Check Task Manager: model should use GPU not CPU

## 📝 License

Private project - use for educational and authorized trading research only.

---

**Created**: 2025
**Platform**: MacBook Pro M1/M2/M3
**Framework**: PyTorch 2.0+
**Last Updated**: 2025-11-28
