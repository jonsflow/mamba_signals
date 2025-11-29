# Quick Start - M1 MacBook Pro

## 60-Second Setup

```bash
# 1. Navigate to project
cd /Users/clutchcoder/working/mamba_signals

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch for M1 (CRITICAL - use CPU wheels!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Verify MPS is available
python3 -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

## Run Training

```bash
python train.py
```

**Expected:**
- First load: ~2-3 minutes (loading 1.3M candles from database)
- Training: ~2-4 hours per 50 epochs on M1 Pro GPU (MPS)
- Output: Models saved to `models/` directory

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main entry point - run this |
| `src/config.py` | All settings (batch_size, epochs, etc.) |
| `src/mamba_model.py` | Model architecture |
| `src/data_loader.py` | Database loading |
| `models/` | Saved checkpoints and best model |

## Configuration (Edit `src/config.py`)

**For slower M1 (MacBook Air):**
```python
batch_size = 8      # Reduce from 16
hidden_dim = 128    # Reduce from 256
num_layers = 2      # Reduce from 4
epochs = 20         # Reduce from 50
```

**For faster M1 (MacBook Pro 16"):**
```python
batch_size = 32     # Increase from 16
hidden_dim = 512    # Increase from 256
num_layers = 6      # Increase from 4
```

## Troubleshooting

**Q: Training is slow**
- A: Check logs for "CPU" vs "MPS" - should say Metal Performance Shaders
- Reduce `batch_size` in config if out of memory

**Q: MPS not available**
- A: Reinstall PyTorch: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu`

**Q: Import errors**
- A: Make sure you're in the virtual environment: `source venv/bin/activate`

## Output

After training completes:
- `models/best_model.pt` - Best model during training
- `models/final_model.pt` - Final model
- `models/training_history.json` - Loss/accuracy curves
- `models/checkpoint_epoch_*.pt` - Intermediate checkpoints

## Next Steps

1. ✅ Run `python train.py`
2. 📊 Check results in `models/training_history.json`
3. 🔧 Tune hyperparameters in `src/config.py`
4. 🧪 Create inference script for predictions

## Documentation

- **Full setup guide**: `SETUP.md`
- **M1 specific guide**: `M1_SETUP.md`
- **Project plan**: `project.md`
