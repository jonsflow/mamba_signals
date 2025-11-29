# M1 MacBook Pro Setup Guide

## Prerequisites

You're running on **Apple Silicon (M1/M2/M3)**, which requires native dependencies and proper PyTorch setup.

## Installation Steps

### 1. Create Virtual Environment (Recommended)
```bash
cd /Users/clutchcoder/working/mamba_signals

# Using Python's built-in venv
python3 -m venv venv
source venv/bin/activate

# Or if using conda
conda create -n mamba_signals python=3.10
conda activate mamba_signals
```

### 2. Install Dependencies

**IMPORTANT: For M1 Macs, PyTorch must be installed specifically for ARM64 architecture**

```bash
# Install PyTorch for M1/M2/M3 Mac
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check PyTorch is correctly installed
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Expected output:
# PyTorch version: 2.x.x
# MPS available: True
```

## Device Support

The training script automatically detects and uses:

1. **MPS (Metal Performance Shaders)** - M1/M2/M3 GPU acceleration (preferred)
2. **CPU** - Fallback if MPS not available

### MPS Performance
- **~2-3x faster** than CPU for training
- Uses unified memory architecture efficiently
- No external GPU drivers needed

## Running Training

```bash
python train.py
```

The script will:
1. Auto-detect MPS availability
2. Log device type (look for "Metal Performance Shaders")
3. Use batch_size=16 (optimized for M1 memory)
4. Train the model with GPU acceleration

### Example Output:
```
================================================================================
MAMBA SIGNALS - TRAINING PIPELINE
================================================================================

1. Loading OHLC data...
Loaded data for 13 symbols

2. Normalizing data...
Processing AAPL...
...

4. Training model...
  Device: Metal Performance Shaders (M1/M2/M3)
  Model: ...

Epoch 1 [Train]: 100%|████████| ... [loss: 0.64, acc: 0.65]
Epoch 1 [Val]: 100%|████████| ... [loss: 0.62, acc: 0.67]
```

## Performance Tips

### Memory Optimization
If you encounter MPS out-of-memory errors:

```python
# In src/config.py, reduce batch size:
batch_size: int = 8  # Instead of 16
```

### CPU-Only Fallback
If you prefer CPU (for debugging/stability):

```bash
# Disable MPS temporarily
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py
```

### Disable GPU Warnings
```python
# Add to train.py if MPS causes warnings:
import warnings
warnings.filterwarnings('ignore')
```

## Troubleshooting

### Issue: "MPS not available"
```
Solution: Ensure you installed PyTorch from the CPU wheel:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "AttributeError: module 'torch.backends' has no attribute 'mps'"
```
Solution: Older PyTorch versions don't support MPS. Update:
pip install --upgrade torch
```

### Issue: Slower than expected training
```
Reasons:
1. Model loaded on CPU instead of MPS - check logs
2. Large batch_size causing MPS memory pressure - reduce to 8
3. Using CPU fallback - ensure MPS is enabled
```

### Issue: "OutOfMemory on MPS device"
```
Solution: Reduce model size or batch size in src/config.py:
hidden_dim: int = 128  # Instead of 256
batch_size: int = 8    # Instead of 16
```

## Hardware Specifications

**M1 Pro/Max MacBook Pro:**
- CPU: 8-10 cores
- GPU: 16-core GPU (M1 Pro) or 32-core GPU (M1 Max)
- Memory: 16GB-32GB unified memory
- Recommended batch_size: 8-32 (16 is balanced)

**M2/M3 and variants:**
- Similar architecture with improved efficiency
- Same batch_size recommendations apply

## Configuration for M1

Key settings in `src/config.py` are pre-optimized:

```python
# Already set for M1:
training.batch_size = 16        # Balanced for M1 memory
training.device = "mps"          # Auto-detected
training.epochs = 50             # Reasonable training time

# Adjust if needed:
model.hidden_dim = 256           # Model capacity
model.num_layers = 4             # Depth (more = slower)
```

## Expected Performance

**M1 Pro (16-core GPU) with MPS:**
- ~500-1000 samples/second for inference
- ~200-400 samples/second for training
- Full dataset (~1M samples): ~2-4 hours per epoch

**M1 CPU (baseline):**
- ~100-200 samples/second
- ~50-100 samples/second training
- Full dataset: ~8-16 hours per epoch

## Next Steps

1. **First run**: `python train.py` (full dataset, ~2-4 hours)
2. **Check results**: `ls -lh models/` for saved checkpoints
3. **Analyze**: See SETUP.md for training history and evaluation
4. **Tune**: Adjust hyperparameters in `src/config.py`

## Additional Resources

- PyTorch M1 Guide: https://pytorch.org/get-started/locally/
- MPS Documentation: https://pytorch.org/docs/stable/notes/mps.html
- M1 GPU Optimization: https://developer.apple.com/metal/pytorch/
