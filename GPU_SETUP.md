# GPU Training Setup

This branch (`gpu-training`) is optimized for running on a machine with NVIDIA GPU (CUDA).

## Configuration for GPU

The default configuration in this branch is optimized for:
- **GPU**: NVIDIA with 12GB VRAM
- **System RAM**: 128GB
- **Training**: Long overnight runs (100+ episodes)

### Key Changes from CPU Version

| Parameter | CPU Version | GPU Version | Notes |
|-----------|------------|------------|-------|
| `batch_size` | 8 | 64 | Larger batches for GPU efficiency |
| `replay_buffer_size` | 1000 | 50000 | More experience data |
| `episodes` | 5 | 100 | More training iterations |
| `hidden_dim` | 64 | 256 | Larger model capacity |
| `num_layers` | 1 | 2 | Deeper model |
| `sequence_length` | 64 | 128 | Longer context window |
| `max_samples` | 5000 | None | Use all available data |
| `target_update_freq` | 100 | 500 | Less frequent target network updates |
| `device` | "mps" | "cuda" | Use NVIDIA GPU |

## Installation

### PyTorch with CUDA Support

Install PyTorch with CUDA 11.8 support (or your CUDA version):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Or for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version:
```bash
nvidia-smi
```

### Other Dependencies

```bash
pip install numpy pandas pydantic tqdm
```

## Verification

Before running training, verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
print(torch.cuda.get_device_properties(0))  # Should show GPU details
```

Or from command line:

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Running Training

```bash
python3 train.py
```

The Agent will automatically detect and use the GPU. You should see:
```
Agent device: cuda
```

## Monitoring GPU Usage

In another terminal, monitor GPU usage:

```bash
nvidia-smi -l 1  # refresh every 1 second
```

Or with better formatting:

```bash
watch -n 1 nvidia-smi
```

## Configuration Adjustments

If you want to adjust parameters for your specific hardware, edit `src/config.py`:

```python
# For less GPU memory:
batch_size = 32  # instead of 64
replay_buffer_size = 25000  # instead of 50000

# For more GPU memory:
batch_size = 128
hidden_dim = 512
num_layers = 3
```

## Expected Performance

With a 12GB GPU:
- Episode time: ~5-10 minutes (depends on data size and batch size)
- Memory usage: ~8-10GB during training
- Total runtime for 100 episodes: ~8-16 hours

## Troubleshooting

### "CUDA out of memory"
Reduce `batch_size` or `replay_buffer_size` in `src/config.py`

### "RuntimeError: CUDA error: device not specified"
Ensure you have the correct PyTorch CUDA version installed matching your GPU driver

### Agent not using GPU
Check that device is set to "cuda" in config.py:
```python
device: str = "cuda"
```

## Notes

- The code will automatically fall back to CPU if CUDA is unavailable
- All data preprocessing happens on CPU, only the neural network runs on GPU
- Batch normalization and dropout will behave slightly differently on GPU vs CPU due to parallel processing
