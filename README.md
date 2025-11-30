# Mamba Signals - RL Trading Agent

A Deep Q-Learning (DQN) trading agent using **Mamba state-space models** to learn buy/sell/hold decisions from OHLCV price data.

## 🚀 Quick Start

### For Mac (M1/M2/M3)
```bash
git clone https://github.com/jonsflow/mamba_signals.git
cd mamba_signals
python3 -m venv venv && source venv/bin/activate
pip install torch
pip install numpy pandas pydantic tqdm
python3 train.py
```

### For NVIDIA GPU
```bash
git clone https://github.com/jonsflow/mamba_signals.git
cd mamba_signals
git checkout gpu-training  # Use GPU-optimized branch
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pydantic tqdm
python3 train.py
```

See **GPU_SETUP.md** for detailed GPU configuration.

## 📊 Dataset

- **1.3M+ OHLCV candles** from 13 major stocks
- **Symbols**: AAPL, AMD, AMZN, GOOGL, IBM, META, MSFT, MSTR, NFLX, NVDA, QQQ, SPY, TSLA
- **Timeframes**: 1-minute, 5-minute, daily
- **SQLite database**: 191MB local file (no API calls)

## 🧠 Architecture

### Mamba Encoder
- **Selective State Space Model**: O(N) complexity (efficient vs O(N²) transformers)
- **Input**: Last N OHLCV candles (Open, High, Low, Close, Volume)
- **Output**: Dense state vector encoding price patterns
- **Layers**: Configurable (1-2 for Mac, 2+ for GPU)

### DQN Agent
- **Main Q-Network**: Maps states to Q-values for [HOLD, BUY, SELL]
- **Target Q-Network**: Stabilizes training (updated periodically)
- **Experience Replay**: Stores (state, action, reward, next_state, done) tuples
- **ε-Greedy Exploration**: Balance exploration vs exploitation

### Training Loop
1. Agent observes OHLCV state
2. Selects action (BUY/SELL/HOLD) using Q-network
3. Environment executes action, returns reward
4. Agent stores experience in replay buffer
5. Training step: sample from buffer, compute loss, backprop
6. Periodically update target network

## 🎯 How It Works

### Reward Function (Configurable)
The agent learns by receiving rewards for good decisions:

**bars_primary** (default - emphasizes patience):
- BUY: -0.50 (discourages random entries)
- SELL: bars_held_profitably + 0.1×profit - 0.25 (rewards patience)
- HOLD: 0 (no signal)

**trading_agent** (simple baseline):
- BUY: 0
- SELL: actual_profit (P&L only)
- HOLD: 0

Switch presets by editing `src/config.py`:
```python
reward_preset: str = "bars_primary"  # or "trading_agent", "continuous_feedback"
```

### Action Space
- **HOLD (0)**: Stay in current position or remain flat
- **BUY (1)**: Open a long position (1 share)
- **SELL (2)**: Close position (max 1 position allowed)

## 🛠 Configuration

### Mac (M1/M2/M3) - Optimized for 8GB RAM

Edit `src/config.py`:
```python
DataConfig:
  max_samples: int = 5000      # Limit data for faster iteration
  sequence_length: int = 64    # Shorter lookback window

ModelConfig:
  hidden_dim: int = 64         # Smaller model
  num_layers: int = 1          # Fewer layers

RLConfig:
  episodes: int = 10           # Quick training cycles
  batch_size: int = 8          # Small batches
  replay_buffer_size: int = 1000
```

### NVIDIA GPU - Optimized for 12GB+ VRAM

Switch to `gpu-training` branch (has these defaults):
```python
DataConfig:
  max_samples: Optional[int] = None  # Use all data
  sequence_length: int = 128         # Longer context

ModelConfig:
  hidden_dim: int = 256              # Larger model
  num_layers: int = 2                # Deeper network

RLConfig:
  episodes: int = 100                # Longer training
  batch_size: int = 64               # Larger batches
  replay_buffer_size: int = 50000
```

## 📁 Project Structure

```
mamba_signals/
├── train.py                      # Main training entry point
├── ohlc_data.db                  # SQLite database (OHLCV data)
│
├── src/
│   ├── config.py                 # Configuration (Data, Model, RL)
│   ├── data_loader.py            # SQLite data loading
│   ├── mamba_model.py            # Mamba encoder + DQN head
│   ├── training.py               # TradingEnvironment + Agent
│   └── reward_config.py          # Reward function presets
│
├── models/                       # Checkpoints & results
│   ├── agent_episode_N.pt        # Model weights
│   └── training_history.json     # Metrics per episode
│
├── README.md                     # This file
├── GPU_SETUP.md                  # GPU-specific guide
├── RL_DESIGN.md                  # Architecture design
└── RL_REWARD_PROBLEM.md         # Reward engineering analysis
```

## 📊 Output & Metrics

After training, check `models/training_history.json`:
```json
{
  "train_profit": [123.34, -91.99, 1290.70, ...],
  "train_loss": [0.0582, 0.3772, 0.8553, ...],
  "train_trades": [2288, 2298, 2449, ...],
  "val_profit": [0.00, 141.48, -0.11, ...],
  "val_trades": [680, 651, 686, ...]
}
```

**Metrics explained:**
- **Profit**: Realized P&L in dollars (from closed trades)
- **Loss**: DQN training loss (lower = better)
- **Trades**: Number of buy/sell transactions
- **Val**: Validation set performance (different from training)

## 🔬 Key Features

✅ **Mamba Encoder** - O(N) state-space model for efficient sequence processing
✅ **DQN Agent** - Deep Q-Learning with experience replay and target networks
✅ **Configurable Rewards** - Swap between reward presets to experiment
✅ **GPU Support** - Automatic device selection (CUDA > MPS > CPU)
✅ **Mac Optimized** - Runs on M1/M2/M3 with MPS acceleration
✅ **GPU Optimized** - 100+ episode training on NVIDIA GPUs
✅ **Type-Safe** - Pydantic configuration, type hints throughout

## 📚 Documentation

- **GPU_SETUP.md** - NVIDIA GPU installation and optimization
- **RL_DESIGN.md** - Initial architecture and design decisions
- **RL_REWARD_PROBLEM.md** - Analysis of reward engineering challenges

## 🚦 Next Steps

1. **Setup**: Follow quick start for your platform (Mac or GPU)
2. **Data**: Ensure `ohlc_data.db` is in the project root
3. **Configure**: Edit `src/config.py` if needed (optional)
4. **Train**: Run `python3 train.py`
5. **Results**: Check `models/training_history.json`
6. **Experiment**: Try different reward presets in `src/reward_config.py`

## 🌿 Branches

- **rl-agent-mamba** - Main development (Mac-friendly, smaller configs)
- **gpu-training** - GPU-optimized (100 episodes, larger models, full data)

Switch branches:
```bash
git checkout gpu-training   # For NVIDIA GPU
git checkout rl-agent-mamba # For Mac or CPU
```

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+ (with appropriate backend)
- NumPy, Pandas, Pydantic, tqdm
- SQLite (built-in)

## ⚙️ Performance

**Mac M1 Pro (8GB RAM, batch_size=8)**
- Per episode: ~5-10 minutes
- 10 episodes: ~1-2 hours

**NVIDIA GPU 12GB (batch_size=64)**
- Per episode: ~2-5 minutes
- 100 episodes: ~3-8 hours

## 🔧 Troubleshooting

### Device not detected
Check automatic device selection:
```python
# src/training.py line ~137-138
# Will try: CUDA (GPU) → MPS (Mac) → CPU
```

### Out of memory
Reduce in `src/config.py`:
```python
batch_size = 16 → 8       # Smaller batches
hidden_dim = 256 → 128    # Smaller model
replay_buffer_size = 50000 → 10000
```

### Training too slow
- Verify correct device is being used (check logs: "Agent device: cuda" or "mps")
- For Mac: ensure MPS is available (PyTorch 1.12+)
- For GPU: ensure CUDA PyTorch is installed (not CPU version)

### Validation profit is $0.00
This is normal in early episodes when exploration is high (epsilon=1.0). Agent learns gradually over multiple episodes.

## 🎓 References

- **Mamba**: [Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- **DQN**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **RL Trading**: [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107)

## 📝 License

MIT - Educational and authorized research use only

---

**Author**: @jonsflow
**Created**: 2025
**PyTorch**: 2.0+
**Branches**: rl-agent-mamba (Mac), gpu-training (NVIDIA GPU)
**Last Updated**: 2025-11-30
