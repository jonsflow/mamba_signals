# CLAUDE.md - Mamba Signals RL Trading Agent

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Current Status (2025-11-30)

**Project**: Mamba-based Deep Q-Learning (DQN) trading agent learning to buy/sell/hold decisions from OHLCV price data.

**Latest Session (Nov 30 Evening)**:
- ✅ Fixed P&L metric contamination (reward penalties were being added to profit)
- ✅ Fixed trade counting (now counts actual executed round-trips, not action signals)
- ✅ Added debug output showing Buys/Sells separately with validation check
- ✅ Removed invalid SELL penalty (-0.1) that was making agent freeze trading
- ✅ Current baseline: `trading_agent` preset with no penalties

**Previous Changes**:
- ✅ Implemented configurable reward system with multiple presets
- ✅ Created GPU-optimized branch (`gpu-training`) for NVIDIA CUDA training
- ✅ Single position constraint (max 1 open position at a time)

**Branches**:
- `rl-agent-mamba` - Main development (Mac-friendly, small configs)
- `gpu-training` - GPU optimized (100 episodes, larger model, full data)

## Architecture Overview

```
SQLite Database (ohlc_data.db, OHLCV data)
    ↓
OHLCDataLoader (src/data_loader.py) - Load & normalize
    ↓
Mamba Encoder (src/mamba_model.py) - Encode last N candles
    ↓
DQN Agent (src/training.py) - Decide BUY/SELL/HOLD
    ↓
TradingEnvironment (src/training.py) - Execute action, return reward
    ↓
Experience Replay & Training - Learn Q-values
    ↓
Checkpoint & History (models/) - Save models and metrics
```

### Key Components

- **src/data_loader.py**: `OHLCDataLoader` loads symbol data from SQLite, applies min-max normalization

- **src/mamba_model.py**:
  - `MambaBlock`: Selective state space layer (O(N) complexity)
  - `MambaEncoder`: Encodes OHLCV sequences into state vectors
  - `DQNHead`: Maps state vectors to Q-values for 3 actions (HOLD, BUY, SELL)
  - `DQNAgent`: Combined encoder + head

- **src/training.py**:
  - `TradingEnvironment`: Simulates trading environment, executes actions, returns rewards
  - `Agent`: DQN agent with main/target networks, experience replay, ε-greedy exploration

- **src/reward_config.py**: Reward function presets (bars_primary, trading_agent, continuous_feedback, conservative_trading)

- **src/config.py**: Pydantic configuration with `DataConfig`, `ModelConfig`, `RLConfig`

### Data

- **Source**: SQLite database with OHLCV candles from multiple stocks (SPY by default)
- **Timeframes**: 1m, 5m, 1d (config defaults to 5m)
- **File**: ohlc_data.db (place in project root)

## Getting Started on GPU Machine

**For NVIDIA GPU training overnight**:

```bash
# Clone and setup
git clone https://github.com/jonsflow/mamba_signals.git
cd mamba_signals

# Switch to GPU-optimized branch
git checkout gpu-training

# Setup environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas pydantic tqdm

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Run training (will use default config with 100 episodes)
python3 train.py

# Monitor GPU
nvidia-smi -l 1  # In another terminal
```

## Running Training

```bash
python3 train.py
```

This:
1. Loads OHLCV data from SQLite
2. Normalizes sequences
3. Splits into train/val/test
4. Trains DQN agent for N episodes
5. Saves checkpoints and metrics to `models/`

## Configuring Training

Edit `src/config.py`:

**For Mac (8GB RAM)**:
```python
DataConfig:
  max_samples = 5000          # Limit data
  sequence_length = 64        # Shorter window

ModelConfig:
  hidden_dim = 64             # Smaller model
  num_layers = 1

RLConfig:
  episodes = 10               # Quick iteration
  batch_size = 8              # Small batches
  replay_buffer_size = 1000
```

**For NVIDIA GPU (12GB VRAM, 128GB RAM)** - Already set in `gpu-training` branch:
```python
DataConfig:
  max_samples = None          # Use all data
  sequence_length = 128       # Longer window

ModelConfig:
  hidden_dim = 256            # Larger model
  num_layers = 2

RLConfig:
  episodes = 100              # More training
  batch_size = 64             # Larger batches
  replay_buffer_size = 50000
```

## Reward Presets

Change reward function by editing `src/config.py`:

```python
RLConfig:
  reward_preset = "bars_primary"  # Default: bars held + small P&L bonus
```

Available presets in `src/reward_config.py`:
- **bars_primary**: BUY=-0.50, SELL=bars+0.1×delta-0.25, HOLD=0
- **trading_agent**: BUY=0, SELL=delta, HOLD=0 (baseline)
- **continuous_feedback**: BUY=0, SELL=delta, HOLD=0.1×unrealized_pnl
- **conservative_trading**: BUY=-0.25, SELL with modest penalties

## Output Files

After training:
- `models/agent_episode_N.pt` - Checkpoint after episode N
- `models/training_history.json` - Metrics per episode:
  ```json
  {
    "train_profit": [...],
    "train_loss": [...],
    "train_trades": [...],
    "val_profit": [...],
    "val_trades": [...]
  }
  ```

## Device Detection

The codebase auto-detects device in `src/training.py` Agent.__init__():
1. Checks for CUDA (NVIDIA GPU)
2. Falls back to MPS (M1/M2/M3 Macs)
3. Falls back to CPU

Verify GPU detection:
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Expected Performance

**Mac M1 Pro (batch_size=8, 10 episodes)**:
- Per episode: 5-10 minutes
- Total: 1-2 hours
- Memory: 4-6GB

**NVIDIA GPU (batch_size=64, 100 episodes)**:
- Per episode: 2-5 minutes
- Total: 3-8 hours overnight
- Memory: 8-10GB VRAM

## Important Notes - Metrics & Trade Counting

**Metrics Fixed (Nov 30)**:
- **P&L Tracking**: Now uses `agent.env.total_profit` (actual realized delta only)
- **Reward Separation**: Reward signal penalties no longer contaminate profit metrics
- **Trade Definition**: A trade = complete round-trip (buy-sell pair), counted as `num_sells`
- **Trade Counting**: Only counts successfully executed trades (inventory actually changed)
- **Invalid Actions**: SELL with no inventory gets reward=0.0 (ignored, not penalized)

**Debug Output**:
Both training and validation print:
```
[TRAIN/VAL DEBUG] Buys: X, Sells: Y, Open position: Z shares (should be X-Y), Total profit: $P
```
This validates that buy/sell accounting is correct (position should equal buys minus sells).

**Reward Presets**:
- **trading_agent** (current): BUY=0, SELL=delta*1.0, HOLD=0, no penalties
- **sparse_trading**: Adds trade_density_penalty (makes agent freeze)
- **min_5bars**: Direction change penalty (needs 5 bars before buy→sell flip)
- **controlled_trading**: Max trades constraint (100/episode)
- **bars_primary**: Complex (bars_held_profitable bonus)

## Important Notes - General

- **Single Position**: Agent can hold max 1 position at a time
- **Overfitting**: Validation profit may diverge from training (normal for RL)
- **Exploration**: Epsilon starts at 1.0 (random), decays to 0.01 (greedy)
- **Current Issue**: Agent makes trades in training but freezes in validation

## Files to Know

**Key Files**:
- `train.py` - Mamba+DQN training (both systems)
- `train_baseline.py` - Dense layer baseline for comparison
- `src/config.py` - Configuration (shared by both scripts)
- `src/training.py` - TradingEnvironment, Agent, step/reward logic
- `src/reward_config.py` - Reward presets

**Documentation**:
- `README.md` - Main guide (setup, architecture, troubleshooting)
- `CLAUDE.md` - This file (for Claude Code sessions)
