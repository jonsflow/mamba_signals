# Mamba+DQN RL Refactor - File-by-File Changes

## Overview
Convert supervised Mamba classifier to RL agent using Trading Agent design pattern. Keep working pieces (data_loader, Mamba encoder, SQLite pipeline), refactor training paradigm to episode-based RL.

---

## File Changes

### src/config.py (MODIFY)
**Current**: DataConfig, LabelingConfig, ModelConfig, TrainingConfig

**Changes**:
- Remove `LabelingConfig` (no labels in RL)
- Rename `TrainingConfig` → `RLConfig` with new parameters:
  - `episodes: int = 5` (for dev)
  - `episode_length: int = 100` (candles per episode)
  - `replay_buffer_size: int = 1000`
  - `batch_size: int = 8`
  - `learning_rate: float = 1e-3`
  - `gamma: float = 0.99` (discount factor)
  - `epsilon: float = 1.0` (exploration)
  - `epsilon_min: float = 0.01`
  - `epsilon_decay: float = 0.995`
  - `target_update_freq: int = 1000` (update target network every N steps)

- Keep `DataConfig` (reuse)
- Modify `ModelConfig`:
  - Keep Mamba params (hidden_dim, num_layers, etc)
  - Remove `output_dim` (will be 3 for [HOLD, BUY, SELL])
  - Add `state_dim: int` output size

### src/mamba_model.py (MODIFY)
**Current**: MambaForecaster (has classification head), MambaLSTMHybrid, create_model factory

**Changes**:
- Remove classification head from `MambaForecaster`
- Rename to `MambaEncoder` - outputs fixed state vector only
- Remove `MambaLSTMHybrid` (not needed for RL)
- Add separate `DQNHead(nn.Module)`:
  ```python
  class DQNHead(nn.Module):
      def __init__(self, state_dim, action_dim=3):
          # state_dim (64) → hidden → action Q-values (3)
  ```
- Keep `create_model` factory, add mode parameter for "encoder" vs "dqn"

### src/training.py (MODIFY - MAJOR REFACTOR)
**Current**: Trainer class with train_epoch, validate, fit methods

**Replace with** RL-style training inspired by Trading Agent:
- Create `RLEnvironment` class:
  - Tracks: cash, shares_held, entry_price, history
  - Methods: step(action), reset(), calculate_reward()

- Create `Agent` class (like Trading Agent):
  - Holds `MambaEncoder` + `DQNHead` networks
  - Main network + target network
  - Experience replay buffer: `deque(maxlen=replay_buffer_size)`
  - Methods:
    - `act(state, is_eval=False)`: ε-greedy action selection
    - `remember(state, action, reward, next_state, done)`: store experience
    - `train_experience_replay(batch_size)`: train on batch
    - `update_target_network()`: copy main → target
    - `save(episode)`: checkpoint

### train.py (MODIFY)
**Current**: Loads data, creates windows, trains supervised model, evaluates test set

**Refactor to episode-based training**:
```python
def main():
    # Load data from SQLite
    loader = OHLCDataLoader(db_path, config.data)
    data = loader.get_ohlcv_data("SPY")  # Get price series

    # Split chronologically: train / val / test
    train_data, val_data, test_data = split_data(data, 0.7, 0.15, 0.15)

    # Create agent
    agent = Agent(config)

    # Episode-based training loop (like Trading Agent)
    for episode in range(config.rl.episodes):
        # Train on train_data
        train_profit = train_episode(agent, train_data, episode, config)

        # Validate on val_data
        val_profit = eval_episode(agent, val_data, is_eval=True)

        # Log: episode, train profit, val profit, avg loss
        print(f"Episode {episode}: Train ${train_profit:.2f}, Val ${val_profit:.2f}")

        if episode % 5 == 0:
            agent.save(episode)

def train_episode(agent, data, episode, config):
    # Like Trading Agent's train_model()
    total_profit = 0
    state = get_state(data, 0, config.data.sequence_length)

    for t in range(len(data) - 1):
        next_state = get_state(data, t + 1, config.data.sequence_length)
        action = agent.act(state)  # ε-greedy

        reward = execute_action(agent.env, action, data[t])
        total_profit += reward if action == SELL else 0

        done = (t == len(data) - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > config.rl.batch_size:
            agent.train_experience_replay(config.rl.batch_size)

        state = next_state

    return total_profit

def get_state(data, idx, sequence_length):
    # Get last sequence_length candles, normalize, encode with Mamba
    candles = data[max(0, idx - sequence_length):idx]
    # Returns: tensor of shape (1, sequence_length, 5) for Mamba encoder
```

### eval.py (CREATE NEW)
**Purpose**: Backtest trained agent on test data, show P&L and trades

**Structure** (from Trading Agent pattern):
```python
def main(model_name, data):
    agent = Agent(pretrained=True, model_name=model_name)
    total_profit = 0
    trades = []

    for t in range(len(data)):
        action = agent.act(state, is_eval=True)

        if action == BUY:
            trades.append(("BUY", data[t]))
        elif action == SELL and len(agent.inventory) > 0:
            entry = agent.inventory.pop()
            pnl = data[t] - entry
            total_profit += pnl
            trades.append(("SELL", data[t], pnl))

    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Trades: {len(trades)}")
    print(trades)
```

### src/windowing.py (DELETE)
**Reason**: Not needed. RL gets raw price series, Mamba handles windowing internally via get_state()

### src/data_loader.py (KEEP - MINIMAL CHANGE)
**Current**: Works fine for loading OHLCV data

**Small change**:
- Add method to return just close prices as list (for train.py)
- Or keep as-is, just use `df['close'].values`

---

## Key Differences from Current Supervised Version

| Aspect | Supervised | RL |
|--------|-----------|-----|
| Input | 128 candles → windowed data | Raw price series |
| Labels | Generated from future price | None (reward from action) |
| Model output | Binary (UP/DOWN) logits | 3 Q-values (HOLD/BUY/SELL) |
| Training | Epochs over static dataset | Episodes over same data with exploration |
| Evaluation | Test accuracy | Backtest profit/loss |
| Action | Prediction only | Actual buy/sell with cash tracking |

---

## Training Flow

```
load_data("SPY")
├── split train/val/test
└── for episode 1 to N:
    ├── train_episode(train_data):
    │   ├── reset agent.env (cash, shares)
    │   ├── for each candle:
    │   │   ├── encode last 64 candles with Mamba
    │   │   ├── DQN picks action (ε-greedy)
    │   │   ├── execute action, get reward
    │   │   ├── store in replay buffer
    │   │   └── train on batch if buffer full
    │   └── return total profit
    ├── eval_episode(val_data)
    │   ├── reset agent.env
    │   ├── for each candle: act (greedy, no exploration)
    │   └── return total profit
    └── save model every N episodes
```

---

## Implementation Steps

1. **src/config.py**: Update RLConfig
2. **src/mamba_model.py**: Remove classifier, add DQNHead
3. **src/training.py**: Build RLEnvironment, Agent, train_experience_replay
4. **train.py**: Rewrite with episode loop and get_state()
5. **eval.py**: Create backtest script
6. **windowing.py**: Delete
7. **Test on dev laptop**: Quick 5-episode run

---

## Success Criteria

- Code runs without errors on M1 Air 8GB
- Agent makes buy/sell actions
- Shows total profit/loss
- Can load/eval trained model
