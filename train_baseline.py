#!/usr/bin/env python3
"""Baseline RL training using dense layers (Trading Agent approach).

This is a simpler baseline that uses dense layers instead of Mamba encoder,
to compare against the Mamba+DQN approach.
"""
import sys
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import OHLCDataLoader
from config_loader import default_config
from training import TradingEnvironment, HOLD, BUY, SELL
from reward_config import get_reward_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DenseQNetwork(nn.Module):
    """Simple dense Q-network baseline (no Mamba)."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # input_dim should already account for flattened size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions: HOLD, BUY, SELL
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, 5)

        Returns:
            Q-values of shape (batch, 3)
        """
        batch_size = x.shape[0]
        # Flatten: (batch, seq_len, 5) -> (batch, seq_len*5)
        x = x.reshape(batch_size, -1)
        return self.net(x)


class BaselineAgent:
    """Simple dense-layer Q-learning agent (Trading Agent baseline)."""

    def __init__(self, config, device=None):
        self.config = config

        # Device selection
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        logger.info(f"Baseline agent device: {self.device}")

        # Input dimension: (sequence_length + 1) * 5 (OHLCV)
        # +1 because we include the current bar, so it's [t-seq_len, ..., t]
        input_dim = (config.data.sequence_length + 1) * 5

        # Networks
        self.model = DenseQNetwork(input_dim, hidden_dim=128).to(self.device)
        self.target_model = DenseQNetwork(input_dim, hidden_dim=128).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.rl.learning_rate)
        self.criterion = nn.HuberLoss(delta=1.0)

        # Experience replay
        self.memory = deque(maxlen=config.rl.replay_buffer_size)
        self.steps = 0

        # Exploration
        self.epsilon = config.rl.epsilon
        self.epsilon_min = config.rl.epsilon_min
        self.epsilon_decay = config.rl.epsilon_decay

        # Environment
        reward_config = get_reward_config(config.rl.reward_preset)
        self.env = TradingEnvironment(
            initial_cash=config.rl.initial_cash,
            share_size=config.rl.share_size,
            reward_config=reward_config
        )

    def act(self, state, is_eval=False):
        """Select action using epsilon-greedy policy."""
        if is_eval or np.random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = state.to(self.device)
                q_values = self.model(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action
        else:
            return np.random.randint(0, 3)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self, batch_size):
        """Train on batch from experience replay."""
        if len(self.memory) < batch_size:
            return 0.0

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Convert to tensors
        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # Current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.config.rl.gamma * next_q_values

        # Loss and backprop
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1

        # Update target network periodically
        if self.steps % self.config.rl.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def get_state(data, idx, sequence_length, normalize=True):
    """Get state tensor from price data."""
    start_idx = max(0, idx - sequence_length)
    window = data.iloc[start_idx:idx + 1]

    if len(window) < sequence_length:
        padding = np.zeros((sequence_length - len(window), 5))
        window_vals = np.vstack([padding, window[['open', 'high', 'low', 'close', 'volume']].values])
    else:
        window_vals = window[['open', 'high', 'low', 'close', 'volume']].values

    if normalize:
        for i in range(5):
            col = window_vals[:, i]
            col_min = col.min()
            col_max = col.max()
            if col_max > col_min:
                window_vals[:, i] = (col - col_min) / (col_max - col_min)

    state = torch.FloatTensor(window_vals).unsqueeze(0)
    return state


def split_data_chronologically(data, train_ratio, val_ratio):
    """Split data chronologically."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end].reset_index(drop=True)
    val_data = data.iloc[train_end:val_end].reset_index(drop=True)
    test_data = data.iloc[val_end:].reset_index(drop=True)

    logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


def train_episode(agent, data, episode_num, total_episodes, config):
    """Train agent for one episode."""
    agent.env.reset()
    total_profit = 0.0
    avg_loss = 0.0
    num_trades = 0
    losses = []

    data_len = len(data)

    with tqdm(total=data_len - config.data.sequence_length, desc=f"Episode {episode_num}/{total_episodes} [TRAIN]", leave=False) as pbar:
        for t in range(config.data.sequence_length, data_len):
            state = get_state(data, t, config.data.sequence_length)
            next_state = get_state(data, t + 1, config.data.sequence_length) if t + 1 < data_len else state

            action = agent.act(state, is_eval=False)

            price = data.iloc[t]['close']
            reward = agent.env.step(action, price)

            if action in [BUY, SELL]:
                num_trades += 1
            if action == SELL:
                total_profit += reward

            done = (t == data_len - 1)

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= config.rl.batch_size:
                loss = agent.train_experience_replay(config.rl.batch_size)
                if loss > 0:
                    losses.append(loss)

            pbar.update(1)

    if losses:
        avg_loss = np.mean(losses)

    logger.info(f"Episode {episode_num}/{total_episodes} - Profit: ${total_profit:.2f}, Trades: {num_trades}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")

    return total_profit, avg_loss, num_trades


def eval_episode(agent, data, config):
    """Evaluate agent (greedy, no exploration)."""
    agent.env.reset()
    total_profit = 0.0
    num_trades = 0

    data_len = len(data)

    for t in range(config.data.sequence_length, data_len):
        state = get_state(data, t, config.data.sequence_length)
        action = agent.act(state, is_eval=True)

        price = data.iloc[t]['close']
        reward = agent.env.step(action, price)

        if action in [BUY, SELL]:
            num_trades += 1
        if action == SELL:
            total_profit += reward

    return total_profit, num_trades


def main():
    """Main training pipeline for baseline."""
    config = default_config
    db_path = Path(__file__).parent / "ohlc_data.db"

    logger.info("=" * 80)
    logger.info("BASELINE RL TRADING AGENT (Dense Layers) - TRAINING")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\n[CONFIGURATION - BASELINE]")
    logger.info(f"Data Config:")
    logger.info(f"  symbol: {config.data.symbol}")
    logger.info(f"  timeframe: {config.data.timeframe}")
    logger.info(f"  sequence_length: {config.data.sequence_length}")
    logger.info(f"  max_samples: {config.data.max_samples}")
    logger.info(f"Model: Dense Layers (no Mamba)")
    logger.info(f"RL Config:")
    logger.info(f"  episodes: {config.rl.episodes}")
    logger.info(f"  batch_size: {config.rl.batch_size}")
    logger.info(f"  replay_buffer_size: {config.rl.replay_buffer_size}")
    logger.info(f"  device: {config.rl.device}")
    logger.info(f"  reward_preset: {config.rl.reward_preset}")
    logger.info("")

    # 1. Load data
    logger.info("\n1. Loading OHLCV data from SQLite...")
    with OHLCDataLoader(db_path, config.data) as loader:
        data = loader.get_ohlcv_data(config.data.symbol)

    if config.data.max_samples:
        data = data.iloc[-config.data.max_samples:].reset_index(drop=True)
        logger.info(f"Limited to last {config.data.max_samples} candles for iteration")

    logger.info(f"Loaded {len(data)} candles for {config.data.symbol}")

    # 2. Split data chronologically
    logger.info("\n2. Splitting data chronologically...")
    train_data, val_data, test_data = split_data_chronologically(
        data,
        config.data.train_ratio,
        config.data.val_ratio
    )

    # 3. Create baseline agent
    logger.info("\n3. Creating baseline agent...")
    agent = BaselineAgent(config)
    model_dir = Path(__file__).parent / "models" / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 4. Training loop
    logger.info("\n4. Starting training loop...")
    training_history = {
        'train_profit': [],
        'train_loss': [],
        'train_trades': [],
        'val_profit': [],
        'val_trades': []
    }

    for episode in tqdm(range(1, config.rl.episodes + 1), desc="Episodes", unit="ep"):
        # Train
        train_profit, train_loss, train_trades = train_episode(agent, train_data, episode, config.rl.episodes, config)

        # Validate
        val_profit, val_trades = eval_episode(agent, val_data, config)

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        training_history['train_profit'].append(float(train_profit))
        training_history['train_loss'].append(float(train_loss))
        training_history['train_trades'].append(int(train_trades))
        training_history['val_profit'].append(float(val_profit))
        training_history['val_trades'].append(int(val_trades))

        # Save checkpoint
        if episode % config.rl.save_every == 0:
            checkpoint = {
                'episode': episode,
                'model_state': agent.model.state_dict(),
                'target_model_state': agent.target_model.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'steps': agent.steps,
            }
            path = model_dir / f"baseline_episode_{episode}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved baseline checkpoint: {path}")

        # Print progress
        logger.info(f"Episode {episode}/{config.rl.episodes} - Train: ${train_profit:.2f} | Val: ${val_profit:.2f} | Trades: {train_trades} | ε: {agent.epsilon:.3f}")

    # Save final history
    import json
    history_path = model_dir / "baseline_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"\nSaved training history to {history_path}")

    # Final test evaluation
    logger.info("\n5. Final evaluation on test set...")
    test_profit, test_trades = eval_episode(agent, test_data, config)
    logger.info(f"Test Results - Profit: ${test_profit:.2f}, Trades: {test_trades}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
