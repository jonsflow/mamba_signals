#!/usr/bin/env python3
"""Main RL training script for Mamba+DQN trading agent."""
import sys
import logging
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import default_config
from data_loader import OHLCDataLoader
from training import Agent, HOLD, BUY, SELL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_state(data, idx, sequence_length, normalize=True):
    """Get state tensor from price data.

    Args:
        data: DataFrame with OHLCV columns
        idx: Current index
        sequence_length: Number of candles to look back
        normalize: Whether to normalize

    Returns:
        State tensor of shape (1, sequence_length, 5)
    """
    start_idx = max(0, idx - sequence_length)
    window = data.iloc[start_idx:idx + 1]

    # Pad if needed
    if len(window) < sequence_length:
        padding = np.zeros((sequence_length - len(window), 5))
        window_vals = np.vstack([padding, window[['open', 'high', 'low', 'close', 'volume']].values])
    else:
        window_vals = window[['open', 'high', 'low', 'close', 'volume']].values

    # Normalize (min-max per feature)
    if normalize:
        for i in range(5):
            col = window_vals[:, i]
            col_min = col.min()
            col_max = col.max()
            if col_max > col_min:
                window_vals[:, i] = (col - col_min) / (col_max - col_min)

    # Convert to tensor (1, seq_len, 5)
    state = torch.FloatTensor(window_vals).unsqueeze(0)
    return state


def split_data_chronologically(data, train_ratio, val_ratio):
    """Split data chronologically into train/val/test.

    Args:
        data: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end].reset_index(drop=True)
    val_data = data.iloc[train_end:val_end].reset_index(drop=True)
    test_data = data.iloc[val_end:].reset_index(drop=True)

    logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    return train_data, val_data, test_data


def train_episode(agent, data, episode_num, total_episodes, config):
    """Train agent for one episode.

    Args:
        agent: Agent instance
        data: Training data
        episode_num: Current episode number
        total_episodes: Total episodes
        config: Configuration

    Returns:
        Tuple of (total_profit, avg_loss, num_trades)
    """
    agent.env.reset()
    total_profit = 0.0
    avg_loss = 0.0
    num_trades = 0
    losses = []

    data_len = len(data)

    with tqdm(total=data_len - config.data.sequence_length, desc=f"Episode {episode_num}/{total_episodes} [TRAIN]", leave=False) as pbar:
        for t in range(config.data.sequence_length, data_len):
            # Get state
            state = get_state(data, t, config.data.sequence_length)
            next_state = get_state(data, t + 1, config.data.sequence_length) if t + 1 < data_len else state

            # Select action
            action = agent.act(state, is_eval=False)

            # Execute action
            price = data.iloc[t]['close']
            reward = agent.env.step(action, price)

            # Track results
            if action in [BUY, SELL]:
                num_trades += 1
            if action == SELL:
                total_profit += reward

            done = (t == data_len - 1)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train if buffer has enough samples
            if len(agent.memory) >= config.rl.batch_size:
                loss = agent.train_experience_replay(config.rl.batch_size)
                if loss > 0:
                    losses.append(loss)

            pbar.update(1)

    if losses:
        avg_loss = np.mean(losses)

    logger.info(f"Episode {episode_num}/{total_episodes} - Profit: ${total_profit:.2f}, Trades: {num_trades}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")

    return total_profit, avg_loss, num_trades


def eval_episode(agent, data, config, show_progress=False):
    """Evaluate agent (greedy, no exploration).

    Args:
        agent: Agent instance
        data: Evaluation data
        config: Configuration
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (total_profit, num_trades, trade_history)
    """
    agent.env.reset()
    total_profit = 0.0
    num_trades = 0
    trade_history = []

    data_len = len(data)
    iterator = range(config.data.sequence_length, data_len)

    if show_progress:
        iterator = tqdm(iterator, desc="[VAL]", leave=False)

    for t in iterator:
        # Get state
        state = get_state(data, t, config.data.sequence_length)

        # Select action (greedy)
        action = agent.act(state, is_eval=True)

        # Execute action
        price = data.iloc[t]['close']
        reward = agent.env.step(action, price)

        # Track results
        if action == BUY:
            trade_history.append(('BUY', price))
            num_trades += 1
        elif action == SELL:
            trade_history.append(('SELL', price, reward))
            total_profit += reward
            num_trades += 1

        done = (t == data_len - 1)

    return total_profit, num_trades, trade_history


def main():
    """Main training pipeline."""
    config = default_config
    db_path = Path(__file__).parent / "ohlc_data.db"

    logger.info("=" * 80)
    logger.info("MAMBA+DQN RL TRADING AGENT - TRAINING")
    logger.info("=" * 80)

    # 1. Load data
    logger.info("\n1. Loading OHLCV data from SQLite...")
    with OHLCDataLoader(db_path, config.data) as loader:
        data = loader.get_ohlcv_data(config.data.symbol)

    # Limit data for quick iteration if max_samples is set
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

    # 3. Create agent
    logger.info("\n3. Creating agent...")
    agent = Agent(config)
    model_dir = Path(__file__).parent / "models"

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
        train_profit, train_loss, train_trades = train_episode(
            agent, train_data, episode, config.rl.episodes, config
        )
        training_history['train_profit'].append(train_profit)
        training_history['train_loss'].append(train_loss)
        training_history['train_trades'].append(train_trades)

        # Validate
        val_profit, val_trades, _ = eval_episode(agent, val_data, config, show_progress=False)
        training_history['val_profit'].append(val_profit)
        training_history['val_trades'].append(val_trades)

        # Decay epsilon after episode
        agent.decay_epsilon()

        tqdm.write(f"Episode {episode}/{config.rl.episodes} - Train: ${train_profit:.2f} | Val: ${val_profit:.2f} | Trades: {val_trades} | ε: {agent.epsilon:.3f}")

        # Save checkpoint
        if episode % config.rl.save_every == 0:
            agent.save(episode, model_dir)

    # Save final model
    agent.save(config.rl.episodes, model_dir)

    # 5. Evaluate on test set
    logger.info("\n5. Evaluating on test set...")
    test_profit, test_trades, test_history = eval_episode(agent, test_data, config)

    logger.info(f"Test Set Results:")
    logger.info(f"  Total Profit: ${test_profit:.2f}")
    logger.info(f"  Total Trades: {test_trades}")
    logger.info(f"  Trade History:")
    for trade in test_history[:10]:  # Show first 10 trades
        if len(trade) == 2:
            action, price = trade
            logger.info(f"    {action} @ ${price:.2f}")
        else:
            action, price, pnl = trade
            logger.info(f"    {action} @ ${price:.2f} (P&L: ${pnl:.2f})")

    # Save training history
    logger.info("\n6. Saving results...")
    history_path = model_dir / "training_history.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history: {history_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
