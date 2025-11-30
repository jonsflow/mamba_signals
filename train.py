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

from data_loader import OHLCDataLoader
from training import Agent, HOLD, BUY, SELL
from config_loader import default_config

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
    """Split data using stratified random sampling within chronological chunks.

    Instead of chronological boundaries (which create distribution mismatch),
    this divides data into chunks and randomly assigns them to train/val/test
    while preserving temporal coherence within each split.

    This ensures each split sees the full range of market conditions.

    Args:
        data: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    num_chunks = 10  # Divide data into 10 chunks
    chunk_size = n // num_chunks

    # Create chunk indices
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else n
        chunks.append(data.iloc[start:end])

    # Shuffle chunks
    import random
    random.seed(42)
    random.shuffle(chunks)

    # Assign chunks to train/val/test based on ratios
    num_train_chunks = max(1, int(num_chunks * train_ratio))
    num_val_chunks = max(1, int(num_chunks * val_ratio))

    train_chunks = chunks[:num_train_chunks]
    val_chunks = chunks[num_train_chunks:num_train_chunks + num_val_chunks]
    test_chunks = chunks[num_train_chunks + num_val_chunks:]

    # Concatenate and sort by index to maintain temporal coherence
    train_data = pd.concat(train_chunks, ignore_index=True).sort_index().reset_index(drop=True)
    val_data = pd.concat(val_chunks, ignore_index=True).sort_index().reset_index(drop=True)
    test_data = pd.concat(test_chunks, ignore_index=True).sort_index().reset_index(drop=True)

    logger.info(f"Data split (stratified random): train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    logger.info(f"  Train chunks: {num_train_chunks}/{num_chunks}, Val chunks: {num_val_chunks}/{num_chunks}")

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
    avg_loss = 0.0
    num_buys = 0
    num_sells = 0
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

            # Track before step to detect successful trades
            inventory_before = len(agent.env.inventory)
            reward = agent.env.step(action, price)
            inventory_after = len(agent.env.inventory)

            # Count actual successful trades separately
            if action == BUY and inventory_after > inventory_before:
                num_buys += 1
            elif action == SELL and inventory_after < inventory_before:
                num_sells += 1

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

    # Use actual total_profit from environment (tracks realized delta)
    total_profit = agent.env.total_profit
    # A trade is a complete round trip (buy-sell pair), so count completed trades (sells)
    num_trades = num_sells

    logger.info(f"  [TRAIN DEBUG] Buys: {num_buys}, Sells: {num_sells}, Open position: {agent.env.shares_held} shares (should be {num_buys - num_sells}), Total profit: ${total_profit:.2f}")
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
    num_buys = 0
    num_sells = 0
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

        # Track before step to detect successful trades
        inventory_before = len(agent.env.inventory)
        reward = agent.env.step(action, price)
        inventory_after = len(agent.env.inventory)

        # Track results - only count successful trades
        if action == BUY and inventory_after > inventory_before:
            trade_history.append(('BUY', price))
            num_buys += 1
        elif action == SELL and inventory_after < inventory_before:
            trade_history.append(('SELL', price, reward))
            num_sells += 1

        done = (t == data_len - 1)

    # Use actual total_profit from environment (tracks realized delta)
    total_profit = agent.env.total_profit
    # A trade is a complete round trip (buy-sell pair), so count completed trades (sells)
    num_trades = num_sells

    return total_profit, num_trades, trade_history


def main():
    """Main training pipeline."""
    config = default_config
    db_path = Path(__file__).parent / "ohlc_data.db"

    logger.info("=" * 80)
    logger.info("MAMBA+DQN RL TRADING AGENT - TRAINING")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\n[CONFIGURATION]")
    logger.info(f"Data Config:")
    logger.info(f"  symbol: {config.data.symbol}")
    logger.info(f"  timeframe: {config.data.timeframe}")
    logger.info(f"  sequence_length: {config.data.sequence_length}")
    logger.info(f"  max_samples: {config.data.max_samples}")
    logger.info(f"Model Config:")
    logger.info(f"  hidden_dim: {config.model.hidden_dim}")
    logger.info(f"  num_layers: {config.model.num_layers}")
    logger.info(f"  state_dim: {config.model.state_dim}")
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
