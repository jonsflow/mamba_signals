#!/usr/bin/env python3
"""Evaluation script for RL trading agent."""
import sys
import logging
from pathlib import Path
import argparse
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import default_config
from data_loader import OHLCDataLoader
from training import Agent, BUY, SELL
from train import get_state, split_data_chronologically

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest_agent(agent, data, config, symbol="SPY"):
    """Run backtest on data and calculate metrics.

    Args:
        agent: Agent instance (pretrained)
        data: Test data
        config: Configuration
        symbol: Stock symbol (for logging)

    Returns:
        Dictionary with backtest results
    """
    agent.env.reset()

    trades = []
    portfolio_values = []
    initial_cash = agent.env.initial_cash
    data_len = len(data)

    for t in range(config.data.sequence_length, data_len):
        # Get state
        state = get_state(data, t, config.data.sequence_length)

        # Select action (greedy)
        action = agent.act(state, is_eval=True)

        # Execute action
        price = data.iloc[t]['close']
        reward = agent.env.step(action, price)

        # Track trades
        if action == BUY:
            trades.append({
                'action': 'BUY',
                'timestamp': t,
                'price': float(price),
                'pnl': 0.0
            })
        elif action == SELL and len(trades) > 0:
            # Find last unmatched BUY
            for i in range(len(trades) - 1, -1, -1):
                if trades[i]['action'] == 'BUY' and trades[i]['pnl'] == 0.0:
                    trades[i]['matched_sell_price'] = float(price)
                    trades[i]['pnl'] = reward
                    trades.append({
                        'action': 'SELL',
                        'timestamp': t,
                        'price': float(price),
                        'pnl': reward
                    })
                    break

        # Track portfolio value
        portfolio_value = agent.env.get_portfolio_value(price)
        portfolio_values.append(portfolio_value)

    # Calculate metrics
    total_profit = agent.env.total_profit
    final_value = agent.env.get_portfolio_value(data.iloc[-1]['close'])

    # Win rate
    closed_trades = [t for t in trades if 'pnl' in t and t['action'] == 'SELL']
    winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
    win_rate = winning_trades / len(closed_trades) if len(closed_trades) > 0 else 0.0

    # Max drawdown
    portfolio_values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # Return metrics
    results = {
        'symbol': symbol,
        'data_points': data_len,
        'num_trades': len(closed_trades),
        'winning_trades': winning_trades,
        'win_rate': float(win_rate),
        'total_profit': float(total_profit),
        'final_portfolio_value': float(final_value),
        'return_pct': float((final_value - initial_cash) / initial_cash * 100),
        'max_drawdown': float(max_drawdown),
        'trades': trades
    }

    return results


def print_backtest_results(results):
    """Print backtest results nicely.

    Args:
        results: Dictionary from backtest_agent
    """
    print("\n" + "=" * 80)
    print(f"BACKTEST RESULTS - {results['symbol']}")
    print("=" * 80)
    print(f"Data points analyzed: {results['data_points']}")
    print(f"Total trades: {results['num_trades']}")
    print(f"Winning trades: {results['winning_trades']}")
    print(f"Win rate: {results['win_rate']:.1%}")
    print(f"Total profit: ${results['total_profit']:.2f}")
    print(f"Final portfolio value: ${results['final_portfolio_value']:.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Max drawdown: {results['max_drawdown']:.2f}%")
    print("=" * 80)

    # Show trades
    print(f"\nTrades (showing first 20):")
    for i, trade in enumerate(results['trades'][:20]):
        if trade['action'] == 'BUY':
            print(f"  {i+1}. BUY @ ${trade['price']:.2f}")
        elif trade['action'] == 'SELL':
            pnl = trade['pnl']
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            print(f"  {i+1}. SELL @ ${trade['price']:.2f} (P&L: {pnl_str})")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Backtest RL trading agent')
    parser.add_argument('--model-path', type=str, default='models/agent_episode_5.pt',
                        help='Path to saved agent checkpoint')
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Stock symbol')
    parser.add_argument('--data-split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which data split to backtest on')
    parser.add_argument('--db-path', type=str, default='ohlc_data.db',
                        help='Path to SQLite database')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RL TRADING AGENT - BACKTEST EVALUATION")
    logger.info("=" * 80)

    # Load config
    config = default_config
    db_path = Path(args.db_path)
    model_path = Path(args.model_path)

    # Load data
    logger.info(f"\nLoading data from {db_path}...")
    with OHLCDataLoader(db_path, config.data) as loader:
        data = loader.get_ohlcv_data(args.symbol)

    logger.info(f"Loaded {len(data)} candles")

    # Split data
    logger.info(f"\nSplitting data...")
    train_data, val_data, test_data = split_data_chronologically(
        data,
        config.data.train_ratio,
        config.data.val_ratio
    )

    # Choose split
    if args.data_split == 'train':
        backtest_data = train_data
        split_name = 'TRAIN'
    elif args.data_split == 'val':
        backtest_data = val_data
        split_name = 'VALIDATION'
    else:
        backtest_data = test_data
        split_name = 'TEST'

    # Load agent
    logger.info(f"\nLoading agent from {model_path}...")
    agent = Agent(config)
    if model_path.exists():
        agent.load(model_path)
        logger.info(f"Agent loaded successfully")
    else:
        logger.error(f"Model not found at {model_path}")
        return

    # Run backtest
    logger.info(f"\nRunning backtest on {split_name} data...")
    results = backtest_agent(agent, backtest_data, config, symbol=args.symbol)

    # Print results
    print_backtest_results(results)

    # Save results
    import json
    results_path = Path('models') / f'backtest_results_{args.data_split}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
