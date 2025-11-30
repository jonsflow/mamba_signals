"""Reward configuration presets for different trading strategies."""
from dataclasses import dataclass
from typing import Literal


@dataclass
class RewardConfig:
    """Configuration for reward signals in trading environment."""

    # BUY action reward
    buy_reward: float

    # SELL action reward components
    sell_pnl_multiplier: float  # multiplier on delta (price change)
    sell_bars_multiplier: float  # multiplier on bars_profitable
    sell_transaction_cost: float  # penalty for selling

    # HOLD action reward
    hold_reward: float  # reward per HOLD action (0 = no reward)

    # Direction change penalty (discourage flipping buy/sell too quickly)
    min_bars_before_direction_change: int = 3  # bars needed before flipping buy→sell or sell→buy
    direction_change_penalty: float = 0.1  # penalty for flipping direction too soon

    # Max trades constraint
    max_trades_per_episode: int = 0  # 0 = unlimited, N = max N trades (buy+sell counts as 2)
    max_trades_penalty: float = 0.0  # penalty for exceeding max trades

    # Trade density penalty (penalize accumulating too many trades)
    trade_density_penalty_multiplier: float = 0.0  # 0 = disabled, >0 = penalty increases with trade count

    # Bars-based reward mechanism (alternative to PNL-based)
    use_bars_only: bool = False  # If True, ignore price delta and only use bars above/below entry
    reward_bars_above_entry: float = 0.0  # reward per bar price is above entry (for profitable holds)
    penalty_bars_below_entry: float = 0.0  # penalty per bar price is below entry (for losing holds)

    name: str = "custom"


# Preset reward configurations
REWARD_PRESETS = {
    # Simple delta-only reward (baseline)
    "simple_delta": RewardConfig(
        name="simple_delta",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,  # Full delta reward on sell only
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=0.0,
    ),

    # Delta with transaction costs
    "with_costs": RewardConfig(
        name="with_costs",
        buy_reward=-0.01,  # Small penalty for buying
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.0,
        sell_transaction_cost=-0.01,  # Small penalty for selling
        hold_reward=0.0,
    ),

    # Conservative (discourage frequent trading)
    "conservative": RewardConfig(
        name="conservative",
        buy_reward=-0.25,
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.05,  # Small bonus for patience
        sell_transaction_cost=0.10,
        hold_reward=0.0,
    ),

    # Continuous feedback on open position
    "continuous": RewardConfig(
        name="continuous",
        buy_reward=0.0,
        sell_pnl_multiplier=0.1,
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=1.0,  # Reward/penalty while holding based on position value
    ),

    # Bars-based (focus on time above/below entry, not PNL)
    "bars_only": RewardConfig(
        name="bars_only",
        buy_reward=0.0,
        sell_pnl_multiplier=0.0,  # Ignore price delta
        sell_bars_multiplier=0.1,
        sell_transaction_cost=0.0,
        hold_reward=0.0,
        use_bars_only=True,
        reward_bars_above_entry=0.05,
        penalty_bars_below_entry=-0.10,
    ),

    # Direction change constraints
    "min_5bars": RewardConfig(
        name="min_5bars",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=0.0,
        min_bars_before_direction_change=5,
        direction_change_penalty=-0.5,
    ),

    # Trade count limits
    "max_100_trades": RewardConfig(
        name="max_100_trades",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=0.0,
        min_bars_before_direction_change=5,
        direction_change_penalty=-0.5,
        max_trades_per_episode=100,
        max_trades_penalty=-1.0,
    ),

    # Density penalty (penalize high trade frequency)
    "sparse": RewardConfig(
        name="sparse",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=0.0,
        trade_density_penalty_multiplier=0.01,
    ),
}


def get_reward_config(preset_name: str) -> RewardConfig:
    """Get a reward configuration by preset name.

    Args:
        preset_name: Name of preset (simple_pnl, trading_agent, bars_primary,
                    continuous_feedback, conservative_trading)

    Returns:
        RewardConfig instance

    Raises:
        ValueError if preset not found
    """
    if preset_name not in REWARD_PRESETS:
        available = ", ".join(REWARD_PRESETS.keys())
        raise ValueError(f"Unknown reward preset: {preset_name}. Available: {available}")

    return REWARD_PRESETS[preset_name]
