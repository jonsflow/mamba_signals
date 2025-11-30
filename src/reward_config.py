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

    name: str = "custom"


# Preset reward configurations
REWARD_PRESETS = {
    "simple_pnl": RewardConfig(
        name="simple_pnl",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,  # Full delta reward
        sell_bars_multiplier=0.0,  # No bonus for bars held
        sell_transaction_cost=0.0,  # No transaction cost
        hold_reward=0.0,
    ),

    "trading_agent": RewardConfig(
        name="trading_agent",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,  # Full delta reward (like Trading Agent repo)
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=0.0,
    ),

    "bars_primary": RewardConfig(
        name="bars_primary",
        buy_reward=-0.50,  # Heavy penalty for buying
        sell_pnl_multiplier=0.1,  # Small bonus for profit magnitude
        sell_bars_multiplier=1.0,  # Primary reward: bars held profitably
        sell_transaction_cost=0.25,  # Transaction cost penalty
        hold_reward=0.0,  # No HOLD reward
    ),

    "continuous_feedback": RewardConfig(
        name="continuous_feedback",
        buy_reward=0.0,
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.0,
        sell_transaction_cost=0.0,
        hold_reward=0.1,  # Continuous feedback on position quality
    ),

    "conservative_trading": RewardConfig(
        name="conservative_trading",
        buy_reward=-0.25,  # Moderate penalty for buying
        sell_pnl_multiplier=1.0,
        sell_bars_multiplier=0.05,  # Small bonus for patience
        sell_transaction_cost=0.10,  # Moderate transaction cost
        hold_reward=0.0,
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
