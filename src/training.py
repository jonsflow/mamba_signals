"""RL training components: Environment, Agent, and training loop."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
import json
import logging

from config import Config
from mamba_model import DQNAgent
from reward_config import RewardConfig, get_reward_config

logger = logging.getLogger(__name__)

# Action constants
HOLD = 0
BUY = 1
SELL = 2


class TradingEnvironment:
    """Trading environment with configurable reward system."""

    def __init__(self, initial_cash: float = 10000.0, share_size: int = 1,
                 reward_config: RewardConfig = None):
        """Initialize environment.

        Args:
            initial_cash: Starting cash
            share_size: Number of shares per action
            reward_config: RewardConfig instance for reward signal design
        """
        self.initial_cash = initial_cash
        self.share_size = share_size
        self.reward_config = reward_config or get_reward_config("bars_primary")

        # State tracking
        self.cash = initial_cash
        self.shares_held = 0
        self.inventory = []  # List of (entry_close, entry_open, bars_held_profitable, bars_high_above, bars_low_below, bars_close_above)
        self.total_profit = 0.0
        self.last_action = HOLD  # Track last action for direction change penalty
        self.bars_since_last_action = 0  # Track bars since last buy/sell
        self.total_trades = 0  # Track total trades this episode

    def reset(self):
        """Reset environment to initial state."""
        self.cash = self.initial_cash
        self.shares_held = 0
        self.inventory = []
        self.total_profit = 0.0
        self.last_action = HOLD
        self.bars_since_last_action = 0
        self.total_trades = 0

    def step(self, action: int, price: float, price_open: float = None, price_high: float = None, price_low: float = None) -> float:
        """Execute action and return intelligent reward.

        Reward system:
        - BUY: reward = 0 (no immediate feedback)
        - SELL: reward = realized P&L (price_delta)
        - HOLD: reward = unrealized P&L feedback (if in position)

        Direction change penalty:
        - If min_bars_before_direction_change is set, penalize flipping buy→sell or sell→buy too quickly

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            price: Current close price
            price_open: Current open price (optional)
            price_high: Current high price (optional)
            price_low: Current low price (optional)

        Returns:
            Reward (intelligent based on position state)
        """
        # Use close as default for all prices if not provided
        if price_open is None:
            price_open = price
        if price_high is None:
            price_high = price
        if price_low is None:
            price_low = price
        reward = 0.0

        # Check for direction change penalty (BUY→SELL or SELL→BUY)
        direction_change_penalty = 0.0
        if self.reward_config.min_bars_before_direction_change > 0:
            # Check if this is a direction change
            is_direction_change = False
            if action == SELL and self.last_action == BUY:
                is_direction_change = True
            elif action == BUY and self.last_action == SELL:
                is_direction_change = True

            # Apply penalty if direction change too soon
            if is_direction_change and self.bars_since_last_action < self.reward_config.min_bars_before_direction_change:
                direction_change_penalty = self.reward_config.direction_change_penalty

        # Check for max trades penalty
        max_trades_penalty = 0.0
        if self.reward_config.max_trades_per_episode > 0 and self.total_trades >= self.reward_config.max_trades_per_episode:
            # Apply penalty if already at/over max trades
            if action == BUY or action == SELL:
                max_trades_penalty = self.reward_config.max_trades_penalty

        # Check for trade density penalty (penalty increases with number of trades)
        trade_density_penalty = 0.0
        if self.reward_config.trade_density_penalty_multiplier > 0:
            if action == BUY or action == SELL:
                trade_density_penalty = -self.reward_config.trade_density_penalty_multiplier * self.total_trades

        if action == BUY:
            # Buy - only if no position open (1 position max)
            if len(self.inventory) == 0:
                # Track entry with all price info: (entry_close, entry_open, bars_profitable, bars_high_above, bars_low_below, bars_close_above)
                self.inventory.append((price, price_open, 0, 0, 0, 0))
                self.shares_held = 1  # 1 share per position
                reward = self.reward_config.buy_reward + direction_change_penalty + max_trades_penalty + trade_density_penalty
                self.last_action = BUY
                self.bars_since_last_action = 0
                self.total_trades += 1  # Count this trade
            else:
                # Already in position, can't buy
                reward = 0.0
                self.bars_since_last_action += 1

        elif action == SELL and len(self.inventory) > 0:
            # Sell - close position with configurable reward
            entry_close, entry_open, bars_profitable, bars_high_above, bars_low_below, bars_close_above = self.inventory.pop(0)  # FIFO
            delta = price - entry_close

            # Calculate reward based on mechanism (bars-only or PNL-based)
            if self.reward_config.use_bars_only:
                # Reward uses all price action data: close above, high touches, low touches
                # bars_close_above = bars where close ended above entry (profitable closes)
                # bars_high_above = bars where high touched above entry (bullish pressure)
                # bars_low_below = bars where low touched below entry (bearish pressure)
                reward = (bars_close_above * self.reward_config.reward_bars_above_entry +
                         bars_low_below * self.reward_config.penalty_bars_below_entry +
                         bars_profitable * self.reward_config.sell_bars_multiplier -
                         self.reward_config.sell_transaction_cost + direction_change_penalty + max_trades_penalty + trade_density_penalty)
            else:
                # Traditional PNL-based: (bars held * multiplier) + (delta * multiplier) - transaction cost
                reward = (bars_profitable * self.reward_config.sell_bars_multiplier +
                         delta * self.reward_config.sell_pnl_multiplier -
                         self.reward_config.sell_transaction_cost + direction_change_penalty + max_trades_penalty + trade_density_penalty)

            self.total_profit += delta
            self.shares_held = 0  # No position now
            self.last_action = SELL
            self.bars_since_last_action = 0
            self.total_trades += 1  # Count this trade

        elif action == HOLD:
            # Hold - track all price action: close above/below, high touches, low touches
            if len(self.inventory) > 0:
                # In a position - track all price metrics
                entry_close, entry_open, bars_profitable, bars_high_above, bars_low_below, bars_close_above = self.inventory[0]

                # Track if close ended above entry (profitable closes)
                if price > entry_close:
                    bars_profitable += 1
                    bars_close_above += 1

                # Track if high touched above entry (bullish pressure)
                if price_high > entry_close:
                    bars_high_above += 1

                # Track if low touched below entry (bearish pressure)
                if price_low < entry_close:
                    bars_low_below += 1

                self.inventory[0] = (entry_close, entry_open, bars_profitable, bars_high_above, bars_low_below, bars_close_above)

                # Reward based on config (unchanged from original)
                unrealized_pnl = price - entry_close
                reward = self.reward_config.hold_reward * unrealized_pnl if self.reward_config.hold_reward > 0 else 0.0
            else:
                # Not in position - no reward for holding cash
                reward = 0.0

            self.bars_since_last_action += 1

        else:
            # Invalid action (SELL when no inventory) - just ignore it
            reward = 0.0
            self.bars_since_last_action += 1

        return reward

    def get_portfolio_value(self, current_price: float) -> float:
        """Get total portfolio value (cash + shares @ current price)."""
        position_value = self.shares_held * current_price
        return self.cash + position_value


class Agent:
    """RL Agent using DQN with experience replay."""

    def __init__(self, config: Config, device: torch.device = None):
        """Initialize agent.

        Args:
            config: Configuration object
            device: torch device (mps, cuda, cpu)
        """
        self.config = config

        # Device
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        logger.info(f"Agent device: {self.device}")

        # Networks
        self.model = DQNAgent(config.model).to(self.device)
        self.target_model = DQNAgent(config.model).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target network doesn't need gradients

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.rl.learning_rate
        )
        # Use Huber loss (more robust than MSE for outliers)
        self.criterion = nn.HuberLoss(delta=1.0)

        # Experience replay
        self.memory = deque(maxlen=config.rl.replay_buffer_size)
        self.steps = 0

        # Exploration
        self.epsilon = config.rl.epsilon
        self.epsilon_min = config.rl.epsilon_min
        self.epsilon_decay = config.rl.epsilon_decay

        # Environment with reward config
        reward_config = get_reward_config(config.rl.reward_preset)
        self.env = TradingEnvironment(
            initial_cash=config.rl.initial_cash,
            share_size=config.rl.share_size,
            reward_config=reward_config
        )

    def act(self, state: torch.Tensor, is_eval: bool = False) -> int:
        """Select action using ε-greedy policy.

        Args:
            state: State tensor of shape (1, seq_len, input_dim)
            is_eval: If True, use greedy policy (no exploration)

        Returns:
            Action (0=HOLD, 1=BUY, 2=SELL)
        """
        if is_eval or np.random.random() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                state_tensor = state.to(self.device)
                q_values = self.model(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action
        else:
            # Random action
            return np.random.randint(0, 3)

    def remember(self, state: torch.Tensor, action: int, reward: float,
                 next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self, batch_size: int) -> float:
        """Train on batch from experience replay.

        Args:
            batch_size: Number of samples to train on

        Returns:
            Loss value
        """
        if len(self.memory) < batch_size:
            return 0.0

        # Sample batch
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
        states = torch.cat(states, dim=0).to(self.device)  # (batch, seq_len, input_dim)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # Current Q-values
        q_values = self.model(states)  # (batch, 3)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1)[0]  # (batch,)
            target_q_values = rewards + (1 - dones) * self.config.rl.gamma * next_q_values

        # Loss and backprop
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update steps (don't decay epsilon here - do it per episode instead)
        self.steps += 1

        # Update target network periodically
        if self.steps % self.config.rl.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, episode: int, model_dir: Path = None):
        """Save model checkpoint.

        Args:
            episode: Episode number
            model_dir: Directory to save to
        """
        if model_dir is None:
            model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        checkpoint = {
            'episode': episode,
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': self.config.dict()
        }

        path = model_dir / f"agent_episode_{episode}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved agent checkpoint: {path}")

    def load(self, checkpoint_path: Path):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"Loaded agent checkpoint: {checkpoint_path}")
