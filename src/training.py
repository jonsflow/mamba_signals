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
        self.inventory = []  # List of (entry_price, bars_held_profitable)
        self.total_profit = 0.0

    def reset(self):
        """Reset environment to initial state."""
        self.cash = self.initial_cash
        self.shares_held = 0
        self.inventory = []
        self.total_profit = 0.0

    def step(self, action: int, price: float) -> float:
        """Execute action and return intelligent reward.

        Reward system:
        - BUY: reward = 0 (no immediate feedback)
        - SELL: reward = realized P&L (price_delta)
        - HOLD: reward = unrealized P&L feedback (if in position)
          - Positive if position is winning (+0.1 per bar above entry)
          - Negative if position is losing (-0.1 per bar below entry)

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            price: Current price

        Returns:
            Reward (intelligent based on position state)
        """
        reward = 0.0

        if action == BUY:
            # Buy - only if no position open (1 position max)
            if len(self.inventory) == 0:
                self.inventory.append((price, 0))  # (entry_price, bars_held_profitably)
                self.shares_held = 1  # 1 share per position
                reward = self.reward_config.buy_reward
            else:
                # Already in position, can't buy
                reward = 0.0

        elif action == SELL and len(self.inventory) > 0:
            # Sell - close position with configurable reward
            entry_price, bars_profitable = self.inventory.pop(0)  # FIFO
            delta = price - entry_price
            # Reward = (bars held * multiplier) + (delta * multiplier) - transaction cost
            reward = (bars_profitable * self.reward_config.sell_bars_multiplier +
                      delta * self.reward_config.sell_pnl_multiplier -
                      self.reward_config.sell_transaction_cost)
            self.total_profit += delta
            self.shares_held = 0  # No position now

        elif action == HOLD:
            # Hold - track bars held profitably for bonus calculation on SELL
            if len(self.inventory) > 0:
                # In a position - track profitable bars
                entry_price, bars_profitable = self.inventory[0]
                unrealized_pnl = price - entry_price
                # Track bars held profitably
                if unrealized_pnl > 0:
                    self.inventory[0] = (entry_price, bars_profitable + 1)
                # Reward based on config
                reward = self.reward_config.hold_reward * unrealized_pnl if self.reward_config.hold_reward > 0 else 0.0
            else:
                # Not in position - no reward for holding cash
                reward = 0.0

        else:
            # Invalid action (SELL when no inventory)
            pass

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
