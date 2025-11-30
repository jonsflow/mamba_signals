# RL Reward Design Problem & Supervised Learning Alternative

## The Core Problem

We're trying to teach an RL agent to trade by rewarding profitable trades, but **every reward structure we've tested has fundamental flaws**:

1. **Sparse Rewards (BUY=0, SELL=delta, HOLD=0)**
   - Agent only gets feedback when closing positions
   - Leads to excessive trading (2300+ trades per 3500 bars)
   - Agent tries random entries hoping for profitable exits
   - No feedback on position quality until exit

2. **Continuous Rewards (HOLD=0.1 × unrealized_pnl)**
   - Agent gets feedback every bar on position quality
   - Encourages patience and holding winners
   - But still causes overtrading - agent buys frequently to collect HOLD rewards
   - Overfits to training data (validation profit collapses)

3. **Hybrid with Transaction Costs (BUY=-0.05, SELL=delta+bonus, HOLD=0)**
   - Penalizes buying to reduce frequency
   - Rewards holding profitably for multiple bars
   - Still doesn't capture what we actually want the agent to learn

## What We Actually Want

The agent should learn to:

1. **Recognize reversals** → generate BUY/SELL signals
2. **Recognize trends** → HOLD and accumulate profits
3. **Confirm signals** → if price continues moving in signal direction for 5-10 bars, it was a good signal
4. **Exit on reversal** → if trend reverses, close position

This is **signal confirmation** - a pattern recognition problem, not an optimization problem.

## The Fundamental Issue with RL Approach

RL optimizes for **maximizing cumulative reward over time**. But our reward signals don't directly encode what constitutes a "good trade":

- A profitable trade could have been due to luck, not good decision-making
- An unprofitable trade could have been the right decision that just didn't work out
- The agent can't distinguish between "I made a good entry that didn't pan out" vs "I made a bad entry"

**RL needs ground truth about action quality, not outcome quality.**

## The Supervised Learning Alternative

Instead of RL, use supervised learning to directly predict: **Is this a reversal or trend continuation?**

### Training Data Generation
- Input: Last 64 candles (OHLCV) - same Mamba encoder
- Output: Binary classification
  - **BUY signal**: Position this as reversal to upside; check if price is higher 5-10 bars later
  - **SELL signal**: Position this as reversal to downside; check if price is lower 5-10 bars later
  - **HOLD**: Price continues in current direction; label as "trend continuation"
- Ground truth: Look forward 10 bars and label based on actual price movement

### Why This Works Better
1. **Clear objective**: Predict signal confirmation, not optimize reward
2. **Uses domain knowledge**: You already know what a good trade looks like
3. **Avoids reward engineering**: No need to tweak bonus multipliers or penalties
4. **Interpretable**: Can measure precision/recall on signal quality
5. **Generalizable**: Learned patterns (reversals vs trends) apply to different assets/timeframes

## Current RL Implementation

**Reward Structure:**
- BUY: -0.05 (penalty to discourage overtrading)
- SELL: delta + (0.05 × bars_held_profitably) - 0.05 (P&L + patience bonus - transaction cost)
- HOLD: 0 (no continuous feedback)

**Constraints:**
- Only 1 position open at a time
- 1 share per position
- Must SELL before opening new position

**Results:**
- Episode 1: -$91.99 train, +$141.48 val (random exploration)
- Episode 2: +$1290.70 train, -$0.11 val (overfitting)
- Trade frequency: ~2300 trades per 3500 bars (~66%, still excessive)

## Decision Point

**Keep exploring RL?**
- Could try: portfolio value change rewards, risk-adjusted (Sharpe-like) rewards, volatility scaling
- Risk: More reward engineering without addressing fundamental problem

**Pivot to Supervised Learning?**
- Cleaner problem formulation
- Better alignment with actual goal (signal confirmation)
- Faster iteration (no reward tuning)

## Notes

The RL approach isn't wrong, but it might not be the right tool for this specific problem. The agent needs to learn **what makes a signal good**, not **how to maximize a reward function**. These are different objectives.
