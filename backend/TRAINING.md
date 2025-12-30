# RL Training Guide

## Running Training in Docker

The RL training can be run in Docker using the `train-rl` service.

### Prerequisites

1. **Dataset**: Ensure `dataset.parquet` exists in the `backend/` directory

   - Generate it by running: `python -m app.scripts.build_dataset` (or via Docker)

2. **MLflow**: The MLflow service must be running
   - It's automatically started with `docker-compose up`

### Running Training

#### Option 1: Using Docker Compose (Recommended)

```bash
# Start MLflow and database services
docker-compose up -d mlflow timescaledb

# Wait for services to be ready (about 10-15 seconds)
sleep 15

# Run training
docker-compose run --rm train-rl
```

#### Option 2: Run in Existing Container

```bash
# Start all services
docker-compose up -d

# Execute training in backend container
docker-compose exec backend python -m app.ml.train_rl
```

#### Option 3: Build and Run Manually

```bash
# Build the backend image
docker-compose build backend

# Run training
docker run --rm \
  --network botrader_botrader-net \
  -v $(pwd)/backend:/app \
  -v $(pwd)/mlruns:/mlruns \
  --env-file .env \
  botrader-backend \
  python -m app.ml.train_rl
```

### Viewing Results

Training results are logged to MLflow. View them at:

- **MLflow UI**: http://localhost:5001
- **Experiment**: "Deep_RL_Trading"
- **Models**: Saved to `backend/app/ml/models/ppo_agent_budget_*.pth`

### Training Configuration

The training script trains models for multiple budget scenarios:

- $250
- $1,000

Each model is trained for 1000 episodes with validation every 100 episodes.

### Troubleshooting

1. **"Dataset not found"**: Run `build_dataset.py` first to generate the dataset
2. **"Connection refused" to MLflow**: Ensure MLflow service is running (`docker-compose up -d mlflow`)
3. **CUDA errors**: Training will fall back to CPU if GPU is not available
4. **Out of memory**: Reduce batch size or number of episodes in `train_rl.py`

### Local Development (Outside Docker)

If running locally (not in Docker), update the MLflow URI in `train_rl.py`:

```python
mlflow.set_tracking_uri("http://localhost:5001")  # Use localhost instead of mlflow
```

---

## Metrics Guide

### Training Metrics (on training data)

#### `train_mean_return`

- **What**: Average total return per training episode over the last 100 episodes
- **Calculation**: `mean((final_portfolio_value - initial_balance) / initial_balance)`
- **Interpretation**:
  - Positive = profitable on average
  - Example: `0.05` = +5% average return per episode
- **Warning**: Can be optimistic due to overfitting; compare with `val_mean_return`

#### `train_sharpe`

- **What**: Risk-adjusted return (annualized)
- **Calculation**: `(mean_return / std_return) * sqrt(252 * 24 * 60)`
- **Interpretation**:
  - < 1.0 = poor risk-adjusted returns
  - 1.0-2.0 = decent
  - > 2.0 = strong
- **Use**: Measures return per unit of risk (volatility)

#### `train_sortino`

- **What**: Risk-adjusted return that only penalizes downside volatility
- **Calculation**: `mean_return / downside_std * sqrt(252 * 24 * 60)`
- **Interpretation**:
  - Higher is better
  - Usually higher than Sharpe (less penalized)
  - > 1.5 = good downside protection
- **Use**: Better than Sharpe when you care more about downside risk

#### `train_profit_factor`

- **What**: Ratio of total wins to total losses (accounts for magnitude)
- **Calculation**: `sum(positive_returns) / abs(sum(negative_returns))`
- **Interpretation**:
  - < 1.0 = losses exceed wins
  - 1.0 = break-even
  - 1.5-2.0 = good
  - > 2.0 = strong
- **Use**: Addresses win_rate limitation; shows if big wins offset losses

---

### Validation Metrics (on held-out data)

#### `val_mean_return`

- **What**: Average total return per validation episode (10 validation windows)
- **Calculation**: Same as training, but on unseen data
- **Interpretation**:
  - More reliable than training metrics
  - Positive = profitable on validation
  - Compare with `train_mean_return` to detect overfitting
- **Good sign**: `val_mean_return` close to `train_mean_return` (within 20%)

#### `val_sharpe` ⭐ (Primary Selection Metric)

- **What**: Risk-adjusted return on validation data
- **Calculation**: Same as training Sharpe, but on validation episodes
- **Interpretation**:
  - Used to select the best model (higher = better)
  - < 1.0 = poor
  - 1.0-2.0 = decent
  - > 2.0 = strong
- **Use**: Primary metric for model selection

#### `val_sortino`

- **What**: Risk-adjusted return penalizing only downside volatility
- **Calculation**: Same as training Sortino, but on validation data
- **Interpretation**:
  - Higher is better
  - Usually higher than Sharpe
  - > 1.5 = good downside protection
- **Use**: Better risk metric when focusing on downside

#### `val_max_drawdown`

- **What**: Largest peak-to-trough decline during validation episodes
- **Calculation**: `min((cumulative - running_max) / running_max)`
- **Interpretation**:
  - Negative value (e.g., `-0.15` = -15% max drawdown)
  - Closer to 0 is better
  - < -0.20 = high risk
- **Use**: Measures worst-case loss from a peak

#### `val_win_rate`

- **What**: Percentage of profitable validation episodes
- **Calculation**: `(episodes with return > 0) / total_episodes`
- **Interpretation**:
  - Range: 0.0 to 1.0 (0% to 100%)
  - > 0.5 = more wins than losses
  - > 0.6 = strong consistency
- **Limitation**: Doesn't account for magnitude (use `val_profit_factor`)

#### `val_profit_factor`

- **What**: Ratio of total wins to total losses (accounts for magnitude)
- **Calculation**: `sum(positive_returns) / abs(sum(negative_returns))`
- **Interpretation**:
  - < 1.0 = losses exceed wins
  - 1.0 = break-even
  - 1.5-2.0 = good
  - > 2.0 = strong
- **Use**: Complements `val_win_rate`; shows if big wins offset losses

#### `val_calmar_ratio`

- **What**: Annualized return divided by maximum drawdown
- **Calculation**: `annualized_return / abs(max_drawdown)`
- **Interpretation**:
  - Higher is better
  - > 1.0 = return exceeds max drawdown
  - > 3.0 = strong risk-adjusted performance
- **Use**: Measures return relative to worst-case loss

---

### Overfitting Detection Metrics

#### `overfitting_gap`

- **What**: Difference between training and validation returns
- **Calculation**: `train_mean_return - val_mean_return`
- **Interpretation**:
  - Close to 0 = good generalization
  - > 0.05 = potential overfitting
  - > 0.10 = significant overfitting
- **Use**: Primary overfitting indicator

#### `train_val_ratio`

- **What**: Ratio of training to validation returns
- **Calculation**: `train_mean_return / val_mean_return`
- **Interpretation**:
  - Close to 1.0 = good generalization
  - 1.0-1.2 = acceptable
  - > 1.5 = overfitting
- **Use**: Alternative overfitting indicator

---

## Quick Interpretation Guide

### ✅ Good Model Signs

- `val_sharpe` > 1.0
- `val_profit_factor` > 1.5
- `overfitting_gap` < 0.05
- `val_mean_return` > 0
- `val_max_drawdown` > -0.20

### ⚠️ Warning Signs

- `overfitting_gap` > 0.10 (overfitting)
- `val_sharpe` < 0.5 (poor risk-adjusted returns)
- `val_profit_factor` < 1.0 (losses exceed wins)
- `val_max_drawdown` < -0.30 (high risk)
- `train_val_ratio` > 1.5 (overfitting)

### 📊 Model Selection Priority

1. `val_sharpe` (primary)
2. `val_profit_factor` (magnitude-aware)
3. `overfitting_gap` (generalization)
4. `val_calmar_ratio` (risk-adjusted)
5. `val_sortino` (downside-focused)

**Note**: The model is saved when `val_sharpe` improves, prioritizing risk-adjusted performance over raw returns.
