# Botrader: AI-Driven Algorithmic Trading System 🚀

Botrader is a locally deployed algorithmic trading platform for Bitcoin and crypto markets. It features live data collection, ML model training with MLflow experiment tracking, paper trading simulation, and a professional React dashboard for monitoring and control.

## 🌟 Current Features (Implemented & Working)

### ✅ Data Infrastructure

- **Live Data Collection**: Binance WebSocket streaming for real-time 1-minute candles
- **TimescaleDB**: Time-series database with 93,000+ historical candles
- **Historical Backfill**: Script to fetch data from any start date to present
- **Technical Indicators**: Auto-computed SMA, EMA, RSI, MACD, Bollinger Bands, ATR

### ✅ Machine Learning

- **Deep Reinforcement Learning**: PPO (Proximal Policy Optimization) agent with continuous action space
- **MLflow Integration**: Experiment tracking and model versioning
- **RL Training**: Simulates realistic trading with overlapping time windows, budget constraints, and fee handling
- **Dataset Builder**: Automated feature engineering from raw OHLCV data
- **Training Script**: `train_rl.py` - Trains PPO agents for multiple budget scenarios ($1K, $10K, $50K, $100K)

### ✅ Paper Trading

- **Live Simulation**: Paper trader monitors database for new candles
- **Current Strategy**: RSI-based (Buy RSI<30, Sell RSI>70)
- **Risk Management**: Position sizing, trade risk validation, circuit breakers
- **Smart Execution**: TWAP/VWAP algorithms for order execution

### ✅ Professional Dashboard

- **Real-time Price Chart**: TradingView-style candlestick chart with multiple intervals
- **Bot Controls**: Start/Stop bot with visual status indicator
- **Performance Metrics**: Total return, Sharpe ratio, drawdown, win rate
- **Trade History**: Real-time display of executed trades
- **AI Insights Tab**: Model architecture and feature importance (ready for ML integration)
- **Strategy Manager**: View and manage trading strategies
- **Audit Log**: Record of all manual interventions

### 📍 Current Status

- **Data Collection**: ✅ Working - Live streaming from Binance
- **Bot Controls**: ✅ Working - Start/Stop with visual feedback
- **Paper Trading**: ✅ Working - RSI strategy executing trades
- **RL Training**: ✅ Implemented - PPO agent with continuous actions [-1, 1]
- **ML Models**: ⚠️ RL models trained but not yet integrated into live trading
- **Dashboard**: ✅ Working - All tabs functional, displays real data

---

## 📘 Quick Start Guide

### 1. Prerequisites

- Docker Desktop installed and running
- Binance Account (for testnet API, no real funds needed)

### 2. Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/botrader.git
cd botrader

# Configure environment
cp .env.example .env
# Edit .env with your testnet API credentials
```

**.env Configuration:**

```env
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
BINANCE_TESTNET=True  # Always keep True for safety!
```

### 3. Launch All Services

```bash
docker-compose up --build -d
```

This starts:

- **Database** (TimescaleDB) on port 5432
- **Backend** (FastAPI) on port 8001
- **Frontend** (Next.js) on port 3001
- **MLflow** (Tracking) on port 5001
- **Collector** (Data ingestion)
- **Paper Trader** (Live simulation)

### 4. Access the Dashboard

Open your browser to **http://localhost:3001**

You'll see:

- **Overview Tab**: Price chart, bot controls, trade history
- **Performance Tab**: Metrics dashboard (shows 0% until trades execute)
- **Strategies Tab**: Available trading strategies
- **AI Insights Tab**: Model information and feature importance
- **Audit Log Tab**: Record of actions

### 5. Start Trading

1. Click the **"Start Bot"** button (green)
2. Button changes to **"Stop Bot"** (red) - bot is now running
3. Wait ~14-15 minutes for RSI calculation window
4. Trades will appear in Trade History when RSI crosses thresholds

---

## 🔬 Advanced Workflows

### Generate Historical Dataset

If you need more historical data:

```bash
# Enter backend container
docker exec -it botrader-backend-1 bash

# Backfill historical data from specific date
python backend/scripts/backfill_historical.py

# Build dataset with indicators
python backend/scripts/build_dataset.py

# Exit container
exit
```

This creates `dataset.parquet` with OHLCV + indicators ready for ML training.

### Train Deep RL Models

The system uses Deep Reinforcement Learning (PPO) to train trading agents that learn optimal trading strategies through simulated experiences.

**Prerequisites:**

1. Ensure `dataset.parquet` exists (generate with `build_dataset.py` if needed)
2. MLflow service must be running

**Run Training:**

```bash
# Option 1: Using Docker Compose (Recommended)
docker-compose up -d mlflow timescaledb
sleep 15  # Wait for services to be ready
docker-compose run --rm train-rl

# Option 2: In existing container
docker-compose exec backend python -m app.ml.train_rl

# Option 3: Local development (update MLflow URI to localhost:5001)
python -m app.ml.train_rl
```

**What Gets Trained:**

- **4 separate PPO agents** for different budget scenarios:
  - $1,000 (small account)
  - $10,000 (medium account)
  - $50,000 (large account)
  - $100,000 (institutional)
- Each agent learns continuous actions in range [-1, 1]:
  - `-1.0` = Sell entire position
  - `0.0` = Hold
  - `+1.0` = Buy with all available cash
  - Values in between = Proportional actions

**Training Process:**

- Uses overlapping 4-hour trading windows (240 candles)
- Simulates realistic trading with 0.1% fees per trade
- Validates on separate data every 100 episodes
- Saves best model based on Sharpe ratio

**Check Training Results:**

- Open MLflow UI: **http://localhost:5001**
- Experiment: **"Deep_RL_Trading"**
- View metrics: Sharpe ratio, returns, drawdown, win rate
- Models saved to: `backend/app/ml/models/ppo_agent_budget_*.pth`

See `backend/TRAINING.md` for detailed training guide.

### Run Backtests

```bash
docker exec -it botrader-backend-1 bash

# Backtest RSI strategy
python -m app.backtest_main --strategy simple_rsi

# Backtest RSI+SMA combination
python -m app.backtest_main --strategy rsi_sma
```

**Interpret Results:**

- **Sharpe Ratio > 1.0** = Good risk-adjusted returns
- **Win Rate > 50%** = More winning trades than losing
- **Max Drawdown** = Worst peak-to-trough loss

---

## 📊 Dashboard Guide

### Overview Tab

- **Price Chart**: Live BTC/USDT price with selectable intervals (1m, 5m, 15m, 1h)
- **Bot Controls**:
  - Green "Start Bot" button activates paper trading
  - Red "Stop Bot" button halts trading
  - Emergency Stop for immediate halt + position close
- **Trade History**: Scrollable table of all executed trades
- **Current Position**: Shows if bot is long, short, or neutral

### Performance Tab

Shows metrics after trades execute:

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Trading Stats**: Total trades, avg win, avg loss, profit factor

### Strategies Tab

- Lists available trading strategies
- Currently shows: "No strategies found" (ML models not yet registered)
- Future: Activate/deactivate ML models here

### AI Insights Tab

- **Active Model**: Currently shows "none" (RSI strategy is rule-based)
- **Feature Importance**: Will show SHAP values when ML model active
- **Model Architecture**: Displays model structure and parameters

### Audit Log Tab

- Records all manual interventions
- Logs strategy changes
- Emergency stop events
- System state changes

---

## 🤖 How Paper Trading Works

**Current Implementation:**

1. **Data Flow**:
   - Collector streams live 1m candles from Binance → TimescaleDB
   - Paper trader polls database every 5 seconds for new candles
2. **Strategy Execution**:

   - Accumulates 14+ candles for RSI calculation
   - Calculates RSI(14) on price series
   - **Buy Signal**: RSI < 30 (oversold)
   - **Sell Signal**: RSI > 70 (overbought)

3. **Risk Management**:

   - Risk manager validates each trade
   - Position sizing based on available capital
   - Circuit breakers prevent excessive losses

4. **Trade Execution**:
   - Execution engine simulates TWAP order placement
   - Virtual portfolio tracks positions and PnL
   - Results logged to trade history

**Why "Active Model: none"?**

The RL models (PPO agents) are trained and stored but not yet integrated into the paper trader. Current trading uses the hardcoded RSI strategy in `paper_main.py`. The RL models use continuous actions and learn from simulated trading experiences with realistic constraints.

---

## 🔧 Integration Next Steps

To use RL models instead of RSI:

1. **Load Trained PPO Agent**:

   ```python
   from app.ml.inference import ActionPredictor

   predictor = ActionPredictor(
       model_path='app/ml/models/ppo_agent_budget_10000.pth',
       budget=10000.0
   )
   ```

2. **Get Continuous Action**:

   ```python
   # Predict continuous action [-1, 1]
   action = predictor.predict(df_history, balance, position_size, avg_entry_price)

   # Or get discrete action for backward compatibility
   discrete_action = predictor.predict_discrete(df_history, ...)
   # Returns: 0 (HOLD), 1 (SELL), 2 (BUY)
   ```

3. **Execute Trade**:

   - Convert continuous action to position sizing
   - Apply risk management constraints
   - Execute through execution engine

4. **Update Dashboard**:
   - Register RL strategy in Strategy Registry
   - Display in Strategies tab
   - Show action probabilities and confidence

---

## 📈 Monitoring Stack

**MLflow Tracking** - `http://localhost:5001`

- View all training experiments
- Compare model metrics
- Download trained model artifacts

**TimescaleDB** - `localhost:5432`

- Database: `trading`
- User: `postgres`
- Tables: `candles`, `indicators`, `trades`

**Backend API** - `http://localhost:8001`

- Interactive docs: `http://localhost:8001/docs`
- API endpoints for bot control, strategies, performance

---

## ⚠️ Risk Warning

**This is educational software for learning algorithmic trading.**

- ✅ **ALWAYS** use `BINANCE_TESTNET=True`
- ✅ Start with paper trading (no real money)
- ✅ Monitor the bot during initial runs
- ❌ **NEVER** use real API keys without extensive testing
- ❌ Don't trade money you can't afford to lose

**RL Training Approach:**

- Uses PPO (Proximal Policy Optimization) for continuous action spaces
- Trains on overlapping 4-hour trading windows
- Learns optimal position sizing and timing
- Accounts for transaction fees (0.1% per trade)
- Validates on separate data to prevent overfitting
- Models are trained but need evaluation before live trading

---

## 🗂️ Project Structure

```
Botrader/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI endpoints (bot control, dashboard, indicators)
│   │   ├── backtest/     # Backtesting engine and strategies
│   │   ├── core/         # Database, config, strategy registry
│   │   ├── execution/    # TWAP/VWAP execution algorithms
│   │   ├── features/     # Indicator calculation (FeatureRegistry)
│   │   ├── ml/           # ML training and inference
│   │   │   ├── rl/       # Reinforcement Learning components
│   │   │   │   ├── env.py           # Trading environment (continuous actions)
│   │   │   │   ├── ppo_agent.py    # PPO agent implementation
│   │   │   │   └── window_generator.py  # Overlapping window generation
│   │   │   ├── train_rl.py         # RL training script
│   │   │   └── inference.py        # Model inference (PPO agent)
│   │   ├── risk/         # Risk manager, portfolio, circuit breakers
│   │   └── services/     # Binance collector (WebSocket)
│   ├── scripts/          # Utility scripts (backfill, dataset builder)
│   ├── TRAINING.md       # Detailed RL training guide
│   └── main.py           # FastAPI application
├── frontend/
│   ├── app/              # Next.js pages (main dashboard)
│   ├── components/       # React components (charts, controls, tables)
│   ├── hooks/            # React hooks (useBotData)
│   └── lib/              # API client
├── docker-compose.yml    # Container orchestration
├── .env                  # Environment configuration
└── README.md             # This file
```

---

## 📝 Recent Updates

**December 30, 2024:**

- ✅ **Deep RL Training System**: Implemented PPO agent with continuous action space
- ✅ **Realistic Trading Simulation**: Environment with budget constraints, fees, and proper position management
- ✅ **Overlapping Windows**: Training on diverse 4-hour trading windows with 25% overlap
- ✅ **Multiple Budget Scenarios**: Trains separate models for $1K, $10K, $50K, $100K accounts
- ✅ **Docker Support**: Added `train-rl` service to docker-compose for easy training
- ✅ **Inference System**: Updated to use PPO agents with continuous actions
- ✅ **Cleanup**: Removed old supervised learning training scripts

**December 4, 2024:**

- ✅ Removed all mock/placeholder data from dashboard
- ✅ Fixed Start/Stop bot button functionality
- ✅ Added bot control API endpoints (`/start`, `/stop`, `/status`)
- ✅ Verified paper trader executes RSI-based trades
- ✅ Fixed paper trader RSI calculation (uses `ta` library)
- ✅ Backend now returns real data or zeros (no fake metrics)

**System Status:** Paper trading operational, RL models trained but not yet integrated into live trading
