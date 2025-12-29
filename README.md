# Botrader: AI-Driven Algorithmic Trading System 🚀

Botrader is a locally deployed algorithmic trading platform for Bitcoin and crypto markets. It features live data collection, ML model training with MLflow experiment tracking, paper trading simulation, and a professional React dashboard for monitoring and control.

## 🌟 Current Features (Implemented & Working)

### ✅ Data Infrastructure
*   **Live Data Collection**: Binance WebSocket streaming for real-time 1-minute candles
*   **TimescaleDB**: Time-series database with 93,000+ historical candles
*   **Historical Backfill**: Script to fetch data from any start date to present
*   **Technical Indicators**: Auto-computed SMA, EMA, RSI, MACD, Bollinger Bands, ATR

### ✅ Machine Learning
*   **MLflow Integration**: Experiment tracking and model versioning
*   **Trained Models**: LightGBM (50% accuracy) and Logistic Regression (51% accuracy)
*   **Dataset Builder**: Automated feature engineering from raw OHLCV data
*   **Models Available**: `train.py`, `train_ensemble.py`, `train_deep.py`, `train_transformer.py`

### ✅ Paper Trading
*   **Live Simulation**: Paper trader monitors database for new candles
*   **Current Strategy**: RSI-based (Buy RSI<30, Sell RSI>70)
*   **Risk Management**: Position sizing, trade risk validation, circuit breakers
*   **Smart Execution**: TWAP/VWAP algorithms for order execution

### ✅ Professional Dashboard
*   **Real-time Price Chart**: TradingView-style candlestick chart with multiple intervals
*   **Bot Controls**: Start/Stop bot with visual status indicator
*   **Performance Metrics**: Total return, Sharpe ratio, drawdown, win rate
*   **Trade History**: Real-time display of executed trades
*   **AI Insights Tab**: Model architecture and feature importance (ready for ML integration)
*   **Strategy Manager**: View and manage trading strategies
*   **Audit Log**: Record of all manual interventions

### 📍 Current Status
- **Data Collection**: ✅ Working - Live streaming from Binance
- **Bot Controls**: ✅ Working - Start/Stop with visual feedback
- **Paper Trading**: ✅ Working - RSI strategy executing trades
- **ML Models**: ⚠️ Trained but not yet integrated into live trading
- **Dashboard**: ✅ Working - All tabs functional, displays real data

---

## 📘 Quick Start Guide

### 1. Prerequisites
*   Docker Desktop installed and running
*   Binance Account (for testnet API, no real funds needed)

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

### Train Machine Learning Models

```bash
# Enter backend container
docker exec -it botrader-backend-1 bash

# Train baseline models (LightGBM + Logistic Regression)
python -m app.ml.train

# Train ensemble meta-learner
python -m app.ml.train_ensemble

# Train deep learning models
python -m app.ml.train_deep

# Exit container
exit
```

**Check Training Results:**
- Open MLflow UI: **http://localhost:5001**
- View experiments, metrics, and model artifacts

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

The ML models (LightGBM, LogReg) are trained and stored in MLflow but not yet integrated into the paper trader. Current trading uses the hardcoded RSI strategy in `paper_main.py`.

---

## 🔧 Integration Next Steps

To use ML models instead of RSI:

1. **Load Model from MLflow**:
   ```python
   import mlflow
   model = mlflow.sklearn.load_model("runs:/RUN_ID/model")
   ```

2. **Replace RSI Logic**:
   - Compute same features model was trained on
   - Get prediction: `pred = model.predict(features)`
   - Execute trade based on prediction

3. **Update Dashboard**:
   - Register model in Strategy Registry
   - Display in Strategies tab
   - Show SHAP explanations in Insights

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

**Baseline Model Performance:**
- LightGBM: 50% accuracy (random baseline)
- LogReg: 51% accuracy (barely above random)
- These models need improvement before real trading

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
│   │   ├── ml/           # Model training scripts
│   │   ├── risk/         # Risk manager, portfolio, circuit breakers
│   │   └── services/     # Binance collector (WebSocket)
│   ├── scripts/          # Utility scripts (backfill, dataset builder)
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

**December 4, 2024:**
- ✅ Removed all mock/placeholder data from dashboard
- ✅ Fixed Start/Stop bot button functionality
- ✅ Added bot control API endpoints (`/start`, `/stop`, `/status`)
- ✅ Verified paper trader executes RSI-based trades
- ✅ Fixed paper trader RSI calculation (uses `ta` library)
- ✅ Backend now returns real data or zeros (no fake metrics)

**System Status:** Paper trading operational, ML models trained but not integrated
