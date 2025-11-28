# Botrader Roadmap: From MVP to AI-Driven Hedge Fund

This document outlines the strategic plan to evolve Botrader into a state-of-the-art algorithmic trading system.
**Constraint:** All components must be fully **Dockerized** and run **locally** (no cloud services).

## Immediate Priorities 🚨
- [ ] **Frontend Modularization**
  - Refactor the single `page.tsx` into a proper Next.js project structure: `app/`, `components/`, `hooks/`, `lib/`.
  - Extract reusable UI building blocks: layout, charts, metrics cards, bot controls.

- [ ] **Mantine UI Integration**
  - Install Mantine and set up a global theme provider.
  - Replace raw HTML with Mantine components (Grid, Card, Table, Button, Badge, Tabs).
  - Add a light/dark toggle and persistent theme preference.

- [ ] **Performance Dashboard (#1 Priority)**
  - Dedicated `/dashboard` route showing:
    - Win / Loss count
    - Cumulative PnL line chart
    - Equity curve vs HODL benchmark
    - Max drawdown, hit ratio, average R/R
  - Connect to the backend metrics API (or local DB) and auto-refresh.

## Phase 1: Data Engineering & Infrastructure 🏗️
*Goal: Build a robust, containerized data foundation.*

- [ ] **Time-Series Database**
  - Deploy **TimescaleDB** (Docker) as the central store for:
    - OHLCV candles at multiple granularities (1s/1m/5m/1h)
    - Trades and mark prices
    - Limit Order Book (LOB) snapshots (Level 1–2 at minimum)
  - Implement retention policies and continuous aggregates for fast queries.

- [ ] **Data Ingestion & Normalization**
  - Separate `collector` container that:
    - Streams live data from Binance (WS + REST) for BTCUSDT.
    - Normalizes timestamps to a single event time (exchange time).
    - Stores both candles and LOB snapshots in TimescaleDB.
  - Add basic data-quality checks (missing candles, gaps, outliers).

- [ ] **Labeling & Dataset Builder**
  - Implement dataset generation scripts that:
    - Resample features at configurable horizons (e.g., 1m, 5m, 15m).
    - Create supervised learning labels (next-k returns, directional move, volatility).
    - Optionally apply more advanced labeling (e.g., barrier-based events).
  - Export datasets as Parquet/Arrow for fast local training.

- [ ] **Alternative Data (Optional but Nice)**
  - Dockerized scraper for simple sentiment proxies (e.g., funding rates, basic sentiment index).
  - Store alternative features alongside market data with consistent timestamps.

- [ ] **Indicator Library**
  - Create a dedicated Python module (or use `ta-lib` / `pandas_ta`) to compute:
    - Trend indicators: SMA, EMA, WMA, VWMA.
    - Momentum: RSI, Stochastic, MACD, ROC.
    - Volatility: ATR, Bollinger Bands, true range.
    - Volume-based: OBV, volume moving averages, volume spikes.
  - All functions must be:
    - Pure (no side effects).
    - Stateless and easily testable (input: DataFrame, output: DataFrame with new columns).

- [ ] **Historical Indicator Backfill**
  - For each symbol/timeframe stored in TimescaleDB:
    - Batch-compute indicators for the whole historical range.
    - Store indicator values in:
      - Either extra columns in existing OHLCV tables, or
      - A separate `indicators_<symbol>_<timeframe>` table linked by timestamp.
  - Handle warmup periods (e.g., discard first N bars where the indicator is not fully defined).

- [ ] **Online Indicator Updates**
  - Integrate indicator computation into the `collector` / `feature` service:
    - On each new bar close, compute/update indicators incrementally.
    - Persist new values with the same timestamp as the bar.
  - Ensure recomputation logic is idempotent (safe to rerun without duplicating).

- [ ] **Data-Quality Checks for Indicators**
  - Add checks to verify:
    - No NaNs in live windows (after warmup).
    - Indicator values are consistent after reload/backfill.
  - Add unit tests that compare indicator outputs with a reference implementation (e.g., `ta-lib`).

## Phase 2: Advanced Backtesting & Simulation 🧪
*Goal: Validate strategies rigorously.*

- [ ] **Backtesting Engine Integration**
  - Integrate **Backtrader** or similar Python backtest framework inside a dedicated `backtest` container.
  - Write an adapter that:
    - Pulls historical data from TimescaleDB.
    - Replays trades with realistic fees and minimum order sizes from Binance.

- [ ] **Execution Realism**
  - Implement:
    - Fee model (maker/taker, Binance testnet fees).
    - Slippage model based on LOB depth (consume L2 snapshots).
    - Minimum tick size and lot size constraints.

- [ ] **Walk-Forward / Rolling Evaluation**
  - Implement rolling train/validation/test splits:
    - Sequential walk-forward windows.
    - Prevent look-ahead and leakage across splits.
  - Export per-window metrics (Sharpe, drawdown, turnover, win rate).

- [ ] **Paper Trading Mode**
  - Run a `paper-trading` container:
    - Subscribes to live prices.
    - Executes orders virtually, logging fills and PnL.
    - Uses the same strategy interface as backtests.

## Phase 3: Modeling & Alpha Generation 🧠
*Goal: Hierarchy of models from simple baselines to SOTA deep learning, all pluggable into the same strategy interface.*

### 3.1 Feature Store & Baselines

- [ ] **Feature Store**
  - Deploy **Feast** (Docker) or a simple custom feature registry:
    - Define reusable features: returns, rolling volatility, volume imbalance, order book imbalance, etc.
    - Ensure consistency between training and live inference.

- [ ] **Baseline Models**
  - Implement a simple baseline pipeline:
    - Logistic regression / linear model on hand-crafted features.
    - Tree-based models (e.g., **LightGBM**) for classification of next-k price direction.
  - Use these as sanity benchmarks for all deep models.

- [ ] **Feature Store**
  - Register features for:
    - Raw: OHLCV, returns, log-returns.
    - Indicators: RSI, SMA, EMA, MACD, Bollinger Bands, ATR, volume-based metrics.
    - Order-book features: bid/ask spread, depth imbalance (when LOB is enabled).
  - Guarantee that the same feature definitions are used in:
    - Offline training.
    - Backtesting.
    - Live inference.

- [ ] **Indicator-Based Baseline Strategy**
  - Implement a simple rule-based strategy using indicators:
    - Example: RSI overbought/oversold + SMA crossover.
    - Run it through the backtester as a sanity check for:
      - Indicator correctness.
      - Backtest execution realism.
  - Compare this simple strategy against:
    - Buy & hold.
    - Model-based strategies, once they exist.

### 3.2 Sequence Models on OHLCV (LSTM / CNN)

- [ ] **Sequence Dataset**
  - Build windowed sequences of OHLCV + engineered features.
  - Configurable window length (e.g., 64–256 timesteps) and prediction horizon (e.g., next 5–15 min).

- [ ] **LSTM / GRU Models**
  - Implement a small LSTM/GRU classifier/regressor for:
    - Directional move (up / down / flat).
    - Or next-k return forecast.
  - Train with early stopping, class balancing, and proper validation splits.

- [ ] **Temporal CNN / ResNet**
  - Implement a 1D CNN / temporal ResNet model on the same sequences.
  - Compare performance vs LSTM/GRU and LightGBM.

### 3.3 Time-Series Transformers (OHLCV)

- [ ] **Transformer Forecasting Models**
  - Implement one or more time-series transformer architectures (via PyTorch or PyTorch Forecasting), e.g.:
    - Temporal Fusion Transformer (TFT).
    - A light PatchTST / Informer-style model.
  - Train for multi-horizon forecasts (distribution over future returns or prices).

- [ ] **Uncertainty & Calibration**
  - Output predictive distributions (quantile loss or variance estimates).
  - Calibrate the outputs and expose “confidence” or “edge” for position sizing.

### 3.4 LOB-Specific Deep Models (High-Frequency Alpha)

- [ ] **LOB Dataset**
  - Build a high-frequency dataset from L2 snapshots:
    - Input: sequences of top N levels (prices + volumes).
    - Labels: short-term mid-price move or next-tick direction.

- [ ] **DeepLOB-Style CNN + LSTM**
  - Implement a DeepLOB-inspired model:
    - Convolutional layers over price levels.
    - Temporal layers (LSTM/GRU) over time.
  - Train and evaluate on the LOB dataset.

- [ ] **Transformer-Based LOB Model**
  - Implement a compact LOB transformer (LiT/TLOB-inspired):
    - Dual attention over price levels and time.
    - Compare vs DeepLOB-style CNN-LSTM.

- [ ] **Toggle Between Regimes**
  - Make LOB models optional and configurable (since they are heavier and data-hungry).
  - Expose an interface to choose candle-based vs LOB-based strategies.

### 3.5 Hybrid & Ensemble Models

- [ ] **LightGBM + Deep Model Ensembles**
  - Implement stacking/blending:
    - Use LightGBM on top of features + deep model outputs (probabilities, hidden embeddings).
    - Or average/weighted-ensemble of multiple models.

- [ ] **Regime-Specific Models**
  - Define simple regime indicators (e.g., volatility, trend, volume).
  - Train separate models per regime and a gating logic that switches between them.

### 3.6 Reinforcement Learning Overlay

- [ ] **Trading Environment**
  - Implement a Gym-compatible environment that:
    - Uses historical data (OHLCV or LOB).
    - Includes transaction costs, slippage, position limits.
    - Exposes observations including model signals (from supervised models) and state features.

- [ ] **RL Agents**
  - Start with **DQN** / **PPO** agents:
    - Action space: discrete {flat, long, short} or {increase/decrease position}.
    - Reward: PnL with penalties for drawdown, volatility, and turnover.

- [ ] **RL vs Supervised Benchmarking**
  - Compare RL policies vs:
    - Thresholded signals from supervised models.
    - Simple rule-based strategies (e.g., moving-average crossover).
  - Keep RL as an optional, “advanced” mode given complexity and fragility.

## Phase 4: Local MLOps & Continuous Learning 🔄
*Goal: Automate the model lifecycle locally, end-to-end.*

- [ ] **Experiment Tracking & Model Registry**
  - Deploy **MLflow** (Docker) with local storage:
    - Track parameters, metrics, artifacts.
    - Register “candidate” models with tags (data range, features, label type).

- [ ] **Model Validation Pipeline**
  - CLI or script to:
    - Train model → log to MLflow → run backtests on a fixed evaluation set.
    - Compute standardized metrics: Sharpe, Sortino, max DD, turnover, hit ratio.
    - Decide if a model is eligible for promotion to “live”.

- [ ] **Model Serving & Inference**
  - Dedicated `inference` service (Docker):
    - Loads the selected model version.
    - Pulls latest features from TimescaleDB/feature store.
    - Returns trading signals with latency constraints.

- [ ] **Monitoring**
  - Deploy **Prometheus** and **Grafana**:
    - Track bot PnL, drawdown, latency, error rates.
    - Track input drift (feature distribution changes) and model performance degradation.


## Phase 5: Risk Management & Execution 🛡️
*Goal: Protect capital and ensure sane execution.*

- [ ] **Position Sizing & Risk Limits**
  - Volatility-targeted position sizing (e.g., target daily vol).
  - Per-trade and per-day loss limits.
  - Max leverage and max notional caps.

- [ ] **Portfolio Optimization (Future Multi-Asset)**
  - Implement mean-variance or risk-parity optimizers.
  - Prepare the codebase to support >1 asset so multi-asset optimization is plug-and-play later.

- [ ] **Smart Execution Algorithms**
  - Implement TWAP/VWAP-like execution strategies for larger orders.
  - Allow the strategy to choose between “aggressive” and “passive” execution modes based on liquidity.

- [ ] **Circuit Breakers & Safety**
  - Global kill switch if:
    - Drawdown exceeds set threshold.
    - Latency or error rates blow up.
  - UI-level “Stop Bot” button that immediately closes positions and halts trading.

## Phase 6: Advanced Dashboard & UI 📊
*Goal: Deep insights and safe human control.*

- [ ] **Model Explainability (for non-DL and DL where possible)**
  - For tree models (LightGBM): SHAP summary and per-trade explanations.
  - For transformers/LSTMs: simpler feature attribution or at least input saliency, where feasible.

- [ ] **Live Risk & Performance Metrics**
  - Display:
    - Live and historical Sharpe/Sortino.
    - Max drawdown and current drawdown.
    - Turnover and fees paid.
  - Per-strategy breakdown if multiple models are live.

- [ ] **Strategy Management UI**
  - List of available models/strategies:
    - Baselines, LSTM/CNN, Transformers, LOB models, RL agent.
  - Toggle which strategy is live, with confirmation dialogs and logs.
  - Show last deployment/change and associated backtest stats.

- [ ] **Manual Override / Kill Switch**
  - Prominent button to:
    - Halt trading immediately.
    - Optionally flatten all positions.
  - Log all manual interventions for audit and debugging.

- [ ] **Chart Overlays with Indicators**
  - In the main chart UI:
    - Add overlays for SMA/EMA on price.
    - Add sub-panels for RSI, MACD, and volume.
  - Allow toggling indicators on/off per chart.
  - Ensure data is pulled from the same indicator tables used by the models.
