# Botrader Roadmap: From MVP to AI-Driven Hedge Fund

This document outlines the strategic plan to evolve Botrader into a state-of-the-art algorithmic trading system.
**Constraint:** All components must be fully **Dockerized** and run **locally** (no cloud services).

## Immediate Priorities 🚨
- [ ] **Frontend Modularization**: Refactor the single `page.tsx` into a proper Next.js project structure with reusable components.
- [ ] **Mantine UI**: Apply [Mantine](https://mantine.dev/) styles and theming for a professional look.
- [ ] **Performance Dashboard**: (#1 Priority) Create a dedicated dashboard page displaying:
    - Wins / Losses
    - Trading Performance Metrics
    - Profit Graph (Cumulative PnL)

## Phase 1: Data Engineering & Infrastructure 🏗️
*Goal: Build a robust, containerized data foundation.*

- [ ] **Time-Series Database**: Deploy **TimescaleDB** (Docker). It offers robust SQL support and efficient time-series handling, making it superior for financial data with complex queries.
- [ ] **Data Pipeline**: Implement **Prefect** (Docker) for orchestration. It is lightweight and Python-native, making it ideal for a local, containerized stack.
- [ ] **Order Book Data**: Collect Level 2 data.
- [ ] **Alternative Data**: Scrape sentiment data locally (e.g., using Selenium/Scrapy in Docker) to feed into models.

## Phase 2: Advanced Backtesting & Simulation 🧪
*Goal: Validate strategies rigorously.*

- [ ] **Backtesting Engine**: Integrate **Backtrader**. It is a pure Python framework that integrates seamlessly with our existing stack.
- [ ] **Realistic Simulation**: Implement slippage and fee models.
- [ ] **Walk-Forward Optimization**: Implement strictly to prevent overfitting.
- [ ] **Paper Trading Mode**: "Shadow mode" running in a separate Docker container.

## Phase 3: Deep Learning & Alpha Generation 🧠
*Goal: Predictive AI models running in inference containers.*

- [ ] **Feature Engineering**: Deploy **Feast** (Docker) with Redis (Online) and PostgreSQL/Parquet (Offline) to manage features consistently between training and inference.
- [ ] **Time-Series Forecasting**:
    - Focus on **Time-Series Transformers** (e.g., Temporal Fusion Transformers) for state-of-the-art performance.
- [ ] **Reinforcement Learning (RL)**:
    - Train **PPO** agents using **Stable Baselines3**.
- [ ] **Training Pipeline**: Create a dedicated `training` container that spins up, trains on GPU (if available/passed to Docker), saves artifacts, and shuts down.

## Phase 4: MLOps & Continuous Learning 🔄
*Goal: Automate the lifecycle locally.*

- [ ] **Model Registry**: Deploy **MLflow** (Docker) with a local artifact store. It is the industry standard for local experiment tracking and model versioning.
- [ ] **Model Serving**: Create a dedicated `inference` service using **ONNX Runtime** wrapped in FastAPI. ONNX provides faster inference and lower latency than TorchServe for this scale.
- [ ] **Monitoring**: Deploy **Prometheus** and **Grafana** (Docker) to scrape metrics from the bot and inference services.

## Phase 5: Risk Management & Execution 🛡️
*Goal: Protect capital.*

- [ ] **Portfolio Optimization**: Implement **Markowitz Mean-Variance Optimization**.
- [ ] **Smart Execution**: TWAP/VWAP execution algorithms.
- [ ] **Circuit Breakers**: Hard-coded safety stops.

## Phase 6: Advanced Dashboard & UI 📊
*Goal: Deep insights.*

- [ ] **Model Explainability**: Visualize SHAP values.
- [ ] **Performance Metrics**: Real-time Sharpe/Sortino ratios.
- [ ] **Manual Override**: "Kill switch" in the UI.
