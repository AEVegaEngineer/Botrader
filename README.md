# Botrader - Bitcoin Algorithmic Trading Bot

Botrader is a full-stack algorithmic trading application designed for Bitcoin trading on Binance. It features a Python FastAPI backend that handles the trading logic and a Next.js frontend for real-time monitoring and control.

## Features

- **Real-time Price Monitoring**: Tracks Bitcoin price updates from Binance.
- **Algorithmic Trading**: Implements a configurable threshold-based strategy (Simple Moving Average logic can be easily extended).
- **Dashboard**: A modern web interface to view price charts, trade history, and bot status.
- **Dockerized**: Fully containerized setup for easy deployment.
- **Testnet Support**: Safely test strategies using Binance Testnet.

## Project Structure

```
Botrader/
├── backend/            # Python FastAPI application
│   ├── bot.py         # Trading logic and Binance client wrapper
│   ├── main.py        # API endpoints
│   ├── config.py      # Configuration loading
│   └── Dockerfile     # Backend container definition
├── frontend/           # Next.js application
│   ├── app/           # React components and pages
│   └── Dockerfile     # Frontend container definition
├── docker-compose.yml  # Docker orchestration
└── .env               # Environment variables (API keys)
```

## Prerequisites

- **Docker Desktop**: Required for running the containerized application.
- **Binance Account**: API Key and Secret required (Testnet recommended for development).

## Setup & Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Configure Environment Variables**:
    Create a `.env` file in the root directory based on `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your Binance credentials:
    ```env
    BINANCE_API_KEY=your_api_key_here
    BINANCE_API_SECRET=your_api_secret_here
    BINANCE_TESTNET=True  # Set to False for real trading
    ```

3.  **Start the Application**:
    Run the following command to build and start the containers:
    ```bash
    docker-compose up --build -d
    ```

## Usage

### Accessing the Dashboard
Once the containers are running, open your browser and navigate to:
**[http://localhost:3001](http://localhost:3001)**

### Controlling the Bot
- **Start Bot**: Click the "Start Bot" button on the dashboard. The bot will initialize and start monitoring prices.
- **Stop Bot**: Click "Stop Bot" to halt trading operations.

### Monitoring
- **Price Chart**: Shows the real-time Bitcoin price.
- **Trade History**: Displays a list of all trades executed by the bot during the current session.
- **Status**: Indicates whether the bot is currently Running or Stopped.

## Configuration

- **Ports**:
    - Backend API: `8001`
    - Frontend Dashboard: `3001`
    - *Note: These ports were configured to avoid conflicts with common defaults.*

- **Strategy**:
    - The current strategy is defined in `backend/bot.py`.
    - It uses a simple threshold mechanism:
        - **Buy**: When price drops 100 units below the initial price.
        - **Sell**: When price rises 100 units above the initial price.

## Development

To view logs for debugging:
```bash
docker-compose logs -f
```

To stop the application:
```bash
docker-compose down
```
