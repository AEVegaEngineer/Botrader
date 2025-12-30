const API_BASE_URL = 'http://localhost:8001';

export const api = {
  getStatus: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/status`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch status:', error);
      return { status: 'Error', is_running: false };
    }
  },
  getHistory: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/history`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      return { trades: data.trades || [] };
    } catch (error) {
      console.error('Failed to fetch history:', error);
      return { trades: [] };
    }
  },
  getPrice: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/price`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch price:', error);
      return { symbol: 'BTCUSDT', price: 0, timestamp: new Date().toISOString() };
    }
  },
  getCandles: async (symbol: string = "BTCUSDT", interval: string = "1m", limit: number = 200, hoursBack?: number) => {
    try {
      let url = `${API_BASE_URL}/api/v1/indicators/candles?symbol=${symbol}&interval=${interval}&limit=${limit}`;
      if (hoursBack !== undefined) {
        url += `&hours_back=${hoursBack}`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch candles:', error);
      return { symbol, interval, candles: [] };
    }
  },
  getSMA: async (symbol: string = "BTCUSDT", period: number = 20, interval: string = "1m", limit: number = 500) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/indicators/sma?symbol=${symbol}&period=${period}&interval=${interval}&limit=${limit}`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch SMA:', error);
      return { symbol, indicator: "SMA", period, interval, data: [] };
    }
  },
  startBot: async () => {
    const res = await fetch(`${API_BASE_URL}/start`, { method: 'POST' });
    return res.json(); // Assuming backend returns something, or just check status
  },
  stopBot: async () => {
    const res = await fetch(`${API_BASE_URL}/stop`, { method: 'POST' });
    return res.json();
  },
  getPerformance: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/performance`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch performance:', error);
      return { wins: 0, losses: 0, total_pnl: 0, win_rate: 0, equity_curve: [] };
    }
  },
  
  // Risk Management Endpoints
  emergencyStop: async (closePositions: boolean = true, reason?: string) => {
    const res = await fetch(`${API_BASE_URL}/api/v1/emergency-stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ close_positions: closePositions, reason })
    });
    return res.json();
  },
  
  resumeTrading: async () => {
    const res = await fetch(`${API_BASE_URL}/api/v1/resume-trading`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmed: true })
    });
    return res.json();
  },
  
  getRiskStatus: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/risk-status`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch risk status:', error);
      return null;
    }
  },
  
  getPortfolio: async () => {
    const res = await fetch(`${API_BASE_URL}/api/v1/portfolio`);
    return res.json();
  },
  
  closePosition: async (symbol: string) => {
    const res = await fetch(`${API_BASE_URL}/api/v1/positions/close/${symbol}`, {
      method: 'POST'
    });
    return res.json();
  },
  
  closeAllPositions: async () => {
    const res = await fetch(`${API_BASE_URL}/api/v1/positions/close-all`, {
      method: 'POST'
    });
    return res.json();
  },

  // Performance & Analytics
  getPerformanceMetrics: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/performance/metrics`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error);
      return {
        total_return: 0.0,
        annualized_return: 0.0,
        sharpe_ratio: 0.0,
        sortino_ratio: 0.0,
        calmar_ratio: 0.0,
        max_drawdown: 0.0,
        current_drawdown: 0.0,
        total_trades: 0,
        win_rate: 0.0,
        avg_win: 0.0,
        avg_loss: 0.0,
        profit_factor: 0.0,
        turnover: 0.0,
        total_fees: 0.0,
        fees_pct_of_pnl: 0.0
      };
    }
  },

  getPerformanceByStrategy: async () => {
    const res = await fetch(`${API_BASE_URL}/api/v1/performance/by-strategy`);
    return res.json();
  },

  // Strategy Management
  getStrategies: async () => {
    const res = await fetch(`${API_BASE_URL}/api/v1/strategies`);
    return res.json();
  },

  activateStrategy: async (id: string, reason: string = "Manual activation") => {
    const res = await fetch(`${API_BASE_URL}/api/v1/strategies/${id}/activate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason })
    });
    return res.json();
  },

  deactivateStrategy: async (id: string) => {
    const res = await fetch(`${API_BASE_URL}/api/v1/strategies/${id}/deactivate`, {
      method: 'POST'
    });
    return res.json();
  },

  // Audit Log
  getInterventions: async (limit: number = 50) => {
    const res = await fetch(`${API_BASE_URL}/api/v1/interventions?limit=${limit}`);
    return res.json();
  },

  // Explainability
  getFeatureImportance: async () => {
    const res = await fetch(`${API_BASE_URL}/api/v1/explainability/feature-importance`);
    return res.json();
  },

  getPredictionExplanation: async () => {
    // Mock request for latest prediction
    const res = await fetch(`${API_BASE_URL}/api/v1/explainability/explain-prediction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: {}, index: 0 })
    });
    return res.json();
  },

  // Strategy Switching (new endpoints)
  getStrategiesNew: async () => {
    const res = await fetch(`${API_BASE_URL}/strategies`);
    return res.json();
  },

  setStrategy: async (strategy: string) => {
    const res = await fetch(`${API_BASE_URL}/strategy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strategy })
    });
    return res.json();
  }
};
