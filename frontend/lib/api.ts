const API_BASE_URL = 'http://localhost:8001';

export const api = {
  getStatus: async () => {
    const res = await fetch(`${API_BASE_URL}/status`);
    return res.json();
  },
  getHistory: async () => {
    const res = await fetch(`${API_BASE_URL}/history`);
    return res.json();
  },
  getPrice: async () => {
    const res = await fetch(`${API_BASE_URL}/price`);
    return res.json();
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
    const res = await fetch(`${API_BASE_URL}/performance`);
    return res.json();
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
    const res = await fetch(`${API_BASE_URL}/api/v1/risk-status`);
    return res.json();
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
    const res = await fetch(`${API_BASE_URL}/api/v1/performance/metrics`);
    return res.json();
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
  }
};
