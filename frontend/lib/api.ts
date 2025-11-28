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
  }
};
