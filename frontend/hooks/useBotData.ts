import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';

export function useBotData() {
  const [status, setStatus] = useState({ status: 'Stopped', is_running: false });
  const [history, setHistory] = useState([]);
  const [price, setPrice] = useState(0);
  const [priceData, setPriceData] = useState<{ time: string; price: number }[]>([]);
  const [performance, setPerformance] = useState({ wins: 0, losses: 0, total_pnl: 0, win_rate: 0, equity_curve: [] });

  const fetchData = useCallback(async () => {
    try {
      const [statusData, historyData, priceData, performanceData] = await Promise.all([
        api.getStatus(),
        api.getHistory(),
        api.getPrice(),
        api.getPerformance()
      ]);

      setStatus(statusData);
      setHistory(historyData);
      setPerformance(performanceData);
      
      const newPrice = priceData.price;
      setPrice(newPrice);

      if (newPrice > 0) {
        setPriceData(prev => {
          const newData = [...prev, { time: new Date().toLocaleTimeString(), price: newPrice }];
          if (newData.length > 50) newData.shift();
          return newData;
        });
      }
    } catch (error) {
      console.error("Failed to fetch bot data", error);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const startBot = async () => {
    try {
      await api.startBot();
      fetchData();
    } catch (error) {
      console.error("Failed to start bot", error);
    }
  };

  const stopBot = async () => {
    try {
      await api.stopBot();
      fetchData();
    } catch (error) {
      console.error("Failed to stop bot", error);
    }
  };

  return {
    status,
    history,
    price,
    priceData,
    performance,
    startBot,
    stopBot,
    refresh: fetchData
  };
}
