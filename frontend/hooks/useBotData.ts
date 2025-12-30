import { useState, useEffect, useCallback } from "react";
import { api } from "../lib/api";

export function useBotData() {
  const [status, setStatus] = useState({
    status: "Stopped",
    is_running: false,
  });
  const [history, setHistory] = useState([]);
  const [price, setPrice] = useState(0);
  const [priceData, setPriceData] = useState<any[]>([]);
  const [smaData, setSmaData] = useState<any[]>([]);
  const [showSMA, setShowSMA] = useState(false);
  const [emaData, setEmaData] = useState<any[]>([]);
  const [showEMA, setShowEMA] = useState(false);
  const [bbData, setBbData] = useState<any[]>([]);
  const [showBB, setShowBB] = useState(false);
  const [rsiData, setRsiData] = useState<any[]>([]);
  const [showRSI, setShowRSI] = useState(false);
  const [performance, setPerformance] = useState({
    wins: 0,
    losses: 0,
    total_pnl: 0,
    win_rate: 0,
    equity_curve: [],
  });
  const [chartInterval, setChartInterval] = useState("1m");

  // Load interval from session storage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedInterval = sessionStorage.getItem("botrader_chart_interval");
      if (savedInterval) {
        setChartInterval(savedInterval);
      }
    }
  }, []);

  // Save interval to session storage when it changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem("botrader_chart_interval", chartInterval);
    }
  }, [chartInterval]);

  const fetchData = useCallback(async () => {
    try {
      // Calculate appropriate limit based on interval to show reasonable amount of data
      // Backend will auto-calculate hours_back, but we can set a good limit
      const limitMap: Record<string, number> = {
        "1m": 120, // ~2 hours
        "3m": 120, // ~6 hours
        "5m": 120, // ~10 hours
        "15m": 96, // ~24 hours
        "30m": 96, // ~48 hours
        "1h": 72, // ~3 days
        "2h": 84, // ~7 days
        "4h": 84, // ~14 days
        "6h": 84, // ~21 days
        "8h": 84, // ~28 days
        "12h": 60, // ~30 days
        "1d": 30, // ~30 days
        "3d": 30, // ~90 days
        "1w": 30, // ~210 days
        "1M": 30, // ~900 days
      };
      const limit = limitMap[chartInterval] || 100;

      const [statusData, historyData, priceData, performanceData, candleData] =
        await Promise.all([
          api.getStatus(),
          api.getHistory(),
          api.getPrice(),
          api.getPerformance(),
          api.getCandles("BTCUSDT", chartInterval, limit),
        ]);

      setStatus(statusData);
      setHistory(historyData.trades || []);
      setPerformance(performanceData);

      const newPrice = priceData.price;
      setPrice(newPrice);

      // Update candle data
      if (candleData && candleData.candles) {
        setPriceData(candleData.candles);
      }

      // Fetch SMA data if enabled
      if (showSMA) {
        try {
          const sma20Data = await api.getSMA("BTCUSDT", 20, chartInterval, limit);
          if (sma20Data && sma20Data.data) {
            setSmaData(sma20Data.data);
          }
        } catch (error) {
          console.error("Failed to fetch SMA data", error);
          setSmaData([]);
        }
      } else {
        setSmaData([]);
      }

      // Fetch EMA data if enabled
      if (showEMA) {
        try {
          const ema50Data = await api.getEMA("BTCUSDT", 50, chartInterval, limit);
          if (ema50Data && ema50Data.data) {
            setEmaData(ema50Data.data);
          }
        } catch (error) {
          console.error("Failed to fetch EMA data", error);
          setEmaData([]);
        }
      } else {
        setEmaData([]);
      }

      // Fetch Bollinger Bands data if enabled
      if (showBB) {
        try {
          const bbData = await api.getBollingerBands("BTCUSDT", 20, 2.0, chartInterval, limit);
          if (bbData && bbData.data) {
            setBbData(bbData.data);
          }
        } catch (error) {
          console.error("Failed to fetch Bollinger Bands data", error);
          setBbData([]);
        }
      } else {
        setBbData([]);
      }

      // Fetch RSI data if enabled
      if (showRSI) {
        try {
          const rsiData = await api.getRSI("BTCUSDT", 14, chartInterval, limit);
          if (rsiData && rsiData.data) {
            setRsiData(rsiData.data);
          }
        } catch (error) {
          console.error("Failed to fetch RSI data", error);
          setRsiData([]);
        }
      } else {
        setRsiData([]);
      }
    } catch (error) {
      console.error("Failed to fetch bot data", error);
    }
  }, [chartInterval, showSMA, showEMA, showBB, showRSI]);

  useEffect(() => {
    fetchData();
    const fetchIntervalId = setInterval(fetchData, 2000);
    return () => clearInterval(fetchIntervalId);
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
    smaData,
    showSMA,
    setShowSMA,
    emaData,
    showEMA,
    setShowEMA,
    bbData,
    showBB,
    setShowBB,
    rsiData,
    showRSI,
    setShowRSI,
    performance,
    interval: chartInterval,
    setInterval: setChartInterval,
    startBot,
    stopBot,
    refresh: fetchData,
  };
}
