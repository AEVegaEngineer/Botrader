"use client";

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, Square, Activity, DollarSign, TrendingUp, Clock } from 'lucide-react';

export default function Dashboard() {
  const [status, setStatus] = useState({ status: 'Stopped', is_running: false });
  const [price, setPrice] = useState(0);
  const [history, setHistory] = useState([]);
  const [priceData, setPriceData] = useState<{ time: string; price: number }[]>([]);
  const [wsConnected, setWsConnected] = useState(false);

  // Fetch status
  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8001/status');
      const data = await res.json();
      setStatus(data);
    } catch (error) {
      console.error("Failed to fetch status", error);
    }
  };

  // Fetch history
  const fetchHistory = async () => {
    try {
      const res = await fetch('http://localhost:8001/history');
      const data = await res.json();
      setHistory(data);
    } catch (error) {
      console.error("Failed to fetch history", error);
    }
  };

  // Fetch price and update chart
  const fetchPrice = async () => {
    try {
      const res = await fetch('http://localhost:8001/price');
      const data = await res.json();
      const newPrice = data.price;
      setPrice(newPrice);
      
      if (newPrice > 0) {
        setPriceData(prev => {
          const newData = [...prev, { time: new Date().toLocaleTimeString(), price: newPrice }];
          if (newData.length > 50) newData.shift(); // Keep last 50 points
          return newData;
        });
      }
    } catch (error) {
      console.error("Failed to fetch price", error);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchHistory();
    const interval = setInterval(() => {
      fetchStatus();
      fetchHistory();
      fetchPrice();
    }, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    try {
      await fetch('http://localhost:8001/start', { method: 'POST' });
      fetchStatus();
    } catch (error) {
      console.error("Failed to start bot", error);
    }
  };

  const handleStop = async () => {
    try {
      await fetch('http://localhost:8001/stop', { method: 'POST' });
      fetchStatus();
    } catch (error) {
      console.error("Failed to stop bot", error);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <div className="flex items-center gap-4">
          <Activity className="w-8 h-8 text-blue-500" />
          <h1 className="title">Botrader Dashboard</h1>
        </div>
        <div className={`status-badge ${status.is_running ? 'status-running' : 'status-stopped'}`}>
          {status.status}
        </div>
      </header>

      <div className="grid">
        <div className="card">
          <div className="card-title flex items-center gap-2">
            <DollarSign size={20} /> Current Price
          </div>
          <div className="card-value">${price.toLocaleString()}</div>
        </div>
        <div className="card">
          <div className="card-title flex items-center gap-2">
            <TrendingUp size={20} /> Total Trades
          </div>
          <div className="card-value">{history.length}</div>
        </div>
        <div className="card">
          <div className="card-title flex items-center gap-2">
            <Clock size={20} /> Uptime
          </div>
          <div className="card-value">--:--</div>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '2fr 1fr' }}>
        <div className="card" style={{ minHeight: '400px' }}>
          <div className="card-title mb-4">Live Price Action</div>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer>
              <LineChart data={priceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis domain={['auto', 'auto']} stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Line type="monotone" dataKey="price" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-title mb-4">Controls</div>
          <div className="flex flex-col gap-4">
            {!status.is_running ? (
              <button onClick={handleStart} className="btn btn-primary flex items-center justify-center gap-2">
                <Play size={20} /> Start Bot
              </button>
            ) : (
              <button onClick={handleStop} className="btn btn-danger flex items-center justify-center gap-2">
                <Square size={20} /> Stop Bot
              </button>
            )}
            
            <div className="mt-4 p-4 bg-slate-800 rounded-lg">
              <h3 className="text-sm font-semibold text-slate-400 mb-2">Strategy Config</h3>
              <div className="text-sm">
                <div className="flex justify-between mb-1">
                  <span>Symbol:</span> <span className="text-white">BTCUSDT</span>
                </div>
                <div className="flex justify-between mb-1">
                  <span>Interval:</span> <span className="text-white">1m</span>
                </div>
                <div className="flex justify-between">
                  <span>Mode:</span> <span className="text-yellow-500">Testnet</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title mb-4">Trade History</div>
        <div className="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Time</th>
                <th>Side</th>
                <th>Price</th>
                <th>Quantity</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center text-slate-500 py-8">
                    No trades executed yet
                  </td>
                </tr>
              ) : (
                history.map((trade: any) => (
                  <tr key={trade.id}>
                    <td>#{trade.id}</td>
                    <td>{new Date(trade.time).toLocaleTimeString()}</td>
                    <td className={trade.side === 'BUY' ? 'text-success' : 'text-danger'}>
                      {trade.side}
                    </td>
                    <td>${trade.price}</td>
                    <td>{trade.quantity}</td>
                    <td>
                      <span className="px-2 py-1 rounded-full bg-slate-700 text-xs">
                        {trade.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
