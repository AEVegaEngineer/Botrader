import { Paper, Text, Group, Grid } from '@mantine/core';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, Bar } from 'recharts';
import { IndicatorControls } from '../Chart/IndicatorControls';
import { useState } from 'react';

interface PriceChartProps {
  data: any[];
}

export function PriceChart({ data }: PriceChartProps) {
  const [indicators, setIndicators] = useState({
    sma: false,
    ema: false,
    bollinger: false,
    rsi: false,
    macd: false
  });

  const handleToggle = (indicator: string, value: boolean) => {
    setIndicators(prev => ({ ...prev, [indicator]: value }));
  };

  return (
    <Grid>
      <Grid.Col span={{ base: 12, md: 9 }}>
        <Paper withBorder p="md" radius="md" h="100%">
          <Group justify="space-between" mb="md">
            <Text size="lg" fw={600}>Live Price Action</Text>
          </Group>
          
          <div style={{ width: '100%', height: 400 }}>
            <ResponsiveContainer>
              <ComposedChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={['auto', 'auto']} stroke="#94a3b8" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                
                {/* Price Line */}
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#8b5cf6" 
                  strokeWidth={2} 
                  dot={false} 
                  isAnimationActive={false}
                />

                {/* Indicators */}
                {indicators.sma && (
                  <Line type="monotone" dataKey="sma" stroke="#fbbf24" strokeWidth={1.5} dot={false} />
                )}
                {indicators.ema && (
                  <Line type="monotone" dataKey="ema" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                )}
                {indicators.bollinger && (
                  <>
                    <Line type="monotone" dataKey="upper_band" stroke="#10b981" strokeDasharray="3 3" dot={false} />
                    <Line type="monotone" dataKey="lower_band" stroke="#10b981" strokeDasharray="3 3" dot={false} />
                  </>
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Sub-charts for Oscillators */}
          {indicators.rsi && (
            <div style={{ width: '100%', height: 150, marginTop: 20 }}>
              <Text size="xs" c="dimmed" mb={4}>RSI (14)</Text>
              <ResponsiveContainer>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 100]} stroke="#94a3b8" fontSize={10} ticks={[30, 70]} />
                  <Line type="monotone" dataKey="rsi" stroke="#f472b6" dot={false} strokeWidth={1.5} />
                  {/* Overbought/Oversold lines */}
                  <line x1="0" y1="70" x2="100%" y2="70" stroke="#ef4444" strokeDasharray="3 3" />
                  <line x1="0" y1="30" x2="100%" y2="30" stroke="#10b981" strokeDasharray="3 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {indicators.macd && (
            <div style={{ width: '100%', height: 150, marginTop: 20 }}>
              <Text size="xs" c="dimmed" mb={4}>MACD</Text>
              <ResponsiveContainer>
                <ComposedChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                  <XAxis dataKey="time" hide />
                  <YAxis stroke="#94a3b8" fontSize={10} />
                  <Bar dataKey="macd_hist" fill="#94a3b8" opacity={0.5} />
                  <Line type="monotone" dataKey="macd" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                  <Line type="monotone" dataKey="signal" stroke="#f97316" dot={false} strokeWidth={1.5} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
        </Paper>
      </Grid.Col>
      
      <Grid.Col span={{ base: 12, md: 3 }}>
        <IndicatorControls indicators={indicators} onToggle={handleToggle} />
      </Grid.Col>
    </Grid>
  );
}
