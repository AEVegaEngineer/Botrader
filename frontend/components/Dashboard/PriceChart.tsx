'use client';

import { Paper, Text, Group, Grid, Button, Modal, SimpleGrid } from '@mantine/core';
import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { IndicatorControls } from '../Chart/IndicatorControls';
import { Calendar } from 'lucide-react';

// Dynamic import for ApexCharts to avoid SSR issues
const ReactApexChart = dynamic(() => import('react-apexcharts'), { ssr: false });

interface PriceChartProps {
  data: any[];
  interval: string;
  smaData?: any[];
  showSMA?: boolean;
  emaData?: any[];
  showEMA?: boolean;
  bbData?: any[];
  showBB?: boolean;
  rsiData?: any[];
  showRSI?: boolean;
  onIntervalChange: (interval: string) => void;
  onToggleSMA?: (show: boolean) => void;
  onToggleEMA?: (show: boolean) => void;
  onToggleBB?: (show: boolean) => void;
  onToggleRSI?: (show: boolean) => void;
}

const INTERVALS = [
  '1m', '3m', '5m', '15m', '30m', 
  '1h', '2h', '4h', '6h', '8h', '12h', 
  '1d', '3d', '1w', '1M'
];

export function PriceChart({ data, interval, smaData = [], showSMA = false, emaData = [], showEMA = false, bbData = [], showBB = false, rsiData = [], showRSI = false, onIntervalChange, onToggleSMA, onToggleEMA, onToggleBB, onToggleRSI }: PriceChartProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  const [indicators, setIndicators] = useState({
    sma: false,
    ema: false,
    bollinger: false,
    rsi: false,
    macd: false
  });

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Sync local indicators state with showSMA, showEMA, showBB, and showRSI props
  useEffect(() => {
    setIndicators(prev => ({ ...prev, sma: showSMA }));
  }, [showSMA]);

  useEffect(() => {
    setIndicators(prev => ({ ...prev, ema: showEMA }));
  }, [showEMA]);

  useEffect(() => {
    setIndicators(prev => ({ ...prev, bollinger: showBB }));
  }, [showBB]);

  useEffect(() => {
    setIndicators(prev => ({ ...prev, rsi: showRSI }));
  }, [showRSI]);

  const handleToggle = (indicator: string, value: boolean) => {
    setIndicators(prev => ({ ...prev, [indicator]: value }));
    if (indicator === 'sma' && onToggleSMA) {
      onToggleSMA(value);
    }
    if (indicator === 'ema' && onToggleEMA) {
      onToggleEMA(value);
    }
    if (indicator === 'bollinger' && onToggleBB) {
      onToggleBB(value);
    }
    if (indicator === 'rsi' && onToggleRSI) {
      onToggleRSI(value);
    }
  };

  const handleIntervalSelect = (newInterval: string) => {
    onIntervalChange(newInterval);
    setIsModalOpen(false);
  };

  // Format data for ApexCharts
  const series: any[] = [{
    name: 'Price',
    data: data.map(d => ({
      x: new Date(d.time).getTime(),
      y: [d.open, d.high, d.low, d.close]
    }))
  }];

  // Add SMA line series if enabled
  if (showSMA && smaData.length > 0) {
    series.push({
      name: 'SMA 20',
      type: 'line',
      data: smaData.map(d => ({
        x: new Date(d.timestamp).getTime(),
        y: d.value
      }))
    });
  }

  // Add EMA line series if enabled
  if (showEMA && emaData.length > 0) {
    series.push({
      name: 'EMA 50',
      type: 'line',
      data: emaData.map(d => ({
        x: new Date(d.timestamp).getTime(),
        y: d.value
      }))
    });
  }

  // Add Bollinger Bands line series if enabled
  if (showBB && bbData.length > 0) {
    series.push({
      name: 'BB Upper',
      type: 'line',
      data: bbData.map(d => ({
        x: new Date(d.timestamp).getTime(),
        y: d.upper
      }))
    });
    series.push({
      name: 'BB Middle',
      type: 'line',
      data: bbData.map(d => ({
        x: new Date(d.timestamp).getTime(),
        y: d.middle
      }))
    });
    series.push({
      name: 'BB Lower',
      type: 'line',
      data: bbData.map(d => ({
        x: new Date(d.timestamp).getTime(),
        y: d.lower
      }))
    });
  }

  const options: any = {
    chart: {
      type: 'candlestick',
      height: 350,
      background: 'transparent',
      toolbar: {
        show: false
      }
    },
    title: {
      text: undefined,
      align: 'left'
    },
    xaxis: {
      type: 'datetime',
      labels: {
        style: {
          colors: '#94a3b8'
        }
      },
      axisBorder: {
        show: false
      },
      axisTicks: {
        show: false
      }
    },
    yaxis: {
      tooltip: {
        enabled: true
      },
      labels: {
        style: {
          colors: '#94a3b8'
        },
        formatter: (value: number) => value.toFixed(2)
      }
    },
    grid: {
      borderColor: '#334155',
      strokeDashArray: 3,
      opacity: 0.3
    },
    theme: {
      mode: 'dark'
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#10b981',
          downward: '#ef4444'
        }
      }
    },
    stroke: {
      width: [0, 2, 2, 1, 1, 1]
    },
    colors: ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#8b5cf6', '#8b5cf6'],
    legend: {
      show: true,
      position: 'top',
      labels: {
        colors: '#94a3b8'
      }
    }
  };

  return (
    <Grid>
      <Grid.Col span={{ base: 12, md: 9 }}>
        <Paper withBorder p="md" radius="md" h="100%">
          <Group justify="space-between" mb="md">
            <Text size="lg" fw={600}>Live Price Action (BTC/USDT)</Text>
            <Button 
              variant="light" 
              size="xs" 
              leftSection={<Calendar size={14} />}
              onClick={() => setIsModalOpen(true)}
            >
              Interval: {interval}
            </Button>
          </Group>
          
          <div style={{ width: '100%', height: 400 }}>
            {isMounted && (
              <ReactApexChart 
                options={options} 
                series={series} 
                type="candlestick" 
                height={350} 
              />
            )}
          </div>

          {/* RSI Sub-Chart */}
          {showRSI && rsiData.length > 0 && isMounted && (
            <div style={{ width: '100%', marginTop: '16px' }}>
              <Text size="sm" fw={600} mb="xs" c="dimmed">RSI (14)</Text>
              <ReactApexChart
                options={{
                  chart: {
                    type: 'line',
                    height: 150,
                    background: 'transparent',
                    toolbar: { show: false },
                    group: 'indicators'
                  },
                  xaxis: {
                    type: 'datetime',
                    labels: {
                      style: { colors: '#94a3b8' }
                    },
                    axisBorder: { show: false },
                    axisTicks: { show: false }
                  },
                  yaxis: {
                    min: 0,
                    max: 100,
                    tickAmount: 5,
                    labels: {
                      style: { colors: '#94a3b8' },
                      formatter: (value: number) => value.toFixed(0)
                    }
                  },
                  grid: {
                    borderColor: '#334155',
                    strokeDashArray: 3,
                    opacity: 0.3
                  },
                  theme: {
                    mode: 'dark'
                  },
                  stroke: {
                    width: 2
                  },
                  colors: ['#8b5cf6'],
                  annotations: {
                    yaxis: [
                      {
                        y: 70,
                        borderColor: '#ef4444',
                        strokeDashArray: 4,
                        label: {
                          text: 'Overbought',
                          style: { color: '#ef4444', fontSize: '12px' }
                        }
                      },
                      {
                        y: 30,
                        borderColor: '#10b981',
                        strokeDashArray: 4,
                        label: {
                          text: 'Oversold',
                          style: { color: '#10b981', fontSize: '12px' }
                        }
                      }
                    ]
                  },
                  legend: {
                    show: false
                  }
                }}
                series={[{
                  name: 'RSI (14)',
                  data: rsiData.map(d => ({
                    x: new Date(d.timestamp).getTime(),
                    y: d.value
                  }))
                }]}
                type="line"
                height={150}
              />
            </div>
          )}
          
        </Paper>
      </Grid.Col>
      
      <Grid.Col span={{ base: 12, md: 3 }}>
        <IndicatorControls indicators={indicators} onToggle={handleToggle} />
      </Grid.Col>

      <Modal 
        opened={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        title="Select Time Interval"
        centered
      >
        <SimpleGrid cols={4}>
          {INTERVALS.map((int) => (
            <Button 
              key={int} 
              variant={interval === int ? "filled" : "outline"} 
              onClick={() => handleIntervalSelect(int)}
              size="sm"
            >
              {int}
            </Button>
          ))}
        </SimpleGrid>
      </Modal>
    </Grid>
  );
}
