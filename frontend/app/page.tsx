"use client";

import React from 'react';
import { Container, Grid, Group, Title, Stack } from '@mantine/core';
import { Activity, DollarSign, TrendingUp, Clock } from 'lucide-react';
import { useBotData } from '../hooks/useBotData';
import { MetricCard } from '../components/Dashboard/MetricCard';
import { StatusBadge } from '../components/Dashboard/StatusBadge';
import { PriceChart } from '../components/Dashboard/PriceChart';
import { BotControls } from '../components/Dashboard/BotControls';
import { TradeTable } from '../components/Dashboard/TradeTable';
import { ThemeToggle } from '../components/ThemeToggle';

export default function Dashboard() {
  const { status, history, price, priceData, startBot, stopBot } = useBotData();

  return (
    <Container size="xl" py="xl">
      <Group justify="space-between" mb="xl">
        <Group>
          <Activity size={32} className="text-blue-500" />
          <Title order={1}>Botrader Dashboard</Title>
        </Group>
        <Group>
          <StatusBadge status={status.status} isRunning={status.is_running} />
          <ThemeToggle />
        </Group>
      </Group>

      <Grid mb="xl">
        <Grid.Col span={{ base: 12, sm: 4 }}>
          <MetricCard 
            title="Current Price" 
            value={`$${price.toLocaleString()}`} 
            icon={DollarSign} 
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 4 }}>
          <MetricCard 
            title="Total Trades" 
            value={history.length} 
            icon={TrendingUp} 
            color="green"
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 4 }}>
          <MetricCard 
            title="Uptime" 
            value="--:--" 
            icon={Clock} 
            color="orange"
          />
        </Grid.Col>
      </Grid>

      <Grid mb="xl">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <PriceChart data={priceData} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <BotControls 
            isRunning={status.is_running} 
            onStart={startBot} 
            onStop={stopBot} 
          />
        </Grid.Col>
      </Grid>

      <TradeTable trades={history} />
    </Container>
  );
}
