"use client";

import React, { useState } from 'react';
import { Container, Grid, Group, Title, Tabs, Paper, Text, Stack, Button, Modal } from '@mantine/core';
import { Activity, LayoutDashboard, Brain, ShieldAlert, History, Settings } from 'lucide-react';
import { useBotData } from '../hooks/useBotData';
import { useAdvancedDashboard } from '../hooks/useAdvancedDashboard';

// Components
import { BotControls } from '../components/Dashboard/BotControls';
import { RiskMonitor } from '../components/Dashboard/RiskMonitor';
import { PerformanceMetricsPanel } from '../components/Dashboard/PerformanceMetricsPanel';
import { StrategyManager } from '../components/Strategy/StrategyManager';
import { AIInsights } from '../components/Explainability/AIInsights';
import { InterventionLog } from '../components/Dashboard/InterventionLog';
import { PriceChart } from '../components/Dashboard/PriceChart';
import { TradeTable } from '../components/Dashboard/TradeTable';
import { ThemeToggle } from '../components/ThemeToggle';
import { StatusBadge } from '../components/Dashboard/StatusBadge';

export default function Dashboard() {
  const { status, history, price, priceData, smaData, showSMA, setShowSMA, startBot, stopBot, interval, setInterval } = useBotData();
  const { 
    riskStatus, 
    performanceMetrics, 
    strategies, 
    interventions,
    actions 
  } = useAdvancedDashboard();

  const [activeTab, setActiveTab] = useState<string | null>('overview');

  return (
    <Container size="xl" py="xl">
      {/* Header */}
      <Group justify="space-between" mb="xl">
        <Group>
          <Activity size={32} className="text-blue-500" />
          <div>
            <Title order={1}>Botrader Pro</Title>
            <Text size="sm" c="dimmed">Advanced Algorithmic Trading System</Text>
          </div>
        </Group>
        <Group>
          <StatusBadge status={status.status} isRunning={status.is_running} />
          <ThemeToggle />
        </Group>
      </Group>

      {/* Main Controls & Risk Header */}
      <Grid mb="xl">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <BotControls 
            isRunning={status.is_running}
            isHalted={riskStatus?.is_halted}
            circuitBreakerReason={riskStatus?.circuit_breaker?.reason}
            onStart={startBot} 
            onStop={stopBot}
            onEmergencyStop={actions.emergencyStop}
            onResume={actions.resumeTrading}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <Paper withBorder p="md" radius="md" h="100%" bg={riskStatus?.is_halted ? 'red.9' : undefined}>
            <Stack gap="xs" justify="center" h="100%">
              <Text size="sm" fw={600} c={riskStatus?.is_halted ? 'red.1' : 'dimmed'}>SYSTEM STATUS</Text>
              <Group>
                <ShieldAlert size={24} color={riskStatus?.is_halted ? 'white' : 'green'} />
                <Title order={3} c={riskStatus?.is_halted ? 'white' : undefined}>
                  {riskStatus?.is_halted ? 'CIRCUIT BREAKER ACTIVE' : 'SYSTEM NOMINAL'}
                </Title>
              </Group>
            </Stack>
          </Paper>
        </Grid.Col>
      </Grid>

      {/* Tabs Navigation */}
      <Tabs value={activeTab} onChange={setActiveTab} mb="xl">
        <Tabs.List>
          <Tabs.Tab value="overview" leftSection={<LayoutDashboard size={16} />}>
            Overview
          </Tabs.Tab>
          <Tabs.Tab value="performance" leftSection={<Activity size={16} />}>
            Performance
          </Tabs.Tab>
          <Tabs.Tab value="strategies" leftSection={<Settings size={16} />}>
            Strategies
          </Tabs.Tab>
          <Tabs.Tab value="insights" leftSection={<Brain size={16} />}>
            AI Insights
          </Tabs.Tab>
          <Tabs.Tab value="audit" leftSection={<History size={16} />}>
            Audit Log
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="overview" pt="xl">
          <Grid>
            <Grid.Col span={{ base: 12, lg: 8 }}>
              <Stack gap="xl">
                <PriceChart 
                  data={priceData} 
                  interval={interval} 
                  smaData={smaData}
                  showSMA={showSMA}
                  onIntervalChange={setInterval}
                  onToggleSMA={setShowSMA}
                />
                <TradeTable trades={history} />
              </Stack>
            </Grid.Col>
            <Grid.Col span={{ base: 12, lg: 4 }}>
              <Stack gap="xl">
                <RiskMonitor 
                  metrics={riskStatus?.metrics} 
                  limits={riskStatus?.limits}
                  isHalted={riskStatus?.is_halted}
                />
                <InterventionLog interventions={interventions.slice(0, 5)} />
              </Stack>
            </Grid.Col>
          </Grid>
        </Tabs.Panel>

        <Tabs.Panel value="performance" pt="xl">
          <PerformanceMetricsPanel metrics={performanceMetrics} />
        </Tabs.Panel>

        <Tabs.Panel value="strategies" pt="xl">
          <StrategyManager />
        </Tabs.Panel>

        <Tabs.Panel value="insights" pt="xl">
          <AIInsights />
        </Tabs.Panel>

        <Tabs.Panel value="audit" pt="xl">
          <InterventionLog interventions={interventions} />
        </Tabs.Panel>
      </Tabs>
    </Container>
  );
}
