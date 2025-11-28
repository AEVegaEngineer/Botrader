"use client";

import React from 'react';
import { Container, Title, Stack } from '@mantine/core';
import { useBotData } from '../../hooks/useBotData';
import { PerformanceSummary } from '../../components/Dashboard/PerformanceSummary';
import { EquityCurve } from '../../components/Dashboard/EquityCurve';

export default function PerformanceDashboard() {
  const { performance } = useBotData();

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Title order={2}>Performance Metrics</Title>
        
        <PerformanceSummary 
          wins={performance.wins} 
          losses={performance.losses} 
          totalPnl={performance.total_pnl} 
          winRate={performance.win_rate} 
        />

        <EquityCurve data={performance.equity_curve} />
      </Stack>
    </Container>
  );
}
