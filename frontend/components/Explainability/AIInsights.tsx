'use client';

import { Card, Title, Text, Group, Badge, Stack } from '@mantine/core';
import { Brain, TrendingUp } from 'lucide-react';

export function AIInsights() {
  return (
    <div>
      <Group justify="space-between" mb="lg">
        <div>
          <Title order={3}>AI Insights</Title>
          <Text c="dimmed" size="sm">Model information and feature importance</Text>
        </div>
      </Group>

      <Stack gap="md">
        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Group mb="xs">
            <Brain size={20} />
            <Text fw={500}>Active Model</Text>
          </Group>
          <Text size="sm" c="dimmed" mb="md">
            Transformer-based Action Classifier
          </Text>
          <Group gap="xs">
            <Badge variant="light" color="blue">Sequence Length: 64</Badge>
            <Badge variant="light" color="green">3 Actions: BUY/SELL/HOLD</Badge>
          </Group>
        </Card>

        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Group mb="xs">
            <TrendingUp size={20} />
            <Text fw={500}>Key Features</Text>
          </Group>
          <Text size="sm" c="dimmed" mb="xs">
            The model uses the following features:
          </Text>
          <Stack gap="xs">
            <Text size="sm">• OHLCV (Open, High, Low, Close, Volume)</Text>
            <Text size="sm">• Technical Indicators: RSI, MACD, Bollinger Bands, ATR</Text>
            <Text size="sm">• Moving Averages: SMA, EMA (20, 50 periods)</Text>
            <Text size="sm">• Log Returns and Volume Changes</Text>
          </Stack>
        </Card>

        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text fw={500} mb="xs">Model Architecture</Text>
          <Text size="sm" c="dimmed">
            The Transformer encoder processes sequences of market data to predict optimal trading actions.
            It uses multi-head attention to capture temporal dependencies and patterns in price movements.
          </Text>
        </Card>
      </Stack>
    </div>
  );
}
