import { Paper, Text, Group, Stack, Progress, Badge, Grid } from '@mantine/core';
import { TrendingDown, Activity, AlertTriangle, Clock } from 'lucide-react';

interface RiskMetrics {
  balance: number;
  peak_balance: number;
  drawdown_pct: number;
  daily_return_pct: number;
  current_volatility: number;
  avg_latency_ms: number;
  max_latency_ms: number;
  error_rate: number;
}

interface RiskLimits {
  max_drawdown_pct: number;
  max_daily_loss_pct: number;
  max_latency_ms: number;
  max_error_rate: number;
  max_trade_notional: number;
}

interface RiskMonitorProps {
  metrics?: RiskMetrics;
  limits?: RiskLimits;
  isHalted?: boolean;
}

export function RiskMonitor({ metrics, limits, isHalted = false }: RiskMonitorProps) {
  if (!metrics || !limits) {
    return (
      <Paper withBorder p="md" radius="md">
        <Text size="lg" fw={600} mb="md">Risk Metrics</Text>
        <Text size="sm" c="dimmed">Loading risk data...</Text>
      </Paper>
    );
  }

  const drawdownPct = (metrics.drawdown_pct * 100);
  const drawdownUsage = (drawdownPct / (limits.max_drawdown_pct * 100)) * 100;
  
  const dailyReturnPct = (metrics.daily_return_pct * 100);
  const dailyLossLimit = limits.max_daily_loss_pct * 100;
  
  const latencyUsage = (metrics.max_latency_ms / limits.max_latency_ms) * 100;
  const errorRateUsage = (metrics.error_rate / limits.max_error_rate) * 100;

  const getProgressColor = (usage: number) => {
    if (usage >= 90) return 'red';
    if (usage >= 70) return 'yellow';
    return 'green';
  };

  return (
    <Paper withBorder p="md" radius="md">
      <Group justify="space-between" mb="md">
        <Text size="lg" fw={600}>Risk Metrics</Text>
        {isHalted && (
          <Badge color="red" variant="filled">
            TRADING HALTED
          </Badge>
        )}
      </Group>

      <Stack gap="lg">
        {/* Drawdown */}
        <div>
          <Group justify="space-between" mb={4}>
            <Group gap="xs">
              <TrendingDown size={16} />
              <Text size="sm" fw={500}>Drawdown</Text>
            </Group>
            <Text size="sm" fw={600} c={drawdownUsage > 70 ? 'red' : 'dimmed'}>
              {drawdownPct.toFixed(2)}% / {(limits.max_drawdown_pct * 100).toFixed(0)}%
            </Text>
          </Group>
          <Progress 
            value={drawdownUsage} 
            color={getProgressColor(drawdownUsage)}
            size="sm"
            radius="xl"
          />
        </div>

        {/* Daily Return */}
        <div>
          <Group justify="space-between" mb={4}>
            <Group gap="xs">
              <Activity size={16} />
              <Text size="sm" fw={500}>Daily Return</Text>
            </Group>
            <Text 
              size="sm" 
              fw={600} 
              c={dailyReturnPct < 0 ? (dailyReturnPct < -dailyLossLimit ? 'red' : 'orange') : 'green'}
            >
              {dailyReturnPct >= 0 ? '+' : ''}{dailyReturnPct.toFixed(2)}%
            </Text>
          </Group>
          {dailyReturnPct < 0 && (
            <Progress 
              value={Math.min((Math.abs(dailyReturnPct) / dailyLossLimit) * 100, 100)} 
              color={Math.abs(dailyReturnPct) > dailyLossLimit ? 'red' : 'yellow'}
              size="sm"
              radius="xl"
            />
          )}
        </div>

        {/* Volatility */}
        <div>
          <Group justify="space-between" mb={4}>
            <Text size="sm" fw={500}>Portfolio Volatility</Text>
            <Text size="sm" fw={600} c="dimmed">
              {(metrics.current_volatility * 100).toFixed(2)}%
            </Text>
          </Group>
        </div>

        <Grid>
          {/* Latency */}
          <Grid.Col span={6}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Group gap="xs" mb={4}>
                <Clock size={14} />
                <Text size="xs" c="dimmed">Latency</Text>
              </Group>
              <Text size="sm" fw={600}>
                {metrics.avg_latency_ms.toFixed(0)}ms
              </Text>
              <Text size="xs" c="dimmed">
                Max: {metrics.max_latency_ms.toFixed(0)}ms
              </Text>
              {latencyUsage > 70 && (
                <Badge size="xs" color="orange" mt="xs">
                  High
                </Badge>
              )}
            </Paper>
          </Grid.Col>

          {/* Error Rate */}
          <Grid.Col span={6}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Group gap="xs" mb={4}>
                <AlertTriangle size={14} />
                <Text size="xs" c="dimmed">Error Rate</Text>
              </Group>
              <Text size="sm" fw={600}>
                {(metrics.error_rate * 100).toFixed(1)}%
              </Text>
              <Text size="xs" c="dimmed">
                Limit: {(limits.max_error_rate * 100).toFixed(0)}%
              </Text>
              {errorRateUsage > 70 && (
                <Badge size="xs" color="orange" mt="xs">
                  High
                </Badge>
              )}
            </Paper>
          </Grid.Col>
        </Grid>

        {/* Balance Info */}
        <Paper bg="dark.6" p="sm" radius="sm">
          <Text size="xs" c="dimmed" mb={4}>Balance</Text>
          <Text size="lg" fw={700}>${metrics.balance.toFixed(2)}</Text>
          <Text size="xs" c="dimmed">
            Peak: ${metrics.peak_balance.toFixed(2)}
          </Text>
        </Paper>
      </Stack>
    </Paper>
  );
}
