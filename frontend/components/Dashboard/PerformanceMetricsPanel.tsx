import { Paper, Text, Grid, Stack, Group, Progress, Badge, Divider } from '@mantine/core';
import { TrendingUp, TrendingDown, Target, Activity, DollarSign, BarChart3 } from 'lucide-react';

interface PerformanceMetricsProps {
  metrics?: {
    total_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    max_drawdown: number;
    current_drawdown: number;
    total_trades: number;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
    profit_factor: number;
    turnover: number;
    total_fees: number;
    fees_pct_of_pnl: number;
  };
}

const MetricCard = ({ 
  icon: Icon, 
  label, 
  value, 
  subtitle, 
  color = 'blue',
  trend
}: any) => (
  <Paper withBorder p="md" radius="md">
    <Group justify="apart" mb="xs">
      <Group gap="xs">
        <Icon size={20} color={`var(--mantine-color-${color}-6)`} />
        <Text size="sm" c="dimmed">{label}</Text>
      </Group>
      {trend && (
        <Badge size="sm" color={trend > 0 ? 'green' : 'red'} variant="light">
          {trend > 0 ? '+' : ''}{(trend * 100).toFixed(1)}%
        </Badge>
      )}
    </Group>
    <Text size="xl" fw={700} c={color}>
      {value}
    </Text>
    {subtitle && (
      <Text size="xs" c="dimmed" mt={4}>{subtitle}</Text>
    )}
  </Paper>
);

export function PerformanceMetricsPanel({ metrics }: PerformanceMetricsProps) {
  if (!metrics) {
    return (
      <Paper withBorder p="md" radius="md">
        <Text size="sm" c="dimmed">Loading performance metrics...</Text>
      </Paper>
    );
  }

  const isProfit = metrics.total_return > 0;
  const riskAdjustedGood = metrics.sharpe_ratio > 1.5;

  return (
    <Stack gap="md">
      <Paper withBorder p="md" radius="md">
        <Group justify="space-between" mb="md">
          <Text size="lg" fw={600}>Performance Metrics</Text>
          <Badge 
            size="lg" 
            color={isProfit ? 'green' : 'red'}
            leftSection={isProfit ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
          >
            {isProfit ? '+' : ''}{(metrics.total_return * 100).toFixed(2)}% Total Return
          </Badge>
        </Group>

        <Grid gutter="md">
          {/* Returns */}
          <Grid.Col span={3}>
            <MetricCard
              icon={TrendingUp}
              label="Annualized Return"
              value={`${(metrics.annualized_return * 100).toFixed(2)}%`}
              color={metrics.annualized_return > 0 ? 'green' : 'red'}
            />
          </Grid.Col>

          {/* Sharpe Ratio */}
          <Grid.Col span={3}>
            <MetricCard
              icon={Target}
              label="Sharpe Ratio"
              value={metrics.sharpe_ratio.toFixed(2)}
              subtitle="Risk-adjusted return"
              color={riskAdjustedGood ? 'green' : 'yellow'}
            />
          </Grid.Col>

          {/* Sortino Ratio */}
          <Grid.Col span={3}>
            <MetricCard
              icon={Activity}
              label="Sortino Ratio"
              value={metrics.sortino_ratio.toFixed(2)}
              subtitle="Downside risk adjusted"
              color={metrics.sortino_ratio > 2 ? 'green' : 'yellow'}
            />
          </Grid.Col>

          {/* Calmar Ratio */}
          <Grid.Col span={3}>
            <MetricCard
              icon={BarChart3}
              label="Calmar Ratio"
              value={metrics.calmar_ratio.toFixed(2)}
              subtitle="Return / Max DD"
              color={metrics.calmar_ratio > 3 ? 'green' : 'yellow'}
            />
          </Grid.Col>
        </Grid>

        <Divider my="md" />

        {/* Drawdown */}
        <Text size="sm" fw={500} mb="xs">Drawdown</Text>
        <Grid gutter="md" mb="md">
          <Grid.Col span={6}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed" mb={4}>Max Drawdown</Text>
              <Text size="lg" fw={700} c="red">
                {(metrics.max_drawdown * 100).toFixed(2)}%
              </Text>
              <Progress 
                value={Math.abs(metrics.max_drawdown) * 100} 
                color="red"
                size="sm"
                mt="xs"
              />
            </Paper>
          </Grid.Col>
          <Grid.Col span={6}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed" mb={4}>Current Drawdown</Text>
              <Text size="lg" fw={700} c={metrics.current_drawdown < 0 ? 'orange' : 'green'}>
                {(metrics.current_drawdown * 100).toFixed(2)}%
              </Text>
              <Progress 
                value={Math.abs(metrics.current_drawdown) * 100} 
                color={metrics.current_drawdown < 0 ? 'orange' : 'green'}
                size="sm"
                mt="xs"
              />
            </Paper>
          </Grid.Col>
        </Grid>

        <Divider my="md" />

        {/* Trading Statistics */}
        <Text size="sm" fw={500} mb="xs">Trading Statistics</Text>
        <Grid gutter="md">
          <Grid.Col span={3}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Total Trades</Text>
              <Text size="lg" fw={700}>{metrics.total_trades}</Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={3}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Win Rate</Text>
              <Text size="lg" fw={700} c={metrics.win_rate > 0.5 ? 'green' : 'red'}>
                {(metrics.win_rate * 100).toFixed(1)}%
              </Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={3}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Avg Win / Loss</Text>
              <Text size="sm" fw={600} c="green">
                ${metrics.avg_win.toFixed(0)}
              </Text>
              <Text size="sm" fw={600} c="red">
                ${metrics.avg_loss.toFixed(0)}
              </Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={3}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Profit Factor</Text>
              <Text size="lg" fw={700} c={metrics.profit_factor > 1.5 ? 'green' : 'yellow'}>
                {metrics.profit_factor.toFixed(2)}
              </Text>
            </Paper>
          </Grid.Col>
        </Grid>

        <Divider my="md" />

        {/* Costs */}
        <Text size="sm" fw={500} mb="xs">Trading Costs</Text>
        <Grid gutter="md">
          <Grid.Col span={4}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Turnover</Text>
              <Text size="lg" fw={700}>{metrics.turnover.toFixed(2)}x</Text>
              <Text size="xs" c="dimmed">Trading intensity</Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={4}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Total Fees</Text>
              <Text size="lg" fw={700}>${metrics.total_fees.toFixed(2)}</Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={4}>
            <Paper bg="dark.6" p="sm" radius="sm">
              <Text size="xs" c="dimmed">Fees % of PnL</Text>
              <Text size="lg" fw={700} c={metrics.fees_pct_of_pnl > 0.1 ? 'orange' : 'green'}>
                {(metrics.fees_pct_of_pnl * 100).toFixed(1)}%
              </Text>
            </Paper>
          </Grid.Col>
        </Grid>
      </Paper>
    </Stack>
  );
}
