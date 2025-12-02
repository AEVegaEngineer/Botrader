import { Paper, Text, Group, Badge, Button, Stack, ThemeIcon, Modal, Grid, RingProgress } from '@mantine/core';
import { Play, Pause, Settings, Activity, TrendingUp, AlertTriangle } from 'lucide-react';
import { useState } from 'react';

interface StrategyStats {
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  win_rate: number;
  total_trades: number;
}

interface Strategy {
  id: string;
  name: string;
  type: string;
  version: string;
  description: string;
  is_active: boolean;
  backtest_stats?: StrategyStats;
}

interface StrategyCardProps {
  strategy: Strategy;
  onActivate: (id: string) => void;
  onDeactivate: (id: string) => void;
  onViewDetails: (id: string) => void;
}

export function StrategyCard({ strategy, onActivate, onDeactivate, onViewDetails }: StrategyCardProps) {
  const [confirmModalOpen, setConfirmModalOpen] = useState(false);

  const handleToggle = () => {
    if (strategy.is_active) {
      onDeactivate(strategy.id);
    } else {
      setConfirmModalOpen(true);
    }
  };

  const handleConfirmActivate = () => {
    onActivate(strategy.id);
    setConfirmModalOpen(false);
  };

  const stats = strategy.backtest_stats;

  return (
    <>
      <Paper withBorder p="md" radius="md" style={{ borderColor: strategy.is_active ? 'var(--mantine-color-green-6)' : undefined, borderWidth: strategy.is_active ? 2 : 1 }}>
        <Group justify="space-between" mb="xs">
          <Group gap="xs">
            <ThemeIcon color={strategy.is_active ? 'green' : 'gray'} variant="light" size="lg">
              <Activity size={20} />
            </ThemeIcon>
            <div>
              <Text fw={600}>{strategy.name}</Text>
              <Group gap={6}>
                <Badge size="xs" variant="dot" color={strategy.is_active ? 'green' : 'gray'}>
                  {strategy.is_active ? 'Active' : 'Inactive'}
                </Badge>
                <Badge size="xs" variant="outline">{strategy.type}</Badge>
                <Text size="xs" c="dimmed">v{strategy.version}</Text>
              </Group>
            </div>
          </Group>
          <Button 
            size="xs" 
            variant={strategy.is_active ? 'light' : 'filled'} 
            color={strategy.is_active ? 'red' : 'green'}
            onClick={handleToggle}
            leftSection={strategy.is_active ? <Pause size={14} /> : <Play size={14} />}
          >
            {strategy.is_active ? 'Stop' : 'Activate'}
          </Button>
        </Group>

        <Text size="sm" c="dimmed" lineClamp={2} mb="md" h={40}>
          {strategy.description}
        </Text>

        {stats ? (
          <Grid gutter="xs" mb="md">
            <Grid.Col span={4}>
              <Paper bg="dark.6" p={6} radius="sm">
                <Text size="xs" c="dimmed">Sharpe</Text>
                <Text fw={600} size="sm">{stats.sharpe_ratio.toFixed(2)}</Text>
              </Paper>
            </Grid.Col>
            <Grid.Col span={4}>
              <Paper bg="dark.6" p={6} radius="sm">
                <Text size="xs" c="dimmed">Return</Text>
                <Text fw={600} size="sm" c={stats.total_return > 0 ? 'green' : 'red'}>
                  {(stats.total_return * 100).toFixed(0)}%
                </Text>
              </Paper>
            </Grid.Col>
            <Grid.Col span={4}>
              <Paper bg="dark.6" p={6} radius="sm">
                <Text size="xs" c="dimmed">Max DD</Text>
                <Text fw={600} size="sm" c="red">
                  {(stats.max_drawdown * 100).toFixed(0)}%
                </Text>
              </Paper>
            </Grid.Col>
          </Grid>
        ) : (
          <Paper bg="dark.6" p="xs" mb="md" ta="center">
            <Text size="xs" c="dimmed">No backtest data available</Text>
          </Paper>
        )}

        <Button fullWidth variant="default" size="xs" onClick={() => onViewDetails(strategy.id)}>
          View Details
        </Button>
      </Paper>

      <Modal 
        opened={confirmModalOpen} 
        onClose={() => setConfirmModalOpen(false)}
        title="Activate Strategy"
        centered
      >
        <Stack>
          <Paper bg="yellow.9" p="sm">
            <Group gap="sm" align="flex-start">
              <AlertTriangle color="orange" size={20} />
              <div>
                <Text size="sm" fw={600} c="yellow.1">Warning</Text>
                <Text size="sm" c="yellow.1">
                  Activating <b>{strategy.name}</b> will stop the currently active strategy. 
                  Ensure you have reviewed the backtest results and risk parameters.
                </Text>
              </div>
            </Group>
          </Paper>
          
          <Group justify="flex-end" mt="md">
            <Button variant="default" onClick={() => setConfirmModalOpen(false)}>Cancel</Button>
            <Button color="green" onClick={handleConfirmActivate}>Confirm Activation</Button>
          </Group>
        </Stack>
      </Modal>
    </>
  );
}
