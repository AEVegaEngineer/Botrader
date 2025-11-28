import { Button, Group, Paper, Text, Stack } from '@mantine/core';
import { Play, Square } from 'lucide-react';

interface BotControlsProps {
  isRunning: boolean;
  onStart: () => void;
  onStop: () => void;
}

export function BotControls({ isRunning, onStart, onStop }: BotControlsProps) {
  return (
    <Paper withBorder p="md" radius="md">
      <Text size="lg" fw={600} mb="md">Controls</Text>
      
      <Stack>
        {isRunning ? (
          <Button 
            color="red" 
            fullWidth 
            leftSection={<Square size={20} />}
            onClick={onStop}
          >
            Stop Bot
          </Button>
        ) : (
          <Button 
            color="green" 
            fullWidth 
            leftSection={<Play size={20} />}
            onClick={onStart}
          >
            Start Bot
          </Button>
        )}

        <Paper bg="dark.6" p="sm" radius="sm">
          <Text size="sm" fw={600} c="dimmed" mb="xs">Strategy Config</Text>
          <Group justify="space-between" mb={4}>
            <Text size="sm">Symbol:</Text>
            <Text size="sm" fw={500}>BTCUSDT</Text>
          </Group>
          <Group justify="space-between" mb={4}>
            <Text size="sm">Interval:</Text>
            <Text size="sm" fw={500}>1m</Text>
          </Group>
          <Group justify="space-between">
            <Text size="sm">Mode:</Text>
            <Text size="sm" c="yellow">Testnet</Text>
          </Group>
        </Paper>
      </Stack>
    </Paper>
  );
}
