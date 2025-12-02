import { Button, Group, Paper, Text, Stack, Badge, Modal } from '@mantine/core';
import { Play, Square, AlertTriangle, RefreshCw } from 'lucide-react';
import { useState } from 'react';

interface BotControlsProps {
  isRunning: boolean;
  isHalted?: boolean;
  circuitBreakerReason?: string;
  onStart: () => void;
  onStop: () => void;
  onEmergencyStop?: () => void;
  onResume?: () => void;
}

export function BotControls({ 
  isRunning, 
  isHalted = false,
  circuitBreakerReason = '',
  onStart, 
  onStop,
  onEmergencyStop,
  onResume
}: BotControlsProps) {
  const [emergencyModalOpen, setEmergencyModalOpen] = useState(false);
  const [resumeModalOpen, setResumeModalOpen] = useState(false);

  const handleEmergencyStop = () => {
    setEmergencyModalOpen(false);
    if (onEmergencyStop) {
      onEmergencyStop();
    }
  };

  const handleResume = () => {
    setResumeModalOpen(false);
    if (onResume) {
      onResume();
    }
  };

  return (
    <>
      <Paper withBorder p="md" radius="md">
        <Group justify="space-between" mb="md">
          <Text size="lg" fw={600}>Controls</Text>
          {isHalted && (
            <Badge color="red" variant="filled" leftSection={<AlertTriangle size={14} />}>
              HALTED
            </Badge>
          )}
        </Group>
        
        <Stack>
          {isHalted ? (
            <>
              <Paper bg="red.9" p="sm" radius="sm" mb="xs">
                <Text size="sm" fw={600} c="red.2" mb="xs">Circuit Breaker Active</Text>
                <Text size="xs" c="red.3">{circuitBreakerReason || 'Trading halted'}</Text>
              </Paper>
              
              <Button 
                color="yellow" 
                fullWidth 
                leftSection={<RefreshCw size={20} />}
                onClick={() => setResumeModalOpen(true)}
                variant="outline"
              >
                Resume Trading
              </Button>
            </>
          ) : isRunning ? (
            <>
              <Button 
                color="red" 
                fullWidth 
                leftSection={<Square size={20} />}
                onClick={onStop}
              >
                Stop Bot
              </Button>
              
              <Button 
                color="orange" 
                fullWidth 
                leftSection={<AlertTriangle size={20} />}
                onClick={() => setEmergencyModalOpen(true)}
                variant="outline"
              >
                Emergency Stop
              </Button>
            </>
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

      {/* Emergency Stop Confirmation Modal */}
      <Modal
        opened={emergencyModalOpen}
        onClose={() => setEmergencyModalOpen(false)}
        title="⚠️ Emergency Stop"
        centered
        overlayProps={{ backgroundOpacity: 0.55, blur: 3 }}
      >
        <Stack>
          <Text size="sm">
            This will immediately halt all trading and close all open positions.
          </Text>
          <Text size="sm" c="dimmed">
            This action cannot be undone. The trading bot will remain stopped until you manually resume.
          </Text>
          <Group justify="flex-end" mt="md">
            <Button variant="default" onClick={() => setEmergencyModalOpen(false)}>
              Cancel
            </Button>
            <Button color="red" onClick={handleEmergencyStop}>
              Confirm Emergency Stop
            </Button>
          </Group>
        </Stack>
      </Modal>

      {/* Resume Trading Confirmation Modal */}
      <Modal
        opened={resumeModalOpen}
        onClose={() => setResumeModalOpen(false)}
        title="Resume Trading"
        centered
      >
        <Stack>
          <Text size="sm">
            Are you sure you want to resume trading?
          </Text>
          <Text size="sm" c="dimmed">
            This will reset the circuit breaker and allow new trades to be executed.
          </Text>
          <Group justify="flex-end" mt="md">
            <Button variant="default" onClick={() => setResumeModalOpen(false)}>
              Cancel
            </Button>
            <Button color="green" onClick={handleResume}>
              Resume Trading
            </Button>
          </Group>
        </Stack>
      </Modal>
    </>
  );
}
