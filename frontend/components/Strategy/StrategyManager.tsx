'use client';

import { Grid, Title, Text, Group, Button, Card, Badge } from '@mantine/core';
import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { CheckCircle, Circle } from 'lucide-react';

interface Strategy {
  id: string;
  name: string;
  description: string;
}

export function StrategyManager() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [activeStrategy, setActiveStrategy] = useState<string>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      const data = await api.getStrategiesNew();
      setStrategies(data.strategies || []);
      setActiveStrategy(data.active || '');
    } catch (error) {
      console.error('Failed to load strategies:', error);
    }
  };

  const handleActivate = async (id: string) => {
    setLoading(true);
    try {
      await api.setStrategy(id);
      setActiveStrategy(id);
    } catch (error) {
      console.error('Failed to activate strategy:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Group justify="space-between" mb="lg">
        <div>
          <Title order={3}>Strategy Management</Title>
          <Text c="dimmed" size="sm">Select and manage trading strategies</Text>
        </div>
      </Group>

      <Grid>
        {strategies.map(strategy => (
          <Grid.Col key={strategy.id} span={{ base: 12, md: 6 }}>
            <Card shadow="sm" padding="lg" radius="md" withBorder>
              <Group justify="space-between" mb="xs">
                <Text fw={500}>{strategy.name}</Text>
                {activeStrategy === strategy.id && (
                  <Badge color="green" variant="light">Active</Badge>
                )}
              </Group>

              <Text size="sm" c="dimmed" mb="md">
                {strategy.description}
              </Text>

              <Button
                fullWidth
                variant={activeStrategy === strategy.id ? "light" : "filled"}
                color={activeStrategy === strategy.id ? "green" : "blue"}
                onClick={() => handleActivate(strategy.id)}
                disabled={loading || activeStrategy === strategy.id}
                leftSection={activeStrategy === strategy.id ? <CheckCircle size={16} /> : <Circle size={16} />}
              >
                {activeStrategy === strategy.id ? 'Active' : 'Activate'}
              </Button>
            </Card>
          </Grid.Col>
        ))}
      </Grid>

      {strategies.length === 0 && (
        <Text c="dimmed" ta="center" mt="xl">No strategies available.</Text>
      )}
    </div>
  );
}
