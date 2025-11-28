import { Paper, Text, Group, ThemeIcon } from '@mantine/core';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  color?: string;
}

export function MetricCard({ title, value, icon: Icon, color = 'blue' }: MetricCardProps) {
  return (
    <Paper withBorder p="md" radius="md">
      <Group justify="space-between">
        <Text size="xs" c="dimmed" fw={700} tt="uppercase">
          {title}
        </Text>
        <ThemeIcon color={color} variant="light" size="lg" radius="md">
          <Icon size="1.2rem" />
        </ThemeIcon>
      </Group>

      <Group align="flex-end" gap="xs" mt={25}>
        <Text fw={700} size="xl">
          {value}
        </Text>
      </Group>
    </Paper>
  );
}
