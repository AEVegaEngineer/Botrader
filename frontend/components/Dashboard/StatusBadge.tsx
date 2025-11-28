import { Badge } from '@mantine/core';

interface StatusBadgeProps {
  status: string;
  isRunning: boolean;
}

export function StatusBadge({ status, isRunning }: StatusBadgeProps) {
  return (
    <Badge 
      color={isRunning ? 'green' : 'red'} 
      variant="light" 
      size="lg"
    >
      {status}
    </Badge>
  );
}
