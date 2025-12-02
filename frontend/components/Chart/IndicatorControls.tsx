import { Paper, Text, Group, Switch, Stack, MultiSelect } from '@mantine/core';
import { Settings } from 'lucide-react';

interface IndicatorControlsProps {
  indicators: {
    sma: boolean;
    ema: boolean;
    bollinger: boolean;
    rsi: boolean;
    macd: boolean;
  };
  onToggle: (indicator: string, value: boolean) => void;
}

export function IndicatorControls({ indicators, onToggle }: IndicatorControlsProps) {
  return (
    <Paper withBorder p="sm" radius="md">
      <Group gap="xs" mb="sm">
        <Settings size={16} />
        <Text size="sm" fw={600}>Chart Indicators</Text>
      </Group>
      
      <Stack gap="xs">
        <Switch 
          label="SMA (20)" 
          size="xs" 
          checked={indicators.sma}
          onChange={(e) => onToggle('sma', e.currentTarget.checked)}
        />
        <Switch 
          label="EMA (50)" 
          size="xs" 
          checked={indicators.ema}
          onChange={(e) => onToggle('ema', e.currentTarget.checked)}
        />
        <Switch 
          label="Bollinger Bands" 
          size="xs" 
          checked={indicators.bollinger}
          onChange={(e) => onToggle('bollinger', e.currentTarget.checked)}
        />
        <Switch 
          label="RSI (14)" 
          size="xs" 
          checked={indicators.rsi}
          onChange={(e) => onToggle('rsi', e.currentTarget.checked)}
        />
        <Switch 
          label="MACD" 
          size="xs" 
          checked={indicators.macd}
          onChange={(e) => onToggle('macd', e.currentTarget.checked)}
        />
      </Stack>
    </Paper>
  );
}
