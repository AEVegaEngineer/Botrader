import { Grid } from '@mantine/core';
import { TrendingUp, TrendingDown, Percent, DollarSign } from 'lucide-react';
import { MetricCard } from './MetricCard';

interface PerformanceSummaryProps {
  wins: number;
  losses: number;
  totalPnl: number;
  winRate: number;
}

export function PerformanceSummary({ wins, losses, totalPnl, winRate }: PerformanceSummaryProps) {
  return (
    <Grid>
      <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
        <MetricCard 
          title="Total PnL" 
          value={`$${totalPnl.toFixed(2)}`} 
          icon={DollarSign}
          color={totalPnl >= 0 ? 'green' : 'red'}
        />
      </Grid.Col>
      <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
        <MetricCard 
          title="Win Rate" 
          value={`${winRate.toFixed(1)}%`} 
          icon={Percent}
          color="blue"
        />
      </Grid.Col>
      <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
        <MetricCard 
          title="Wins" 
          value={wins} 
          icon={TrendingUp}
          color="green"
        />
      </Grid.Col>
      <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
        <MetricCard 
          title="Losses" 
          value={losses} 
          icon={TrendingDown}
          color="red"
        />
      </Grid.Col>
    </Grid>
  );
}
