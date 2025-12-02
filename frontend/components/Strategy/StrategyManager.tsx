import { Grid, Title, Text, Group, Button, TextInput, Select } from '@mantine/core';
import { Search, Filter } from 'lucide-react';
import { useState } from 'react';
import { StrategyCard } from './StrategyCard';

interface StrategyManagerProps {
  strategies: any[];
  onActivate: (id: string) => void;
  onDeactivate: (id: string) => void;
}

export function StrategyManager({ strategies, onActivate, onDeactivate }: StrategyManagerProps) {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);

  const filteredStrategies = strategies.filter(s => {
    const matchesSearch = s.name.toLowerCase().includes(search.toLowerCase()) || 
                          s.description.toLowerCase().includes(search.toLowerCase());
    const matchesType = typeFilter ? s.type === typeFilter : true;
    return matchesSearch && matchesType;
  });

  const strategyTypes = Array.from(new Set(strategies.map(s => s.type)));

  return (
    <div>
      <Group justify="space-between" mb="lg">
        <div>
          <Title order={3}>Strategy Management</Title>
          <Text c="dimmed" size="sm">Manage and deploy trading algorithms</Text>
        </div>
        <Group>
          <TextInput 
            placeholder="Search strategies..." 
            leftSection={<Search size={16} />}
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
          />
          <Select
            placeholder="Filter by type"
            data={strategyTypes}
            value={typeFilter}
            onChange={setTypeFilter}
            clearable
            leftSection={<Filter size={16} />}
            w={150}
          />
        </Group>
      </Group>

      <Grid>
        {filteredStrategies.map(strategy => (
          <Grid.Col key={strategy.id} span={{ base: 12, md: 6, lg: 4 }}>
            <StrategyCard 
              strategy={strategy}
              onActivate={onActivate}
              onDeactivate={onDeactivate}
              onViewDetails={(id) => console.log('View details', id)}
            />
          </Grid.Col>
        ))}
      </Grid>

      {filteredStrategies.length === 0 && (
        <Text c="dimmed" ta="center" mt="xl">No strategies found matching your criteria.</Text>
      )}
    </div>
  );
}
