import { Paper, Text, Table, Badge, ScrollArea, Group, ActionIcon, TextInput } from '@mantine/core';
import { History, Search, Download } from 'lucide-react';
import { useState } from 'react';

interface Intervention {
  id: string;
  timestamp: string;
  type: string;
  user: string;
  action: string;
  reason: string;
}

interface InterventionLogProps {
  interventions: Intervention[];
}

export function InterventionLog({ interventions }: InterventionLogProps) {
  const [search, setSearch] = useState('');

  const filtered = interventions.filter(i => 
    i.action.toLowerCase().includes(search.toLowerCase()) || 
    i.reason.toLowerCase().includes(search.toLowerCase()) ||
    i.user.toLowerCase().includes(search.toLowerCase())
  );

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'emergency_stop': return 'red';
      case 'resume_trading': return 'green';
      case 'strategy_change': return 'blue';
      case 'manual_trade': return 'orange';
      default: return 'gray';
    }
  };

  return (
    <Paper withBorder p="md" radius="md">
      <Group justify="space-between" mb="md">
        <Group gap="xs">
          <History size={20} />
          <Text size="lg" fw={600}>Audit Log</Text>
        </Group>
        <Group>
          <TextInput 
            placeholder="Search logs..." 
            leftSection={<Search size={14} />}
            size="xs"
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
          />
          <ActionIcon variant="light" color="gray" size="input-xs">
            <Download size={14} />
          </ActionIcon>
        </Group>
      </Group>

      <ScrollArea h={300}>
        <Table striped highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Time</Table.Th>
              <Table.Th>Type</Table.Th>
              <Table.Th>Action</Table.Th>
              <Table.Th>User</Table.Th>
              <Table.Th>Reason</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {filtered.map((item) => (
              <Table.Tr key={item.id}>
                <Table.Td style={{ whiteSpace: 'nowrap' }}>
                  <Text size="xs" c="dimmed">
                    {new Date(item.timestamp).toLocaleString()}
                  </Text>
                </Table.Td>
                <Table.Td>
                  <Badge size="xs" color={getTypeColor(item.type)} variant="light">
                    {item.type.replace('_', ' ')}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  <Text size="sm">{item.action}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs">{item.user}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs" c="dimmed" style={{ maxWidth: 200 }} truncate="end">
                    {item.reason}
                  </Text>
                </Table.Td>
              </Table.Tr>
            ))}
            {filtered.length === 0 && (
              <Table.Tr>
                <Table.Td colSpan={5} ta="center">
                  <Text c="dimmed" size="sm" py="md">No logs found</Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Paper>
  );
}
