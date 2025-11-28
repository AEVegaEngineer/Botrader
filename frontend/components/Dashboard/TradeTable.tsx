import { Table, Paper, Text, Badge } from '@mantine/core';

interface Trade {
  id: number;
  time: string;
  side: string;
  price: number;
  quantity: number;
  status: string;
}

interface TradeTableProps {
  trades: Trade[];
}

export function TradeTable({ trades }: TradeTableProps) {
  const rows = trades.map((trade) => (
    <Table.Tr key={trade.id}>
      <Table.Td>#{trade.id}</Table.Td>
      <Table.Td>{new Date(trade.time).toLocaleTimeString()}</Table.Td>
      <Table.Td>
        <Text c={trade.side === 'BUY' ? 'green' : 'red'} fw={500}>
          {trade.side}
        </Text>
      </Table.Td>
      <Table.Td>${trade.price}</Table.Td>
      <Table.Td>{trade.quantity}</Table.Td>
      <Table.Td>
        <Badge size="sm" variant="light" color="gray">
          {trade.status}
        </Badge>
      </Table.Td>
    </Table.Tr>
  ));

  return (
    <Paper withBorder p="md" radius="md">
      <Text size="lg" fw={600} mb="md">Trade History</Text>
      <Table.ScrollContainer minWidth={500}>
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>ID</Table.Th>
              <Table.Th>Time</Table.Th>
              <Table.Th>Side</Table.Th>
              <Table.Th>Price</Table.Th>
              <Table.Th>Quantity</Table.Th>
              <Table.Th>Status</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {trades.length === 0 ? (
              <Table.Tr>
                <Table.Td colSpan={6} align="center">
                  <Text c="dimmed" py="xl">No trades executed yet</Text>
                </Table.Td>
              </Table.Tr>
            ) : (
              rows
            )}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Paper>
  );
}
