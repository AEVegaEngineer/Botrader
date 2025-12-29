import { Table, Paper, Text, Badge } from '@mantine/core';

interface Trade {
  time: string;
  symbol: string;
  side: string;
  price: number;
  quantity: number;
  pnl?: number;
  capital_after?: number;
  strategy?: string;
}

interface TradeTableProps {
  trades: Trade[];
}

export function TradeTable({ trades }: TradeTableProps) {
  const rows = trades.map((trade, index) => (
    <Table.Tr key={`${trade.time}-${trade.symbol}-${index}`}>
      <Table.Td>{new Date(trade.time).toLocaleString()}</Table.Td>
      <Table.Td>
        <Text c={trade.side === 'BUY' ? 'green' : 'red'} fw={500}>
          {trade.side}
        </Text>
      </Table.Td>
      <Table.Td>${trade.price.toFixed(2)}</Table.Td>
      <Table.Td>{trade.quantity.toFixed(6)}</Table.Td>
      <Table.Td>
        {trade.pnl !== null && trade.pnl !== undefined ? (
          <Text c={trade.pnl >= 0 ? 'green' : 'red'} fw={500}>
            ${trade.pnl.toFixed(2)}
          </Text>
        ) : (
          <Text c="dimmed">-</Text>
        )}
      </Table.Td>
      <Table.Td>
        {trade.strategy && (
          <Badge size="sm" variant="light" color="blue">
            {trade.strategy}
          </Badge>
        )}
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
              <Table.Th>Time</Table.Th>
              <Table.Th>Side</Table.Th>
              <Table.Th>Price</Table.Th>
              <Table.Th>Quantity</Table.Th>
              <Table.Th>PnL</Table.Th>
              <Table.Th>Strategy</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {trades.length === 0 ? (
              <Table.Tr>
                <Table.Td colSpan={6} style={{ textAlign: 'center' }}>
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
