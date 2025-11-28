import { Group, Code, Title, Container } from '@mantine/core';
import { Activity } from 'lucide-react';
import Link from 'next/link';
import { ThemeToggle } from './ThemeToggle';
import classes from './Navbar.module.css';

export function Navbar() {
  return (
    <header className={classes.header}>
      <Container size="xl" className={classes.inner}>
        <Group gap={5}>
          <Activity size={28} className="text-blue-500" />
          <Title order={3}>Botrader</Title>
          <Code fw={700}>v1.0</Code>
        </Group>

        <Group gap={5} visibleFrom="xs">
          <Link href="/" className={classes.link}>Live Trading</Link>
          <Link href="/dashboard" className={classes.link}>Performance</Link>
        </Group>

        <ThemeToggle />
      </Container>
    </header>
  );
}
