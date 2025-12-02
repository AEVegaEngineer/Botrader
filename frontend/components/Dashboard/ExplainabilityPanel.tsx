import { Paper, Text, Group, Stack, Progress, Select, Button, Loader } from '@mantine/core';
import { Brain, HelpCircle, RefreshCw } from 'lucide-react';
import { useState } from 'react';

interface FeatureImportance {
  feature: string;
  importance: number; // 0-1
}

interface PredictionExplanation {
  base_value: number;
  prediction: number;
  contributions: {
    feature: string;
    value: number;
    contribution: number;
  }[];
}

interface ExplainabilityPanelProps {
  featureImportance?: {
    features: string[];
    importance: number[];
    method: string;
  };
  predictionExplanation?: PredictionExplanation;
  isLoading?: boolean;
  onRefresh?: () => void;
}

export function ExplainabilityPanel({ 
  featureImportance, 
  predictionExplanation, 
  isLoading = false,
  onRefresh 
}: ExplainabilityPanelProps) {
  const [viewMode, setViewMode] = useState<'global' | 'local'>('global');

  if (isLoading && !featureImportance) {
    return (
      <Paper withBorder p="md" radius="md" h="100%" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Loader size="sm" />
      </Paper>
    );
  }

  // Process feature importance for display
  const features = featureImportance ? featureImportance.features.map((f, i) => ({
    name: f,
    value: featureImportance.importance[i]
  })).sort((a, b) => b.value - a.value).slice(0, 10) : [];

  const maxImportance = features.length > 0 ? features[0].value : 1;

  return (
    <Paper withBorder p="md" radius="md" h="100%">
      <Group justify="space-between" mb="md">
        <Group gap="xs">
          <Brain size={20} color="var(--mantine-color-grape-6)" />
          <Text size="lg" fw={600}>Model Insights</Text>
        </Group>
        <Group>
          <Select 
            size="xs"
            value={viewMode}
            onChange={(v) => setViewMode(v as 'global' | 'local')}
            data={[
              { value: 'global', label: 'Global Importance' },
              { value: 'local', label: 'Last Prediction' }
            ]}
            w={140}
          />
          {onRefresh && (
            <Button variant="subtle" size="xs" onClick={onRefresh}>
              <RefreshCw size={14} />
            </Button>
          )}
        </Group>
      </Group>

      {viewMode === 'global' ? (
        <Stack gap="sm">
          <Text size="xs" c="dimmed" mb={4}>
            Top features driving model decisions ({featureImportance?.method || 'SHAP'})
          </Text>
          
          {features.map((feature, idx) => (
            <div key={idx}>
              <Group justify="space-between" mb={2}>
                <Text size="sm" fw={500}>{feature.name}</Text>
                <Text size="xs" c="dimmed">{feature.value.toFixed(4)}</Text>
              </Group>
              <Progress 
                value={(feature.value / maxImportance) * 100} 
                color="grape" 
                size="sm" 
                radius="xl"
              />
            </div>
          ))}
          
          {features.length === 0 && (
            <Text c="dimmed" size="sm" ta="center" py="xl">
              No feature importance data available
            </Text>
          )}
        </Stack>
      ) : (
        <Stack gap="sm">
           <Text size="xs" c="dimmed" mb={4}>
            Feature contributions to last prediction
          </Text>
          
          {predictionExplanation ? (
            <>
              <Paper bg="dark.6" p="xs" radius="sm" mb="xs">
                <Group justify="space-between">
                  <Text size="sm">Prediction:</Text>
                  <Text fw={700} c={predictionExplanation.prediction > 0.5 ? 'green' : 'red'}>
                    {predictionExplanation.prediction.toFixed(4)}
                  </Text>
                </Group>
                <Group justify="space-between">
                  <Text size="xs" c="dimmed">Base Value:</Text>
                  <Text size="xs" c="dimmed">{predictionExplanation.base_value.toFixed(4)}</Text>
                </Group>
              </Paper>

              {predictionExplanation.contributions.slice(0, 8).map((item, idx) => (
                <div key={idx}>
                  <Group justify="space-between" mb={2}>
                    <Text size="sm">{item.feature} <span style={{ opacity: 0.5, fontSize: '0.8em' }}>({item.value.toFixed(2)})</span></Text>
                    <Text size="xs" c={item.contribution > 0 ? 'green' : 'red'}>
                      {item.contribution > 0 ? '+' : ''}{item.contribution.toFixed(4)}
                    </Text>
                  </Group>
                  <Progress 
                    value={Math.min(Math.abs(item.contribution) * 500, 100)} // Scale for visibility
                    color={item.contribution > 0 ? 'green' : 'red'}
                    size="xs" 
                    radius="xl"
                  />
                </div>
              ))}
            </>
          ) : (
             <Text c="dimmed" size="sm" ta="center" py="xl">
              No prediction explanation available
            </Text>
          )}
        </Stack>
      )}
    </Paper>
  );
}
