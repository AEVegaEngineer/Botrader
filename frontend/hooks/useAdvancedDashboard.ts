import { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';

export function useAdvancedDashboard() {
  const [riskStatus, setRiskStatus] = useState<any>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null);
  const [strategies, setStrategies] = useState<any[]>([]);
  const [activeStrategy, setActiveStrategy] = useState<any>(null);
  const [interventions, setInterventions] = useState<any[]>([]);
  const [featureImportance, setFeatureImportance] = useState<any>(null);
  const [predictionExplanation, setPredictionExplanation] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [
        riskData,
        perfData,
        strategiesData,
        interventionsData,
        featImpData
      ] = await Promise.all([
        api.getRiskStatus(),
        api.getPerformanceMetrics(),
        api.getStrategies(),
        api.getInterventions(20),
        api.getFeatureImportance()
      ]);

      setRiskStatus(riskData);
      setPerformanceMetrics(perfData);
      setStrategies(strategiesData.strategies || []);
      setActiveStrategy(strategiesData.strategies?.find((s: any) => s.is_active));
      setInterventions(interventionsData.interventions || []);
      setFeatureImportance(featImpData);
      
      setIsLoading(false);
    } catch (error) {
      console.error("Failed to fetch advanced dashboard data", error);
      setIsLoading(false);
    }
  }, []);

  const fetchPredictionExplanation = async () => {
    try {
      const data = await api.getPredictionExplanation();
      setPredictionExplanation(data);
    } catch (error) {
      console.error("Failed to fetch prediction explanation", error);
    }
  };

  useEffect(() => {
    fetchData();
    fetchPredictionExplanation();
    
    const interval = setInterval(fetchData, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, [fetchData]);

  const activateStrategy = async (id: string) => {
    try {
      await api.activateStrategy(id);
      fetchData();
    } catch (error) {
      console.error("Failed to activate strategy", error);
    }
  };

  const deactivateStrategy = async (id: string) => {
    try {
      await api.deactivateStrategy(id);
      fetchData();
    } catch (error) {
      console.error("Failed to deactivate strategy", error);
    }
  };

  const handleEmergencyStop = async () => {
    try {
      await api.emergencyStop(true, "Manual Emergency Stop");
      fetchData();
    } catch (error) {
      console.error("Failed to trigger emergency stop", error);
    }
  };

  const handleResumeTrading = async () => {
    try {
      await api.resumeTrading();
      fetchData();
    } catch (error) {
      console.error("Failed to resume trading", error);
    }
  };

  return {
    riskStatus,
    performanceMetrics,
    strategies,
    activeStrategy,
    interventions,
    featureImportance,
    predictionExplanation,
    isLoading,
    actions: {
      activateStrategy,
      deactivateStrategy,
      emergencyStop: handleEmergencyStop,
      resumeTrading: handleResumeTrading,
      refreshPrediction: fetchPredictionExplanation,
      refreshAll: fetchData
    }
  };
}
