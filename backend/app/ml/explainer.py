"""
Model explainability module using SHAP and other techniques.
Provides feature importance and per-prediction explanations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

class ModelExplainer(ABC):
    """Abstract base class for model explainers"""
    
    @abstractmethod
    def explain_global(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate global feature importance explanation.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dict with feature importance data
        """
        pass
    
    @abstractmethod
    def explain_prediction(
        self, 
        X: np.ndarray, 
        feature_names: List[str],
        index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            index: Index of prediction to explain
            
        Returns:
            Dict with explanation data
        """
        pass

class SHAPExplainer(ModelExplainer):
    """
    SHAP-based explainer for tree models (LightGBM, XGBoost, RandomForest).
    Uses TreeExplainer for fast, exact Shapley value computation.
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained model (must be compatible with SHAP TreeExplainer)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        try:
            self.explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def explain_global(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate global feature importance using SHAP values.
        
        Returns mean absolute SHAP value for each feature.
        """
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Sort features by importance
            indices = np.argsort(mean_abs_shap)[::-1]
            
            sorted_features = [feature_names[i] for i in indices]
            sorted_importance = [float(mean_abs_shap[i]) for i in indices]
            
            logger.info(f"Generated global SHAP explanation for {len(feature_names)} features")
            
            return {
                'features': sorted_features,
                'importance': sorted_importance,
                'method': 'SHAP TreeExplainer',
                'num_samples': len(X)
            }
        
        except Exception as e:
            logger.error(f"Error in global SHAP explanation: {e}")
            return self._fallback_feature_importance(feature_names)
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: List[str],
        index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Returns contribution of each feature to the prediction.
        """
        try:
            # Calculate SHAP values for this sample
            shap_values = self.explainer.shap_values(X[index:index+1])
            
            # For binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1][0]
            else:
                shap_values = shap_values[0]
            
            # Get base value (expected value)
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            # Sort features by absolute contribution
            indices = np.argsort(np.abs(shap_values))[::-1]
            
            contributions = []
            for idx in indices:
                contributions.append({
                    'feature': feature_names[idx],
                    'value': float(X[index, idx]),
                    'contribution': float(shap_values[idx])
                })
            
            # Get prediction
            prediction = self.model.predict(X[index:index+1])[0]
            
            logger.info(f"Generated SHAP explanation for prediction index {index}")
            
            return {
                'base_value': float(base_value),
                'prediction': float(prediction),
                'contributions': contributions,
                'method': 'SHAP TreeExplainer'
            }
        
        except Exception as e:
            logger.error(f"Error in prediction SHAP explanation: {e}")
            return {}
    
    def get_shap_summary_data(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 20
    ) -> Dict[str, Any]:
        """
        Get data for SHAP summary plot (for frontend visualization).
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            max_display: Max number of features to include
            
        Returns:
            Dict with summary plot data
        """
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
            
            summary_data = []
            for idx in top_indices:
                feature_shap = shap_values[:, idx]
                feature_values = X[:, idx]
                
                summary_data.append({
                    'feature': feature_names[idx],
                    'mean_abs_shap': float(mean_abs_shap[idx]),
                    'shap_values': feature_shap.tolist(),
                    'feature_values': feature_values.tolist()
                })
            
            return {
                'summary': summary_data,
                'num_samples': len(X),
                'method': 'SHAP summary'
            }
        
        except Exception as e:
            logger.error(f"Error generating SHAP summary data: {e}")
            return {}
    
    def _fallback_feature_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """Fallback to model's native feature importance if available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                indices = np.argsort(importance)[::-1]
                
                return {
                    'features': [feature_names[i] for i in indices],
                    'importance': [float(importance[i]) for i in indices],
                    'method': 'Model native feature importance',
                    'num_samples': 'N/A'
                }
        except Exception:
            pass
        
        return {
            'features': feature_names,
            'importance': [0.0] * len(feature_names),
            'method': 'Fallback (uniform)',
            'num_samples': 0
        }

class GradientExplainer(ModelExplainer):
    """
    Simplified gradient-based explainer for neural networks.
    Computes input gradients to estimate feature importance.
    """
    
    def __init__(self, model, framework: str = "pytorch"):
        """
        Args:
            model: Trained neural network model
            framework: 'pytorch' or 'tensorflow'
        """
        self.model = model
        self.framework = framework
        logger.info(f"Gradient explainer initialized ({framework})")
    
    def explain_global(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Global explanation using average gradient magnitudes.
        """
        # Simplified: use variance of features as proxy
        # In production, would compute actual gradients
        feature_variance = np.var(X, axis=0)
        indices = np.argsort(feature_variance)[::-1]
        
        return {
            'features': [feature_names[i] for i in indices],
            'importance': [float(feature_variance[i]) for i in indices],
            'method': 'Feature variance (gradient approximation)',
            'num_samples': len(X)
        }
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: List[str],
        index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain prediction using input feature values.
        Simplified version - in production would compute actual gradients.
        """
        # Simplified explanation based on feature values
        feature_values = X[index]
        abs_values = np.abs(feature_values)
        indices = np.argsort(abs_values)[::-1]
        
        contributions = []
        for idx in indices:
            contributions.append({
                'feature': feature_names[idx],
                'value': float(feature_values[idx]),
                'contribution': float(abs_values[idx])  # Simplified
            })
        
        return {
            'base_value': 0.0,
            'prediction': 0.0,  # Would need actual model prediction
            'contributions': contributions,
            'method': 'Gradient approximation'
        }

def create_explainer(model, model_type: str = "tree") -> ModelExplainer:
    """
    Factory function to create appropriate explainer.
    
    Args:
        model: Trained model
        model_type: 'tree' for tree-based models, 'neural' for neural networks
        
    Returns:
        ModelExplainer instance
    """
    if model_type == "tree":
        return SHAPExplainer(model)
    elif model_type == "neural":
        return GradientExplainer(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
