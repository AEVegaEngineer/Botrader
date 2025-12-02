import torch
import joblib
import numpy as np
import os
from app.ml.models.deep import LSTMClassifier, CNNClassifier
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "app/ml/models"

class EnsembleModel:
    def __init__(self, input_dim, device):
        self.device = device
        self.input_dim = input_dim
        
        # Load Base Models
        self.lgbm = self._load_lgbm()
        self.lstm = self._load_lstm()
        self.cnn = self._load_cnn()
        
        # Meta Learner (to be trained)
        self.meta_learner = None
        
    def _load_lgbm(self):
        path = os.path.join(MODEL_DIR, "lgbm_baseline.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        return None

    def _load_lstm(self):
        path = os.path.join(MODEL_DIR, "lstm_model.pth")
        if os.path.exists(path):
            model = LSTMClassifier(self.input_dim, 64, 2, 2).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model
        return None

    def _load_cnn(self):
        path = os.path.join(MODEL_DIR, "cnn_model.pth")
        if os.path.exists(path):
            model = CNNClassifier(self.input_dim, 64, 2).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model
        return None

    def predict_proba(self, X_df, X_tensor):
        """
        X_df: DataFrame for LightGBM
        X_tensor: Tensor (batch, seq_len, features) for Deep Models
        """
        preds = []
        
        # 1. LightGBM
        if self.lgbm:
            # LGBM expects 2D input (batch, features). 
            # If X_df is sequence, we might need to take the last row or flatten.
            # Assuming X_df here is already aligned with the target (e.g. last row features).
            # But wait, SequenceDataset gives sequences.
            # For LGBM, we usually use the features at time t to predict t+k.
            # So we should take the last timestep of the sequence.
            
            # X_tensor shape: (batch, seq_len, features)
            # We can extract the last timestep features from X_tensor and convert to numpy
            last_step_features = X_tensor[:, -1, :].cpu().numpy()
            p_lgbm = self.lgbm.predict_proba(last_step_features)[:, 1] # Prob of class 1
            preds.append(p_lgbm)
        else:
            preds.append(np.zeros(X_tensor.size(0)))

        # 2. LSTM
        if self.lstm:
            with torch.no_grad():
                out = self.lstm(X_tensor)
                p_lstm = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds.append(p_lstm)
        else:
            preds.append(np.zeros(X_tensor.size(0)))

        # 3. CNN
        if self.cnn:
            with torch.no_grad():
                out = self.cnn(X_tensor)
                p_cnn = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds.append(p_cnn)
        else:
            preds.append(np.zeros(X_tensor.size(0)))
            
        # Stack predictions: (batch, 3)
        stacked_preds = np.column_stack(preds)
        
        # Meta Learner
        if self.meta_learner:
            return self.meta_learner.predict_proba(stacked_preds)[:, 1]
        else:
            # Simple Average
            return np.mean(stacked_preds, axis=1)

    def train_meta_learner(self, X_tensor, y):
        """
        Train Logistic Regression on stacked predictions.
        """
        # Generate base predictions
        preds = []
        
        # LightGBM
        if self.lgbm:
            last_step_features = X_tensor[:, -1, :].cpu().numpy()
            p_lgbm = self.lgbm.predict_proba(last_step_features)[:, 1]
            preds.append(p_lgbm)
        else:
            preds.append(np.zeros(len(y)))
            
        # LSTM
        if self.lstm:
            with torch.no_grad():
                out = self.lstm(X_tensor)
                p_lstm = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds.append(p_lstm)
        else:
            preds.append(np.zeros(len(y)))
            
        # CNN
        if self.cnn:
            with torch.no_grad():
                out = self.cnn(X_tensor)
                p_cnn = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds.append(p_cnn)
        else:
            preds.append(np.zeros(len(y)))
            
        stacked_X = np.column_stack(preds)
        
        self.meta_learner = LogisticRegression()
        self.meta_learner.fit(stacked_X, y)
        
        # Save meta learner
        joblib.dump(self.meta_learner, os.path.join(MODEL_DIR, "ensemble_meta.joblib"))
