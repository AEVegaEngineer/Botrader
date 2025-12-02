import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
from app.features.registry import FeatureRegistry
from app.ml.dataset import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"

import mlflow
import mlflow.sklearn

# ... imports ...

def train_models():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Baseline_Models")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data()
    if df is None:
        return

    # Features and Target
    feature_cols = FeatureRegistry.get_feature_names()
    target_col = 'dir_5m' # 1 for up, 0 for down
    
    # Check if columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns: {missing_cols}")
        return

    X = df[feature_cols]
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 1. Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression"):
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(X_train, y_train)
        
        y_pred_lr = lr.predict(X_test)
        acc = accuracy_score(y_test, y_pred_lr)
        
        logger.info("Logistic Regression Results:")
        logger.info(classification_report(y_test, y_pred_lr))
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(lr, "model")
        
        joblib.dump(lr, os.path.join(MODEL_DIR, "logreg_baseline.joblib"))
    
    # 2. LightGBM
    with mlflow.start_run(run_name="LightGBM"):
        logger.info("Training LightGBM...")
        lgbm = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
        lgbm.fit(X_train, y_train)
        
        y_pred_lgbm = lgbm.predict(X_test)
        acc = accuracy_score(y_test, y_pred_lgbm)
        
        logger.info("LightGBM Results:")
        logger.info(classification_report(y_test, y_pred_lgbm))
        
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(lgbm, "model")
        
        joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm_baseline.joblib"))
    
    logger.info("Models saved.")

if __name__ == "__main__":
    train_models()
