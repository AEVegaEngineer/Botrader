import asyncio
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    model_uri = os.getenv("MODEL_URI", "models:/ProductionModel/Production")
    logger.info(f"Loading model from {model_uri}...")
    try:
        # In a real scenario, we might want to load a specific run or version
        # For now, we'll try to load from a local path or a dummy if not found for testing
        # model = mlflow.pyfunc.load_model(model_uri)
        # logger.info("Model loaded successfully.")
        pass # Placeholder until we have a real model in registry
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We might want to exit or continue with a dummy model
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference service...")

app = FastAPI(lifespan=lifespan)

# Instrument with Prometheus
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "Botrader Inference Service Running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        # For demonstration purposes if no model is loaded
        import random
        return PredictionResponse(prediction=random.random(), model_version="dummy-v1")
    
    try:
        df = pd.DataFrame([request.features])
        prediction = model.predict(df)
        return PredictionResponse(prediction=float(prediction[0]), model_version="loaded-model")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
