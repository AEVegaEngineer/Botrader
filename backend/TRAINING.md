# RL Training Guide

## Running Training in Docker

The RL training can be run in Docker using the `train-rl` service.

### Prerequisites

1. **Dataset**: Ensure `dataset.parquet` exists in the `backend/` directory
   - Generate it by running: `python -m app.scripts.build_dataset` (or via Docker)

2. **MLflow**: The MLflow service must be running
   - It's automatically started with `docker-compose up`

### Running Training

#### Option 1: Using Docker Compose (Recommended)

```bash
# Start MLflow and database services
docker-compose up -d mlflow timescaledb

# Wait for services to be ready (about 10-15 seconds)
sleep 15

# Run training
docker-compose run --rm train-rl
```

#### Option 2: Run in Existing Container

```bash
# Start all services
docker-compose up -d

# Execute training in backend container
docker-compose exec backend python -m app.ml.train_rl
```

#### Option 3: Build and Run Manually

```bash
# Build the backend image
docker-compose build backend

# Run training
docker run --rm \
  --network botrader_botrader-net \
  -v $(pwd)/backend:/app \
  -v $(pwd)/mlruns:/mlruns \
  --env-file .env \
  botrader-backend \
  python -m app.ml.train_rl
```

### Viewing Results

Training results are logged to MLflow. View them at:
- **MLflow UI**: http://localhost:5001
- **Experiment**: "Deep_RL_Trading"
- **Models**: Saved to `backend/app/ml/models/ppo_agent_budget_*.pth`

### Training Configuration

The training script trains models for multiple budget scenarios:
- $1,000
- $10,000
- $50,000
- $100,000

Each model is trained for 1000 episodes with validation every 100 episodes.

### Troubleshooting

1. **"Dataset not found"**: Run `build_dataset.py` first to generate the dataset
2. **"Connection refused" to MLflow**: Ensure MLflow service is running (`docker-compose up -d mlflow`)
3. **CUDA errors**: Training will fall back to CPU if GPU is not available
4. **Out of memory**: Reduce batch size or number of episodes in `train_rl.py`

### Local Development (Outside Docker)

If running locally (not in Docker), update the MLflow URI in `train_rl.py`:

```python
mlflow.set_tracking_uri("http://localhost:5001")  # Use localhost instead of mlflow
```

