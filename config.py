"""
config.py — Centralised Configuration for the Federated Learning System
========================================================================

All hyperparameters, paths, and server settings in one place.
Override any value by setting the corresponding environment variable.

Environment variables:
    FL_SERVER_IP       Override the server IP for clients
    FL_SERVER_PORT     Override the Flower gRPC port (default: 8080)
    FL_API_PORT        Override the FastAPI port (default: 5000)
    FL_NUM_ROUNDS      Override the number of FL rounds (default: 30)
"""

import os

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
SERVER_IP: str = os.environ.get("FL_SERVER_IP", "10.133.98.49")
"""Server IP address. Clients connect to this. Set FL_SERVER_IP env var to override."""

SERVER_FL_PORT: int = int(os.environ.get("FL_SERVER_PORT", 8080))
"""Flower gRPC port used for federated learning communication."""

SERVER_API_PORT: int = int(os.environ.get("FL_API_PORT", 5000))
"""FastAPI REST port used for the prediction and monitoring endpoints."""

SERVER_ADDRESS: str = f"{SERVER_IP}:{SERVER_FL_PORT}"
"""Combined address string passed to Flower start_server / start_numpy_client."""

# ---------------------------------------------------------------------------
# Federated Learning
# ---------------------------------------------------------------------------
NUM_ROUNDS: int = int(os.environ.get("FL_NUM_ROUNDS", 30))
"""Total number of FL training rounds."""

ROUND_TIMEOUT: int = 120
"""Per-round timeout in seconds. Clients that don't respond are skipped."""

MIN_FIT_CLIENTS: int = 2
"""Minimum number of clients required to start a training round."""

MIN_AVAILABLE_CLIENTS: int = 2
"""Minimum number of clients that must be connected before training begins."""

MIN_EVALUATE_CLIENTS: int = 2
"""Minimum number of clients used for evaluation after aggregation."""

FRACTION_FIT: float = 1.0
"""Fraction of available clients selected for training each round."""

FRACTION_EVALUATE: float = 1.0
"""Fraction of available clients selected for evaluation each round."""

PROXIMAL_MU: float = 0.1
"""FedProx proximal term coefficient. Higher → less client drift, slower convergence."""

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 8
"""Number of input features (Pima Indians Diabetes dataset columns)."""

NUM_CLASSES: int = 2
"""Number of output classes: 0 = No Diabetes, 1 = Diabetes."""

LEAKY_RELU_SLOPE: float = 0.1
"""Negative slope for LeakyReLU activations."""

DROPOUT_L1: float = 0.5
"""Dropout probability for Layer 1 (input layer — most regularisation)."""

DROPOUT_L2: float = 0.4
"""Dropout probability for Layer 2."""

DROPOUT_L3: float = 0.3
"""Dropout probability for Layer 3 (least regularisation near output)."""

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 0.0017
"""Adam optimiser learning rate."""

WEIGHT_DECAY: float = 1e-4
"""L2 regularisation coefficient (also applied explicitly in the loss)."""

BATCH_SIZE: int = 32
"""Mini-batch size for local client training."""

LOCAL_EPOCHS: int = 1
"""Number of local training epochs per FL round. Keep low to avoid client drift."""

LABEL_SMOOTHING: float = 0.1
"""Label smoothing coefficient. 0 = no smoothing, 0.1 = 10% smoothing."""

LR_SCHEDULER_FACTOR: float = 0.5
"""Factor by which LR is reduced when validation loss plateaus."""

LR_SCHEDULER_PATIENCE: int = 2
"""Number of non-improving epochs before LR is reduced."""

EARLY_STOPPING_PATIENCE: int = 3
"""Number of non-improving validation epochs before local training stops."""

# ---------------------------------------------------------------------------
# Dataset Paths
# ---------------------------------------------------------------------------
DATASET_CLIENT_1: str = "diabetes_non_negative_part1_2000.csv"
"""CSV file loaded by Client 1."""

DATASET_CLIENT_2: str = "diabetes_non_negative_part2_2000.csv"
"""CSV file loaded by Client 2."""

FEATURE_COLUMNS: list = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
"""Expected feature column names in the dataset CSV."""

TARGET_COLUMN: str = "Outcome"
"""Target column name (0 = No Diabetes, 1 = Diabetes)."""

TEST_SIZE: float = 0.2
"""Fraction of data reserved for local client test sets."""

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
FINAL_MODEL_PATH: str = "global_model_final.pth"
"""Path where the final trained global model is saved."""

SCALER_PATH: str = "global_scaler.pkl"
"""Path to the pre-fitted StandardScaler used for inference."""

CHECKPOINTS_DIR: str = "checkpoints"
"""Directory where per-round model checkpoints are saved."""

TRAINING_HISTORY_PATH: str = "training_history.json"
"""JSON file where per-round metrics are appended during training."""

CLIENT_LOG_PATH: str = "client.log"
"""Log file written by the FL client."""

# ---------------------------------------------------------------------------
# Risk Thresholds (used by the prediction API)
# ---------------------------------------------------------------------------
LOW_RISK_THRESHOLD: float = 25.0
"""Risk percentage below this → Low risk."""

HIGH_RISK_THRESHOLD: float = 75.0
"""Risk percentage above this → High risk. Between thresholds → Moderate."""
