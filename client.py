"""
client.py — Federated Learning Client for Diabetes Risk Prediction
==================================================================

Implements the Flower (flwr) federated learning client. Each client represents
a hospital or clinic with its own local patient dataset (Pima Indians Diabetes).

Usage::

    python client.py --client-id 1 [--server-address HOST:PORT] [--batch-size N]

Environment variables (override config.py defaults)::

    FL_SERVER_IP=192.168.1.100 python client.py --client-id 1

See Also:
    server.py      — FL server and prediction API
    config.py      — All hyperparameters and paths
"""

import os
import sys
import argparse
import logging
import socket
import time
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import flwr as fl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.CLIENT_LOG_PATH),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app (client-side micro-service)
# ---------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.26.65.217:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_diabetes_data(
    file_path: str = config.DATASET_CLIENT_1,
    test_size: float = config.TEST_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load, validate, and preprocess the Pima Indians Diabetes CSV dataset.

    Steps:
        1. Read CSV with pandas.
        2. Extract 8 feature columns (or fall back to first 8 columns).
        3. Extract ``Outcome`` column (or last column as fallback).
        4. Binarise multi-class targets.
        5. Stratified 80/20 train/test split (seed=42).
        6. Fit ``StandardScaler`` on train, transform both splits.
        7. Convert to ``torch.FloatTensor`` / ``torch.LongTensor``.

    Args:
        file_path: Path to the CSV file.
        test_size: Fraction reserved for the test set (default 0.2).

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)`` as PyTorch tensors.

    Raises:
        ValueError: If the CSV file is empty.
        FileNotFoundError: If ``file_path`` does not exist.
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"Dataset at '{file_path}' is empty.")

        logger.info("Dataset loaded: shape=%s", df.shape)

        # Extract features
        missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Missing expected columns %s — using first 8 columns.", missing)
            X = df.iloc[:, :8].values
        else:
            X = df[config.FEATURE_COLUMNS].values

        # Extract target
        if config.TARGET_COLUMN in df.columns:
            y = df[config.TARGET_COLUMN].values
        else:
            y = df.iloc[:, -1].values

        logger.info("Features: %d | Samples: %d", X.shape[1], y.shape[0])

        # Binarise multi-class targets
        if len(np.unique(y)) > 2:
            y = (y > 0).astype(int)

    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Standardise (fit on train only to prevent data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_train),
        torch.LongTensor(y_test),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DiabetesModel(nn.Module):
    """Feed-forward neural network for binary diabetes risk classification.

    Identical architecture to the server-side model to ensure weight compatibility.

    Args:
        input_size: Number of input features. Default: 8.
        num_classes: Number of output classes. Default: 2.
    """

    def __init__(self, input_size: int = config.INPUT_SIZE, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(config.LEAKY_RELU_SLOPE),
            nn.BatchNorm1d(32),
            nn.Dropout(config.DROPOUT_L1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(config.LEAKY_RELU_SLOPE),
            nn.BatchNorm1d(16),
            nn.Dropout(config.DROPOUT_L2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(config.LEAKY_RELU_SLOPE),
            nn.BatchNorm1d(8),
            nn.Dropout(config.DROPOUT_L3),
        )
        self.output = nn.Linear(8, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming Normal init for Linear; identity init for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Auto-unsqueezes 1-D single-sample input.

        Args:
            x: Shape ``(batch_size, input_size)`` or ``(input_size,)``.

        Returns:
            Raw logits, shape ``(batch_size, num_classes)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Flower Client
# ---------------------------------------------------------------------------

class FlowerClient(fl.client.NumPyClient):
    """Flower FL client for local diabetes model training and evaluation.

    Communicates only model weight arrays with the server — raw patient
    data never leaves this machine.

    Args:
        model: Local DiabetesModel instance.
        X_train, y_train: Training tensors.
        X_test, y_test: Test/validation tensors.
    """

    def __init__(
        self,
        model: DiabetesModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.label_smoothing = config.LABEL_SMOOTHING
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )

        logger.info("Client ready on device: %s", self.device)

    def get_parameters(self, config: Optional[Dict] = None) -> List[np.ndarray]:
        """Serialise model state dict to a list of numpy arrays.

        Args:
            config: Unused config dict from Flower.

        Returns:
            List of numpy arrays — one per state_dict entry.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load server-provided parameters into the local model.

        Handles shape mismatches (extra dims, element-count matches, truncation,
        zero-padding) that can occur across FL framework versions.

        Args:
            parameters: List of numpy arrays from the server.
        """
        try:
            state_dict = self.model.state_dict()
            param_dict = {
                name: param for name, param in zip(state_dict.keys(), parameters)
            }

            for name, param in state_dict.items():
                if name not in param_dict:
                    continue
                server_param = param_dict[name]
                if isinstance(server_param, np.ndarray):
                    server_param = torch.from_numpy(server_param)

                if param.shape == server_param.shape:
                    state_dict[name] = server_param.to(self.device)
                elif len(server_param.shape) > len(param.shape) and server_param.shape[0] == 1:
                    state_dict[name] = server_param.squeeze(0).to(self.device)
                elif len(server_param.shape) > len(param.shape) and server_param.shape[1] == 1:
                    state_dict[name] = server_param.squeeze(1).to(self.device)
                elif server_param.numel() == param.numel():
                    state_dict[name] = server_param.reshape_as(param).to(self.device)
                else:
                    logger.warning(
                        "Shape mismatch for %s: expected %s, got %s",
                        name, param.shape, server_param.shape,
                    )
                    if server_param.numel() >= param.numel():
                        state_dict[name] = (
                            server_param.flatten()[: param.numel()]
                            .reshape(param.shape)
                            .to(self.device)
                        )
                    else:
                        temp = torch.zeros_like(param, device=self.device)
                        temp.flatten()[: server_param.numel()] = server_param.flatten()
                        state_dict[name] = temp

            self.model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            logger.error("set_parameters failed: %s — trying simplified load.", e)
            try:
                for i, (_, param) in enumerate(self.model.named_parameters()):
                    if i < len(parameters):
                        param.data = torch.from_numpy(parameters[i]).to(self.device)
                logger.info("Simplified parameter load succeeded.")
            except Exception as e2:
                logger.critical("Critical parameter load error: %s — reinitialising.", e2)
                self.model.apply(self._init_weights)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Load global params, run local training, return updated weights.

        FIX: X_train / y_train are already PyTorch tensors from load_diabetes_data().
        Previously they were redundantly wrapped in torch.FloatTensor() /
        torch.LongTensor() constructors on every round, creating unnecessary copies.
        Now they are used directly.

        Training applies:
            - CrossEntropy + L2 regularisation + label smoothing + FedProx term
            - Early stopping (patience from config)
            - ReduceLROnPlateau scheduling

        Args:
            parameters: Global model weights from the server.
            config: Training config dict with epochs, batch_size, proximal_mu, weight_decay.

        Returns:
            (updated_params, num_train_samples, {})
        """
        self.set_parameters(parameters)

        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        proximal_mu = config.get("proximal_mu", 0.1)
        weight_decay = config.get("weight_decay", 1e-4)

        # Snapshot global params for FedProx proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()] if proximal_mu > 0 else []

        # FIX: X_train / y_train are already tensors — no redundant constructor call
        train_dataset = data.TensorDataset(self.X_train, self.y_train)
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.train()
        best_params = None
        best_val_loss = float("inf")
        patience = 3
        no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # Explicit L2 regularisation
                l2_reg = sum(torch.norm(p) for p in self.model.parameters())
                loss = loss + weight_decay * l2_reg

                # Label smoothing
                if self.label_smoothing > 0:
                    smooth = -F.log_softmax(outputs, dim=1).mean(dim=1).mean()
                    loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth

                # FedProx proximal term
                if proximal_mu > 0 and global_params:
                    prox = sum(
                        (lp - gp).norm(2)
                        for lp, gp in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (proximal_mu / 2) * prox

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            val_loss, val_metrics = self._evaluate_validation()
            self.scheduler.step(val_loss)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1, epochs,
                epoch_loss / len(self.X_train),
                val_loss,
                val_metrics["accuracy"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                    if best_params:
                        self.set_parameters(best_params)
                    break

        return self.get_parameters(), len(self.X_train), {}

    def _evaluate_validation(self) -> Tuple[float, Dict[str, float]]:
        """Compute validation loss and accuracy on the local test split.

        Returns:
            (val_loss, {"accuracy": float})
        """
        self.model.eval()
        with torch.no_grad():
            X = self.X_test.to(self.device)
            y = self.y_test.to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y).sum().item() / y.size(0)
        self.model.train()
        return loss, {"accuracy": acc}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict[str, float]]:
        """Load global params and evaluate on the local test set.

        Args:
            parameters: Global model weights to evaluate.
            config: Evaluation config.

        Returns:
            (loss, num_test_samples, metrics_dict)
            metrics_dict keys: accuracy, precision, recall, f1_score
        """
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            X = self.X_test.to(self.device)
            y = self.y_test.to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).sum().item() / len(y)
            y_true = y.cpu().numpy()
            y_pred = predicted.cpu().numpy()

        if len(np.unique(y_pred)) == 1:
            precision = recall = f1 = 0.0
        else:
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        logger.info(
            "Eval — loss=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
            loss, accuracy, precision, recall, f1,
        )
        return float(loss), len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_local_ip() -> str:
    """Return the machine's primary LAN IP address, or '0.0.0.0' on failure."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning("Could not determine local IP: %s", e)
        return "0.0.0.0"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_client() -> None:
    """Parse CLI args, load dataset, and start the Flower FL client.

    CLI arguments:
        --server-address  HOST:PORT of the Flower server. Defaults to config value.
                          Override server IP with FL_SERVER_IP env var.
        --client-id       Required. Determines dataset partition (1 or 2).
        --batch-size      Mini-batch size for local training.

    Raises:
        SystemExit: On fatal data or connection errors.
    """
    parser = argparse.ArgumentParser(
        description="Federated Learning Client — Diabetes Risk Prediction"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=config.SERVER_ADDRESS,
        help=f"Server address host:port (default: {config.SERVER_ADDRESS}). "
             f"Set FL_SERVER_IP env var to change the IP.",
    )
    parser.add_argument(
        "--client-id", type=int, required=True,
        help="Unique client ID (1 or 2). Determines which dataset partition is used.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help=f"Training batch size (default: {config.BATCH_SIZE}).",
    )

    args = parser.parse_args()

    # Select dataset partition based on client ID
    dataset_map = {
        1: config.DATASET_CLIENT_1,
        2: config.DATASET_CLIENT_2,
    }
    dataset_file = dataset_map.get(args.client_id, config.DATASET_CLIENT_1)
    logger.info("Client %d → dataset: %s", args.client_id, dataset_file)

    try:
        logger.info("=" * 50)
        logger.info("Starting FL Client (ID: %d)", args.client_id)
        logger.info("=" * 50)

        X_train, X_test, y_train, y_test = load_diabetes_data(file_path=dataset_file)

        logger.info(
            "Dataset ready — train=%d test=%d features=%d",
            len(X_train), len(X_test), X_train.shape[1],
        )

        model = DiabetesModel(input_size=X_train.shape[1])
        client = FlowerClient(model, X_train, y_train, X_test, y_test)

        logger.info("Connecting to %s …", args.server_address)
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
        logger.info("Client finished successfully.")

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_client()