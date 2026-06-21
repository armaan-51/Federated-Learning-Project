"""
client.py — Federated Learning Client for Heart Disease / Diabetes Prediction
==============================================================================

This module implements the Flower (flwr) federated learning client. Each client
represents a hospital or clinic with its own local patient dataset.

Responsibilities:
    - Load and preprocess a local CSV dataset (Pima Indians Diabetes format).
    - Define the ``DiabetesModel`` neural network (identical to the server model).
    - Implement ``FlowerClient`` which performs local training and evaluation.
    - Connect to the central Flower server and participate in FL rounds.

Usage::

    python client.py --client-id 1 [--server-address HOST:PORT] [--batch-size N]

Client IDs determine which dataset partition is loaded:
    - Client 1 → ``diabetes_non_negative_part1_2000.csv``
    - Client 2 → ``diabetes_non_negative_part2_2000.csv``
    - Other IDs → defaults to part 1

Notes:
    - Update ``SERVER_IP`` at the top of this file before running on a network.
    - No raw data ever leaves this machine — only model weights are transmitted.

See Also:
    server.py      — FL server and prediction API
    docs/architecture.md — In-depth architecture documentation
"""

# client.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import flwr as fl
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging
from fastapi import FastAPI, HTTPException
import uvicorn
import threading
import time
import sys
import argparse
import socket
import torch.utils.data as data

# ---------------------------------------------------------------------------
# Server IP Configuration — UPDATE THIS BEFORE NETWORK DEPLOYMENT
# ---------------------------------------------------------------------------
SERVER_IP = "10.133.98.49"  # Replace with your server machine's actual LAN IP

# Configure logging to both console and a rotating log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("client.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI (used by the client-side prediction micro-service if needed)
app = FastAPI()

# Enable CORS so the React frontend can make cross-origin requests
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.26.65.217:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_diabetes_data(
    file_path: str = 'diabetes_non_negative_part1_2000.csv',
    test_size: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load, validate, and preprocess the Pima Indians Diabetes CSV dataset.

    Steps performed:
        1. Read CSV with ``pandas``.
        2. Extract the 8 standard feature columns (or fall back to first 8 cols).
        3. Extract the ``Outcome`` target column (or last column as fallback).
        4. Convert multi-class targets to binary (disease / no disease).
        5. Stratified 80/20 train/test split (``random_state=42``).
        6. Standardise features with ``StandardScaler`` (fit on train, transform both).
        7. Convert arrays to ``torch.FloatTensor`` / ``torch.LongTensor``.

    Args:
        file_path (str): Path to the CSV file. Must contain the 8 feature columns
            or at least 9 columns total. Default: ``diabetes_non_negative_part1_2000.csv``.
        test_size (float): Fraction of samples reserved for the test set.
            Must be between 0 and 1. Default: ``0.2`` (20 % test).

    Returns:
        tuple: ``(X_train, X_test, y_train, y_test)`` — four tensors:
            - ``X_train`` (FloatTensor): Training features, shape ``(n_train, 8)``.
            - ``X_test``  (FloatTensor): Test features,     shape ``(n_test,  8)``.
            - ``y_train`` (LongTensor):  Training labels,   shape ``(n_train,)``.
            - ``y_test``  (LongTensor):  Test labels,       shape ``(n_test,)``.

    Raises:
        ValueError: If the CSV file is empty.
        FileNotFoundError: If ``file_path`` does not exist.

    Example::

        X_train, X_test, y_train, y_test = load_diabetes_data("data.csv", test_size=0.2)
        print(X_train.shape)  # torch.Size([1600, 8])
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError("The dataset is empty. Please check the file path.")

        logger.info(f"Dataset shape: {df.shape}")
        logger.info("\nDataset info:")
        logger.info(df.info())
        logger.info("\nFirst few rows of the dataset:")
        logger.info(df.head())

        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            logger.warning(f"Warning: Missing expected features: {missing_features}")
            logger.warning("Using first 8 columns as features")
            X = df.iloc[:, :8].values
        else:
            X = df[feature_columns].values

        if 'Outcome' in df.columns:
            y = df['Outcome'].values
        else:
            y = df.iloc[:, -1].values

        logger.info(f"Loaded {X.shape[1]} features and {y.shape[0]} samples")

        # Binarise target if more than 2 unique values exist
        if len(np.unique(y)) > 2:
            y = (y > 0).astype(int)

    except Exception as e:
        logger.error(f"Error loading the dataset: {e}")
        raise

    # Stratified split preserves class balance in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Fit scaler on training data only to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test


class DiabetesModel(nn.Module):
    """Feed-forward neural network for binary diabetes / heart disease classification.

    This is the **same architecture** as the server-side model, ensuring that
    weight tensors are compatible when the server sends global parameters to clients.

    Architecture:
        - Layer 1: Linear(input_size → 32) + LeakyReLU + BatchNorm + Dropout(0.5)
        - Layer 2: Linear(32 → 16)         + LeakyReLU + BatchNorm + Dropout(0.4)
        - Layer 3: Linear(16 → 8)          + LeakyReLU + BatchNorm + Dropout(0.3)
        - Output:  Linear(8 → num_classes)

    Args:
        input_size (int): Number of input features. Default is 8.
        num_classes (int): Number of output classes. Default is 2 (binary).
    """

    def __init__(self, input_size: int = 8, num_classes: int = 2):
        super(DiabetesModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.3)
        )
        self.output = nn.Linear(8, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with Kaiming Normal; BatchNorm with identity."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, handles single-sample 1-D input automatically.

        Args:
            x (torch.Tensor): Shape ``(batch_size, input_size)`` or ``(input_size,)``.

        Returns:
            torch.Tensor: Raw logits, shape ``(batch_size, num_classes)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


class FlowerClient(fl.client.NumPyClient):
    """Flower federated learning client for local model training and evaluation.

    Implements the ``fl.client.NumPyClient`` interface so Flower can orchestrate
    training across multiple clients. The client communicates only model weights
    (numpy arrays) with the server — raw data never leaves this machine.

    Training details:
        - Optimizer: Adam with weight decay (L2 regularisation).
        - Loss: CrossEntropyLoss + optional label smoothing + FedProx proximal term.
        - LR Scheduler: ReduceLROnPlateau (halves LR after 2 non-improving epochs).
        - Early Stopping: stops local training after 3 non-improving validation epochs.

    Args:
        model (DiabetesModel): Local model instance to train.
        X_train (torch.Tensor): Training features, shape ``(n_train, n_features)``.
        y_train (torch.Tensor): Training labels,   shape ``(n_train,)``.
        X_test  (torch.Tensor): Validation/test features, shape ``(n_test, n_features)``.
        y_test  (torch.Tensor): Validation/test labels,   shape ``(n_test,)``.
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

        self.label_smoothing = 0.1
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0017,
            weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        self.best_val_loss = float('inf')
        self.patience = 3
        self.no_improve_epochs = 0

        logger.info(f"Client initialized on device: {self.device}")

    def get_parameters(self, config: Optional[Dict] = None) -> List[np.ndarray]:
        """Serialise the local model's state dict into a list of numpy arrays.

        This is called by Flower after training (``fit``) to send updated weights
        back to the server.

        Args:
            config (dict, optional): Configuration dict from the server (unused).

        Returns:
            list[np.ndarray]: One array per model parameter tensor (in state_dict order).
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load server-provided parameters into the local model.

        Handles common shape mismatches that can occur between server and client
        model versions:
            1. Exact shape match         → direct assignment
            2. Extra leading dim [1, N]  → squeeze(0)
            3. Extra trailing dim [N, 1] → squeeze(1)
            4. Same element count        → reshape
            5. Server has more elements  → truncate + reshape
            6. Server has fewer elements → zero-pad + reshape

        Args:
            parameters (list[np.ndarray]): Parameter arrays from the server,
                in the same order as ``model.state_dict()``.
        """
        try:
            state_dict = self.model.state_dict()
            param_dict = {name: param for name, param in zip(state_dict.keys(), parameters)}

            for name, param in state_dict.items():
                if name in param_dict:
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
                            f"Shape mismatch for {name}: expected {param.shape}, got {server_param.shape}"
                        )
                        if server_param.numel() >= param.numel():
                            state_dict[name] = (
                                server_param.flatten()[:param.numel()].reshape(param.shape).to(self.device)
                            )
                        else:
                            temp = torch.zeros_like(param, device=self.device)
                            flat_temp = temp.flatten()
                            flat_temp[:server_param.numel()] = server_param.flatten()
                            state_dict[name] = flat_temp.reshape(param.shape)

            self.model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            logger.error(f"Error in set_parameters: {str(e)}")
            logger.error("Attempting to load parameters with simplified method...")
            try:
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i < len(parameters):
                        param.data = torch.from_numpy(parameters[i]).to(self.device)
                logger.info("Successfully loaded parameters with simplified method")
            except Exception as e2:
                logger.critical(f"Critical error in parameter loading: {str(e2)}")
                logger.info("Reinitializing model weights...")
                self.model.apply(self._init_weights)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Load global parameters, run local training, return updated weights.

        Training applies:
            - Standard cross-entropy loss
            - Explicit L2 regularisation (weight decay term)
            - Label smoothing (10 % by default)
            - FedProx proximal term (``μ/2 · ‖w_local − w_global‖²``)
            - Early stopping with patience=3 based on validation loss

        Args:
            parameters (list[np.ndarray]): Global model weights from the server.
            config (dict): Training config with optional keys:
                - ``epochs`` (int): Local epochs per round. Default: 1.
                - ``batch_size`` (int): Mini-batch size. Default: 32.
                - ``proximal_mu`` (float): FedProx μ. Default: 0.2.
                - ``weight_decay`` (float): L2 coefficient. Default: 1e-4.

        Returns:
            tuple: ``(updated_parameters, num_train_examples, metrics)``
                where ``metrics`` is an empty dict (metrics reported in ``evaluate``).
        """
        self.set_parameters(parameters)

        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        proximal_mu = config.get("proximal_mu", 0.2)
        weight_decay = config.get("weight_decay", 1e-4)

        # Snapshot of global params for FedProx proximal term
        if proximal_mu > 0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        train_dataset = data.TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        self.model.train()
        best_params = None
        best_val_loss = float('inf')
        patience = 3
        no_improve_epochs = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)

                # Explicit L2 regularisation (supplements weight_decay in Adam)
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += weight_decay * l2_reg

                # Label smoothing: blend hard targets with uniform distribution
                if self.label_smoothing > 0:
                    smooth_loss = -F.log_softmax(outputs, dim=1).mean(dim=1).mean()
                    loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

                # FedProx proximal term: discourage client drift from global model
                if proximal_mu > 0:
                    proximal_term = 0.
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += (local_param - global_param).norm(2)
                    loss += (proximal_mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            val_loss, val_metrics = self._evaluate_validation()
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {epoch_loss/len(self.X_train):.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}, loading best model...")
                    if best_params is not None:
                        self.set_parameters(best_params)
                    break

        return self.get_parameters(), len(self.X_train), {}

    def _evaluate_validation(self) -> Tuple[float, Dict[str, float]]:
        """Compute validation loss and accuracy on the local test split.

        Temporarily switches the model to eval mode (disables dropout/batchnorm
        training behaviour), then restores train mode.

        Returns:
            tuple: ``(val_loss, metrics_dict)`` where ``metrics_dict`` contains
                ``{"accuracy": float}`` with value in [0, 1].
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_test_tensor = torch.LongTensor(self.y_test).to(self.device)

            outputs = self.model(X_test_tensor)
            val_loss = self.criterion(outputs, y_test_tensor).item()

            _, predicted = torch.max(outputs.data, 1)
            total = y_test_tensor.size(0)
            correct = (predicted == y_test_tensor).sum().item()

        self.model.train()
        return val_loss, {"accuracy": correct / total if total > 0 else 0.0}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict[str, float]]:
        """Load global parameters and evaluate on the local test set.

        Computes a comprehensive set of classification metrics. Handles the edge
        case where the model predicts only one class (sets precision/recall/F1 to 0).

        Args:
            parameters (list[np.ndarray]): Global model weights to evaluate.
            config (dict): Evaluation config (``round`` key present).

        Returns:
            tuple: ``(loss, num_test_examples, metrics)`` where ``metrics`` is a
                dict with keys: ``accuracy``, ``precision``, ``recall``, ``f1_score``.
        """
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_test_tensor = torch.LongTensor(self.y_test).to(self.device)

            outputs = self.model(X_test_tensor)
            loss = self.criterion(outputs, y_test_tensor).item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test_tensor)

            y_true = y_test_tensor.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            if len(np.unique(y_pred)) == 1:
                # Model collapsed to predicting a single class — metrics undefined
                precision = recall = f1 = 0.0
            else:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            logger.info(
                f"Evaluation - Loss: {loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1: {f1:.4f}"
            )

        return float(loss), len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }


def get_local_ip() -> str:
    """Determine the machine's primary LAN IP address.

    Uses a UDP socket connect trick — connecting to an external address causes
    the OS to select the appropriate outgoing interface without actually sending
    any data.

    Returns:
        str: Local IP address string (e.g. ``"192.168.1.105"``), or
            ``"0.0.0.0"`` if the address cannot be determined.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP address: {e}")
        return "0.0.0.0"


def run_client() -> None:
    """Entry point: parse CLI args, load data, and start the Flower client.

    Argument parsing:
        --server-address (str): ``HOST:PORT`` of the Flower server.
            Default: ``SERVER_IP:8080``.
        --client-id (int): Unique identifier for this client. *Required.*
            Determines which dataset partition file to load.
        --batch-size (int): Mini-batch size for local training.
            Default: ``32``.

    Raises:
        SystemExit: If a fatal error occurs during data loading or server connection.
    """
    parser = argparse.ArgumentParser(
        description='Federated Learning Client for Diabetes Prediction'
    )
    parser.add_argument(
        '--server-address', type=str, default=f"{SERVER_IP}:8080",
        help='Address of the server in the format host:port (default: SERVER_IP:8080)'
    )
    parser.add_argument(
        '--client-id', type=int, required=True,
        help='Unique identifier for this client (required)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for training (default: 32)'
    )

    args = parser.parse_args()

    # Select dataset partition based on client ID
    if args.client_id == 1:
        dataset_file = 'diabetes_non_negative_part1_2000.csv'
        logger.info(f"Client {args.client_id}: Loading dataset part 1")
    elif args.client_id == 2:
        dataset_file = 'diabetes_non_negative_part2_2000.csv'
        logger.info(f"Client {args.client_id}: Loading dataset part 2")
    else:
        dataset_file = 'diabetes_non_negative_part1_2000.csv'
        logger.info(f"Client {args.client_id}: Loading default dataset part 1")

    try:
        logger.info("=" * 50)
        logger.info(f"Starting Federated Learning Client (ID: {args.client_id})")
        logger.info("=" * 50)

        X_train, X_test, y_train, y_test = load_diabetes_data(file_path=dataset_file)

        logger.info("Dataset loaded successfully:")
        logger.info(f"- Training samples: {len(X_train)}")
        logger.info(f"- Test samples: {len(X_test)}")
        logger.info(f"- Number of features: {X_train.shape[1]}")

        input_size = X_train.shape[1]
        model = DiabetesModel(input_size=input_size)

        logger.info("\nModel Architecture:")
        logger.info(model)

        client = FlowerClient(model, X_train, y_train, X_test, y_test)

        logger.info("\nClient Configuration:")
        logger.info("-" * 20)
        logger.info(f"Client ID: {args.client_id}")
        logger.info(f"Server: {args.server_address}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Device: {client.device}")
        logger.info("-" * 20)

        logger.info("\nConnecting to server...")
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )

        logger.info("Client finished successfully!")

    except Exception as e:
        logger.error(f"Error in client: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    run_client()