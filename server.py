"""
server.py — Federated Learning Server for Diabetes Risk Prediction
==================================================================

This module starts the Flower (flwr) federated learning server and a FastAPI
prediction endpoint.

Responsibilities:
    - Define the global ``DiabetesModel`` neural network architecture.
    - Implement ``CustomFedProx`` aggregation strategy (extends FedProx),
      which handles non-IID data distributions across hospital clients.
    - Coordinate multiple FL training rounds (default: 30).
    - Expose REST endpoints on port 5000 via FastAPI + Uvicorn:
        GET  /         — HTML prediction form
        POST /predict  — Diabetes risk prediction
        GET  /health   — Server and training status
        GET  /metrics  — Latest per-round training metrics
    - Save best model checkpoints per round to checkpoints/
    - Append per-round metrics to training_history.json

Usage::

    python server.py

See Also:
    client.py      — FL client implementation
    config.py      — All hyperparameters and paths
    docs/architecture.md — In-depth architecture documentation
"""

import os
import json
import time
import flwr as fl
import torch.nn as nn
import torch
import socket
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedProx
from flwr.server.history import History
from flwr.common import Parameters, Scalar, FitRes, NDArrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training state (shared between FL callbacks and FastAPI endpoints)
# ---------------------------------------------------------------------------
_training_state: Dict = {
    "current_round": 0,
    "total_rounds": config.NUM_ROUNDS,
    "training_complete": False,
    "best_accuracy": 0.0,
    "latest_metrics": {},
    "history": [],
}


class DiabetesModel(nn.Module):
    """Feed-forward neural network for binary diabetes risk classification.

    Architecture (input → hidden → output):
        Layer 1: Linear(input_size→32) + LeakyReLU + BatchNorm + Dropout(0.5)
        Layer 2: Linear(32→16)         + LeakyReLU + BatchNorm + Dropout(0.4)
        Layer 3: Linear(16→8)          + LeakyReLU + BatchNorm + Dropout(0.3)
        Output:  Linear(8→num_classes) — raw logits

    Weights initialised with Kaiming Normal (optimal for LeakyReLU).

    Args:
        input_size (int): Number of input features. Default: 8 (Pima dataset).
        num_classes (int): Number of output classes. Default: 2 (binary).
    """

    def __init__(self, input_size: int = config.INPUT_SIZE, num_classes: int = config.NUM_CLASSES):
        super(DiabetesModel, self).__init__()
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
        """Initialise weights with Kaiming Normal; BatchNorm with identity."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Handles single-sample 1-D input by unsqueezing.

        Args:
            x: Shape ``(batch_size, input_size)`` or ``(input_size,)``.

        Returns:
            Raw logits of shape ``(batch_size, num_classes)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Global model (initialised once; kept in sync after every aggregation)
# ---------------------------------------------------------------------------
model = DiabetesModel(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES)
logger.info("Server model architecture:\n%s", model)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_initial_parameters(model: nn.Module) -> list:
    """Extract model parameters as detached CPU numpy arrays.

    Args:
        model: PyTorch model to extract parameters from.

    Returns:
        List of numpy arrays, one per trainable parameter tensor.
    """
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Compute weighted mean accuracy across clients.

    Args:
        metrics: List of ``(num_examples, metrics_dict)`` tuples.

    Returns:
        ``{"accuracy": float}`` — weighted mean accuracy.
    """
    accuracies = [n * m["accuracy"] for n, m in metrics]
    examples = [n for n, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Build training config sent to clients before each fit round.

    Args:
        server_round: Current FL round number (1-indexed).

    Returns:
        Config dict with ``round``, ``epochs``, and ``proximal_mu``.
    """
    _training_state["current_round"] = server_round
    return {
        "round": server_round,
        "epochs": config.LOCAL_EPOCHS,
        "proximal_mu": config.PROXIMAL_MU,
        "batch_size": config.BATCH_SIZE,
        "weight_decay": config.WEIGHT_DECAY,
    }


def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Build evaluation config sent to clients before each evaluate round.

    Args:
        server_round: Current FL round number (1-indexed).

    Returns:
        Config dict with ``round`` key.
    """
    return {"round": server_round}


def _save_checkpoint(round_num: int, accuracy: float) -> None:
    """Save a model checkpoint if accuracy improved.

    Checkpoints are written to ``config.CHECKPOINTS_DIR/model_round_{N}.pth``.
    Only saves if the current round accuracy beats the best seen so far.

    Args:
        round_num: Current FL round number.
        accuracy: Aggregated accuracy for this round.
    """
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    if accuracy > _training_state["best_accuracy"]:
        _training_state["best_accuracy"] = accuracy
        path = os.path.join(config.CHECKPOINTS_DIR, f"model_round_{round_num}.pth")
        torch.save(model.state_dict(), path)
        logger.info("✅ New best model saved: %s (accuracy=%.4f)", path, accuracy)


def _append_history(round_num: int, metrics: Dict) -> None:
    """Append round metrics to training_history.json and in-memory state.

    Args:
        round_num: Current FL round number.
        metrics: Aggregated metrics dict for this round.
    """
    entry = {"round": round_num, **metrics}
    _training_state["history"].append(entry)
    _training_state["latest_metrics"] = entry

    try:
        history = []
        if os.path.exists(config.TRAINING_HISTORY_PATH):
            with open(config.TRAINING_HISTORY_PATH, "r") as f:
                history = json.load(f)
        history.append(entry)
        with open(config.TRAINING_HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning("Could not write training history: %s", e)


# ---------------------------------------------------------------------------
# Custom FedProx Strategy
# ---------------------------------------------------------------------------

class CustomFedProx(FedProx):
    """FedProx strategy that syncs the in-memory global model and saves checkpoints.

    Extends Flower's ``FedProx`` to:
    - Update the global ``DiabetesModel`` after every aggregation.
    - Save model checkpoints when accuracy improves.
    - Persist per-round metrics to ``training_history.json``.
    - Correctly pass the ``proximal_mu`` config to clients via ``FitIns``.

    Args:
        *args, **kwargs: Forwarded to ``FedProx.__init__``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = model.to(self.device)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates, sync global model, save checkpoint.

        Args:
            server_round: Current FL round number.
            results: Successful (client_proxy, FitRes) pairs.
            failures: Failed clients or exceptions.

        Returns:
            (aggregated_parameters, metrics_aggregated) or (None, {}) if no results.
        """
        if not results:
            return None, {}

        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Sync global model with new aggregated weights
            parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = {
                k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in params_dict
            }
            self.global_model.load_state_dict(state_dict, strict=True)

            # Save checkpoint if accuracy improved
            accuracy = metrics_aggregated.get("accuracy", 0.0)
            _save_checkpoint(server_round, accuracy)
            _append_history(server_round, {"accuracy": accuracy})

        return aggregated_parameters, metrics_aggregated

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure clients for the upcoming fit round.

        FIX: Previously built a ``config`` dict but never passed it to the parent,
        so ``proximal_mu`` was never sent to clients. Now correctly creates
        ``FitIns`` with the config and returns it directly.

        Args:
            server_round: Current FL round number.
            parameters: Current global model parameters.
            client_manager: Manages available client proxies.

        Returns:
            List of (client_proxy, FitIns) for each sampled client.
        """
        config_dict = {
            "round": server_round,
            "epochs": config.LOCAL_EPOCHS,
            "proximal_mu": config.PROXIMAL_MU,
            "batch_size": config.BATCH_SIZE,
            "weight_decay": config.WEIGHT_DECAY,
        }
        fit_ins = fl.common.FitIns(parameters, config_dict)
        clients = client_manager.sample(
            num_clients=max(
                config.MIN_FIT_CLIENTS,
                int(self.fraction_fit * client_manager.num_available()),
            ),
            min_num_clients=config.MIN_FIT_CLIENTS,
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure clients for post-aggregation evaluation.

        Args:
            server_round: Current FL round number.
            parameters: Aggregated global model parameters.
            client_manager: Manages available client proxies.

        Returns:
            List of (client_proxy, EvaluateIns) for each sampled client.
        """
        eval_config = {"round": server_round}
        clients = client_manager.sample(
            num_clients=min(config.MIN_EVALUATE_CLIENTS, client_manager.num_available()),
            min_num_clients=config.MIN_EVALUATE_CLIENTS,
        )
        return [(client, fl.common.EvaluateIns(parameters, eval_config)) for client in clients]


# ---------------------------------------------------------------------------
# Strategy instantiation
# ---------------------------------------------------------------------------
strategy = CustomFedProx(
    min_fit_clients=config.MIN_FIT_CLIENTS,
    min_available_clients=config.MIN_AVAILABLE_CLIENTS,
    min_evaluate_clients=config.MIN_EVALUATE_CLIENTS,
    fraction_fit=config.FRACTION_FIT,
    fraction_evaluate=config.FRACTION_EVALUATE,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(model)),
    proximal_mu=config.PROXIMAL_MU,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start FastAPI prediction server and Flower FL server.

    Execution order:
        1. Create FastAPI app with ``/``, ``/predict``, ``/health``, ``/metrics``.
        2. Launch FastAPI + Uvicorn in a background daemon thread on API port.
        3. Start Flower gRPC server on FL port for NUM_ROUNDS rounds.
        4. Save final global model to ``global_model_final.pth``.
        5. Enter keep-alive loop (Ctrl+C to exit).
    """
    import threading
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi import Request
    import uvicorn

    app = FastAPI(
        title="Diabetes Risk Prediction API",
        description=(
            "Federated learning-based diabetes risk assessment using the "
            "Pima Indians Diabetes Dataset. Powered by PyTorch + Flower."
        ),
        version="2.0.0",
    )

    # Mount static files if the directory exists
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    templates = Jinja2Templates(directory="templates")

    # ------------------------------------------------------------------
    # GET /
    # ------------------------------------------------------------------
    @app.get("/", summary="Serve the prediction web form")
    async def read_root(request: Request):
        """Serve the Jinja2 HTML form for manual diabetes risk predictions."""
        return templates.TemplateResponse("index.html", {"request": request})

    # ------------------------------------------------------------------
    # POST /predict
    # ------------------------------------------------------------------
    @app.post("/predict", summary="Predict diabetes risk from Pima patient features")
    async def predict(data: dict):
        """Accept patient health features and return a diabetes risk assessment.

        Expected JSON body keys (all Pima Indians Diabetes Dataset columns):
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age

        Missing fields default to 0.

        Returns:
            JSON with: status, risk_percentage, risk_level, interpretation, recommendation.

        Raises:
            HTTPException 500: if model inference fails.
        """
        try:
            input_features = [
                float(data.get("Pregnancies", 0)),
                float(data.get("Glucose", 0)),
                float(data.get("BloodPressure", 0)),
                float(data.get("SkinThickness", 0)),
                float(data.get("Insulin", 0)),
                float(data.get("BMI", 0)),
                float(data.get("DiabetesPedigreeFunction", 0)),
                float(data.get("Age", 0)),
            ]

            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor([input_features])
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                risk_percentage = round(probabilities[0][1].item() * 100, 2)

            logger.debug("Model output: %s", output)
            logger.debug("Probabilities: %s", probabilities)
            logger.info("Prediction: risk=%.2f%%", risk_percentage)

            if risk_percentage < config.LOW_RISK_THRESHOLD:
                risk_level = "Low"
                recommendation = (
                    "Maintain a healthy lifestyle with regular exercise and a balanced diet. "
                    "Annual check-ups are recommended."
                )
            elif risk_percentage < config.HIGH_RISK_THRESHOLD:
                risk_level = "Moderate"
                recommendation = (
                    "Consider lifestyle changes and regular blood glucose monitoring. "
                    "Consult a healthcare provider for personalised advice."
                )
            else:
                risk_level = "High"
                recommendation = (
                    "Please consult an endocrinologist for a comprehensive evaluation "
                    "and diabetes management plan."
                )

            return {
                "status": "success",
                "risk_percentage": f"{risk_percentage}%",
                "risk_level": risk_level,
                "interpretation": (
                    f"Based on the provided health data, the patient has a {risk_percentage}% "
                    f"probability of having diabetes. This is considered {risk_level} risk."
                ),
                "recommendation": recommendation,
            }

        except Exception as e:
            logger.exception("Prediction error")
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health", summary="Server and training status")
    async def health():
        """Return the current server and FL training status.

        Returns:
            JSON with: status, current_round, total_rounds, training_complete,
            best_accuracy, uptime_seconds.
        """
        return {
            "status": "ok",
            "current_round": _training_state["current_round"],
            "total_rounds": _training_state["total_rounds"],
            "training_complete": _training_state["training_complete"],
            "best_accuracy": round(_training_state["best_accuracy"], 4),
        }

    # ------------------------------------------------------------------
    # GET /metrics
    # ------------------------------------------------------------------
    @app.get("/metrics", summary="Latest per-round training metrics")
    async def metrics():
        """Return aggregated metrics from the most recent completed FL round.

        Returns:
            JSON with: latest_round metrics and full training history list.
        """
        return {
            "latest": _training_state["latest_metrics"],
            "history": _training_state["history"],
        }

    # ------------------------------------------------------------------
    # Start FastAPI in daemon thread
    # ------------------------------------------------------------------
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=config.SERVER_API_PORT)

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    machine_ip = socket.gethostbyname(socket.gethostname())

    print("\n" + "=" * 55)
    print("  🫀 Diabetes Risk Prediction — FL Server")
    print("=" * 55)
    print(f"  Flower gRPC : 0.0.0.0:{config.SERVER_FL_PORT}")
    print(f"  FastAPI     : http://0.0.0.0:{config.SERVER_API_PORT}")
    print(f"  Machine IP  : {machine_ip}")
    print(f"  Rounds      : {config.NUM_ROUNDS}")
    print(f"  Min clients : {config.MIN_FIT_CLIENTS}")
    print(f"  Clients connect to → {machine_ip}:{config.SERVER_FL_PORT}")
    print("=" * 55 + "\n")

    try:
        history = fl.server.start_server(
            server_address=f"0.0.0.0:{config.SERVER_FL_PORT}",
            config=fl.server.ServerConfig(
                num_rounds=config.NUM_ROUNDS,
                round_timeout=config.ROUND_TIMEOUT,
            ),
            strategy=strategy,
        )

        _training_state["training_complete"] = True

        torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
        logger.info("Final global model saved → %s", config.FINAL_MODEL_PATH)

        print("\n" + "=" * 55)
        print("  ✅ Training complete!")
        print(f"  Best accuracy : {_training_state['best_accuracy']:.4f}")
        print(f"  Final model   : {config.FINAL_MODEL_PATH}")
        print(f"  History       : {config.TRAINING_HISTORY_PATH}")
        print(f"  API running   : http://{machine_ip}:{config.SERVER_API_PORT}/predict")
        print("  Press Ctrl+C to stop.")
        print("=" * 55 + "\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()