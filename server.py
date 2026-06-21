"""
server.py — Federated Learning Server for Heart Disease / Diabetes Prediction
==============================================================================

This module starts the Flower (flwr) federated learning server and a FastAPI
prediction endpoint.

Responsibilities:
    - Define the global ``DiabetesModel`` neural network architecture.
    - Implement the ``CustomFedProx`` aggregation strategy (extends FedProx),
      which handles non-IID data distributions across hospital clients.
    - Coordinate multiple FL training rounds (default: 30).
    - Expose a REST prediction API on port 5000 via FastAPI + Uvicorn.
    - Save the final aggregated global model to disk after training.

Usage::

    python server.py

The server will:
    1. Bind the Flower gRPC server to ``0.0.0.0:8080`` (clients connect here).
    2. Start FastAPI on ``0.0.0.0:5000`` (prediction endpoint).
    3. Wait for at least 2 clients before starting training rounds.

See Also:
    client.py      — FL client implementation
    docs/architecture.md — In-depth architecture documentation
"""

import time
import flwr as fl
import torch.nn as nn
import torch
import socket
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedProx
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.common import Parameters, Scalar, FitRes, Parameters, NDArrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesModel(nn.Module):
    """Feed-forward neural network for binary diabetes / heart disease classification.

    Architecture:
        - Layer 1: Linear(input_size → 32) + LeakyReLU + BatchNorm + Dropout(0.5)
        - Layer 2: Linear(32 → 16)        + LeakyReLU + BatchNorm + Dropout(0.4)
        - Layer 3: Linear(16 → 8)         + LeakyReLU + BatchNorm + Dropout(0.3)
        - Output:  Linear(8 → num_classes)

    Dropout rates decrease with depth to allow the network to form increasingly
    abstract representations with less noise in later layers.

    Args:
        input_size (int): Number of input features. Default is 8 (Pima dataset).
        num_classes (int): Number of output classes. Default is 2 (binary).

    Example::

        model = DiabetesModel(input_size=8, num_classes=2)
        x = torch.randn(16, 8)   # batch of 16 samples
        logits = model(x)         # shape: (16, 2)
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

        # Initialize weights using Kaiming Normal (optimal for LeakyReLU)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise layer weights using Kaiming Normal for Linear layers.

        BatchNorm layers are initialised with weight=1 and bias=0 (identity
        transform), which is standard practice to avoid disrupting the initial
        feature distributions.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers.

        Automatically handles single-sample (1-D) input by unsqueezing to add
        a batch dimension, which is required by BatchNorm1d.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_size)``
                or ``(input_size,)`` for a single sample.

        Returns:
            torch.Tensor: Raw logits of shape ``(batch_size, num_classes)``.
                Apply ``torch.softmax`` to convert to probabilities.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


# Initialize global model with 8 features (Pima dataset columns)
model = DiabetesModel(input_size=8, num_classes=2)


def init_weights(m: nn.Module) -> None:
    """Apply Xavier Uniform initialisation to Linear layers.

    Used as an alternative weight init applied via ``model.apply(init_weights)``.
    Xavier Uniform works well for symmetric activation functions (tanh, sigmoid).

    Args:
        m (nn.Module): A module whose weights will be initialised in-place.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


model.apply(init_weights)

logger.info("Server model architecture:")
logger.info(model)


def get_initial_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of numpy arrays.

    Parameters are detached from the computation graph and moved to CPU before
    conversion to avoid issues with GPU tensors and in-place modifications.

    Args:
        model (nn.Module): The PyTorch model whose parameters to extract.

    Returns:
        list[np.ndarray]: List of parameter arrays, one per trainable layer.
    """
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Compute a weighted average of per-client accuracy metrics.

    Weights are proportional to the number of examples each client evaluated on,
    ensuring larger clients contribute proportionally more to the aggregate metric.

    Args:
        metrics (list): List of ``(num_examples, metrics_dict)`` tuples from each
            client's ``evaluate()`` call.

    Returns:
        dict: ``{"accuracy": float}`` — the weighted mean accuracy across all clients.

    Example::

        metrics = [(100, {"accuracy": 0.80}), (200, {"accuracy": 0.90})]
        result = weighted_average(metrics)
        # result == {"accuracy": 0.8667}  (200-sample client weighs more)
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Build the training configuration dict sent to clients before each fit round.

    Args:
        server_round (int): The current FL round number (1-indexed).

    Returns:
        dict: Configuration with keys ``round`` and ``epochs``.
    """
    return {
        "round": server_round,
        "epochs": 1,
    }


def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Build the evaluation configuration dict sent to clients before each evaluate round.

    Args:
        server_round (int): The current FL round number (1-indexed).

    Returns:
        dict: Configuration with key ``round``.
    """
    return {"round": server_round}


class CustomFedProx(FedProx):
    """Custom FedProx strategy with in-memory global model updates.

    Extends Flower's built-in ``FedProx`` strategy to additionally update an
    in-memory ``DiabetesModel`` instance after every aggregation step. This
    global model is used by the FastAPI prediction endpoint.

    FedProx adds a proximal term ``(μ/2) · ‖w_local − w_global‖²`` to each
    client's loss function, preventing client drift on non-IID data.

    Args:
        *args: Passed through to ``FedProx.__init__``.
        **kwargs: Passed through to ``FedProx.__init__``.

    Attributes:
        global_test_loader: Reserved for optional server-side evaluation loader.
        device (torch.device): CPU or CUDA device for the global model.
        global_model (DiabetesModel): In-memory global model kept in sync with
            the aggregated Flower parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_test_loader = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = model.to(self.device)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client model updates and sync the in-memory global model.

        Delegates the actual aggregation to the parent ``FedProx.aggregate_fit``,
        then loads the resulting parameters into ``self.global_model`` so that the
        FastAPI prediction endpoint always serves the latest aggregated weights.

        Args:
            server_round (int): Current FL round number.
            results (list): Successful client (proxy, FitRes) pairs.
            failures (list): Failed clients or exceptions from this round.

        Returns:
            tuple: ``(aggregated_parameters, metrics_aggregated)`` where
                ``aggregated_parameters`` is a Flower ``Parameters`` object or
                ``None`` if no results were received.
        """
        if not results:
            return None, {}

        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                         for k, v in params_dict}
            self.global_model.load_state_dict(state_dict, strict=True)

        return aggregated_parameters, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure clients for the upcoming fit round.

        Passes the ``proximal_mu`` hyperparameter to clients so they can apply
        the FedProx proximal regularisation term in their local loss.

        Args:
            server_round (int): Current FL round number.
            parameters (Parameters): Current global model parameters.
            client_manager (ClientManager): Manages available client proxies.

        Returns:
            list: ``[(client_proxy, FitIns), ...]`` for each sampled client.
        """
        config = {
            "round": server_round,
            "epochs": 1,
            "proximal_mu": 0.1,
        }
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure a subset of clients for evaluation after aggregation.

        Args:
            server_round (int): Current FL round number.
            parameters (Parameters): Aggregated global model parameters to evaluate.
            client_manager (ClientManager): Manages available client proxies.

        Returns:
            list: ``[(client_proxy, EvaluateIns), ...]`` for each sampled client.
        """
        config = {"round": server_round}
        clients = client_manager.sample(
            num_clients=min(self.min_evaluate_clients, client_manager.num_available()),
            min_num_clients=self.min_evaluate_clients,
        )
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]


# Define FedProx strategy — requires 2 clients minimum
strategy = CustomFedProx(
    min_fit_clients=2,        # Require 2 clients for training
    min_available_clients=2,  # Need 2 clients to be available
    min_evaluate_clients=2,   # Evaluate on both clients
    fraction_fit=1.0,         # Use 100% of available clients for training
    fraction_evaluate=1.0,    # Evaluate on 100% of available clients
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(model)),
    proximal_mu=0.1,
)


class PersistentServer(fl.server.Server):
    """Flower server that stays alive after all training rounds complete.

    The default Flower server exits once ``num_rounds`` rounds finish. This
    subclass overrides ``run()`` to keep the process alive, allowing new clients
    to connect for additional training or evaluation after the initial run.

    Args:
        *args: Passed through to ``fl.server.Server.__init__``.
        **kwargs: Passed through to ``fl.server.Server.__init__``.

    Attributes:
        keep_running (bool): Set to ``False`` externally to shut down the loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_running = True

    def run(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run all FL rounds then enter an idle keep-alive loop.

        Args:
            num_rounds (int): Number of FL rounds to execute.
            timeout (float or None): Per-round timeout in seconds.

        Returns:
            History: Flower training history containing per-round metrics.
        """
        history = super().run(num_rounds, timeout)
        print("\nTraining completed. Keeping server alive for new connections...")
        while self.keep_running:
            time.sleep(1)
        return history


def main() -> None:
    """Entry point: start FastAPI prediction server and Flower FL server.

    Execution order:
        1. Create FastAPI app with ``/`` (HTML form) and ``/predict`` endpoints.
        2. Launch FastAPI + Uvicorn in a background daemon thread on port 5000.
        3. Start Flower gRPC server on port 8080 for 30 FL rounds.
        4. Save the final global model to ``global_model_final.pth``.
        5. Enter infinite keep-alive loop (Ctrl+C to exit).
    """
    import threading
    from fastapi import FastAPI, HTTPException
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi import Request

    app = FastAPI(
        title="Diabetes Risk Prediction API",
        description="Federated learning-based diabetes risk assessment",
        version="1.0.0"
    )

    app.mount("/static", StaticFiles(directory="static"), name="static")

    import os
    templates = Jinja2Templates(directory="templates")

    @app.get("/", summary="Serve the prediction web form")
    async def read_root(request: Request):
        """Serve the Jinja2 HTML form for manual predictions."""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/predict", summary="Predict diabetes risk from patient features")
    async def predict(data: dict):
        """Accept patient health features and return a diabetes risk assessment.

        The model expects 8 numeric features matching the Pima Indians Diabetes
        dataset columns. Missing fields default to 0.

        Args:
            data (dict): JSON body with keys: ``Pregnancies``, ``Glucose``,
                ``BloodPressure``, ``SkinThickness``, ``Insulin``, ``BMI``,
                ``DiabetesPedigreeFunction``, ``Age``.

        Returns:
            dict: ``status``, ``risk_percentage``, ``risk_level``,
                ``interpretation``, ``recommendation``.

        Raises:
            HTTPException: 500 if model inference fails.
        """
        try:
            input_features = [
                float(data.get('Pregnancies', 0)),
                float(data.get('Glucose', 0)),
                float(data.get('BloodPressure', 0)),
                float(data.get('SkinThickness', 0)),
                float(data.get('Insulin', 0)),
                float(data.get('BMI', 0)),
                float(data.get('DiabetesPedigreeFunction', 0)),
                float(data.get('Age', 0))
            ]

            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor([input_features])
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                risk_percentage = round(probabilities[0][1].item() * 100, 2)

                logger.info(f"Model output: {output}")
                logger.info(f"Probabilities: {probabilities}")
                logger.info(f"Risk percentage: {risk_percentage}%")

            if risk_percentage < 25:
                risk_level = "Low"
                recommendation = "Maintain a healthy lifestyle with regular exercise and balanced diet."
            elif risk_percentage < 75:
                risk_level = "Moderate"
                recommendation = "Consider lifestyle changes and regular monitoring. Consult a healthcare provider."
            else:
                risk_level = "High"
                recommendation = "Please consult with an endocrinologist for a comprehensive evaluation and management plan."

            return {
                "status": "success",
                "risk_percentage": f"{risk_percentage}%",
                "risk_level": risk_level,
                "interpretation": (
                    f"Based on the provided health data, the patient has a {risk_percentage}% "
                    f"risk of developing diabetes. This is considered {risk_level} risk."
                ),
                "recommendation": recommendation
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def run_fastapi():
        """Run Uvicorn in a daemon thread (non-blocking)."""
        uvicorn.run(app, host="0.0.0.0", port=5000)

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    server_ip = '0.0.0.0'
    machine_ip = socket.gethostbyname(socket.gethostname())

    print("\n" + "="*50)
    print("Starting Diabetes Prediction Server")
    print("="*50)
    print(f"Flower server binding to: {server_ip}:8080")
    print(f"Machine IP Address: {machine_ip}")
    print(f"FastAPI server running on: http://{server_ip}:5000")
    print(f"\nClients should connect to: {machine_ip}:8080")
    print("\nServer is running. Will work with 1 or 2 clients...")
    print("="*50)

    try:
        history = fl.server.start_server(
            server_address=f"{server_ip}:8080",
            config=fl.server.ServerConfig(
                num_rounds=30,
                round_timeout=120,
            ),
            strategy=strategy
        )

        torch.save(model.state_dict(), 'global_model_final.pth')
        print("\nFinal global model saved to global_model_final.pth")

        print("\n" + "="*50)
        print("Training completed! Starting server test...")
        print("="*50)

        import subprocess
        import sys
        python = sys.executable
        subprocess.Popen([python, "test_server.py"])

        print("\nServer is running and ready for predictions.")
        print("Press Ctrl+C to stop the server.")
        print("Test the prediction endpoint with: python test_predict.py")
        print("="*50 + "\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()