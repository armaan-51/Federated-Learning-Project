# Architecture Deep-Dive

This document explains the internals of the Federated Learning system for heart disease / diabetes prediction.

---

## Table of Contents

- [Federated Learning Overview](#federated-learning-overview)
- [Why FedProx?](#why-fedprox)
- [Round Lifecycle](#round-lifecycle)
- [Model Architecture](#model-architecture)
- [Data Flow](#data-flow)
- [Server Components](#server-components)
- [Client Components](#client-components)
- [API Layer](#api-layer)
- [Design Decisions](#design-decisions)

---

## Federated Learning Overview

Traditional centralised ML:
```
Hospital A data ─┐
Hospital B data ─┼──► Central Server ──► Trained Model
Hospital C data ─┘      (stores all raw patient data)
```

**Problem:** Raw patient data must leave the hospital — a major privacy and compliance risk (HIPAA, GDPR).

Federated Learning:
```
Hospital A ──► local training ──► gradients/weights ─┐
Hospital B ──► local training ──► gradients/weights ─┼──► Server (aggregates weights)
Hospital C ──► local training ──► gradients/weights ─┘         │
         ▲                                                       │
         └─────────────── updated global model ─────────────────┘
```

**Key guarantee:** Raw data **never** leaves the local institution. Only model weight updates travel the network.

---

## Why FedProx?

The standard **FedAvg** algorithm assumes client datasets are identically distributed (IID). In practice, one hospital might see mostly elderly patients while another sees mostly young patients — creating **non-IID data**.

FedAvg with non-IID data suffers from **client drift**: each client's local model drifts far from the global optimum, making aggregation unreliable.

**FedProx** solves this by adding a proximal term to each client's local loss function:

```
L_local(w) = L_CE(w) + (μ/2) · ‖w − w_global‖²
```

- `L_CE` is the standard cross-entropy loss
- `μ` (proximal_mu = 0.1) penalises the local model for drifting too far from the last global model `w_global`
- Higher μ → more conservative local updates, slower but more stable convergence
- Lower μ → more local optimisation, faster but risks divergence

---

## Round Lifecycle

Each federated round goes through these phases:

```
┌─────────────────────────────────────────────────────────────┐
│  Round N                                                     │
│                                                              │
│  1. SERVER: sample clients (min_fit_clients = 2)            │
│       │                                                      │
│  2. SERVER → CLIENT: send global model weights + config     │
│       │                                                      │
│  3. CLIENT: set_parameters() — load global weights          │
│       │                                                      │
│  4. CLIENT: fit() — local training for `epochs` epochs      │
│       │   • forward pass                                      │
│       │   • cross-entropy + L2 + FedProx proximal term      │
│       │   • Adam optimizer step                              │
│       │   • validation + early stopping                      │
│       │                                                      │
│  5. CLIENT → SERVER: send updated weights + num_examples    │
│       │                                                      │
│  6. SERVER: aggregate_fit()                                  │
│       │   • weighted average of all client weights           │
│       │   • update global DiabetesModel                     │
│       │                                                      │
│  7. SERVER → CLIENT: send aggregated weights for evaluation │
│       │                                                      │
│  8. CLIENT: evaluate() — compute loss, accuracy, F1, etc.  │
│       │                                                      │
│  9. SERVER: log round metrics, proceed to Round N+1        │
└─────────────────────────────────────────────────────────────┘
```

Total rounds: **30** (configurable via `num_rounds` in `server.py`).

---

## Model Architecture

`DiabetesModel` is a 4-layer feed-forward neural network:

```
Input Layer
  size: 8 (one neuron per feature)

Layer 1
  Linear(8 → 32)
  LeakyReLU(negative_slope=0.1)   ← avoids dying ReLU
  BatchNorm1d(32)                  ← stabilises training
  Dropout(p=0.5)                   ← regularisation

Layer 2
  Linear(32 → 16)
  LeakyReLU(0.1)
  BatchNorm1d(16)
  Dropout(p=0.4)

Layer 3
  Linear(16 → 8)
  LeakyReLU(0.1)
  BatchNorm1d(8)
  Dropout(p=0.3)

Output Layer
  Linear(8 → 2)                   ← 2 logits: [no_disease, disease]
  (softmax applied at inference)
```

**Weight initialisation:** Kaiming Normal (`fan_out`, `leaky_relu`) for Linear layers; ones/zeros for BatchNorm — matches the LeakyReLU activation function mathematically.

**Parameter count:** approximately 1,650 trainable parameters (small by design — fast to aggregate over the network).

---

## Data Flow

### Client-side preprocessing

```
Raw CSV
  └──► pd.read_csv()
         └──► Extract 8 feature columns
                └──► train_test_split (80/20, stratified, seed=42)
                       └──► StandardScaler.fit_transform (train)
                              └──► StandardScaler.transform (test)
                                     └──► torch.FloatTensor / LongTensor
```

Each client fits its **own scaler** on its own training data — this is intentional, as centralising the scaler would require sharing data statistics. A shared global scaler (`global_scaler.pkl`) is pre-computed and used for inference via the API.

### Server-side aggregation

```
Client 1 weights (n₁ examples)  ─┐
Client 2 weights (n₂ examples)  ─┼──► weighted_average()
                                  │         w_global = Σ(nᵢ · wᵢ) / Σnᵢ
                                  └──► DiabetesModel.load_state_dict()
```

---

## Server Components

### `DiabetesModel`
Identical architecture to the client model. Holds the global (aggregated) state.

### `CustomFedProx(FedProx)`
Extends Flower's built-in `FedProx` strategy with:
- `aggregate_fit()`: updates the in-memory global model after each round
- `configure_fit()`: passes the `proximal_mu` hyperparameter to clients
- `configure_evaluate()`: samples clients for evaluation

### `PersistentServer`
Keeps the Flower server alive after all training rounds complete so it remains ready for new client connections.

### FastAPI App
Runs in a daemon thread on **port 5000**, independent of the FL server on port 8080:
- `GET /` — serves the Jinja2 HTML prediction form
- `POST /predict` — runs the global model on submitted patient features

---

## Client Components

### `FlowerClient(fl.client.NumPyClient)`
Implements the Flower client interface:

| Method | Purpose |
|---|---|
| `get_parameters()` | Serialize model weights → list of numpy arrays |
| `set_parameters()` | Deserialize received weights → load into model (with shape mismatch handling) |
| `fit()` | Local training loop with FedProx, early stopping, LR scheduling |
| `evaluate()` | Compute loss + accuracy + precision + recall + F1 on local test set |
| `_evaluate_validation()` | Internal helper used during training for early stopping |

### Shape Mismatch Handling
`set_parameters()` includes robust logic for shape mismatches that can occur when models evolve across FL rounds:
1. Exact shape match → direct load
2. Extra leading dimension `[1, N]` → squeeze
3. Same element count, different shape → reshape
4. More server params than local → truncate
5. Fewer server params than local → zero-pad

---

## API Layer

```
POST /predict
     │
     ▼
Extract 8 input features from JSON
     │
     ▼
torch.FloatTensor([features])
     │
     ▼
model.eval() + torch.no_grad()
     │
     ▼
DiabetesModel.forward(x)
     │
     ▼
torch.softmax(logits, dim=1)
     │
     ▼
risk_percentage = probabilities[0][1] * 100
     │
     ▼
Threshold:
  < 25%  → Low    risk
  25-75% → Moderate risk
  > 75%  → High   risk
     │
     ▼
Return JSON response
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **FedProx over FedAvg** | Clients have non-IID data (different hospital populations) |
| **1 epoch per round** | Minimises client drift; more FL rounds preferred over deep local training |
| **LeakyReLU** | Prevents dying neurons; important with BatchNorm and Dropout combinations |
| **Dropout decreasing 0.5→0.4→0.3** | Heavier regularisation in early layers where features are more redundant |
| **Stratified train/test split** | Preserves class balance in imbalanced diabetes datasets |
| **Daemon FastAPI thread** | API stays responsive without blocking the FL training loop |
| **Model saved after training** | Enables offline inference without restarting FL training |
