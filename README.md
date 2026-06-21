# 🩺 Federated Learning for Diabetes Risk Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C?logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-FL%20Framework-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A **privacy-preserving machine learning system** that predicts **diabetes risk** using federated learning. Multiple hospitals or clinics can collaboratively train a shared model **without ever sharing raw patient data** — each institution keeps its data local.

> **Dataset:** Pima Indians Diabetes Dataset — 8 clinical features including Glucose, BMI, Insulin, and Age.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Configuration](#-configuration)
- [Running the Project](#-running-the-project)
- [API Reference](#-api-reference)
- [Tests](#-tests)
- [Contributing](#-contributing)

---

## 🔍 Overview

Traditional machine learning requires centralising all patient data on one server — a massive privacy risk in healthcare. This project uses **Federated Learning (FL)** to solve that problem:

1. A central **server** coordinates training and holds the global model
2. Multiple **clients** (hospitals/clinics) each train on their local diabetes data
3. Only **model weights** (never raw data) are sent to the server
4. The server **aggregates** the weights using the **FedProx** algorithm
5. The improved global model is sent back to all clients

After training, the server exposes a **FastAPI REST API** for real-time diabetes risk predictions.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING                    │
│                                                          │
│  ┌─────────────┐         ┌────────────────────────────┐  │
│  │  CLIENT 1   │         │         SERVER             │  │
│  │  Hospital A │◄───────►│  • FedProx Aggregation     │  │
│  │  (local DB) │         │  • Global DiabetesModel    │  │
│  └─────────────┘         │  • POST /predict           │  │
│                          │  • GET  /health            │  │
│  ┌─────────────┐         │  • GET  /metrics           │  │
│  │  CLIENT 2   │◄───────►│  • Flower gRPC :8080       │  │
│  │  Hospital B │         └────────────────────────────┘  │
│  │  (local DB) │                                         │
│  └─────────────┘                                         │
│                                                          │
│  ← Only model weights travel the network, never data →  │
└──────────────────────────────────────────────────────────┘
```

### Model Architecture — `DiabetesModel`

```
Input (8 features: Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age)
    │
    ▼
Linear(8→32) → LeakyReLU → BatchNorm → Dropout(0.5)
    │
    ▼
Linear(32→16) → LeakyReLU → BatchNorm → Dropout(0.4)
    │
    ▼
Linear(16→8)  → LeakyReLU → BatchNorm → Dropout(0.3)
    │
    ▼
Linear(8→2)  ← logits: [No Diabetes, Diabetes]
    │
    ▼
Softmax → risk probability %
```

---

## 🌟 Features

| Feature | Details |
|---|---|
| 🔒 **Privacy-Preserving** | Raw patient data never leaves the local machine |
| 🧠 **FedProx Strategy** | Handles non-IID data distributions across hospitals |
| 🌐 **REST API** | FastAPI endpoints for predictions, health checks, and metrics |
| 📊 **Rich Metrics** | Accuracy, Precision, Recall, F1-score per FL round |
| 💾 **Model Checkpointing** | Best model saved automatically each round |
| 📈 **Training History** | Per-round metrics persisted to `training_history.json` |
| ⚡ **Early Stopping** | Prevents client overfitting during local training |
| 🔁 **LR Scheduling** | ReduceLROnPlateau adapts learning rate automatically |
| ⚙️ **Centralised Config** | All hyperparameters in one `config.py` file |
| 🌍 **Env Var Support** | Override server IP, ports, and rounds via env vars |

---

## 📁 Project Structure

```
Federated-Learning-Project/
│
├── server.py                    # FL server + FastAPI prediction API
├── client.py                    # FL client with local training logic
├── config.py                    # ⭐ Centralised configuration (all hyperparameters)
├── network_utils.py             # Network scanning and IP discovery
├── setup_scaler.py              # Pre-compute and save the global StandardScaler
├── analyze_class_imbalance.py   # Dataset class imbalance analysis tool
├── check_network.py             # Network connectivity checker
│
├── test_server.py               # Manual server integration test
├── test_predict.py              # Manual /predict endpoint test
├── final_audit.py               # Full system audit script
│
├── checkpoints/                 # Per-round model checkpoints (auto-created)
├── training_history.json        # Per-round metrics log (auto-created)
├── global_model_final.pth       # Final saved model (post-training)
├── global_scaler.pkl            # Saved StandardScaler for inference
│
├── frontend/                    # React frontend for the prediction UI
├── templates/                   # Jinja2 HTML templates for FastAPI
│
├── tests/                       # pytest test suite
│   ├── conftest.py              # Shared fixtures
│   ├── test_model.py            # DiabetesModel unit tests
│   ├── test_data_loading.py     # Data pipeline tests
│   ├── test_network_utils.py    # Network utility tests
│   ├── test_server_utils.py     # Server helper function tests
│   └── test_flower_client.py   # FlowerClient integration tests
│
├── docs/
│   └── architecture.md          # Deep-dive architecture documentation
│
├── requirements.txt             # Python dependencies
├── pytest.ini                   # pytest configuration
└── README.md                    # This file
```

---

## 📋 Prerequisites

- **Python 3.7+**
- **pip** (Python package manager)
- Two machines (or two terminal windows) for client-server deployment

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/armaan-51/Federated-Learning-Project.git
cd Federated-Learning-Project
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv fl_env
fl_env\Scripts\activate

# macOS / Linux
python -m venv fl_env
source fl_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The model uses the **Pima Indians Diabetes Dataset** — a classic benchmark for binary diabetes classification.

| Feature | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of pregnancies | count |
| `Glucose` | Plasma glucose concentration (2-hour OGTT) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skinfold thickness | mm |
| `Insulin` | 2-Hour serum insulin | μU/mL |
| `BMI` | Body mass index | kg/m² |
| `DiabetesPedigreeFunction` | Genetic diabetes risk score | score |
| `Age` | Age | years |
| `Outcome` | **Target** — 0 = No Diabetes, 1 = Diabetes | binary |

**Expected dataset files (place in project root):**
- `diabetes_non_negative_part1_2000.csv` → Client 1
- `diabetes_non_negative_part2_2000.csv` → Client 2

---

## ⚙️ Configuration

All settings live in [`config.py`](config.py). Environment variables override defaults:

```bash
# Override server IP for multi-machine deployment
FL_SERVER_IP=192.168.1.100 python client.py --client-id 1

# Override FL rounds
FL_NUM_ROUNDS=50 python server.py
```

| Parameter | Default | Env Var | Description |
|---|---|---|---|
| `SERVER_IP` | `10.133.98.49` | `FL_SERVER_IP` | Server machine's LAN IP |
| `SERVER_FL_PORT` | `8080` | `FL_SERVER_PORT` | Flower gRPC port |
| `SERVER_API_PORT` | `5000` | `FL_API_PORT` | FastAPI port |
| `NUM_ROUNDS` | `30` | `FL_NUM_ROUNDS` | FL training rounds |
| `PROXIMAL_MU` | `0.1` | — | FedProx μ coefficient |
| `LEARNING_RATE` | `0.0017` | — | Adam LR |
| `BATCH_SIZE` | `32` | — | Mini-batch size |
| `LOCAL_EPOCHS` | `1` | — | Local epochs per round |

---

## 🏃 Running the Project

### Single Machine (Development / Demo)

Open **three** terminal windows:

**Terminal 1 — Server:**
```bash
python server.py
```

**Terminal 2 — Client 1:**
```bash
python client.py --client-id 1
```

**Terminal 3 — Client 2:**
```bash
python client.py --client-id 2
```

### Multi-Machine (Network Deployment)

```bash
# Set server IP via environment variable — no code changes needed
FL_SERVER_IP=192.168.1.50 python client.py --client-id 1
FL_SERVER_IP=192.168.1.50 python client.py --client-id 2
```

### Client CLI Options

```
python client.py --help

  --server-address TEXT   Server address host:port  [default: configured IP:8080]
  --client-id INTEGER     Unique client ID (1 or 2)  [required]
  --batch-size INTEGER    Training batch size         [default: 32]
```

---

## 🌐 API Reference

After training starts, the FastAPI server is immediately available at `http://localhost:5000`.

### `POST /predict` — Diabetes Risk Prediction

**Request:**
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "status": "success",
  "risk_percentage": "78.43%",
  "risk_level": "High",
  "interpretation": "Based on the provided health data, the patient has a 78.43% probability of having diabetes. This is considered High risk.",
  "recommendation": "Please consult an endocrinologist for a comprehensive evaluation and diabetes management plan."
}
```

**Risk Levels:**

| Risk % | Level | Action |
|---|---|---|
| < 25% | 🟢 Low | Maintain healthy lifestyle |
| 25–75% | 🟡 Moderate | Lifestyle changes + monitoring |
| > 75% | 🔴 High | Consult endocrinologist |

---

### `GET /health` — Server Status

```json
{
  "status": "ok",
  "current_round": 15,
  "total_rounds": 30,
  "training_complete": false,
  "best_accuracy": 0.7812
}
```

### `GET /metrics` — Training Metrics

```json
{
  "latest": { "round": 15, "accuracy": 0.7812 },
  "history": [
    { "round": 1, "accuracy": 0.6543 },
    { "round": 2, "accuracy": 0.6901 },
    ...
  ]
}
```

### Test the API

```bash
# Manual test with correct diabetes features
python test_predict.py

# Against a remote server
python test_predict.py --server http://192.168.1.100:5000

# curl example
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

---

## 🧪 Tests

Run the full test suite (no real server or dataset needed):

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

| Test File | Coverage |
|---|---|
| `test_model.py` | Forward pass, output shape, weight init, dropout mode |
| `test_data_loading.py` | CSV loading, split ratio, normalization, edge cases |
| `test_network_utils.py` | IP detection, ping mocking |
| `test_server_utils.py` | Weighted average, config helpers, parameter extraction |
| `test_flower_client.py` | get/set params roundtrip, evaluate/fit output format |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Built with ❤️ using <a href="https://flower.dev/">Flower</a> and <a href="https://pytorch.org/">PyTorch</a></p>