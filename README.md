# 🫀 Federated Learning for Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C?logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-FL%20Framework-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A **privacy-preserving machine learning system** that predicts heart disease / diabetes risk using federated learning. Multiple hospitals or clinics can collaboratively train a shared model **without ever sharing raw patient data** — each institution keeps its data local.

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

Traditional machine learning requires centralizing all patient data on one server — a massive privacy risk in healthcare. This project uses **Federated Learning (FL)** to solve that problem:

1. A central **server** coordinates training and holds the global model
2. Multiple **clients** (hospitals/clinics) each train on their local data
3. Only **model weights** (never raw data) are sent to the server
4. The server **aggregates** the weights using the **FedProx** algorithm
5. The improved global model is sent back to all clients

After training, the server exposes a **FastAPI REST endpoint** for real-time diabetes risk predictions.

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
│  └─────────────┘         │  • FastAPI /predict        │  │
│                          │    (port 5000)             │  │
│  ┌─────────────┐         │  • Flower gRPC             │  │
│  │  CLIENT 2   │◄───────►│    (port 8080)             │  │
│  │  Hospital B │         └────────────────────────────┘  │
│  │  (local DB) │                                         │
│  └─────────────┘                                         │
│                                                          │
│  ← Only model weights travel the network, never data →   │
└──────────────────────────────────────────────────────────┘
```

### Model Architecture — `DiabetesModel`

```
Input (8 features)
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
Linear(8→2)  ← output logits (No Disease / Disease)
```

Weights are initialised with **Kaiming Normal** (suitable for LeakyReLU activations).

---

## 🌟 Features

| Feature | Details |
|---|---|
| 🔒 **Privacy-Preserving** | Raw patient data never leaves the local machine |
| 🧠 **FedProx Strategy** | Handles non-IID data distributions across clients |
| 🌐 **REST API** | FastAPI endpoint for real-time predictions |
| 📊 **Rich Metrics** | Accuracy, Precision, Recall, F1-score per round |
| ⚡ **Early Stopping** | Prevents client overfitting during local training |
| 🔁 **LR Scheduling** | ReduceLROnPlateau adapts learning rate automatically |
| 🏥 **Multi-Client** | Supports 2+ hospital/clinic clients simultaneously |
| 💾 **Model Checkpointing** | Best global model saved after training completes |

---

## 📁 Project Structure

```
Federated-Learning-for-Heart-Disease-Prediction/
│
├── server.py                    # Flower FL server + FastAPI prediction API
├── client.py                    # Flower FL client with local training logic
├── fixed_server.py              # Bug-fixed server variant
├── fixed_client.py              # Bug-fixed client variant
├── improved_server.py           # Performance-improved server
├── improved_client.py           # Performance-improved client
├── tuned_client.py              # Hyperparameter-tuned client
├── network_utils.py             # Network scanning and IP utilities
├── setup_scaler.py              # Pre-compute and save the global scaler
├── analyze_class_imbalance.py   # Dataset class imbalance analysis
├── check_network.py             # Network connectivity checker
│
├── test_server.py               # Manual server integration test
├── test_predict.py              # Manual prediction endpoint test
├── fixed_test_server.py         # Test for fixed server
├── final_audit.py               # Full system audit script
│
├── global_model_final.pth       # Saved global model (post-training)
├── fixed_global_model.pth       # Saved fixed-variant model
├── improved_global_model.pth    # Saved improved-variant model
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
- Two machines (or two terminal windows on one machine) for client-server setup

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/armaan-51/Federated-Learning-Project.git
cd Federated-Learning-Project
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv fl_env

# Activate on Windows
fl_env\Scripts\activate

# Activate on macOS/Linux
source fl_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The model expects **CSV files** with the following 8 feature columns (Pima Indians Diabetes dataset format):

| Column | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of pregnancies | count |
| `Glucose` | Plasma glucose concentration | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skinfold thickness | mm |
| `Insulin` | 2-Hour serum insulin | μU/mL |
| `BMI` | Body mass index | kg/m² |
| `DiabetesPedigreeFunction` | Diabetes pedigree function | score |
| `Age` | Age | years |
| `Outcome` | Target variable (0 = No Diabetes, 1 = Diabetes) | binary |

**Client dataset files (place in project root):**
- `diabetes_non_negative_part1_2000.csv` → used by Client 1
- `diabetes_non_negative_part2_2000.csv` → used by Client 2

---

## ⚙️ Configuration

### Server IP (client.py)

Before running on multiple machines, update the server IP at the top of `client.py`:

```python
# client.py — line 25
SERVER_IP = "192.168.1.100"   # ← Replace with your server's actual IP
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_rounds` | `30` | Number of federated learning rounds |
| `round_timeout` | `120s` | Timeout per round |
| `proximal_mu` | `0.1` | FedProx proximal term weight |
| `min_fit_clients` | `2` | Minimum clients needed to start a round |
| `learning_rate` | `0.0017` | Adam optimizer learning rate |
| `batch_size` | `32` | Training batch size per client |
| FL port | `8080` | Flower gRPC port |
| API port | `5000` | FastAPI prediction endpoint port |

---

## 🏃 Running the Project

### Single Machine (development)

Open **three** terminal windows:

**Terminal 1 — Start the server:**
```bash
python server.py
```

**Terminal 2 — Start client 1:**
```bash
python client.py --client-id 1
```

**Terminal 3 — Start client 2:**
```bash
python client.py --client-id 2
```

### Multi-Machine (network deployment)

1. Set `SERVER_IP` in `client.py` to the server's actual LAN IP
2. Run `python server.py` on the server machine
3. Run `python client.py --client-id 1` on each client machine

### Client Options

```bash
python client.py --help

Options:
  --server-address TEXT   Server address host:port  [default: SERVER_IP:8080]
  --client-id INTEGER     Unique client identifier  [required]
  --batch-size INTEGER    Training batch size        [default: 32]
```

---

## 🌐 API Reference

After training completes, the server exposes a prediction endpoint:

### `POST /predict`

**Request body:**

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 85,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 35
}
```

**Response:**

```json
{
  "status": "success",
  "risk_percentage": "34.72%",
  "risk_level": "Moderate",
  "interpretation": "Based on the provided health data, the patient has a 34.72% risk of developing diabetes. This is considered Moderate risk.",
  "recommendation": "Consider lifestyle changes and regular monitoring. Consult a healthcare provider."
}
```

**Risk Levels:**

| Risk % | Level | Recommendation |
|---|---|---|
| < 25% | 🟢 Low | Maintain healthy lifestyle |
| 25–75% | 🟡 Moderate | Lifestyle changes + monitoring |
| > 75% | 🔴 High | Consult an endocrinologist |

### Test the API manually:

```bash
python test_predict.py
# or
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":120,"BloodPressure":70,"SkinThickness":25,"Insulin":85,"BMI":28.5,"DiabetesPedigreeFunction":0.45,"Age":35}'
```

---

## 🧪 Tests

Run the full test suite (no real server or dataset needed):

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

Test categories:

| Test File | Coverage |
|---|---|
| `test_model.py` | Forward pass, output shape, weight init, dropout |
| `test_data_loading.py` | CSV loading, 80/20 split, normalization, edge cases |
| `test_network_utils.py` | IP detection, address format validation |
| `test_server_utils.py` | Weighted average, config helpers, parameter extraction |
| `test_flower_client.py` | get/set parameters, fit/evaluate output format |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using <a href="https://flower.dev/">Flower</a> and <a href="https://pytorch.org/">PyTorch</a></p>
