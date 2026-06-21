# Federated Learning 

Federated Learning A privacy-preserving machine learning system that predicts diabetes risk using federated learning.
This approach enables collaborative model training across multiple institutions while keeping sensitive patient data decentralized and secure. 
This project implements a federated learning system using Flower to predict heart disease while keeping patient data decentralized. 

## Project Structure 
- `server.py` - Flower server for federated learning coordination 
- `client.py` - Client implementation with heart disease prediction model 
- `requirements.txt` - Python dependencies 
 
## Running the Project 
1. Start the server in one terminal: ```bash python server.py ``` 
2. Run one or more clients in separate terminals: ```bash python client.py ``` 

## Dependencies - Python 3.7+ - PyTorch - Flower - scikit-learn - pandas - zeroconf 
 
## 🌟 Features 
- **Privacy-Preserving**: Patient data remains on local devices 
- **Real-time Predictions**: Web interface for instant risk assessment 
- **Federated Learning**: Collaborative model training without sharing raw data 
- **Detailed Risk Analysis**: Provides risk percentage and interpretation 
- **Scalable Architecture**: Supports multiple clients (hospitals/clinics) 

## 🚀 Getting Started 

### Prerequisites 
- Python 3.7 or higher 
- pip (Python package manager) 
