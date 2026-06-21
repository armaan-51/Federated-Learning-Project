"""
tests/conftest.py — Shared pytest fixtures for the federated learning test suite.

Fixtures provided:
    - ``synthetic_csv``  : Temporary CSV file with 200 rows of diabetes-like data.
    - ``sample_model``   : Fresh DiabetesModel instance (input_size=8, num_classes=2).
    - ``sample_tensors`` : Pre-split (X_train, X_test, y_train, y_test) tensors.
    - ``sample_client``  : FlowerClient initialised with synthetic tensor data.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Make the project root importable from within the tests/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import DiabetesModel, FlowerClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    """Create a temporary CSV file with 200 rows of synthetic diabetes data.

    The file contains all 8 feature columns plus the ``Outcome`` column.
    Values are randomly generated within clinically plausible ranges so that
    the scaler and split logic work as expected.

    Scope: ``session`` — the file is created once and reused across all tests.

    Returns:
        pathlib.Path: Path to the temporary CSV file.
    """
    tmp_path = tmp_path_factory.mktemp("data")
    csv_path = tmp_path / "diabetes_test.csv"

    np.random.seed(42)
    n = 200
    data = {
        "Pregnancies":              np.random.randint(0, 15, n),
        "Glucose":                  np.random.randint(70, 200, n),
        "BloodPressure":            np.random.randint(40, 120, n),
        "SkinThickness":            np.random.randint(0, 60, n),
        "Insulin":                  np.random.randint(0, 400, n),
        "BMI":                      np.round(np.random.uniform(15.0, 55.0, n), 1),
        "DiabetesPedigreeFunction": np.round(np.random.uniform(0.05, 2.5, n), 3),
        "Age":                      np.random.randint(18, 80, n),
        "Outcome":                  np.random.randint(0, 2, n),
    }

    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_model():
    """Return a freshly initialised DiabetesModel with default hyperparameters.

    Scope: ``function`` — a new model is created for every test that uses this
    fixture, preventing state leakage between tests.

    Returns:
        DiabetesModel: Model with ``input_size=8``, ``num_classes=2``.
    """
    return DiabetesModel(input_size=8, num_classes=2)


@pytest.fixture
def sample_tensors():
    """Return small synthetic (X_train, X_test, y_train, y_test) tensors.

    Generates 160 training and 40 test samples with 8 features each.
    Labels are binary (0 or 1). Values are standardised to approximate
    the output of ``StandardScaler``.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) as PyTorch tensors.
    """
    torch.manual_seed(0)
    X_train = torch.randn(160, 8)
    X_test = torch.randn(40, 8)
    y_train = torch.randint(0, 2, (160,))
    y_test = torch.randint(0, 2, (40,))
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_client(sample_model, sample_tensors):
    """Return a FlowerClient initialised with a fresh model and synthetic tensors.

    Returns:
        FlowerClient: Ready-to-use client instance on CPU.
    """
    X_train, X_test, y_train, y_test = sample_tensors
    return FlowerClient(sample_model, X_train, y_train, X_test, y_test)
