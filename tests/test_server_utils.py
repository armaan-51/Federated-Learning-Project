"""
tests/test_server_utils.py — Unit tests for server.py helper functions.

Tests cover:
    - weighted_average(): correct weighted mean across single and multiple clients
    - fit_config(): returns a dict with the correct 'round' and 'epochs' keys
    - evaluate_config(): returns a dict with the correct 'round' key
    - get_initial_parameters(): returns a list of numpy arrays matching model params
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server import weighted_average, fit_config, evaluate_config, get_initial_parameters, DiabetesModel


class TestWeightedAverage:
    """Tests for the weighted_average() metrics aggregation function."""

    def test_single_client_accuracy(self):
        """With one client, the weighted average should equal that client's accuracy."""
        metrics = [(100, {"accuracy": 0.85})]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.85) < 1e-6

    def test_two_equal_clients(self):
        """Two clients with equal sample sizes: result should be plain mean."""
        metrics = [
            (100, {"accuracy": 0.80}),
            (100, {"accuracy": 0.90}),
        ]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.85) < 1e-6

    def test_two_unequal_clients(self):
        """Larger client should contribute proportionally more to the average."""
        metrics = [
            (100, {"accuracy": 0.80}),   # small client
            (300, {"accuracy": 0.60}),   # large client
        ]
        # Expected: (100*0.80 + 300*0.60) / 400 = 260/400 = 0.65
        result = weighted_average(metrics)
        expected = (100 * 0.80 + 300 * 0.60) / 400
        assert abs(result["accuracy"] - expected) < 1e-6

    def test_returns_accuracy_key(self):
        """Result dict must contain the 'accuracy' key."""
        metrics = [(50, {"accuracy": 0.75})]
        result = weighted_average(metrics)
        assert "accuracy" in result

    def test_result_is_between_zero_and_one(self):
        """Weighted average accuracy must always be in [0, 1]."""
        metrics = [
            (200, {"accuracy": 0.95}),
            (200, {"accuracy": 0.05}),
        ]
        result = weighted_average(metrics)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_perfect_accuracy(self):
        """1.0 accuracy from all clients should aggregate to exactly 1.0."""
        metrics = [(n, {"accuracy": 1.0}) for n in [50, 100, 200]]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 1.0) < 1e-9

    def test_zero_accuracy(self):
        """0.0 accuracy from all clients should aggregate to exactly 0.0."""
        metrics = [(n, {"accuracy": 0.0}) for n in [50, 100, 200]]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.0) < 1e-9


class TestFitConfig:
    """Tests for fit_config() — configuration sent to clients before training."""

    def test_returns_dict(self):
        """fit_config() should return a dict."""
        config = fit_config(1)
        assert isinstance(config, dict)

    def test_contains_round_key(self):
        """Config dict must have a 'round' key."""
        config = fit_config(3)
        assert "round" in config

    def test_round_value_matches_argument(self):
        """'round' value in config must equal the server_round argument."""
        for round_num in [1, 5, 10, 30]:
            config = fit_config(round_num)
            assert config["round"] == round_num

    def test_contains_epochs_key(self):
        """Config dict must have an 'epochs' key."""
        config = fit_config(1)
        assert "epochs" in config

    def test_epochs_is_positive_int(self):
        """'epochs' must be a positive integer."""
        config = fit_config(1)
        assert isinstance(config["epochs"], int)
        assert config["epochs"] >= 1


class TestEvaluateConfig:
    """Tests for evaluate_config() — configuration sent to clients before evaluation."""

    def test_returns_dict(self):
        """evaluate_config() should return a dict."""
        config = evaluate_config(1)
        assert isinstance(config, dict)

    def test_contains_round_key(self):
        """Config dict must have a 'round' key."""
        config = evaluate_config(2)
        assert "round" in config

    def test_round_value_matches_argument(self):
        """'round' value must equal the server_round argument passed in."""
        for round_num in [1, 15, 30]:
            config = evaluate_config(round_num)
            assert config["round"] == round_num


class TestGetInitialParameters:
    """Tests for get_initial_parameters() — model weight extraction."""

    def test_returns_list(self):
        """get_initial_parameters() should return a list."""
        model = DiabetesModel()
        params = get_initial_parameters(model)
        assert isinstance(params, list)

    def test_returns_numpy_arrays(self):
        """Each element in the returned list should be a numpy array."""
        model = DiabetesModel()
        params = get_initial_parameters(model)
        for i, p in enumerate(params):
            assert isinstance(p, np.ndarray), f"Parameter {i} is not a numpy array"

    def test_param_count_matches_model(self):
        """Number of parameter arrays must equal the number of model parameters."""
        model = DiabetesModel()
        params = get_initial_parameters(model)
        expected_count = len(list(model.parameters()))
        assert len(params) == expected_count, (
            f"Expected {expected_count} parameter arrays, got {len(params)}"
        )

    def test_parameters_are_copies(self):
        """Modifying the returned arrays should not affect the model's parameters."""
        import torch
        model = DiabetesModel()
        params = get_initial_parameters(model)

        # Mutate the first returned array
        original_val = params[0][0, 0] if params[0].ndim > 1 else params[0][0]
        if params[0].ndim > 1:
            params[0][0, 0] = 999.0
        else:
            params[0][0] = 999.0

        # Verify the model's first parameter is unchanged
        model_first_param = list(model.parameters())[0].detach().cpu().numpy()
        if model_first_param.ndim > 1:
            assert model_first_param[0, 0] != 999.0, (
                "get_initial_parameters() should return copies, not views"
            )
