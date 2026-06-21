"""
tests/test_flower_client.py — Integration-style tests for FlowerClient.

No real Flower server is required — all tests exercise the client's local logic:
    - get_parameters() serialises model state to numpy arrays
    - set_parameters() roundtrip preserves model weights
    - evaluate() returns (float, int, dict) with correct metric keys
    - fit() returns (list, int, dict) — no server connection needed

These tests use the ``sample_client`` fixture from conftest.py, which is
initialised with 160 synthetic training samples and 40 test samples.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client import DiabetesModel, FlowerClient


class TestGetParameters:
    """Tests for FlowerClient.get_parameters()."""

    def test_returns_list(self, sample_client):
        """get_parameters() should return a list."""
        params = sample_client.get_parameters()
        assert isinstance(params, list)

    def test_each_element_is_numpy_array(self, sample_client):
        """Every element in the list should be a numpy ndarray."""
        params = sample_client.get_parameters()
        for i, p in enumerate(params):
            assert isinstance(p, np.ndarray), f"Parameter {i} is not a numpy array"

    def test_length_matches_model_state_dict(self, sample_client):
        """Number of arrays must equal the number of entries in model.state_dict()."""
        params = sample_client.get_parameters()
        expected = len(sample_client.model.state_dict())
        assert len(params) == expected, (
            f"Expected {expected} parameter arrays, got {len(params)}"
        )

    def test_params_not_empty(self, sample_client):
        """Returned parameter list should not be empty."""
        params = sample_client.get_parameters()
        assert len(params) > 0


class TestSetParameters:
    """Tests for FlowerClient.set_parameters()."""

    def test_roundtrip_preserves_values(self, sample_client):
        """get → set → get should yield the same arrays (within float32 tolerance)."""
        original_params = sample_client.get_parameters()
        sample_client.set_parameters(original_params)
        restored_params = sample_client.get_parameters()

        for i, (orig, rest) in enumerate(zip(original_params, restored_params)):
            assert np.allclose(orig, rest, atol=1e-6), (
                f"Parameter {i} changed after set_parameters() roundtrip"
            )

    def test_set_zero_parameters(self, sample_client):
        """Setting all-zero parameters should result in all-zero model weights."""
        zero_params = [np.zeros_like(p) for p in sample_client.get_parameters()]
        sample_client.set_parameters(zero_params)
        restored = sample_client.get_parameters()
        for i, p in enumerate(restored):
            assert np.allclose(p, 0.0, atol=1e-6), (
                f"Parameter {i} is not zero after setting zero parameters"
            )

    def test_set_parameters_updates_model(self, sample_client, sample_model):
        """After set_parameters(), the model weights should differ from the initial weights."""
        # Create a second model with different random weights
        other_model = DiabetesModel(input_size=8, num_classes=2)
        # Re-initialise to guarantee different weights
        for param in other_model.parameters():
            torch.nn.init.uniform_(param, -1.0, 1.0)

        new_params = [val.cpu().numpy() for _, val in other_model.state_dict().items()]
        sample_client.set_parameters(new_params)

        # The client model should now have the new weights
        client_params = sample_client.get_parameters()
        for i, (new, client) in enumerate(zip(new_params, client_params)):
            assert np.allclose(new, client, atol=1e-5), (
                f"Parameter {i} was not updated correctly"
            )


class TestEvaluate:
    """Tests for FlowerClient.evaluate()."""

    def test_returns_three_values(self, sample_client):
        """evaluate() should return a tuple of exactly 3 elements."""
        params = sample_client.get_parameters()
        result = sample_client.evaluate(params, config={"round": 1})
        assert len(result) == 3

    def test_loss_is_float(self, sample_client):
        """The first return value (loss) must be a Python float."""
        params = sample_client.get_parameters()
        loss, _, _ = sample_client.evaluate(params, config={"round": 1})
        assert isinstance(loss, float)

    def test_loss_is_positive(self, sample_client):
        """Cross-entropy loss must be non-negative."""
        params = sample_client.get_parameters()
        loss, _, _ = sample_client.evaluate(params, config={"round": 1})
        assert loss >= 0.0, f"Loss should be non-negative, got {loss}"

    def test_num_examples_is_int(self, sample_client):
        """The second return value (num_examples) must be a positive int."""
        params = sample_client.get_parameters()
        _, num_examples, _ = sample_client.evaluate(params, config={"round": 1})
        assert isinstance(num_examples, int)
        assert num_examples > 0

    def test_num_examples_equals_test_set_size(self, sample_client):
        """num_examples should equal the size of the local test set."""
        params = sample_client.get_parameters()
        _, num_examples, _ = sample_client.evaluate(params, config={"round": 1})
        assert num_examples == len(sample_client.X_test)

    def test_metrics_dict_has_required_keys(self, sample_client):
        """The metrics dict must contain accuracy, precision, recall, f1_score."""
        params = sample_client.get_parameters()
        _, _, metrics = sample_client.evaluate(params, config={"round": 1})
        required_keys = {"accuracy", "precision", "recall", "f1_score"}
        assert required_keys.issubset(metrics.keys()), (
            f"Missing metric keys: {required_keys - metrics.keys()}"
        )

    def test_accuracy_in_valid_range(self, sample_client):
        """Accuracy metric must be between 0.0 and 1.0."""
        params = sample_client.get_parameters()
        _, _, metrics = sample_client.evaluate(params, config={"round": 1})
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_all_metrics_are_floats(self, sample_client):
        """Every value in the metrics dict must be a Python float."""
        params = sample_client.get_parameters()
        _, _, metrics = sample_client.evaluate(params, config={"round": 1})
        for key, val in metrics.items():
            assert isinstance(val, float), f"Metric '{key}' is not a float: {type(val)}"


class TestFit:
    """Tests for FlowerClient.fit() — local training round."""

    def test_returns_three_values(self, sample_client):
        """fit() should return a tuple of exactly 3 elements."""
        params = sample_client.get_parameters()
        result = sample_client.fit(params, config={"epochs": 1, "batch_size": 32})
        assert len(result) == 3

    def test_first_return_is_list(self, sample_client):
        """The first return value (updated params) must be a list."""
        params = sample_client.get_parameters()
        updated_params, _, _ = sample_client.fit(
            params, config={"epochs": 1, "batch_size": 32}
        )
        assert isinstance(updated_params, list)

    def test_updated_params_are_numpy_arrays(self, sample_client):
        """Every updated parameter must be a numpy array."""
        params = sample_client.get_parameters()
        updated_params, _, _ = sample_client.fit(
            params, config={"epochs": 1, "batch_size": 32}
        )
        for i, p in enumerate(updated_params):
            assert isinstance(p, np.ndarray), f"Updated param {i} is not ndarray"

    def test_num_examples_is_train_set_size(self, sample_client):
        """fit() should report the number of training examples used."""
        params = sample_client.get_parameters()
        _, num_examples, _ = sample_client.fit(
            params, config={"epochs": 1, "batch_size": 32}
        )
        assert num_examples == len(sample_client.X_train)

    def test_params_count_unchanged_after_fit(self, sample_client):
        """The number of parameter arrays should not change after fit()."""
        params = sample_client.get_parameters()
        original_count = len(params)
        updated_params, _, _ = sample_client.fit(
            params, config={"epochs": 1, "batch_size": 32}
        )
        assert len(updated_params) == original_count

    def test_fit_modifies_parameters(self, sample_client):
        """After fit(), at least some parameters should differ from the initial values."""
        params = sample_client.get_parameters()
        initial_flat = np.concatenate([p.flatten() for p in params])

        updated_params, _, _ = sample_client.fit(
            params, config={"epochs": 1, "batch_size": 32}
        )
        updated_flat = np.concatenate([p.flatten() for p in updated_params])

        # Training for 1 epoch on 160 samples should change weights
        assert not np.allclose(initial_flat, updated_flat, atol=1e-8), (
            "Parameters should change after one training epoch"
        )
