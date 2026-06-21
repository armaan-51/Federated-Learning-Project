"""
tests/test_model.py — Unit tests for the DiabetesModel neural network.

Tests cover:
    - Forward pass output shape for batched input
    - Single-sample (1-D) input handling via unsqueeze
    - Correct number of output logits (num_classes=2)
    - Weight initialisation (not all-zero after init)
    - Dropout behaviour difference between train and eval mode
    - Model parameter count consistency
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client import DiabetesModel


class TestDiabetesModelForward:
    """Tests for DiabetesModel.forward()."""

    def test_forward_pass_output_shape(self, sample_model):
        """Batch of 16 samples should produce output shape (16, 2)."""
        x = torch.randn(16, 8)
        sample_model.eval()
        out = sample_model(x)
        assert out.shape == (16, 2), f"Expected (16, 2), got {out.shape}"

    def test_forward_single_sample_1d(self, sample_model):
        """A 1-D input (8,) should be auto-expanded and return shape (1, 2)."""
        x = torch.randn(8)  # Single sample without batch dimension
        sample_model.eval()
        out = sample_model(x)
        assert out.shape == (1, 2), f"Expected (1, 2) for 1-D input, got {out.shape}"

    def test_output_num_classes(self, sample_model):
        """Output should have exactly 2 columns (binary classification)."""
        x = torch.randn(4, 8)
        sample_model.eval()
        out = sample_model(x)
        assert out.shape[1] == 2, "Model should output 2 logits for binary classification"

    def test_forward_returns_tensor(self, sample_model):
        """Forward pass should return a torch.Tensor."""
        x = torch.randn(8, 8)
        sample_model.eval()
        out = sample_model(x)
        assert isinstance(out, torch.Tensor)

    def test_forward_no_nan_or_inf(self, sample_model):
        """Output should not contain NaN or Inf values on normal input."""
        x = torch.randn(32, 8)
        sample_model.eval()
        out = sample_model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_custom_input_size(self):
        """Model should work with non-default input sizes."""
        model = DiabetesModel(input_size=12, num_classes=3)
        x = torch.randn(8, 12)
        model.eval()
        out = model(x)
        assert out.shape == (8, 3)


class TestDiabetesModelWeights:
    """Tests for weight initialisation."""

    def test_linear_weights_not_all_zero(self, sample_model):
        """Linear layer weights should be non-zero after Kaiming init."""
        for name, param in sample_model.named_parameters():
            if "weight" in name and len(param.shape) >= 2:
                assert param.abs().sum().item() > 0, (
                    f"Weights for {name} are all zero after initialisation"
                )

    def test_batchnorm_weight_ones(self, sample_model):
        """BatchNorm scale parameters (weight) should be initialised to 1."""
        for name, module in sample_model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                assert torch.allclose(
                    module.weight, torch.ones_like(module.weight)
                ), f"BatchNorm weight not 1.0 for {name}"

    def test_batchnorm_bias_zeros(self, sample_model):
        """BatchNorm shift parameters (bias) should be initialised to 0."""
        for name, module in sample_model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                assert torch.allclose(
                    module.bias, torch.zeros_like(module.bias)
                ), f"BatchNorm bias not 0.0 for {name}"


class TestDiabetesModelTrainEval:
    """Tests for train/eval mode behaviour (e.g. dropout stochasticity)."""

    def test_eval_mode_deterministic(self, sample_model):
        """Two forward passes in eval mode should produce identical results."""
        x = torch.randn(16, 8)
        sample_model.eval()
        with torch.no_grad():
            out1 = sample_model(x)
            out2 = sample_model(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_train_mode_stochastic(self, sample_model):
        """Two forward passes in train mode should (usually) differ due to dropout."""
        x = torch.randn(64, 8)
        sample_model.train()
        with torch.no_grad():
            out1 = sample_model(x)
            out2 = sample_model(x)
        # With dropout p=0.5 on 64 samples, this should almost always differ
        assert not torch.allclose(out1, out2), (
            "Train mode outputs should differ due to dropout stochasticity"
        )

    def test_model_has_correct_layer_count(self, sample_model):
        """Model should have exactly 4 Sequential/Linear top-level children."""
        children = list(sample_model.children())
        # layer1, layer2, layer3, output = 4 direct children
        assert len(children) == 4, f"Expected 4 children, got {len(children)}"
