"""
tests/test_data_loading.py — Unit tests for the load_diabetes_data() pipeline.

All tests use a synthetic CSV fixture (from conftest.py) so no real dataset
files are required. Tests cover:
    - Correct return types (4 PyTorch tensors)
    - Train/test split ratio
    - Feature standardisation (mean ≈ 0, std ≈ 1 after StandardScaler)
    - Fallback behaviour when expected column names are missing
    - ValueError raised on empty CSV
    - Binary target enforcement for multi-class targets
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client import load_diabetes_data


class TestLoadDiabetesDataReturnShape:
    """Tests for return types and shapes from load_diabetes_data()."""

    def test_returns_four_tensors(self, synthetic_csv):
        """Function should return exactly four values."""
        result = load_diabetes_data(str(synthetic_csv))
        assert len(result) == 4, "Expected 4 return values (X_train, X_test, y_train, y_test)"

    def test_x_train_is_float_tensor(self, synthetic_csv):
        """X_train should be a FloatTensor."""
        X_train, _, _, _ = load_diabetes_data(str(synthetic_csv))
        assert X_train.dtype == torch.float32

    def test_x_test_is_float_tensor(self, synthetic_csv):
        """X_test should be a FloatTensor."""
        _, X_test, _, _ = load_diabetes_data(str(synthetic_csv))
        assert X_test.dtype == torch.float32

    def test_y_train_is_long_tensor(self, synthetic_csv):
        """y_train should be a LongTensor (required by CrossEntropyLoss)."""
        _, _, y_train, _ = load_diabetes_data(str(synthetic_csv))
        assert y_train.dtype == torch.int64

    def test_y_test_is_long_tensor(self, synthetic_csv):
        """y_test should be a LongTensor."""
        _, _, _, y_test = load_diabetes_data(str(synthetic_csv))
        assert y_test.dtype == torch.int64

    def test_feature_count_is_eight(self, synthetic_csv):
        """Both X_train and X_test should have exactly 8 feature columns."""
        X_train, X_test, _, _ = load_diabetes_data(str(synthetic_csv))
        assert X_train.shape[1] == 8, f"X_train has {X_train.shape[1]} features, expected 8"
        assert X_test.shape[1] == 8, f"X_test has {X_test.shape[1]} features, expected 8"


class TestLoadDiabetesDataSplit:
    """Tests for the 80/20 train/test split logic."""

    def test_correct_split_ratio(self, synthetic_csv):
        """With 200 samples and test_size=0.2, expect ~160 train, ~40 test."""
        X_train, X_test, y_train, y_test = load_diabetes_data(str(synthetic_csv), test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == 200, f"Total samples should be 200, got {total}"
        # Allow ±2 samples due to stratified rounding
        assert abs(len(X_test) - 40) <= 2, f"Expected ~40 test samples, got {len(X_test)}"

    def test_train_larger_than_test(self, synthetic_csv):
        """Training set must be larger than the test set for default test_size=0.2."""
        X_train, X_test, _, _ = load_diabetes_data(str(synthetic_csv))
        assert len(X_train) > len(X_test)

    def test_custom_test_size(self, synthetic_csv):
        """test_size=0.3 should give ~60 test samples out of 200."""
        X_train, X_test, _, _ = load_diabetes_data(str(synthetic_csv), test_size=0.3)
        assert abs(len(X_test) - 60) <= 2, f"Expected ~60 test samples, got {len(X_test)}"

    def test_y_shapes_match_x(self, synthetic_csv):
        """Label tensors must have the same length as the corresponding feature tensors."""
        X_train, X_test, y_train, y_test = load_diabetes_data(str(synthetic_csv))
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestLoadDiabetesDataNormalization:
    """Tests for StandardScaler normalisation."""

    def test_train_features_near_zero_mean(self, synthetic_csv):
        """After scaling, training feature means should be close to 0."""
        X_train, _, _, _ = load_diabetes_data(str(synthetic_csv))
        col_means = X_train.numpy().mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=0.1), (
            f"Training feature means not near 0: {col_means}"
        )

    def test_train_features_near_unit_std(self, synthetic_csv):
        """After scaling, training feature std devs should be close to 1."""
        X_train, _, _, _ = load_diabetes_data(str(synthetic_csv))
        col_stds = X_train.numpy().std(axis=0)
        assert np.allclose(col_stds, 1.0, atol=0.1), (
            f"Training feature stds not near 1: {col_stds}"
        )


class TestLoadDiabetesDataEdgeCases:
    """Tests for fallback and error-handling behaviour."""

    def test_missing_column_names_uses_fallback(self, tmp_path):
        """CSV without expected column names should fall back to first 8 columns."""
        csv_path = tmp_path / "no_names.csv"
        np.random.seed(1)
        df = pd.DataFrame(np.random.rand(100, 9))  # 9 generic columns
        # Last column used as Outcome (all binary)
        df.iloc[:, -1] = np.random.randint(0, 2, 100)
        df.to_csv(csv_path, index=False)

        X_train, X_test, y_train, y_test = load_diabetes_data(str(csv_path))
        assert X_train.shape[1] == 8, "Fallback should use first 8 columns as features"

    def test_empty_csv_raises_value_error(self, tmp_path):
        """An empty CSV file (headers only or completely empty) should raise ValueError."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")  # Truly empty file
        with pytest.raises(Exception):  # pd.read_csv raises EmptyDataError -> caught as ValueError
            load_diabetes_data(str(empty_csv))

    def test_binary_target_values(self, synthetic_csv):
        """All target values should be 0 or 1 (binary classification)."""
        _, _, y_train, y_test = load_diabetes_data(str(synthetic_csv))
        all_labels = torch.cat([y_train, y_test])
        unique_labels = torch.unique(all_labels).tolist()
        for label in unique_labels:
            assert label in [0, 1], f"Unexpected label value: {label}"

    def test_multiclass_target_binarised(self, tmp_path):
        """Targets with values > 1 should be binarised to 0/1."""
        csv_path = tmp_path / "multiclass.csv"
        np.random.seed(99)
        data = {
            "Pregnancies": np.random.randint(0, 10, 100),
            "Glucose": np.random.randint(70, 200, 100),
            "BloodPressure": np.random.randint(40, 120, 100),
            "SkinThickness": np.random.randint(0, 60, 100),
            "Insulin": np.random.randint(0, 400, 100),
            "BMI": np.random.uniform(15, 55, 100),
            "DiabetesPedigreeFunction": np.random.uniform(0.05, 2.5, 100),
            "Age": np.random.randint(18, 80, 100),
            "Outcome": np.random.randint(0, 5, 100),  # Multi-class (0–4)
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        _, _, y_train, y_test = load_diabetes_data(str(csv_path))
        all_labels = torch.cat([y_train, y_test])
        unique_labels = torch.unique(all_labels).tolist()
        for label in unique_labels:
            assert label in [0, 1], f"Multi-class target not binarised: label={label}"
