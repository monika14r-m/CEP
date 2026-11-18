"""Regression tests for the probability adapters used by the IDS pipeline."""

import ast
from pathlib import Path

import numpy as np

from model_wrappers import AnomalyProbabilityWrapper, ReconstructionProbabilityWrapper


class FixedDetector:
    def decision_function(self, X):
        return np.asarray(X).reshape(-1)


class ZeroReconstruction:
    def predict(self, X):
        return np.zeros_like(X)


def test_anomaly_wrapper_maps_normality_to_bounded_attack_risk():
    X = np.array([[-2.0], [0.0], [2.0]])
    probabilities = AnomalyProbabilityWrapper(FixedDetector(), -2.0, 2.0).predict_proba(X)

    np.testing.assert_allclose(probabilities[:, 1], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_reconstruction_wrapper_maps_error_to_attack_risk():
    X = np.array([[0.0], [1.0], [2.0]])
    wrapper = ReconstructionProbabilityWrapper(ZeroReconstruction(), 0.0, 4.0)

    probabilities = wrapper.predict_proba(X)

    np.testing.assert_allclose(probabilities[:, 1], [0.0, 0.25, 1.0])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(wrapper.predict(X), [0, 0, 1])


def test_reconstruction_wrapper_clamps_out_of_range_errors():
    X = np.array([[3.0]])
    probabilities = ReconstructionProbabilityWrapper(ZeroReconstruction(), 0.0, 1.0).predict_proba(X)

    np.testing.assert_allclose(probabilities, [[0.0, 1.0]])


def test_training_pipeline_imports_both_probability_adapters():
    source = Path(__file__).parents[1] / "train_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "model_wrappers"
        for alias in node.names
    }

    assert {"AnomalyProbabilityWrapper", "ReconstructionProbabilityWrapper"} <= imported
