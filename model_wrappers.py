
import numpy as np

class AnomalyProbabilityWrapper:
    """Adapter that exposes predict_proba for anomaly detectors.

    The underlying detector's decision_function is higher for more normal
    samples. We map it to a bounded attack-risk score using training-time
    score quantiles. This is a deployment adapter, not a calibrated
    probability estimate.
    """
    def __init__(self, detector, low, high):
        self.detector = detector
        self.low = float(low)
        self.high = float(high)

    def predict_proba(self, X):
        scores = np.asarray(self.detector.decision_function(X), dtype=float)
        denom = max(self.high - self.low, 1e-9)
        risk = np.clip((self.high - scores) / denom, 0.0, 1.0)
        return np.column_stack([1.0 - risk, risk])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ReconstructionProbabilityWrapper:
    """Autoencoder-style reconstruction error -> attack-risk adapter."""
    def __init__(self, model, low, high):
        self.model = model
        self.low = float(low)
        self.high = float(high)

    def predict_proba(self, X):
        recon = self.model.predict(X)
        err = np.mean((np.asarray(X) - np.asarray(recon)) ** 2, axis=1)
        denom = max(self.high - self.low, 1e-9)
        risk = np.clip((err - self.low) / denom, 0.0, 1.0)
        return np.column_stack([1.0 - risk, risk])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
