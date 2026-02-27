import joblib
import pandas as pd

class AnomalyDetector:
    def __init__(self, model_path: str):
        self.pipe = joblib.load(model_path)
        self.expected_features = set(self.pipe.feature_names_in_)

    def score(self, X):
        missing = self.expected_features - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return self.pipe.decision_function(X)

    def predict(self, X, threshold: float = -0.15):
        missing = self.expected_features - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        scores = self.pipe.decision_function(X)

        results = X.copy()
        results["anomaly_score"] = scores
        results["is_anomaly"] = (scores < threshold).astype(int)

        return results