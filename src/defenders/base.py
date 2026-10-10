from abc import ABC, abstractmethod
import numpy as np

class Defender(ABC):
    def __init__(self, name: str):
        self.name = name
        self.threshold = 0.5  # Default threshold
        
    @abstractmethod
    def fit(self, X_train: list[str], y_train: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Returns array of shape (n_samples, 2) where column 1 is P(spam)"""
        pass
        
    def set_threshold(self, tau: float):
        self.threshold = tau
        
    def predict(self, X: list[str]) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
        
    def get_config(self) -> dict:
        return {
            "name": self.name,
            "threshold": self.threshold
        }
