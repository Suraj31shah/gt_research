from .base import Defender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

class LogisticDefender(Defender):
    def __init__(self, name="LogisticRegression"):
        super().__init__(name)
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
    def fit(self, X_train: list[str], y_train: np.ndarray):
        self.pipeline.fit(X_train, y_train)
        
    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)
