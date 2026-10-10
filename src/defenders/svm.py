from .base import Defender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np

class SVMDefender(Defender):
    def __init__(self, name="SVM"):
        super().__init__(name)
        # We need probability=True to use Platt scaling for predict_proba
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english', max_features=10000)),
            ('clf', SVC(kernel='linear', probability=True, class_weight='balanced', max_iter=2000))
        ])
        
    def fit(self, X_train: list[str], y_train: np.ndarray):
        self.pipeline.fit(X_train, y_train)
        
    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)
