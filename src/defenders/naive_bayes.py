from .base import Defender
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

class NaiveBayesDefender(Defender):
    def __init__(self, name="NaiveBayes"):
        super().__init__(name)
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(stop_words='english', max_features=10000)),
            ('clf', MultinomialNB())
        ])
        
    def fit(self, X_train: list[str], y_train: np.ndarray):
        self.pipeline.fit(X_train, y_train)
        
    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)
