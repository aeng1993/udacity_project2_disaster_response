import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CountLength(BaseEstimator, TransformerMixin):
    '''
    Define the CountLength class to pass into the machine learning pipeline
    This is used to count the number of sentence in a document (message), which will be used as a feature.
    '''
    def length_count(self, text):
        return len(text)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.length_count)
        return pd.DataFrame(X_tagged)
