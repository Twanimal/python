from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import re


class PrepProcesor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.salesImputer = SimpleImputer()

    def fit(self, X, y=None):
        self.salesImputer.fit(X[['sales']])
        return self

    def transform(self, X, y=None):
        X['sales'] = self.salesImputer.transform(X[['sales']])
        return X

    def predict(self, X):
        return self.model.predict(X)



columns = ['sales','customer_rate','performance']