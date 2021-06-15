import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from trd.params import RATINGS_MAP
import re
from datetime import date

TODAY = date.today().year



class RatingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return np.vectorize(RATINGS_MAP.get)(X)

    def fit(self, X, y=None):
        return self



class RunTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def clean_runtime(self, row):
        x = row[0]
        counter = 0
        if 'h' in x:
            counter += int(x[0]) * 60
            x = re.sub('.*h', '', x).strip()
        x = x.replace('min', '').replace(',', '').strip()
        counter += int(x)
        return [counter]

    def transform(self, X, y=None):
        return [self.clean_runtime(row) for row in X]

    def fit(self, X, y=None):
        return self


class LanguageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return np.array([[1] if 'English' in  x else [0] for x in X])

    def fit(self, X, y=None):
        return self


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        final = np.array([re.findall("[a-zA-Z]+", x[0]) for x in X])
        return final

    def fit(self, X, y=None):
        return self


class CountryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def include_us(self,row):
        x = row[0]
        usa = ['United States', 'USA']
        for name in usa:
            if name in x:
                return [1]
        return [0]

    def transform(self, X, y=None):
        final = np.array([self.include_us(row) for row in X])
        return final

    def fit(self, X, y=None):
        return self


class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def age(self, row):
        x = row[0]
        return [int(TODAY) - int(x)]

    def transform(self, X, y=None):
        final = np.array([self.age(row) for row in X])
        return final

    def fit(self, X, y=None):
        return self


class ProductionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return np.array([row[0].lower().replace('film', '') for row in X])

    def fit(self, X, y=None):
        return self


if __name__ == '__main__':
    print(TODAY)
