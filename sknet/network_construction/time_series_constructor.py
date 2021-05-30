import pandas as pd
import networkx as nx
import numpy as np

from scipy.stats import pearsonr
from abc import ABCMeta, abstractmethod


class TimeSeriesBaseConstructor(metaclass=ABCMeta):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        if X.shape[1] != 1:
            raise Exception('For multivariate series, use other constructor')

        if self.X is not None:
            X = np.vstack((self.X, X))

        self.add_nodes(X)

    @abstractmethod
    def add_nodes(self, X, y=None):
        pass


class UnivariateCorrelationConstructor(TimeSeriesBaseConstructor):
    def __init__(self, r, L):
        self.r = r
        self.L = L
        self.X = None

    def add_nodes(self, X, y=None):
        C = np.zeros((1, 1))
        # Create the segments of size L

        # Make the correlation matrix

        # Make the D matrix
        C[C < self.r] = 0
        C[C >= self.r] = 1

        self.G = nx.from_numpy_matrix(C)


class MultivariateCorrelationConstructor(TimeSeriesBaseConstructor):
    def __init__(self):
        pass
