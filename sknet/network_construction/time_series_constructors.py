import warnings
import pandas as pd
import networkx as nx
import numpy as np

from scipy.stats import pearsonr
from abc import ABCMeta, abstractmethod


class TimeSeriesBaseConstructor(metaclass=ABCMeta):
    """
    This class allows to transform a time series into a networkx
    complex network by using the several different transformation
    methods

    Do not use this abstract class, use derived classes instead
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit the constructor creating the NetworkX graph

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : ignored, used just for API convention
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        self.X = None

        self.add_nodes(X)

    def transform(self):
        """Returns the networkX graph after the constructor is fitted

        Returns
        -----
        G : NetworkX graph
            The network version of the inserted time series data
        """
        try:
            return self.G
        except AttributeError:
            raise Exception("Transformer is not fitted")

    def get_network(self):
        """Retrieves the network generated in the constructor class
        """
        return self.G

    def fit_transform(self, X, y=None):
        """Fit the constructor creating the NetworkX graph and returns the graph

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : ignored, used just for API convention

        Returns
        -------
        G : NetworkX graph
            The network version of the inserted time series data
        """
        self.fit(X, y)
        return self.G

    @abstractmethod
    def add_nodes(self, X, y=None):
        pass


class UnivariateCorrelationConstructor(TimeSeriesBaseConstructor):
    """
    Creates a networkX complex network from a univariate time series
    by splitting it into segments of length L and generating the correlation
    between those segments

    Parameters
    ----------
    r : float
        The minimun correlation threshold between two segments
        to create an edge between them on the network. Value must be
        between 0 and 1
    L : int
        The lenght of each segment to be considered on the correlations

    Attributes
    ----------
    G : NetworkX graph
        The network version of the inserted time series data

    Examples
    --------
    >>> from sknet.network_construction import UnivariateCorrelationConstructor
    >>> r = 0.5
    >>> L = 10
    >>> constructor = UnivariateCorrelationConstructor(r, L)
    >>> constructor.fit(X)
    >>> G = constructor.transform()

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in
    Complex Networks. 10.1007/978-3-319-17290-3.

    Yang, Y., Yang, H.: Complex network-based time series
    analysis. Physica A 387, 1381–1386 (2008)

    """
    def __init__(self, r, L):
        self.r = r
        self.L = L
        self.X = None

    def add_nodes(self, X, y=None):
        """Add nodes to an existing network inside a fitted transformer
        object

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, 1)
            The input data.
        y : ignored, used just for API convention
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        if X.shape[1] != 1:
            warnings.warn(
                """More than one feature identified in the series.
                   For multivariate time series use the
                   MultivariateCorrelationConstructor"""
            )

        if self.X is not None:
            X = np.vstack((self.X, X))

        # Create the segments of size L
        segments = []
        for i in range(len(X)):
            segment = X[i:self.L + i]
            if len(segment) < self.L:
                continue
            segments.append(segment)
        C = np.zeros((len(segments), len(segments)))

        # Make the correlation matrix
        # Turn into list comprehension later
        for i in range(len(segments)):
            for j in range(len(segments)):
                C[i][j] = pearsonr(np.array(segments[i]).flatten(),
                                   np.array(segments[j]).flatten())[0]

        # Make the D matrix
        C[C < self.r] = 0
        C[C >= self.r] = 1

        self.G = nx.from_numpy_matrix(C)

        self.X = X


class MultivariateCorrelationConstructor(TimeSeriesBaseConstructor):
    """
    Creates a networkX complex network from a multivariate time series
    by creating edges between highly correlated series

    Parameters
    ----------
    r : float
        The minimun correlation threshold between two series
        to create an edge between them on the network. Value must be
        between 0 and 1

    Attributes
    ----------
    G : NetworkX graph
        The network version of the inserted time series data

    Examples
    --------
    >>> from sknet.network_construction import MultivariateCorrelationConstructor  # noqa: E501
    >>> r = 0.5
    >>> L = 10
    >>> constructor = UnivariateCorrelationConstructor(r, L)
    >>> constructor.fit(X)
    >>> G = constructor.transform()

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in
    Complex Networks. 10.1007/978-3-319-17290-3.

    Yang, Y., Yang, H.: Complex network-based time series
    analysis. Physica A 387, 1381–1386 (2008)

    """
    def __init__(self, r):
        self.r = r

    def add_nodes(self, X, y=None):
        """Add nodes to an existing network inside a fitted transformer
        object

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : ignored, used just for API convention
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        if X.shape[1] == 1:
            warnings.warn(
                """Only one feature identified in the series.
                   For univariate time series use the
                   UnivariateCorrelationConstructor"""
            )

        if self.X is not None:
            X = np.vstack((self.X, X))

        C = np.zeros((X.shape[1], X.shape[1]))

        # Make the correlation matrix
        # Turn into list comprehension later
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                C[i][j] = pearsonr(X[:, i], X[:, j])[0]

        # Make the D matrix
        C[C < self.r] = 0
        C[C >= self.r] = 1

        self.G = nx.from_numpy_matrix(C)

        self.X = X
