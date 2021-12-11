import warnings
import pandas as pd
import networkx as nx
import numpy as np

from scipy.stats import pearsonr
from abc import ABCMeta, abstractmethod
from sklearn.metrics import pairwise_distances
from gtda.time_series import SingleTakensEmbedding


class TimeSeriesBaseConstructor(metaclass=ABCMeta):
    """
    This class allows to transform a time series into a networkx
    complex network by using the several different transformation
    methods

    Do not use this abstract class, use derived classes instead
    """

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

        self.X_ = None

        self.add_nodes(X)

        return self

    def transform(self):
        """Returns the networkX graph after the constructor is fitted

        Returns
        -----
        G_ : NetworkX graph
            The network version of the inserted time series data
        """
        try:
            return self.G_
        except AttributeError:
            raise Exception("Transformer is not fitted")

    def get_network(self):
        """Retrieves the network generated in the constructor class
        """
        return self.G_

    def fit_transform(self, X, y=None):
        """Fit the constructor creating the NetworkX graph and returns the graph

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : ignored, used just for API convention

        Returns
        -------
        G_ : NetworkX graph
            The network version of the inserted time series data
        """
        self.fit(X, y)
        return self.G_

    @abstractmethod
    def add_nodes(self, X, y=None):
        """Adds a node to the graph"""

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


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
    >>> G_ = constructor.transform()

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in
    Complex Networks. 10.1007/978-3-319-17290-3.

    Yang, Y., Yang, H.: Complex network-based time series
    analysis. Physica A 387, 1381–1386 (2008)

    """
    def __init__(self, r=0.5, L=10):
        self.r = r
        self.L = L
        self.X_ = None

    def get_params(self, deep=True):
        return {"r": self.r, 'L': self.L}

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

        if self.X_ is not None:
            X = np.vstack((self.X_, X))

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

        self.G_ = nx.from_numpy_array(C)

        self.X_ = X


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
    G_ : NetworkX graph
        The network version of the inserted time series data

    Examples
    --------
    >>> from sknet.network_construction import MultivariateCorrelationConstructor  # noqa: E501
    >>> r = 0.5
    >>> L = 10
    >>> constructor = MultivariateCorrelationConstructor(r, L)
    >>> constructor.fit(X)
    >>> G_ = constructor.transform()

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in
    Complex Networks. 10.1007/978-3-319-17290-3.

    Yang, Y., Yang, H.: Complex network-based time series
    analysis. Physica A 387, 1381–1386 (2008)

    """
    def __init__(self, r=0.5):
        self.r = r

    def get_params(self, deep=True):
        return {"r": self.r}

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

        if self.X_ is not None:
            X = np.vstack((self.X_, X))

        C = np.zeros((X.shape[1], X.shape[1]))

        # Make the correlation matrix
        # Turn into list comprehension later
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                C[i][j] = pearsonr(X[:, i], X[:, j])[0]

        # Make the D matrix
        C[C < self.r] = 0
        C[C >= self.r] = 1

        self.G_ = nx.from_numpy_array(C)

        self.X_ = X


class UnivariateRecurrenceNetworkConstructor(TimeSeriesBaseConstructor):
    """
    Creates a networkX complex network from a univariate time series
    by creating edges between recurrent (close) states on the phase space
    of the series.

    The phase space is constructed using the Takens Embedding Theorem.

    Parameters
    ----------
    epsilon : float, optional (default=0.1)
        The required distance between two states for them to be considered a
        recurrence and hence be connected
    d : int, optional (default=None)
        The dimension of the embedding to be used for the Takens embedding.
        If None and tau is also None, the best parameter will be
        automatically chosen.
    tau : int, optional (default=None)
        The time delay to be used on the Takens Embedding. If None and
        tau is also None, the best parameter will be automatically chosen.
    metric : str, optional (default='euclidean')
        The distance metric to be used to calculate the distance between
        two points on the phase space
    n_jobs : int, optional (default=None)
        The number of parallel processes to be used when applying the
        Takens embedding and when calculating the distances between the
        states. None means 1 core will be used, -1 means use all cores.
    Attributes
    ----------
    G_ : NetworkX graph
        The network version of the inserted time series data

    Examples
    --------
    >>> from sknet.network_construction import UnivariateRecurrenceNetworkConstructor  # noqa: E501
    >>> constructor = UnivariateRecurrenceNetworkConstructor(10)
    >>> constructor.fit(X)
    >>> G_ = constructor.transform()

    References
    ----------
    Donner, R.V., Zou, Y., Donges, J.F., Marwan, N., Kurths, J.: Recurrence
    networks – a novel paradigm for nonlinear time series analysis.
    New J. Phys. 12, 033025 (2010)

    """
    def __init__(self, epsilon=0.1, d=None, tau=None, metric='euclidean',
                 n_jobs=None):
        self.epsilon = epsilon
        self.d = d
        self.tau = tau
        self.metric = metric
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'epsilon': self.epsilon, 'd': self.d, 'tau': self.tau,
                'metric': self.metric, 'n_jobs': self.n_jobs}

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

        if X.shape[1] > 1:
            warnings.warn(
                """More than one feature identified in the series.
                   Recurrence should not be used on multivariate series"""
            )

        if self.X_ is not None:
            X = np.vstack((self.X_, X))

        # Generate the state space of the time series X
        if self.d is None and self.tau is None:
            takens_dict = {'parameters_type': 'search'}
        else:
            takens_dict = {'parameters_type': 'fixed', 'dimension': self.d,
                           'time_delay': self.tau}

        takens_dict['n_jobs'] = self.n_jobs

        takens = SingleTakensEmbedding(**takens_dict)

        space = takens.fit_transform(X)

        # Get distance metric
        R = pairwise_distances(space, metric=self.metric, n_jobs=self.n_jobs)

        # Generate the recurrence matrix
        R[R <= self.epsilon] = 1
        R[R > self.epsilon] = 0

        # Remove self-loops
        for i in range(R.shape[0]):
            R[i, i] = 0

        self.G_ = nx.from_numpy_array(R)

        self.X_ = X
