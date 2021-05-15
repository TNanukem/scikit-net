import numpy as np
import pandas as pd
import networkx as nx

from abc import ABCMeta, abstractmethod
from sklearn.neighbors import KDTree, BallTree


class BaseConstructor(metaclass=ABCMeta):
    """
    This class allows to transform a dataset into a networkx
    complex network by using the several different transformation
    methods

    Do not use this abstract class, use derived classes instead

    """

    def __init__(self, k, epsilon, metric, leaf_size=40, sep_comp=True):
        self.k = k
        self.epsilon = epsilon
        self.metric = metric
        self.leaf_size = leaf_size
        self.sep_comp = sep_comp
        self.X = None
        self.y = None

    @abstractmethod
    def add_nodes(self, X, y=None):
        """Add nodes to an existing network inside a fitted transformer
        object

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The true classes.

        Notes
        -----
        If y is set, then the class of each node will be inserted into
        the node information under the label 'class'. If sep_comp is true
        then each class will be a separated component of the network.

        If by some reason the transformer is not fitted, this will generate
        an error.

        After the new nodes are added, one should use the get_network
        function to retrieve the network with the new nodes.

        """
        pass

    def fit(self, X, y=None):
        """Fit the constructor creating the NetworkX graph

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The true classes.

        Notes
        -----
        If y is set, then the class of each node will be inserted into
        the node information under the label 'class' and each class will
        be a separated component of the network

        """

        self.G = nx.Graph()
        self.node_count = 0
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        self.X = X
        self.y = y
        self.fitting = True
        self.add_nodes(self.X, self.y)
        self.fitting = False

    def transform(self):
        """Returns the networkX graph after the constructor is fitted

        Returns
        -----
        G : NetworkX graph
            The network version of the inserted tabular data
        """
        try:
            return self.G
        except AttributeError:
            raise Exception("Transformer is not fitted")

    def fit_transform(self, X, y=None):
        """Fit the constructor creating the NetworkX graph and returns the graph

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The predicted classes.

        Returns
        -------
        G : NetworkX graph
            The network version of the inserted tabular data

        Notes
        -----
        If y is set, then the class of each node will be inserted
        into the node information under the label 'class'

        """
        self.fit(X, y)
        return self.G

    def get_network(self):
        """Retrieves the network generated in the constructor class
        """
        return self.G

    def set_sep_comp(self, sep_comp):
        self.sep_comp = sep_comp


class KNNConstructor(BaseConstructor):
    """
    Using a k-nearest neighbors algorithm, defines an
    networkx complex network

    Parameters
    ----------
    k : int, default=5
        The number of neighbors to be connected to any given node
        of the network.
    metric : str or DistanceMetric object, default='minkowski'
        The distance metric to use for the neighborhood tree. Refer
        to the DistanceMetric class documentation from sklearn for a list
        of available metrics
    leaf_size : int, default=40
        Number of points to switch to brute-force search of neighbors
    sep_comp : boolean, default=True
        If True and if y is not None, then each class of the dataset
        will be a separated component, so nodes from one class will only
        be connected to those of the same class. If False then this
        restriction is not applied.

    Attributes
    ----------
    k : int
        The k being used to construct the network
    metric : str or DistanceMetric object
        The distance metric being used
    leaf_size : int
        The leaf_size being used
    G : NetworkX graph
        The network version of the inserted tabular data

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from dataset_constructors import KNNConstructor
    >>> X, y = load_iris(return_X_y = True)
    >>> knn_c = KNNConstructor(k=3)
    >>> knn_c.fit(X, y)
    >>> G = knn_c.transform()
    >>> print(len(G.nodes))
    150

    Notes
    -----

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in
    Complex Networks. 10.1007/978-3-319-17290-3.

    """
    def __init__(self, k=5, metric='minkowski', leaf_size=40, sep_comp=True):
        super().__init__(k, None, metric, leaf_size, sep_comp)

    def add_nodes(self, X, y=None):
        """Add nodes to an existing network inside a fitted transformer
        object

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The true classes.

        Notes
        -----
        If y is set, then the class of each node will be inserted into
        the node information under the label 'class'. If sep_comp is true
        then each class will be a separated component of the network.

        If by some reason the transformer is not fitted, this will generate
        an error.

        After the new nodes are added, one should use the get_network
        function to retrieve the network with the new nodes.

        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        # Each class will be a separated component
        if self.y is None:
            classes = [0]
        else:
            classes = np.unique(self.y)

        for class_ in classes:

            if self.y is None:
                nodes = [node for node in range(self.node_count, len(X) + self.node_count)]  # noqa: E501
                X_ = X
                self.tree = _tree_selector(self.X, self.leaf_size)
                label_ind = [i for i in range(len(X))]

            else:
                if self.sep_comp:
                    # Verifies if someone to be added is from class
                    X_component = np.take(X, np.where(y == class_), axis=0)[0]
                    if len(X_component) == 0:
                        continue

                    # Calculating the distances for guys on the same component
                    if self.fitting:
                        total_y = self.y
                        total_X = self.X
                    else:
                        total_y = np.append(self.y, y)
                        total_X = np.vstack((self.X, X))
                    label_ind = np.where(total_y == class_)

                    X_ = np.take(total_X, label_ind, axis=0)[0]
                    nodes = [(node, {'class': class_}) for node in range(self.node_count, len(X_component) + self.node_count)]  # noqa: E501

                    label_ind = label_ind[0].tolist()

                else:
                    X_ = X
                    label_ind = [i for i in range(len(X))]
                    nodes = [(node, {'class': y[node - self.node_count]}) for node in range(self.node_count, len(X_) + self.node_count)]  # noqa: E501

                self.tree = _tree_selector(X_, self.leaf_size)

            neighbors = [self.tree.query(x.reshape(1, -1), k=self.k+1, return_distance=True) for x in X_]  # noqa: E501
            distances_aux = [neigh[0] for neigh in neighbors]
            indexes_aux = [neigh[1] for neigh in neighbors]
            indexes = [node[0] for node in indexes_aux]
            distances = [node[0] for node in distances_aux]
            edges = [(label_ind[node[0]], label_ind[node[j]], distances[i][j]) for i, node in enumerate(indexes) for j in range(1, self.k+1)]  # noqa: E501

            self.G.add_nodes_from(nodes)
            self.G.add_weighted_edges_from(edges)
            self.node_count += len(nodes)

            if self.sep_comp is False:
                break

        if not np.array_equal(self.X, X):
            self.X = np.vstack((self.X, X))
            if self.y is not None:
                self.y = np.append(self.y, y)


class EpsilonRadiusConstructor(BaseConstructor):
    """
    Using an epsilon-radius algorithm, defines an
    networkx complex network

    Parameters
    ----------
    epsilon : float
        The radius to define which neighbors should be connected.
    metric : str or DistanceMetric object, default='minkowski'
        The distance metric to use for the neighborhood tree. Refer
        to the DistanceMetric class documentation from sklearn for a list
        of available metrics
    leaf_size : int, default=40
        Number of points to switch to brute-force search of neighbors
    sep_comp : boolean, default=True
        If True and if y is not None, then each class of the dataset
        will be a separated component, so nodes from one class will only
        be connected to those of the same class. If False then this
        restriction is not applied.

    Attributes
    ----------
    epsilon : float
        The epsilon being used to construct the network
    metric : str or DistanceMetric object
        The distance metric being used
    leaf_size : int
        The leaf_size being used
    G : NetworkX graph
        The network version of the inserted tabular data

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from dataset_constructors import EpsilonRadiusConstructor
    >>> X, y = load_iris(return_X_y = True)
    >>> eps_c = EpsilonRadiusConstructor(epsilon=3)
    >>> eps_c.fit(X, y)
    >>> G = eps_c.transform()
    >>> print(len(G.nodes))
    150

    Notes
    -----

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in
    Complex Networks. 10.1007/978-3-319-17290-3.

    """
    def __init__(self, epsilon, metric='minkowski', leaf_size=40,
                 sep_comp=True):
        super().__init__(None, epsilon, metric, leaf_size, sep_comp)

    def add_nodes(self, X, y=None):
        """Add nodes to an existing network inside a fitted transformer
        object

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The true classes.

        Notes
        -----
        If y is set, then the class of each node will be inserted into
        the node information under the label 'class'. If sep_comp is true
        then each class will be a separated component of the network.

        If by some reason the transformer is not fitted, this will generate
        an error.

        After the new nodes are added, one should use the get_network
        function to retrieve the network with the new nodes.

        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        # Each class will be a separated component
        if self.y is None:
            classes = [0]
        else:
            classes = np.unique(self.y)

        for class_ in classes:
            if self.y is None:
                nodes = [node for node in range(self.node_count, len(X) + self.node_count)]  # noqa: E501
                X_ = X
                self.tree = _tree_selector(self.X, self.leaf_size)
                label_ind = [i for i in range(len(X))]

            else:
                if self.sep_comp:
                    # Verifies if someone to be added is from class
                    X_component = np.take(X, np.where(y == class_), axis=0)[0]
                    if len(X_component) == 0:
                        continue

                    # Calculating the distances for guys on the same component
                    if self.fitting:
                        total_y = self.y
                        total_X = self.X
                    else:
                        total_y = np.append(self.y, y)
                        total_X = np.vstack((self.X, X))
                    label_ind = np.where(total_y == class_)

                    X_ = np.take(total_X, label_ind, axis=0)[0]
                    nodes = [(node, {'class': class_}) for node in range(self.node_count, len(X_component) + self.node_count)]  # noqa: E501

                    label_ind = label_ind[0].tolist()

                else:
                    X_ = X
                    label_ind = [i for i in range(len(X))]
                    nodes = [(node, {'class': y[node - self.node_count]}) for node in range(self.node_count, len(X_) + self.node_count)]  # noqa: E501

                self.tree = _tree_selector(X_, self.leaf_size)

            neighbors = [self.tree.query_radius(x.reshape(1, -1), r=self.epsilon, return_distance=True, sort_results=True) for x in X_]  # noqa: E501

            indexes_aux = [neigh[0] for neigh in neighbors]
            distances_aux = [neigh[1] for neigh in neighbors]
            distances = [node[0] for node in distances_aux]
            indexes = [node[0] for node in indexes_aux]

            edges = [(label_ind[node[0]], label_ind[node[j]], distances[i][j]) for i, node in enumerate(indexes) for j in range(1, len(node))]  # noqa: E501

            self.G.add_nodes_from(nodes)
            self.G.add_weighted_edges_from(edges)

            # Removing self-loops
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
            self.node_count += len(nodes) + 1

            if self.sep_comp is False:
                break

        if not np.array_equal(self.X, X):
            self.X = np.vstack((self.X, X))
            if self.y is not None:
                self.y = np.vstack((self.y, y))


class KNNEpislonRadiusConstructor(BaseConstructor):
    """
    Using a k-nearest neighbors algorithm, defines an
    networkx complex network

    Parameters
    ----------
    k : int, default=5
        The number of neighbors to be connected to any given node
        of the network.
    epsilon : float, default=0.1
        The radius to define which neighbors should be connected.
    metric : str or DistanceMetric object, default='minkowski'
        The distance metric to use for the neighborhood tree. Refer
        to the DistanceMetric class documentation from sklearn for a list
        of available metrics
    leaf_size : int, default=40
        Number of points to switch to brute-force search of neighbors
    sep_comp : boolean, default=True
        If True and if y is not None, then each class of the dataset
        will be a separated component, so nodes from one class will only
        be connected to those of the same class. If False then this
        restriction is not applied.

    Attributes
    ----------
    k : int
        The k being used to construct the network
    epsilon : float
        The epsilon being used to construct the network
    metric : str or DistanceMetric object
        The distance metric being used
    leaf_size : int
        The leaf_size being used
    G : NetworkX graph
        The network version of the inserted tabular data

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from dataset_constructors import KNNEpislonRadiusConstructor
    >>> X, y = load_iris(return_X_y = True)
    >>> ke_c = KNNEpislonRadiusConstructor(k=3, epsilon=0.3)
    >>> ke_c.fit(X, y)
    >>> G = ke_c.transform()
    >>> print(len(G.nodes))
    150

    Notes
    -----
    The KNN is used for sparse regions while the Epsilon-Radius is used for
    dense regions. This approach hopes to overcome the limitations of the
    individual components, allowing for a better network construction. The
    equation that runs this method is defined as:

    ``neighbor(v_i) = epsilon-radius(v_i) if |epsilon-radius(v_i)| >
    k else knn(v_i)``

    References
    ----------
    Silva, T.C.; Liang Zhao (2012). Network-Based High Level Data
    Classification., 23(6), –. doi:10.1109/tnnls.2012.2195027
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex Networks.
    10.1007/978-3-319-17290-3.

    """
    def __init__(self, k=5, epsilon=0.1, metric='minkowski', leaf_size=40,
                 sep_comp=True):
        super().__init__(k, epsilon, metric, leaf_size, sep_comp)

    def add_nodes(self, X, y=None):
        """Add nodes to an existing network inside a fitted transformer
        object

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The true classes.

        Notes
        -----
        If y is set, then the class of each node will be inserted into
        the node information under the label 'class'. If sep_comp is true
        then each class will be a separated component of the network.

        If by some reason the transformer is not fitted, this will generate
        an error.

        After the new nodes are added, one should use the get_network
        function to retrieve the network with the new nodes.

        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X)

        # Each class will be a separated component
        if self.y is None:
            classes = [0]
        else:
            classes = np.unique(self.y)

        for class_ in classes:

            if self.y is None:
                nodes = [node for node in range(self.node_count, len(X) + self.node_count)]  # noqa: E501
                X_ = X
                self.tree = _tree_selector(self.X, self.leaf_size)
                label_ind = [i for i in range(len(X))]

            else:
                if self.sep_comp:
                    # Verifies if someone to be added is from class
                    X_component = np.take(X, np.where(y == class_), axis=0)[0]
                    if len(X_component) == 0:
                        continue

                    # Calculating the distances for guys on the same component
                    if self.fitting:
                        total_y = self.y
                        total_X = self.X
                    else:
                        total_y = np.append(self.y, y)
                        total_X = np.vstack((self.X, X))
                    label_ind = np.where(total_y == class_)

                    X_ = np.take(total_X, label_ind, axis=0)[0]
                    nodes = [(node, {'class': class_}) for node in range(self.node_count, len(X_component) + self.node_count)]  # noqa: E501

                    label_ind = label_ind[0].tolist()

                else:
                    X_ = X
                    label_ind = [i for i in range(len(X))]
                    nodes = [(node, {'class': y[node - self.node_count]}) for node in range(self.node_count, len(X_) + self.node_count)]  # noqa: E501

                self.tree = _tree_selector(X_, self.leaf_size)

            radius_neighbors = [self.tree.query_radius(x.reshape(1, -1), r=self.epsilon, return_distance=True, sort_results=True) for x in X_]  # noqa: E501
            k_neighbors = [self.tree.query(x.reshape(1, -1), k=self.k+1, return_distance=True) for x in X_]  # noqa: E501

            # Auxiliar lists
            indexes_radius_aux = [neigh[0] for neigh in radius_neighbors]
            distances_radius_aux = [neigh[1] for neigh in radius_neighbors]  # noqa: E501
            distances_radius = [node[0] for node in distances_radius_aux]
            indexes_radius = [node[0] for node in indexes_radius_aux]

            distances_k_aux = [neigh[0] for neigh in k_neighbors]
            indexes_k_aux = [neigh[1] for neigh in k_neighbors]  # noqa: E501
            indexes_k = [node[0] for node in indexes_k_aux]
            distances_k = [node[0] for node in distances_k_aux]

            # Nodes with neighbors inside radius greater than k
            greater_than_k_indices = [index for index, neighbors in enumerate(indexes_radius) if len(neighbors) - 1 > self.k]  # noqa: E501

            final_k = [neighbors for index, neighbors in enumerate(indexes_k) if index not in greater_than_k_indices]  # noqa: E501
            final_radius = [neighbors for index, neighbors in enumerate(indexes_radius) if index in greater_than_k_indices]  # noqa: E501
            final_k_distances = [dist for index, dist in enumerate(distances_k) if index not in greater_than_k_indices]  # noqa: E501
            final_radius_distances = [distance for index, distance in enumerate(distances_radius) if index in greater_than_k_indices]  # noqa: E501

            assert len(final_k) + len(final_radius) == len(nodes)

            edges_radius = [(label_ind[node[0]], label_ind[node[j]], final_radius_distances[i][j]) for i, node in enumerate(final_radius) for j in range(1, len(node))]  # noqa: E501
            edges_k = [(label_ind[node[0]], label_ind[node[j]], final_k_distances[i][j]) for i, node in enumerate(final_k) for j in range(1, self.k+1)]  # noqa: E501

            self.G = nx.Graph()
            self.G.add_nodes_from(nodes)
            self.G.add_weighted_edges_from(edges_radius)
            self.G.add_weighted_edges_from(edges_k)

            # Removing self-loops
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
            self.node_count += len(nodes) + 1

            if self.sep_comp is False:
                break

        if not np.array_equal(self.X, X):
            self.X = np.vstack((self.X, X))
            if self.y is not None:
                self.y = np.vstack((self.y, y))


def _tree_selector(X, leaf_size=40, metric='minkowski'):
    """
    Selects the better tree approach for given data

    Parameters
    ----------
    X : {array-like, pandas dataframe} of shape (n_samples, n_features)
        The input data.
    leaf_size : int, default=40
        Number of points to switch to brute-force search of neighbors
    metric : str or DistanceMetric object, default='minkowski'
        The distance metric to use for the neighborhood tree. Refer
        to the DistanceMetric class documentation from sklearn for a list
        of available metrics

    Returns
    -------
    tree : {KDTree or BallTree}
        The best tree to be used to find neighbors given data
    """

    # Low dimensional spaces are fit to KD-Tree
    if X.shape[1] < 30:
        return KDTree(X, leaf_size=leaf_size, metric=metric)

    # High dimensional spaces are fit to Ball Tree
    if X.shape[1] >= 30:
        return BallTree(X, leaf_size=leaf_size, metric=metric)
