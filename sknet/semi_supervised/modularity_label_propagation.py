import numpy as np
import networkx as nx

from sknet.network_construction import KNNConstructor


class ModularityLabelPropagation():
    """
    Semi-supervised method that propagates labels to instances not
    classified using the Modularity Propagation method.

    Parameters
    ----------
    reduction_factor : None or list of floats, optional (default=None)
        If not None, the aggregation algorithm proposed by Silva & Zhao will be
        applied to reduce the network and speed up the processing. The values
        on the list will be the reduction factor for each class

    Attributes
    ----------
    generated_y_ : {ndarray, pandas series}, shape (n_samples, 1)
        The label list
    generated_G_ : NetworkX Network
        The constructed network on the fit of the model

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sknet.network_construction import KNNConstructor
    >>> from sknet.semi_supervised import ModularityLabelPropagation
    >>> X, y = load_iris(return_X_y = True)
    >>> knn_c = KNNConstructor(k=5, sep_comp=False)
    >>> y[10:20] = np.nan
    >>> y[70:80] = np.nan
    >>> y[110:120] = np.nan
    >>> propagator = ModularityLabelPropagation()
    >>> propagator.fit(X, y, constructor=knn_c)
    >>> propagator.generated_y_
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
       1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

    References
    ----------
    Silva, Thiago & Zhao, Liang. (2012). Semi-Supervised Learning Guided
    by the Modularity Measure in Complex Networks. Neurocomputing. 78.
    30-37. 10.1016/j.neucom.2011.04.042.

    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex
    Networks. 10.1007/978-3-319-17290-3.

    """
    def __init__(self, reduction_factor=None, random_state=None):
        self.estimator_type = 'classifier'
        self.reduction_factor = reduction_factor
        self.random_state = random_state
        np.random.seed(random_state)  # Arrumar

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {'reduction_factor': self.reduction_factor,
                'random_state': self.random_state}

    def fit(self, X=None, y=None, G=None,
            constructor=KNNConstructor(5, sep_comp=False)):
        """Fit the propagator by using the modularity measure
        to propagate the labels to non-labeled examples

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape
        (n_samples, n_features), optional (default=None)
            The input data samples. Can be None if G is set.
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), optional (default=None)
            The target classes. Can be None if G is set. Missing labels
            should have the np.nan value
        G : NetworkX Network, optional (default=None)
            The network with missing labels to be propagated. Can be
            None if X and y are not None in which case the constructor
            will be used to generate the network. Labels must be into
            the data of each node with the 'class' key. Missing labels
            should be valued np.nan
        constructor : BaseConstructor inhrerited class, optional(default=
            KNNConstructor(5, sep_comp=False))
            A constructor class to transform the tabular data into a
            network. It can be set to None if a complex network is directly
            passed to the ``fit`` method. Notice that you should use 'sep_com'
            as False on the constructor.

        """
        self.constructor = constructor
        if y is None and G is None:
            raise Exception('Both y and G are None!')

        if self.constructor is None and G is None:
            raise Exception(
                'You either have to set the constructor or the network'
            )

        if y is not None and self.constructor is not None:
            G = self.constructor.fit_transform(X, y)
        elif y is None and G is not None:
            y = np.array([node[1]['class'] for node in G.nodes(data=True)])

        if self.reduction_factor is not None:
            if not isinstance(self.reduction_factor, list):
                raise Exception('Reduction_factor must be a list or None')

            if np.max(self.reduction_factor) > 1 or np.min(self.reduction_factor) < 0:  # noqa: E501
                raise Exception('Reduction_factor must be between 0 and 1')

            if len(np.unique(y[~np.isnan(y)])) != len(self.reduction_factor):
                raise Exception('The number of reduction factors must be equal'
                      ' to the number of classes')

        missing_elements = len(y[np.isnan(y)])

        if self.reduction_factor is not None:
            original_G = G.copy()
            original_y = y.copy()

            G = self._reduce_graph(G, y)

            positions_dict = {i: node for i, node in enumerate(list(G.nodes()))}  # noqa: E501
            G = nx.convert_node_labels_to_integers(G)
            y = np.array([node[1]['class'] for node in G.nodes(data=True)])

        # Generate modularity matrix
        Q = self._increment_modularity_matrix(G)

        while missing_elements != 0:
            propagated = False

            while not propagated:
                # Select the i and j of argmax
                i, j = np.unravel_index(Q.argmax(), Q.shape)

                Q[i][j] = -np.inf
                Q[j][i] = -np.inf

                if y[i] != y[j]:
                    if (~np.isnan(y[i])) and (~np.isnan(y[j])):
                        continue
                    if np.isnan(y[i]):
                        y[i] = y[j]
                        G.nodes[i]['class'] = y[i]
                        propagated = True
                    if np.isnan(y[j]):
                        y[j] = y[i]
                        G.nodes[j]['class'] = y[j]
                        propagated = True
                else:
                    continue

            missing_elements = len(y[np.isnan(y)])

        if self.reduction_factor is not None:
            for key in positions_dict:
                original_y[positions_dict[key]] = y[key]
                original_G.nodes[positions_dict[key]]['class'] = y[key]  # noqa: E501

            y = original_y
            G = original_G

        self.generated_y_ = y
        self.generated_G_ = G

        return self

    def get_propagated_labels(self):
        """
        Return the labels list with the propagated classes

        Returns
        -------
        generated_y_ : {ndarray, pandas series}, shape (n_samples, 1)
            The label list
        """

        return self.generated_y_

    def get_propagated_network(self):
        """
        Returns the generated network with the propagated labels

        Returns
        --------
        generated_G_ : NetworkX Network
            The constructed network on the fit of the model"""

        return self.generated_G_

    def _increment_modularity_matrix(self, G):
        N = len(G.nodes)
        E = len(G.edges)
        k = [val for (node, val) in G.degree()]

        Q = [[0 for i in range(N)] for j in range(N)]

        for i in range(N):
            for j in range(N):
                if i not in G.neighbors(j):
                    Q[i][j] = 0
                else:
                    Q[i][j] = (1/(2*E)) - (k[i]*k[j])/((2*E)**2)
        return np.array(Q)

    def _reduce_graph(self, G, y):
        """
        Reduce the graph using the algorithm from Silva & Zhao (2012)

        Parameters
        ----------
        G : NetworkX Network
            The network to be reduced
        y : {ndarray, pandas series}, shape (n_samples,)
            The label list

        Returns
        -------
        G : NetworkX Network
            The reduced network
        """
        G = G.copy()
        classes = np.unique(y[~np.isnan(y)])
        classes.sort()
        for idx, class_ in enumerate(classes):
            factor = self.reduction_factor[idx]

            if factor == 0:
                continue

            N = len([i for i in G.nodes(data=True) if i[1]['class'] == class_])  # noqa: E501
            N_tilda = N

            if factor != 1:
                desired_value = round((1-factor) * N)
            else:
                desired_value = 1

            while N_tilda != desired_value:
                # Randomly select two nodes from the class
                nodes = np.random.choice(
                    [i[0] for i in G.nodes(data=True) if i[1]['class'] == class_],  # noqa: E501
                    size=2,
                    replace=False)

                # Get the edges from first node
                edges = [i for i in G.edges(nodes[0])]

                # Remove the first node from the network
                G.remove_node(nodes[0])

                # Remove self-loops
                G.remove_edges_from(nx.selfloop_edges(G))

                # Redistribute the edges from the first node to the second node
                for edge in edges:
                    # Avoid self-loops
                    if edge[0] == edge[1]:
                        continue
                    if edge[0] == nodes[0]:
                        G.add_edge(nodes[1], edge[1])
                    else:
                        G.add_edge(edge[0], nodes[1])

                N_tilda = len([i for i in G.nodes(data=True) if i[1]['class'] == class_])  # noqa: E501

        # Remove any possible remaining self-loop
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
