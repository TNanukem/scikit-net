import numpy as np


class ModularityLabelPropagation():
    """
    Semi-supervised method that propagates labels to instances not
    classified using the Modularity Propagation method.

    Parameters
    ----------
    constructor : BaseConstructor inhrerited class, optional(default=None)
        A constructor class to transform the tabular data into a
        network. It can be set to None if a complex network is directly
        passed to the ``fit`` method. Notice that you should use 'sep_com' as
        False on the constructor.

    Attributes
    ----------
    generated_y : {ndarray, pandas series}, shape (n_samples, 1)
        The label list
    generated_G : NetworkX Network
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
    >>> propagator = ModularityLabelPropagation(knn_c)
    >>> propagator.fit(X, y)
    >>> propagator.generated_y
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
    def __init__(self, constructor=None):
        self.constructor = constructor

    def fit(self, X=None, y=None, G=None):
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

        """

        if y is None and G is None:
            raise('Both y and G are None!')

        if self.constructor is None and G is None:
            raise('You either have to set the constructor or the network')

        if y is not None and self.constructor is not None:
            G = self.constructor.fit_transform(X, y)
        elif y is None and G is not None:
            y = np.array([node[1]['class'] for node in G.nodes(data=True)])

        missing_elements = len(y[np.isnan(y)])

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
        self.generated_y = y
        self.generated_G = G

    def get_propagated_labels(self):
        """
        Return the labels list with the propagated classes

        Returns
        -------
        generated_y : {ndarray, pandas series}, shape (n_samples, 1)
            The label list
        """

        return self.generated_y

    def get_propagated_network(self):
        """
        Returns the generated network with the propagated labels

        Returns
        --------
        generated_G : NetworkX Network
            The constructed network on the fit of the model"""

        return self.generated_G

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
