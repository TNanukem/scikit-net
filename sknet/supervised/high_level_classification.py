import copy
import numpy as np
from tqdm import tqdm

from sknet.utils import NetworkMetricsHandler
from sknet.utils import LowLevelModelsHandler

from sknet.network_construction import KNNConstructor


class HighLevelClassifier():
    """
    Classifies a dataset using a high-level approach where the predictions
    from a low-level model (standard ML) and a high-level model (Complex
    Network) are combined to generate a final inference about the class of
    each data point.

    Parameters
    ----------
    low_level : str, optional(default='random_forest')
        The low-level model to be used. See available options on the
        low_level_models_handler documentation
    p : float, optional(default=0.5)
        The weight to be used on the ponderation between the
        low-level and the high-level model predictions. The formula
        is:
        ``(1 - p) * low_level + p * high_level``
        This number should be less or equal than one
    alphas : list of floats, optional(default=[0.5, .5])
        The weight to be used on each high-level metric for the
        classification. This list should sum up to one.
    metrics: list of str, optional(default=['clustering_coefficient',
                                            'assortativity'])
        Which complex networks metrics to use to generate the high-level
        prediction. See available options on the network_metrics_handler
    low_level_parameters : dict, optional(default={})
        Parameters to be set on the low-level classifier

    Attributes
    ----------
    constructor_ : BaseConstructor inhrerited class
        The transformer used to transform the tabular data into network
    low_level_pred_ : {ndarray, pandas series}, shape (n_samples, n_classes)
        The probability of each class from the low-level prediction
    high_level_pred_ : {ndarray, pandas series}, shape (n_samples, n_classes)
        The probability of each class from the high-level prediction
    original_constructor_ : NetworkX Network
        The constructed network on the fit of the model

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sknet.network_construction import KNNConstructor
    >>> from sknet.supervised import HighLevelClassifier
    >>> X, y = load_iris(return_X_y = True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33)
    >>> knn_c = KNNConstructor(k=5)
    >>> classifier = HighLevelClassifier(t=5)
    >>> classifier.fit(X_train, y_train, constructor=knn_c)
    >>> pred = classifier.predict(X_test)

    References
    ----------
    Silva, T.C., Zhao, L.: Network-based high level data classification.
    IEEE Trans. Neural Netw. Learn. Syst. 23(6), 954–970 (2012)

    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex
    Networks. 10.1007/978-3-319-17290-3.

    """
    _estimator_type = 'classifier'

    def __init__(self, low_level='random_forest',
                 p=0.5, alphas=[0.5, 0.5],
                 metrics=['clustering_coefficient', 'assortativity'],
                 low_level_parameters={}):
        self.p = p
        self.alphas = alphas
        self.metrics = metrics
        self.low_level = low_level
        self.low_level_parameters = low_level_parameters
        self.metrics = metrics

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {'p': self.p, 'alphas': self.alphas, 'metrics': self.metrics,
                'low_level': self.low_level,
                'low_level_parameters': self.low_level_parameters,
                'self.metrics': self.metrics}

    def fit(self, X, y, G=None, constructor=KNNConstructor(5)):
        """Fit the classifier by fitting the low-level model and
        creating the high-level classification network

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data samples
        y : {ndarray, pandas series}, shape (n_samples,) or
        (n_samples, n_classes), default=None
            The true classes
        G : NetworkX Graph, default=None
            If the graph was already generated, then this parameter will
            make as so the transformer is not called. Notice that each class
            should be formed of only one class
        constructor : BaseConstructor inhrerited class, optional(default=
        KNNConstructor(5))
            A constructor class to transform the tabular data into a
            network

        """
        self.constructor_ = constructor

        # Basic configuration
        self.metrics_handler = NetworkMetricsHandler()
        self.low_level_handler = LowLevelModelsHandler()

        self.low_level_model = self.low_level_handler.get_model(
            self.low_level, self.low_level_parameters
        )
        self.metric_func = []
        self.default_values = []
        for metric in self.metrics:
            self.metric_func.append(self.metrics_handler.get_metric(metric))
            self.default_values.append(self.metrics_handler.get_default_value(
                metric)
            )

        assert self.p <= 1

        if np.sum(self.alphas) != 1:
            raise ValueError('Alphas should sum to one')

        # Fits the constructor to generate the network
        if G is not None:
            self.G_ = G
        else:
            self.constructor_.set_sep_comp(True)
            self.G_ = self.constructor_.fit_transform(X, y)

        # Fits the low level model
        self.low_level_model.fit(X, y)

        self.X = X
        self.y = y

        return self

    def predict_proba(self, X_test):
        """Predicts the probability, for each test sample
        that it belongs to any of the training classes

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data samples

        """
        # Gets the low level predictions
        self.low_level_pred_ = self.low_level_model.predict_proba(X_test)

        classes = np.unique(self.y)
        self.high_level_pred_ = np.zeros((len(X_test), len(classes)))
        total_training_nodes = len(self.G_.nodes)
        class_proportions = np.zeros((len(classes)))

        # We need to keep the original constructor
        self.original_constructor = copy.deepcopy(self.constructor_)

        for i, class_ in enumerate(classes):
            label_ind = np.where(self.y == class_)
            X_ = np.take(self.X, label_ind, axis=0)[0]

            class_proportions[i] = len(X_) / total_training_nodes

        for i, x in tqdm(enumerate(X_test)):
            original_G = self.original_constructor.get_network()

            delta_G = np.zeros((len(self.metric_func), len(classes)))
            # Tries to put the node into each component
            for class_id, class_ in enumerate(classes):

                # Selects subset of original G
                original_G_sub = self._get_subgraph(original_G, class_)

                # Adds the node to the network on the component of the class
                singleton = False
                self.constructor_.add_nodes([x], [class_])
                new_G = self.constructor_.get_network()
                new_G_sub = self._get_subgraph(new_G, class_)

                # Verifies if the added node has a neighbor
                node = list(new_G_sub.nodes())[-1]

                if new_G_sub.adj[node] == {}:
                    singleton = True

                f = np.zeros((len(self.metric_func)))

                # Gets the variation from the addition
                for idx, metric in enumerate(self.metric_func):
                    if not singleton:
                        original = metric(original_G_sub)
                        new = metric(new_G_sub)
                        delta_G[idx][class_id] = original - new
                    else:
                        delta_G[idx][class_id] = self.default_values[idx]

                # Return the original constructor
                self.constructor_ = copy.deepcopy(self.original_constructor)

            delta_G / delta_G.sum(axis=1)[:, np.newaxis]
            f = delta_G * class_proportions

            for k, f_ in enumerate(f):

                self.high_level_pred_[i] = self.alphas[k] * (1 - f_)

        # Normalize the high_level_pred
        self.high_level_pred_ = (
            self.high_level_pred_ / self.high_level_pred_.sum(
                axis=1)[:, np.newaxis]
            )

        final_pred = (
            (1 -
                self.p) * self.low_level_pred_ + self.p * self.high_level_pred_
        )

        return final_pred

    def predict(self, X_test):
        """Predicts the class for each test sample

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape (n_samples, n_features)
            The input data samples

        """
        predictions = np.argmax(self.predict_proba(X_test), axis=1)
        return predictions

    def _get_subgraph(self, G, class_):
        nodes = (node for node, data in G.nodes(data=True)
                 if data.get('class') == class_)

        return G.subgraph(nodes)
