import pytest
import pandas as pd

from sknet.network_construction import dataset_constructors
from sknet.network_construction import time_series_constructors


@pytest.fixture
def X_y_generator():

    X = pd.DataFrame([
      (-2.24, -1.19),
      (-3.17, -0.67),
      (1.92, 0.57),
      (1.6, 1.97),
      (3.32, 1.51),
      (1.12, 1.21),
      (-1.32, -2.39),
      (-2.88, -1.83),
      (-2.56, 4.01),
      (-3.36, 3.25),
      (-5.64, 2.57),
      (-4.14, 2.85),
      (-3.04, 2.15)])

    y = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2])

    return X, y


@pytest.fixture
def X_time_series_generator():

    X_uni = pd.DataFrame([-5, -3, 2, 4, -5, 1, 4, 6, 7, -3, 3, 2])
    X_multi = pd.DataFrame([
        (-5, 4, 3),
        (-3, 8, 4),
        (2, -5, 5),
        (3, 2, 6),
        (4, 3, 4),
        (-5, 4, 7),
        (1, 6, 7),
        (4, -2, -3),
        (6, 5, 8),
        (7, -4, 4),
        (-3, -4, 6),
        (3, 4, -6),
        (2, 2, 4)
    ])

    return X_uni, X_multi


def test_knn_fit(X_y_generator):

    knn = dataset_constructors.KNNConstructor(k=3, sep_comp=False)

    with pytest.raises(Exception):
        knn.transform()

    knn.fit(X_y_generator[0], X_y_generator[1])

    G = knn.transform()

    expected_nodes = [i for i in range(13)]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 7), (0, 1), (0, 6),
                      (1, 7), (1, 6), (2, 5),
                      (2, 3), (2, 4), (3, 5),
                      (3, 4), (4, 5), (6, 7),
                      (8, 9), (8, 12), (8, 11),
                      (9, 11), (9, 12), (9, 10),
                      (10, 11), (10, 12), (11, 12)]
    assert list(G.edges) == expected_edges


def test_epsilon_radius_fit(X_y_generator):

    eps = dataset_constructors.EpsilonRadiusConstructor(epsilon=1,
                                                        sep_comp=False)

    with pytest.raises(Exception):
        eps.transform()

    eps.fit(X_y_generator[0], X_y_generator[1])

    G = eps.transform()

    expected_nodes = [i for i in range(13)]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 7), (3, 5), (9, 11)]

    assert list(G.edges) == expected_edges


def test_epsilon_radius_fit_true_sep_comp(X_y_generator):

    eps = dataset_constructors.EpsilonRadiusConstructor(epsilon=1,
                                                        sep_comp=True)

    with pytest.raises(Exception):
        eps.transform()

    eps.fit(X_y_generator[0], X_y_generator[1])

    G = eps.transform()

    expected_nodes = [0, 1, 2, 3, 7, 5, 6, 8, 10, 11, 12, 13, 14, 9]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 7), (3, 5), (11, 9)]

    assert list(G.edges) == expected_edges


def test_clustering_heuristics_fit(X_y_generator):
    clustering = dataset_constructors.SingleLinkageHeuristicConstructor(
        sep_comp=False)

    with pytest.raises(Exception):
        clustering.transform()

    clustering.fit(X_y_generator[0], X_y_generator[1])

    G = clustering.transform()

    expected_nodes = [0, 11, 2, 9, 3, 4, 5, 6, 7, 8, 10, 12]

    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 11), (0, 2), (0, 3), (0, 4),
                      (0, 6), (0, 7), (11, 8), (11, 10),
                      (11, 12), (2, 9), (2, 3), (2, 4),
                      (2, 5), (9, 3), (9, 8), (9, 10),
                      (9, 12), (3, 4), (3, 5), (4, 5),
                      (5, 6), (5, 8), (6, 7), (8, 10),
                      (8, 12)]

    assert list(G.edges) == expected_edges


def test_clustering_heuristics_fit_true_sep_comp(X_y_generator):
    clustering = dataset_constructors.SingleLinkageHeuristicConstructor(
        sep_comp=True)

    with pytest.raises(Exception):
        clustering.transform()

    clustering.fit(X_y_generator[0], X_y_generator[1])

    G = clustering.transform()

    expected_nodes = [2, 3, 4, 5, 0, 6, 7, 9, 8, 11, 10, 12]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5),
                      (4, 5), (0, 6), (0, 7), (6, 7), (9, 8),
                      (9, 10), (9, 12), (8, 11), (8, 10),
                      (8, 12), (11, 10), (11, 12)]

    assert list(G.edges) == expected_edges


def test_knn_epsilon_fit(X_y_generator):

    eps_knn = dataset_constructors.KNNEpislonRadiusConstructor(
      k=2, epsilon=1.5, sep_comp=False)

    with pytest.raises(Exception):
        eps_knn.transform()

    eps_knn.fit(X_y_generator[0], X_y_generator[1])

    G = eps_knn.transform()

    expected_nodes = [i for i in range(13)]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 7), (0, 1), (0, 6),
                      (1, 7), (2, 5), (2, 3),
                      (2, 4), (3, 5), (3, 4),
                      (6, 7), (8, 9), (8, 12),
                      (9, 11), (9, 12), (9, 10),
                      (10, 11), (11, 12)]

    assert list(G.edges) == expected_edges


def test_knn_epsilon_fit_true_sep_comp(X_y_generator):

    eps_knn = dataset_constructors.KNNEpislonRadiusConstructor(
      k=2, epsilon=1.5, sep_comp=True)

    with pytest.raises(Exception):
        eps_knn.transform()

    eps_knn.fit(X_y_generator[0], X_y_generator[1])

    G = eps_knn.transform()

    expected_nodes = [10, 11, 12, 13, 14, 9, 8]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(10, 11), (10, 9), (11, 9), (11, 12), (12, 9),
                      (12, 8), (9, 8)]

    assert list(G.edges) == expected_edges


def test_univariate_series_fit(X_time_series_generator):
    constructor = (
        time_series_constructors.UnivariateCorrelationConstructor(
            0.3, 4
        )
    )

    constructor.fit(X_time_series_generator[0])
    G = constructor.transform()

    expected_nodes = [i for i in range(9)]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 0), (0, 4), (0, 5), (1, 1),
                      (1, 6), (2, 2), (2, 7), (3, 3),
                      (3, 8), (4, 4), (4, 5), (5, 5),
                      (6, 6), (7, 7), (8, 8)]

    assert list(G.edges) == expected_edges

    G = constructor.fit_transform(X_time_series_generator[0])
    assert list(G.nodes) == expected_nodes
    assert list(G.edges) == expected_edges


def test_multivariate_series_fit(X_time_series_generator):
    constructor = (
        time_series_constructors.MultivariateCorrelationConstructor(
            0.1
        )
    )

    constructor.fit(X_time_series_generator[1])
    G = constructor.transform()

    expected_nodes = [i for i in range(3)]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 0), (1, 1), (2, 2)]

    assert list(G.edges) == expected_edges

    G = constructor.fit_transform(X_time_series_generator[1])
    assert list(G.nodes) == expected_nodes
    assert list(G.edges) == expected_edges


def test_get_set_params():
    # Time series constructors
    constructor = (
        time_series_constructors.UnivariateCorrelationConstructor()
    )
    param_dict = {'r': 0.3, 'L': 5}
    constructor.set_params(**param_dict)
    assert param_dict == constructor.get_params()

    constructor = (
        time_series_constructors.MultivariateCorrelationConstructor()
    )
    param_dict = {'r': 0.3}
    constructor.set_params(**param_dict)
    assert param_dict == constructor.get_params()

    # Dataset constructors
    param_dict = {"k": 3, "epsilon": None, "metric": 'minkowski',
                  "leaf_size": 40, "sep_comp": True}
    constructor = dataset_constructors.KNNConstructor()
    constructor.set_params(**param_dict)
    assert param_dict == constructor.get_params()

    param_dict['k'] = None
    param_dict['epsilon'] = 0.1
    constructor = dataset_constructors.EpsilonRadiusConstructor()
    constructor.set_params(**param_dict)
    assert param_dict == constructor.get_params()

    param_dict['k'] = 2
    constructor = dataset_constructors.KNNEpislonRadiusConstructor()
    constructor.set_params(**param_dict)
    assert param_dict == constructor.get_params()

    constructor = dataset_constructors.SingleLinkageHeuristicConstructor()
    param_dict = {'k': 3, 'lambda_': 0.1,
                  'n_jobs': 2, 'sep_comp': True,
                  'metric': 'euclidean'}
    constructor.set_params(**param_dict)
    assert param_dict == constructor.get_params()


def test_not_fitted_raise():
    with pytest.raises(Exception):
        dataset_constructors.KNNConstructor().transform()

    with pytest.raises(Exception):
        dataset_constructors.EpsilonRadiusConstructor().transform()

    with pytest.raises(Exception):
        dataset_constructors.KNNEpislonRadiusConstructor().transform()

    with pytest.raises(Exception):
        dataset_constructors.ClusteringHeuristicConstructor().transform()

    with pytest.raises(Exception):
        time_series_constructors.UnivariateCorrelationConstructor().transform()

    with pytest.raises(Exception):
        time_series_constructors.MultivariateCorrelationConstructor(
        ).transform()
