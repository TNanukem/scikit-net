import pytest
import pandas as pd

from sknet.network_construction import dataset_constructors

@pytest.fixture
def X_y_generator():

    X = pd.DataFrame([(-2.24, -1.19),
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

def test_knn_fit(X_y_generator):

    knn = dataset_constructors.KNNConstructor(k=3)

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

    eps = dataset_constructors.EpsilonRadiusConstructor(epsilon=1)
    
    with pytest.raises(Exception):
        eps.transform()

    eps.fit(X_y_generator[0], X_y_generator[1])

    G = eps.transform()

    expected_nodes = [i for i in range(13)]
    assert list(G.nodes) == expected_nodes

    expected_edges = [(0, 7), (3, 5), (9, 11)]

    assert list(G.edges) == expected_edges

def test_knn_epsilon_fit(X_y_generator):

    eps_knn = dataset_constructors.KNNEpislonRadiusConstructor(k=2, epsilon=1.5)
    
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