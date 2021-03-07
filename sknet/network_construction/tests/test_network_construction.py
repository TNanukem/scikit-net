import pytest

from sknet.network_construction import dataset_constructors

@pytest.fixture
def X_y_generator():
    X = 'abc'
    y = 'cde'
    return (X, y)

def test_knn_fit(X_y_generator):

    knn = dataset_constructors.KNNConstructor()

    with pytest.raises(Exception):
        knn.transform()

def test_epsilon_radius_fit(X_y_generator):

    eps = dataset_constructors.EpsilonRadiusConstructor(epsilon=0.1)
    
    with pytest.raises(Exception):
        eps.transform()


def test_knn_epsilon_fit(X_y_generator):

    eps_knn = dataset_constructors.KNNEpislonRadiusConstructor()
    
    with pytest.raises(Exception):
        eps_knn.transform()
