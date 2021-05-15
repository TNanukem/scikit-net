import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sknet.network_construction import KNNConstructor
from sknet.supervised import HighLevelClassifier


@pytest.fixture
def X_y_generator():

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def module_generator(X_y_generator):
    knn = KNNConstructor(k=3)
    classifier = HighLevelClassifier(
        knn, 'random_forest', 0.5, [0.5, 0.5],
        ['clustering_coefficient', 'assortativity']
    )
    classifier.fit(X_y_generator[0], X_y_generator[1])

    return classifier


def test_predict(module_generator, X_y_generator):

    expected = [1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0,
                1, 2, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0,
                0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 1, 1, 0, 0,
                1, 1, 2, 1, 2]
    pred = module_generator.predict(X_y_generator[2])
    np.testing.assert_equal(expected, pred)
