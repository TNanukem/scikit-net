import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sknet.network_construction import KNNConstructor
from sknet.supervised import EaseOfAccessClassifier


@pytest.fixture
def X_y_generator():

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def module_generator_eigen_classifier(X_y_generator):
    knn = KNNConstructor(k=3)
    classifier = EaseOfAccessClassifier(t=5)
    classifier.fit(X_y_generator[0], X_y_generator[1], constructor=knn)

    return classifier


@pytest.fixture
def module_generator_power_classifier(X_y_generator):
    knn = KNNConstructor(k=3)
    classifier = EaseOfAccessClassifier(t=5, method='power')
    classifier.fit(X_y_generator[0], X_y_generator[1], constructor=knn)

    return classifier


@pytest.fixture
def class_generator_classifier(module_generator_eigen_classifier,
                               module_generator_power_classifier,
                               X_y_generator):

    pred_eig = module_generator_eigen_classifier.predict(X_y_generator[2])
    pred_power = module_generator_power_classifier.predict(X_y_generator[2])

    return (module_generator_eigen_classifier,
            module_generator_power_classifier, pred_eig, pred_power)


def test__stationary_distribution(class_generator_classifier):
    np.testing.assert_almost_equal(class_generator_classifier[0].P_inf,
                                   class_generator_classifier[1].P_inf)

    pd.testing.assert_frame_equal(class_generator_classifier[0].tau_,
                                  class_generator_classifier[1].tau_)


def test_predictions(class_generator_classifier):

    expected = [1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0, 1,
                2, 1, 1, 2, 0, 1, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 2, 2,
                1, 2]

    eigen_pred = class_generator_classifier[2]
    power_pred = class_generator_classifier[3]

    assert eigen_pred == power_pred == expected


def test_raise_on_predict(X_y_generator):

    knn = KNNConstructor(k=3)
    classifier = EaseOfAccessClassifier(t=5, method='something')

    with pytest.raises(Exception):
        classifier.fit(X_y_generator[0], X_y_generator[1], constructor=knn)
        classifier.predict(X_y_generator[2])


def test_set_get_params(module_generator_eigen_classifier):
    classifier = module_generator_eigen_classifier
    classifier.set_params(t=5)
    assert classifier.get_params()['t'] == 5
