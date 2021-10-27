import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from sknet.network_construction import KNNConstructor
from sknet.supervised import EaseOfAccessClassifier, EaseOfAccessRegressor


@pytest.fixture
def X_y_generator_classification():

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def X_y_generator_regression():

    X, y = load_boston(return_X_y=True)
    X = X[:150, :]
    y = y[:150]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def module_generator_eigen_classifier(X_y_generator_classification):
    knn = KNNConstructor(k=3)
    classifier = EaseOfAccessClassifier(t=5)
    classifier.fit(X_y_generator_classification[0],
                   X_y_generator_classification[1], constructor=knn)

    return classifier


@pytest.fixture
def module_generator_power_classifier(X_y_generator_classification):
    knn = KNNConstructor(k=3)
    classifier = EaseOfAccessClassifier(t=5, method='power')
    classifier.fit(X_y_generator_classification[0],
                   X_y_generator_classification[1], constructor=knn)

    return classifier


@pytest.fixture
def module_generator_eigen_regressor(X_y_generator_regression):
    knn = KNNConstructor(k=3)
    regressor = EaseOfAccessRegressor(t=5)
    regressor.fit(X_y_generator_regression[0],
                  X_y_generator_regression[1], constructor=knn)

    return regressor


@pytest.fixture
def module_generator_power_regressor(X_y_generator_regression):
    knn = KNNConstructor(k=3)
    regressor = EaseOfAccessRegressor(t=5, method='power')
    regressor.fit(X_y_generator_regression[0],
                  X_y_generator_regression[1], constructor=knn)

    return regressor


@pytest.fixture
def class_generator_classifier(module_generator_eigen_classifier,
                               module_generator_power_classifier,
                               X_y_generator_classification):

    pred_eig = module_generator_eigen_classifier.predict(
        X_y_generator_classification[2]
    )
    pred_power = module_generator_power_classifier.predict(
        X_y_generator_classification[2]
    )

    return (module_generator_eigen_classifier,
            module_generator_power_classifier, pred_eig, pred_power)


def test__stationary_distribution_classifier(class_generator_classifier):
    np.testing.assert_almost_equal(class_generator_classifier[0].P_inf,
                                   class_generator_classifier[1].P_inf)

    pd.testing.assert_frame_equal(class_generator_classifier[0].tau_,
                                  class_generator_classifier[1].tau_)


@pytest.fixture
def class_generator_regressor(module_generator_eigen_regressor,
                              module_generator_power_regressor,
                              X_y_generator_regression):

    pred_eig = module_generator_eigen_regressor.predict(
        X_y_generator_regression[2]
    )
    pred_power = module_generator_power_regressor.predict(
        X_y_generator_regression[2]
    )

    return (module_generator_eigen_regressor,
            module_generator_power_regressor, pred_eig, pred_power)


def test__stationary_distribution_regressor(class_generator_regressor):
    np.testing.assert_almost_equal(class_generator_regressor[0].P_inf,
                                   class_generator_regressor[1].P_inf)

    pd.testing.assert_frame_equal(class_generator_regressor[0].tau_,
                                  class_generator_regressor[1].tau_)


def test_predictions_classifier(class_generator_classifier):

    expected = [1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0, 1,
                2, 1, 1, 2, 0, 1, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 2, 2,
                1, 2]

    eigen_pred = class_generator_classifier[2]
    power_pred = class_generator_classifier[3]

    assert eigen_pred == power_pred == expected


def test_predictions_regressor(class_generator_regressor):

    expected = [21.28, 18.0, 17.26, 20.74, 17.2, 15.62, 20.98,
                17.759999999999998, 19.52, 23.479999999999997,
                20.080000000000002, 19.64, 18.76, 15.62, 18.22,
                22.779999999999998, 18.560000000000002, 19.52,
                21.619999999999997, 17.759999999999998, 15.440000000000001,
                16.04, 15.440000000000001, 16.66, 16.860000000000003, 18.0,
                16.259999999999998, 17.619999999999997, 25.080000000000002,
                15.5, 14.62, 18.6, 21.660000000000004, 17.619999999999997,
                20.1, 18.0, 19.860000000000003, 17.5, 21.24, 17.5,
                17.619999999999997, 25.22, 25.340000000000003,
                19.639999999999997, 14.66, 19.619999999999997,
                16.9, 16.04, 22.52, 17.6]

    eigen_pred = class_generator_regressor[2]
    power_pred = class_generator_regressor[3]

    assert eigen_pred == power_pred == expected


def test_raise_on_predict(X_y_generator_classification):

    knn = KNNConstructor(k=3)
    classifier = EaseOfAccessClassifier(t=5, method='something')

    with pytest.raises(Exception):
        classifier.fit(X_y_generator_classification[0],
                       X_y_generator_classification[1], constructor=knn)
        classifier.predict(X_y_generator_classification[2])


def test_set_get_params(module_generator_eigen_classifier):
    classifier = module_generator_eigen_classifier
    classifier.set_params(t=5)
    assert classifier.get_params()['t'] == 5
