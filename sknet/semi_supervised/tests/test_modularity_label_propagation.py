import pytest
import numpy as np

from sklearn.datasets import load_iris

from sknet.network_construction import KNNConstructor
from sknet.semi_supervised import ModularityLabelPropagation


@pytest.fixture
def X_y_generator():

    X, y = load_iris(return_X_y=True)
    y = np.array(y, dtype='float32')
    y[10:40] = np.nan
    y[60:70] = np.nan
    y[110:140] = np.nan

    return X, y


@pytest.fixture
def result_generator():
    result = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
              1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
              2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 1.,
              2., 2., 2., 1., 2., 2., 1., 1., 2., 2., 2., 2.,
              2., 1., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2.,
              2., 2., 2., 2., 2., 2.]

    return result


def test_fit_y(X_y_generator, result_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)
    ML = ModularityLabelPropagation()
    ML.fit(X_y_generator[0], X_y_generator[1], constructor=knn_c)

    np.testing.assert_equal(result_generator,
                            np.array(ML.get_propagated_labels(),
                                     dtype='float32')
                            )


def test_fit_G(X_y_generator, result_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)
    G = knn_c.fit_transform(X_y_generator[0], X_y_generator[1])
    ML = ModularityLabelPropagation()
    ML.fit(G=G)

    np.testing.assert_equal(result_generator,
                            np.array(ML.get_propagated_labels(),
                                     dtype='float32')
                            )


def test_set_get_params():
    ML = ModularityLabelPropagation()
    ML.set_params(reduction_factor=None)
    assert ML.get_params() == {'reduction_factor': None,
                               'random_state': None}


def test_raise_on_fit_1(X_y_generator):
    ML = ModularityLabelPropagation()
    with pytest.raises(Exception):
        ML.fit(X=X_y_generator[0], y=None, G=None)


def test_raise_on_fit_2(X_y_generator):
    ML = ModularityLabelPropagation()
    with pytest.raises(Exception):
        ML.fit(X=X_y_generator[0], y=X_y_generator[1], G=None,
               constructor=None)


def test_raises_on_aggregation(X_y_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)

    ML = ModularityLabelPropagation(reduction_factor=0.3)
    with pytest.raises(Exception):
        ML.fit(X_y_generator[0], X_y_generator[1], constructor=knn_c)

    ML = ModularityLabelPropagation(reduction_factor=[0.5, 0.2])
    with pytest.raises(Exception):
        ML.fit(X_y_generator[0], X_y_generator[1], constructor=knn_c)

    ML = ModularityLabelPropagation(reduction_factor=[2, 13, 9])
    with pytest.raises(Exception):
        ML.fit(X_y_generator[0], X_y_generator[1], constructor=knn_c)


def test_aggregation(X_y_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)
    ML = ModularityLabelPropagation(reduction_factor=[0.5, 0.5, 0],
                                    random_state=42)
    ML.fit(X_y_generator[0], X_y_generator[1], constructor=knn_c)

    print(ML.get_propagated_labels())
    print(len(ML.get_propagated_labels()))

    expected_result = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0.,  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                       1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 2.,
                       2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2.,
                       2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                       2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]
    np.testing.assert_equal(expected_result,
                            np.array(ML.get_propagated_labels(),
                                     dtype='float32')
                            )
