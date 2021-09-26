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
                            np.array(ML.generated_y_,
                                     dtype='float32')
                            )


def test_fit_G(X_y_generator, result_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)
    G = knn_c.fit_transform(X_y_generator[0], X_y_generator[1])
    ML = ModularityLabelPropagation()
    ML.fit(G=G)

    np.testing.assert_equal(result_generator,
                            np.array(ML.generated_y_,
                                     dtype='float32')
                            )
