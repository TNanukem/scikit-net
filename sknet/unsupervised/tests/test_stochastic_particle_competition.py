import pytest
import numpy as np

from sklearn.datasets import load_iris

from sknet.network_construction import KNNConstructor
from sknet.unsupervised import StochasticParticleCompetition


@pytest.fixture
def X_y_generator():

    X, y = load_iris(return_X_y=True)
    y = np.array(y, dtype='float32')

    return X, y


@pytest.fixture
def result_generator():
    result = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return result


def test_fit_X(X_y_generator, result_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)
    SPC = StochasticParticleCompetition(random_state=42, n_iter=3)
    SPC.fit(X_y_generator[0], constructor=knn_c)

    np.testing.assert_equal(result_generator,
                            np.array(SPC.clusters_,
                                     dtype='float32')
                            )


def test_fit_G(X_y_generator, result_generator):
    knn_c = KNNConstructor(k=5, sep_comp=False)
    G = knn_c.fit_transform(X_y_generator[0], X_y_generator[1])
    SPC = StochasticParticleCompetition(random_state=42, n_iter=3)
    SPC.fit(G=G)

    np.testing.assert_equal(result_generator,
                            np.array(SPC.clusters_,
                                     dtype='float32')
                            )
