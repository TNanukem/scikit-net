from sklearn.datasets import load_iris
from sknet.network_construction import KNNConstructor
from sknet.unsupervised import StochasticParticleCompetition
X, y = load_iris(return_X_y = True)

knn_c = KNNConstructor(k=5, sep_comp=False)
sc = StochasticParticleCompetition(knn_c, 3, 0.5, 0.1, 0.01, 10, 1)
sc.fit(X, y)