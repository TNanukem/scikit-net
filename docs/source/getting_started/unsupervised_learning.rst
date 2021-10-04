Unsupervised Learning
=====================

The unsupervised learning methods, when applied to complex networks, are usually
called community dectection methods. Their focus is to find groups of nodes where
the number of edges intra-community is way greater than the number of edges extra-community.

The following code snippet shows how one can use one of the unsupervised methods of the sknet
to clusterize some dataset:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.unsupervised import StochasticParticleCompetition
    X, y = load_iris(return_X_y = True)
    knn_c = KNNConstructor(k=5, sep_comp=False)
    SCP = StochasticParticleCompetition()
    SCP.fit(X, y, constructor=knn_c)
    SCP.clusters_
