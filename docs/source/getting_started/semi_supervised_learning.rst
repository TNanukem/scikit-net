Semi Supervised Learning
========================

The semi-supervised algorithms try to leverage great amounts of unlabeled data
with a smaller amount of labeled data. One can use a semi-supervised algorithm
available on the sknet as follows.

.. code-block:: python

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.semi_supervised import ModularityLabelPropagation
    X, y = load_iris(return_X_y = True)
    knn_c = KNNConstructor(k=5, sep_comp=False)
    y[10:20] = np.nan
    y[70:80] = np.nan
    y[110:120] = np.nan
    propagator = ModularityLabelPropagation()
    propagator.fit(X, y, constructor=knn_c)
    propagator.generated_y
