Transforming Data
=================

The sknet provides classes to allow data transformation between different kinds. Since
the implemented algorithms may require an specific data type to work, those classes
allow the user to freely transform data and use any of the methods.

So far, the following transformations are available:

- Tabular data -> Complex networks
- Time series tabular data -> Complex networks

Below there is an example of how one can use one of the tabular datasets constructor
to turn tabular data into a complex network.

.. code-block:: python

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNEpislonRadiusConstructor
    X, y = load_iris(return_X_y = True)
    ke_c = KNNEpislonRadiusConstructor(k=3, epsilon=0.3)
    ke_c.fit(X, y)
    G = ke_c.transform()

And below an example of how one can use one of the time series constructor to turn a
time series into a complex network:

.. code-block:: python

    from sknet.network_construction import UnivariateCorrelationConstructor
    r = 0.5
    L = 10
    constructor = UnivariateCorrelationConstructor(r, L)
    constructor.fit(X)
    G = constructor.transform()