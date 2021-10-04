Supervised Learning
===================
On a supervised learning setting, the sknet has two main focuses:
- Performing a supervised learning task on a complex network
- Use complex networks to improve the performance of other machine learning algorithms

We will briefly show two examples of supervised learning algorithms available on sknet.

Heuristic of Ease of Access
---------------------------

This is a learning algorithm to be applied on complex networks and consists on verifying
how new samples affects the similarity between nodes when added to the component related
to a given class.

The following code snippet shows how to run it using a tabular dataset:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.supervised import EaseOfAccessClassifier
    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33)
    knn_c = KNNConstructor(k=5)
    classifier = EaseOfAccessClassifier(t=5)
    classifier.fit(X_train, y_train, constructor=knn_c)
    ease = classifier.predict(X_test)

If you want to run it on a Complex Network, then the following snippet shows how to:

.. code-block:: python

    from sknet.supervised import EaseOfAccessClassifier

    classifier = EaseOfAccessClassifier(t=5)
    classifier.fit(G=G)
    ease = classifier.predict(G_test)

High Leval Data Classification
------------------------------

This algorithm leverages both, traditional tabular Machine Learning and Complex
Networks Machine Learning to generate a classifier with better accuracy. In order
to use this method, you must use a tabular dataset with the desired features. 

This algorithm will use the low-level (traditional Machine Learning model) model to
predict the classes probabilities and then will do the same using a Complex Network
method. Then, both of the predictions will be united generating a single probability
prediction.

The following snippet shows how to use it:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.supervised import HighLevelClassifier
    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33)
    knn_c = KNNConstructor(k=5)
    classifier = HighLevelClassifier()
    classifier.fit(X_train, y_train, constructor=knn_c)
    pred = classifier.predict(X_test)