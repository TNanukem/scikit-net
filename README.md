![sknet Logo](https://github.com/TNanukem/sknet/blob/develop/docs/source/_static/full_logo.png "sknet Logo")

![Codecov branch](https://img.shields.io/codecov/c/github/tnanukem/sknet/develop?token=PIQ338YNK1)

The sknet project is a scikit-learn and NetworkX compatible framework for machine learning in complex networks. It provides learning algorithms for complex networks, as well as transforming methods to turn tabular data into complex networks.

It started in 2021 as a project from volunteers to help to improve the development of research on the interface between complex networks and machine learning. It main focus
is to help researchers and students to develop solutions using machine learning on complex networks.

## :computer: Installation

The sknet installation is available via PiPy:

    pip install scikit-net

## :high_brightness: Quickstart

The following code snippet shows how one can transform tabular data into a complex network and then use it to create a classifier:

    from sklearn.model_selection import train_test_split
    from sklean.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.supervised import EaseOfAccessClassifier

    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # The constructor responsible for transforming the tabular data into a complex network
    knn_c = KNNConstructor(k=5)

    classifier = EaseOfAccessClassifier()
    classifier.fit(X_train, y_train, constructor=knn_c)
    y_pred = classifier.predict(X_test)
    accuracy_score(y_test, y_pred)

## :pencil: Documentation

We provide an extensive API documentation as well with some user guides. The documentation is available on https://tnanukem.github.io/scikit-net/
