---
title: 'sknet: A Python framework for Machine Learning in Complex Networks'
tags:
  - Python
  - complex networks
  - machine learning
  - graph learning
  - graphs
authors:
  - name: Tiago Toledo Jr
    orcid: 0000-0001-5675-554X
    affiliation: "1, 2"
affiliations:
 - name: Instituto de Ciências Matemáticas e Computação, Universidade de São Paulo
   index: 1
 - name: Big Data
   index: 2
date: x October 2021
bibliography: paper.bib

---

# Summary

Recent advances in Machine Learning, an area that leverages data to identify patterns,
and in Complex Networks, an area which leverages connections between entities to identify
complex phenomena and can be considered as an extension to the graph theory, have been
changing the society. Both of those areas can be related to the task of 'learning' from data [@book].

It seems, however to be a gap on the interface between those two research areas.
It was already shown that one can leverage both, using complex networks to improve
machine learning methods, and using machine learning to exploit the information on
complex networks to achieve better results.
However, little to no implementation of methods that can be used on both areas has been
open-sourced. And for those who did, it was not done in any unified way.
`sknet` comes a library to solve this gap.

# Statement of need

`sknet` is a sklearn [@sklearn_api] and NetworkX [@SciPyProceedings_11] compatible Python package for machine learning
in complex networks. 

`sknet` was designed to be used by both researchers and by students in courses
on Machine Learning or Complex Networks. As far as the author knows, no unified
package was developed focusing on Machine Learning on Complex Networks, altough
NetworkX presents some algorithms that could be considered Machine Learning methods.

# Library overview

The `sknet` tries to maintain, as much as possible, the known API structure from
the `scikit-learn`. It main focus is in transforming data from of kind of representation
to the other and allowing combined methods from the Complex Networks and Machine Learning
areas to allow the users to find patterns on their data.

![`sknet` packages structure.\label{fig:packages}](sknet_packages.png)

The main structure of the library is represented on \autoref{fig:packages} and is as follows:

- A `constructor` package responsible for transforming data from different types, such
as transforming tabular data or time series data into complex networks representations.  

- A `supervised` package responsible for supervised learning tasks where one has labeled data.  

- A `unsupervised` package responsible for unsupervised learning tasks where one does not have labeled data.  

- A `semi-supervised` package responsible for semi-supervised learning tasks where one have a small set of labeled data.  

- A `utils` package with some auxiliar functions for the other packages.

As of the version 0.1.0, the following algorithms are implemented:

- Stochastic Particle Competition (Unsupervised) by @Unsupervised.  

- Heuristic of Ease of Access (Supervised) by @Heuristic.  

- High Level Data Classification (Supervised) by @HighLevel.  

- Modularity Label Propagation (Semi-Supervised) by @SemiSupervised.  


 The library was implemented with extensive API documentation and with an user-guide
 that aims to be a basic introduction to people learning more about the area.

## Usage Example

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.supervised import EaseOfAccessClassifier

    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # The constructor responsible for transforming the
    # tabular data into a complex network
    knn_c = KNNConstructor(k=5)

    classifier = EaseOfAccessClassifier()
    classifier.fit(X_train, y_train, constructor=knn_c)
    y_pred = classifier.predict(X_test)
    accuracy_score(y_test, y_pred)

# References