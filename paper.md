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
complex phenomena, have been changing the society. Both of those areas can be related
to the task of 'learning' from data (ML in CN).

It seems, however to be a gap on the interface between those two research areas.
It was already shown that one can leverage both, using complex networks to improve
machine learning methods and using machine learning to exploit the information on
complex networks to achieve better results.
However, little to no implementation of methods that can be used on both areas has been
open-sourced. `sknet` comes a library to solve this gap.

# Statement of need

`sknet` is sklearn () and NetworkX () compatible Python package for machine learning
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

![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

The main structure of the library is as follows:
- A `constructor` package responsible for transforming data from different types, such
as transforming tabular data or time series data into complex networks representations.
- A `supervised` package,
- A `unsupervised` package,
- A `semi-supervised` package,

As of the version 0.1.0, the following algorithms are implemented:
- Stochastic Particle Competition ()
- Heuristic of Ease of Access ()
- High Level Data Classification ()
- Modularity Label Propagation ()

[Also, some auxiliar methods have been implemented on the `utils` package such as the
 T () and HT () centrality measures that are not available on the NetworkX package.]

## Usage Example

    from sklearn.datasets import load_iris
    from sknet.network_construction import KNNConstructor
    from sknet.supervised import EaseOfAccessClassifier

    X, y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    knn_c = KNNConstructor(k=5)

    classifier = EaseOfAccessClassifier()
    classifier.fit(X_train, y_train, constructor=knn_c)
    ease = classifier.predict(X_test)
    accuracy_score(y_test, ease)

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# References