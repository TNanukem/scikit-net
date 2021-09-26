.. sknet documentation master file, created by
   sphinx-quickstart on Fri Mar  5 05:44:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

User Guide
**********

This section will introduce the main modules of the sknet and show some examples as well as explaining the theory
behind the implemented algorithms.

The sknet main structure divide the classes into two main types: auxiliar methods such as utilities and transformations and
machine learning methods which are divided into supervised, unsupervised and semi supervised methods.

Most of the Machine Learning methods can work both with tabular data (in form of a Pandas Dataframe or a Numpy Array) and with graph data
(in form of a NetworkX complex network), exceptions will be explicit on the documentation.

Transformation methods
======================

These are the backbones of the inner workings of the sknet. The transformation classes are responsible for transforming data from one
type to another. To this date, the following transformations are possible:

- Tabular Data -> Complex Network
- Time Series -> Complex Network

The Machine Learning classes are responsible for transforming data to the appropriate format for each one, however, one can always
insert the already transformed data into the class.

Dataset Constructors
--------------------

Those are the methods responsible for transforming tabular data, from the Pandas DataFrame or the Numpy Array format into a
NetworkX complex network.

When dealing with Dataset Constructors, one may have the classes of the tabular data availabe (such as on a supervised method),
on that case, one may set the constructor so it will generate separated components for each class. Some Machine Learning models
will require this while others will require that no separated component is generated. Look up for the documentation of each method
to be aware of the requirements for each method.

KNN Constructor
^^^^^^^^^^^^^^^

The KNN Constructor uses a k-Nearest Neighbors algorithm to create edges between the instances (rows) of our tabular dataset. For that
the distance between each instance of the dataset is calculated using some distance metric, like the Euclidean Distance, and then, for each
instance, the k closest instances are selected and edges are created between them.

Notice that this methodology does not create a symmetric network since, given node ``i``, node ``j`` could be one of the k closest points to it but
the contrary may not be true.

.. image:: images/knn.png
   :alt: KNN Constructor

Also, this method does not allow for singletons to be created. If a node is too far away from the others on the generated space, it will
create at least k edges with k other nodes.

The main drawback of this methodology is that, for dense regions where there are too many nodes close to each other, the degree of each node
will be underestimated.

Epsilon-Radius Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Epsilon-Radius constructor, for each node (row) of the dataset, connects it to all nodes that are inside a circle of radius epsilon.  For that
the distance between each instance of the dataset is calculated using some distance metric, like the Euclidean Distance.

.. image:: images/epsilon.png
   :alt: Epsilon Radius Constructor

This methodology will create a symmetric network since that, for a node to be inside the radius of another, the contrary must also be true at all times. However
this method allows singletons to be created since it may be that there are no nodes inside the radius, which is a big drawback of this method.

KNN Epsilon-Radius Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To overcome the drawbacks of both, the KNN Constructor and the Epsilon-Radius constructor, the KNN Epsilon-Radius Constructor tries to sum-up the strenghts
of both methods. This constructor will use the Epsilon-Radius method for dense regions of the space and the K-NN method for sparse regions according to the
following equation:

.. math::

   \left\{\begin{matrix}
      \epsilon\text{-radius}(v_i), & \text{if} |\epsilon\text{-radius}| > k \\ 
      k\text{-NN}(v_i), & \text{otherwise} 
   \end{matrix}\right.

The idea behind this strategy is to add more edges on dense regions that should be more densely connected and to avoid singletons being created on sparse
regions. This way, the generated network will be connected and will have a variable degree level, better representing real world networks.

.. image:: images/k-eps.png
   :alt: KNN Epsilon-Radius Constructor

Time Series Constructors
------------------------

Univariate Correlation Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multivariate Correlation Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supervised Methods
==================

Supervised methods have one objective: given a labeled dataset, learn the data patterns to the able to predict the label (continous or discrete)
of new, unseen, data samples.

Heuristic of Ease of Access
---------------------------

This algorithm...

High Level Data Classification
------------------------------

Unsupervised Methods
====================

Unsupervised methods, usually called community detection methods on the Complex Network area, are algorithms that try to find patterns on
the data so to group up data samples.

Stochastic Particle Competition
-------------------------------

Semi Supervised Methods
=======================

These are methods designed to work with large amounts of unlabeled data given a small amount of labeled data. Usually this kind of method
works towards spreading labels from labeled examples to unlabeled examples.

Modularity Label Propagation
----------------------------
