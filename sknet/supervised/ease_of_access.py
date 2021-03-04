import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import mode
from abc import ABCMeta, abstractmethod
from sklearn.neighbors import DistanceMetric

class EaseOfAccess(metaclass=ABCMeta):
	"""
	Ease of Access method to learn network patterns on data

    Parameters
    ----------
    transformer : BaseConstructor inhrerited class
        A transformer class to transform the tabular data into a
        network
    epsilon : float, default=0.2
    	The perturbance to be applied to the weights matrix after the
    	insertion of the test data
    t : int, deafult=3
    	Number of points on the convergence probabilities vector
    	to classify the test data
    method : str, default='eigenvalue'
    	Which method to use to compute the markov chain limiting
    	probabilties. Options are 'eigenvalue' and 'power'.
    
    Attributes
    ----------
    transformer : BaseConstructor inhrerited class
    	The transformer used to transform the tabular data into network
    epsilon : float
    	The disturbance applied to the weights matrix
    t : int
    	Number of points used on the convergence probabilities
    method : str
    	Method used to compute the limiting probabilities of the Markov chain
    G : NetworkX network
		The network generated from the tabular data
    W : {array-like, pandas dataframe} of shape (n_samples, n_samples)
		The adjacency matrix of the network G
    X : {array-like, pandas dataframe} of shape (n_samples, n_features)
		The used tabular data features
    y : {ndarray, pandas series}, shape (n_samples,) or (n_samples, n_classes)
    	The classes of each node

    Notes
    -----
    Do not use this abstract class, use derived classes instead

    References
    ----------
    Cupertino, T.H., Zhao, L., Carneiro, M.G.: Network-based supervised data classification by 
    using an heuristic of ease of access. Neurocomputing 149(Part A), 86–92 (2015)
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex Networks. 10.1007/978-3-319-17290-3.
	    
	"""
	
	def __init__(self, transformer, epsilon=0.2, t=3, method='eigenvalue'):
		self.transformer = transformer
		self.epsilon = epsilon
		self.t = t
		self.method = method

	def fit(self, X, y, G=None):
		"""
		Fit the model, internalizing the graph that should be used

	    Parameters
	    ----------
	    X : {array-like, pandas dataframe} of shape (n_samples, n_features)
	        The input data.
	    y : {ndarray, pandas series}, shape (n_samples,) or (n_samples, n_classes)
	        The true classes.
	    G : NetworkX Graph, default=None
	    	If the graph was already generated, then this parameter will make as so the
	    	transformer is not called

	    Notes
	    -----
	    Even though the G can be passed directly, X is required so the distance between
	    test classes and the other nodes on the graph can be calculated
	    
	    """

		if G is none:
			# Generates the graph from X and y
			self.G = self.transformer.fit_transform(X, y)
		else:
			self.G = G

		# Transforms X into undirected
		if nx.is_directed(self.G):
			self.G = self.G.to_undirected()

		# Generates W matrix
		self.W = nx.to_numpy_matrix(self.G)

		self.X = X
		self.y = y

	def predict(self, X):
		"""
		Predicts the labels of the test samples from X

	    Parameters
	    ----------
	    X : {array-like, pandas dataframe} of shape (n_samples, n_features)
	        The test data to be predicted.

	    Returns
	    -------
	    predictions : array-like of shape (n_samples)
	    	The predicted label for each sample
	    """
		predictions = []
		dist = self._get_distance_metric()
		
		for x in X:

			# For each instance, calculates the s vector
			s = [dist.pairwise((x, x_t))[1][0] for x_t in self.X]
			L = len(s)

			# Generates the S_tilda vector
			S_tilda = np.zeros((L, L))

			for i in range(L):
				S_tilda[i] = [s[i]] * L

			# Alter the weight matrix
			w_tilda = self.W + self.epsilon * S_tilda

			# Generates probability matrix
			norm_factor = np.sum(w_tilda, axis=1)
			P = [x / norm_factor[i] for i, x in enumerate(w_tilda)]
			P = np.array(P).reshape(w_tilda.shape)

			# Computes the convergence
			P_inf = self._stationary_distribution(P, self.method)

			# Associates each class with the probabilities
			res = pd.DataFrame()
			res['prob'] = P_inf
			res['y'] = self.y
			res.sort_values('prob', inplace=True, ascending=False)

			# Gets the t classes from P_inf and set to the majority
			tau = res.iloc[:self.t]

			predictions.append(self._aggregation_method(tau))

		return predictions

	@abstractmethod
	def _aggregation_method(self, tau):
		pass

	def _get_distance_metric(self):
		metric = self.transformer.metric

		if type(metric) is str:
			return DistanceMetric.get_metric(metric)

		return metric

	def _stationary_distribution(self, W, method):

		if method == 'power':
			
			return np.linalg.matrix_power(np.array(W)(np.array(W), 50))[0]

		elif method == 'eigenvalue':
			evals, evecs = np.linalg.eig(np.array(W).T)
			evec1 = evecs[:,np.isclose(evals, 1)]

			evec1 = evec1[:,0]

			stationary = evec1 / evec1.sum()

			return stationary

		else:
			raise Exception("{} is not an available method to calculate the markov chain convergence. Available methods are 'power' and 'eigenvalue'".format(method))

class EaseOfAccessClassifier(EaseOfAccess):
	"""
	Ease of Access Classifier

	Classifier that uses the heuristic of ease of access to classify
	new instances inside a network

    Parameters
    ----------
    transformer : BaseConstructor inhrerited class
        A transformer class to transform the tabular data into a
        network
    epsilon : float, default=0.2
    	The perturbance to be applied to the weights matrix after the
    	insertion of the test data
    t : int, deafult=3
    	Number of points on the convergence probabilities vector
    	to classify the test data
    method : str, default='eigenvalue'
    	Which method to use to compute the markov chain limiting
    	probabilties. Options are 'eigenvalue' and 'power'.
    
    Attributes
    ----------
    transformer : BaseConstructor inhrerited class
    	The transformer used to transform the tabular data into network
    epsilon : float
    	The disturbance applied to the weights matrix
    t : int
    	Number of points used on the convergence probabilities
    method : str
    	Method used to compute the limiting probabilities of the Markov chain
    G : NetworkX network
		The network generated from the tabular data
    W : {array-like, pandas dataframe} of shape (n_samples, n_samples)
		The adjacency matrix of the network G
    X : {array-like, pandas dataframe} of shape (n_samples, n_features)
		The used tabular data features
    y : {ndarray, pandas series}, shape (n_samples,) or (n_samples, n_classes)
    	The classes of each node
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from dataset_constructors import KNNConstructor
    >>> from ease_of_access import EaseOfAccessClassifier
	>>> X, y = load_iris(return_X_y = True)
	>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	>>> knn_c = KNNConstructor(k=5)
	>>> classifier = EaseOfAccessClassifier(knn_c, t=5)
	>>> classifier.fit(X_train, y_train)
	>>> ease = classifier.predict(X_test)
	>>> accuracy_score(y_test, ease)
	0.92

    Notes
    -----

    References
    ----------
    Cupertino, T.H., Zhao, L., Carneiro, M.G.: Network-based supervised data classification by 
    using an heuristic of ease of access. Neurocomputing 149(Part A), 86–92 (2015)
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex Networks. 10.1007/978-3-319-17290-3.
	    
	"""

	def __init__(self, transformer, epsilon=0.2, t=3, method='eigenvalue'):
		super().__init__(transformer, epsilon, t, method)

	def _aggregation_method(self, tau):
		return mode(tau['y'])[0][0]

class EaseOfAccessRegressor(EaseOfAccess):
		"""
	Ease of Access Regressor

	Regressor that uses the heuristic of ease of access to classify
	the real-value of the target of new instances inside a network

    Parameters
    ----------
    transformer : BaseConstructor inhrerited class
        A transformer class to transform the tabular data into a
        network
    epsilon : float, default=0.2
    	The perturbance to be applied to the weights matrix after the
    	insertion of the test data
    t : int, deafult=3
    	Number of points on the convergence probabilities vector
    	to classify the test data
    method : str, default='eigenvalue'
    	Which method to use to compute the markov chain limiting
    	probabilties. Options are 'eigenvalue' and 'power'.
    
    Attributes
    ----------
    transformer : BaseConstructor inhrerited class
    	The transformer used to transform the tabular data into network
    epsilon : float
    	The disturbance applied to the weights matrix
    t : int
    	Number of points used on the convergence probabilities
    method : str
    	Method used to compute the limiting probabilities of the Markov chain
    G : NetworkX network
		The network generated from the tabular data
    W : {array-like, pandas dataframe} of shape (n_samples, n_samples)
		The adjacency matrix of the network G
    X : {array-like, pandas dataframe} of shape (n_samples, n_features)
		The used tabular data features
    y : {ndarray, pandas series}, shape (n_samples,) or (n_samples, n_classes)
    	The classes of each node
    
    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from dataset_constructors import KNNConstructor
    >>> from ease_of_access import EaseOfAccessRegressor
	>>> X, y = load_boston(return_X_y = True)
	>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	>>> knn_c = KNNConstructor(k=5)
	>>> reg = EaseOfAccessRegressor(knn_c, t=5)
	>>> reg.fit(X_train, y_train)
	>>> ease = reg.predict(X_test)

    Notes
    -----

    References
    ----------
    Cupertino, T.H., Zhao, L., Carneiro, M.G.: Network-based supervised data classification by 
    using an heuristic of ease of access. Neurocomputing 149(Part A), 86–92 (2015)
    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex Networks. 10.1007/978-3-319-17290-3.
	    
	"""

	def __init__(self, transformer, epsilon=0.2, t=3, method='eigenvalue'):
		super().__init__(transformer, epsilon, t, method)

	def _aggregation_method(self, tau):
		return tau['y'].mean()
