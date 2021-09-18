import numpy as np
import networkx as nx


class StochasticParticleCompetition():
    """
    Non supervised method that uses a stochastic particle competition to
    group the data into K clusters.

    This class still has major performance issues, taking too long to
    converge. Further optimization shall happen, be advised when using

    Parameters
    ----------
    constructor : BaseConstructor inhrerited class, optional(default=None)
        A constructor class to transform the tabular data into a
        network. It can be set to None if a complex network is directly
        passed to the ``fit`` method. Notice that you should use 'sep_com' as
        False on the constructor.
    K : int, optional(default=3)
        The number of particles to compete which will be the number of
        resulting clusters
    lambda_ : float, optional(default=0.5)
        The probability of a particle choosing the preferential movement
        (exploitation) against the random movement (exploration)
    delta : int, optional(default=10)
        The amount of energy gained at each step for each particle
    omega_max : float, optional(default=10)
        The maximum amount of energy that a particle can have at any given time
    omega_min : float, optional(default=1)
        The minimum amount of energy before a particle is exhausted
    epsilon : float, optional(default=0.01)
        The minimum difference between the dominance matrix variation
        before finishing the competition.
    n_iters : int, optional(default=500)
        The maximum number of steps before finishing the competition.
        The process will stop when either the convergence happens given epsilon
        or the maximum number of steps is reached

    Attributes
    ----------
    clusters : {ndarray, pandas series}, shape (n_samples, 1)
        The cluster of each sample

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sknet.network_construction import KNNConstructor
    >>> from sknet.unsupervised import StochasticParticleCompetition
    >>> X, y = load_iris(return_X_y = True)
    >>> knn_c = KNNConstructor(k=5, sep_comp=False)
    >>> SCP = StochasticParticleCompetition(knn_c)
    >>> SCP.fit(X, y)
    >>> SCP.clusters
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
       1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
       2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

    References
    ----------
    T. C. Silva and L. Zhao, "Stochastic Competitive Learning in Complex
    Networks," in IEEE Transactions on Neural Networks and Learning
    Systems, vol. 23, no. 3, pp. 385-398, March 2012,
    doi: 10.1109/TNNLS.2011.2181866.

    Silva, Thiago & Zhao, Liang. (2016). Machine Learning in Complex
    Networks. 10.1007/978-3-319-17290-3.

    """

    def __init__(self, constructor=None, K=3, lambda_=0.5, delta=0.1,
                 omega_max=10, omega_min=1, epsilon=0.01, n_iters=500,
                 random_state=None):
        self.constructor = constructor
        self.K = K
        self.lambda_ = lambda_
        self.delta = delta
        self.epsilon = epsilon
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.n_iters = n_iters
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X=None, y=None, G=None):
        """Fit the algorithms by using the particle competition
        to cluster the data points

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape
            (n_samples, n_features), optional (default=None)
            The input data samples. Can be None if G is set.
        y : {ndarray, pandas series}, shape (n_samples,) or
            (n_samples, n_classes), optional (default=None)
            The target classes. Ignored for this class, used only
            to keep API consistency
        G : NetworkX Network, optional (default=None)
            The network to have its communities detected. Can be
            None if X is not None in which case the constructor
            will be used to generate the network.

        """

        if X is None and G is None:
            raise Exception('X or G must be defined')

        if X is None and G is not None:
            self.G = G
        else:
            self.G = self.constructor.fit_transform(X, y)

        A = nx.to_numpy_array(self.G)
        self.V = A.shape[0]

        P_pref = np.zeros((self.V, self.V, self.K))

        P_rean = np.zeros((self.V, self.V, self.K))

        P_rand = self._create_p_rand(A)

        # Set the initial random position of the particles
        node_list = np.array(list(self.G))
        p = np.random.choice(node_list, self.K, False)

        # Calculate initial N
        N = self._calculate_initial_N(p)

        N_bar = self._calculate_initial_N_bar(N)

        # Calculate initial E
        initial_energy = self.omega_min + (
            (self.omega_max - self.omega_min) / self.K
        )
        E = np.array([initial_energy] * self.K)

        # Calculate initial S
        S = np.zeros(self.K)

        P_tran = np.zeros((self.V, self.V, self.K))

        convergence = False
        t = 0
        while not convergence and t < self.n_iters:

            # Updates the movement matrices
            P_pref = self._calculate_P_pref(A, N_bar)

            P_rean = self._calculate_P_rean(N_bar)

            P_tran = self._calculate_P_tran(P_rand,
                                            P_pref, P_rean, S, -1)

            p = self._choose_next_vertices(P_tran, p)

            N = self._update_N(p, N)
            old_N_Bar = N_bar.copy()
            N_bar = self._update_N_bar(N)
            E = self._update_E(E, N_bar, p)
            S = self._update_S(E)

            # Update time and verify convergence
            t += 1
            convergence = self._verify_convergence(N_bar, old_N_Bar)

        self.clusters = np.argmax(N_bar, axis=1)

    def predict(self, X=None, G=None):
        """
        Returns the clusters after the model was fitted.

        Parameters
        ----------

        X : {array-like, pandas dataframe} of shape
            (n_samples, n_features), optional (default=None)
            Ignored on this method
        G : NetworkX Network, optional (default=None)
            Ignored on this method
        """
        return self.clusters

    def fit_predict(self, X=None, y=None, G=None):
        """Fit the algorithms by using the particle competition
        to cluster the data points

        Parameters
        ----------
        X : {array-like, pandas dataframe} of shape
            (n_samples, n_features), optional (default=None)
            The input data samples. Can be None if G is set.
        y : {ndarray, pandas series}, shape (n_samples,) or
            (n_samples, n_classes), optional (default=None)
            The target classes. Ignored for this class, used only
            to keep API consistency
        G : NetworkX Network, optional (default=None)
            The network to have its communities detected. Can be
            None if X is not None in which case the constructor
            will be used to generate the network.

        Returns
        -------
        clusters : {array-like} of shape (n_samples)
                   The cluster of each data point

        """
        self.fit(X, y, G)
        return self.predict()

    def _verify_convergence(self, N_bar, old_N_bar):
        diff = np.sum(np.abs(N_bar - old_N_bar))
        print(f'Convergence: {diff}')
        return diff < self.epsilon

    def _create_p_rand(self, A):
        P_rand = A / A.sum(axis=1, keepdims=True)
        return P_rand

    def _calculate_initial_N(self, p):
        N = np.ones((self.V, self.K))
        for k, i in enumerate(p):
            N[int(i)][k] = 2
        return N

    def _calculate_initial_N_bar(self, N):
        N_bar = N/N.sum(axis=1, keepdims=True)
        return N_bar

    def _calculate_P_pref(self, A, N_bar):
        aux = np.zeros((self.V, self.V, self.K))

        num = [[[A[i, j] * N_bar[j, k] for k in range(self.K)
                 ] for j in range(self.V)] for i in range(self.V)]
        den = [[[np.sum([
            A[i, l_]*N_bar[l_, k] for l_ in range(self.V)
            ]) for k in range(self.K)] for j in range(self.V)
            ] for i in range(self.V)]
        aux[:, :, :] = np.divide(np.array(num), np.array(den))

        return aux

    def _calculate_P_rean(self, N_bar):
        aux = np.zeros((self.V, self.V, self.K))

        den = [np.sum(
                [np.argmax(N_bar[u, :]) == k for u in range(self.V)]
            ) for k in range(self.K)]

        num = [
            [np.sum(np.argmax(N_bar[j, :]) == k) for j in range(self.V)
             ] for k in range(self.K)
        ]
        aux[:, :, :] = [np.array(num)/np.array(den) for i in range(self.V)]

        return aux

    def _calculate_P_tran(self, P_rand, P_pref, P_rean, S, t):
        aux = np.zeros((self.V, self.V, self.K))
        for k in range(self.K):

            non_exhausted = (
                1 - S[k]) * (
                    self.lambda_ * P_pref[:, :, k] + (
                        1 - self.lambda_) * P_rand
                    )

            exhausted = S[k] * P_rean[:, :, k]
            aux[:, :, k] = non_exhausted + exhausted

        return aux

    def _choose_next_vertices(self, P_tran, p):
        aux = np.zeros(self.K)
        for k in range(self.K):
            aux[k] = np.random.choice(
                [i for i in range(self.V)],
                p=P_tran[int(p[k]), :, k]
            )

        return aux

    def _update_N(self, p, N):
        aux = N.copy()
        for k, i in enumerate(p):
            aux[int(i), k] = aux[int(i), k] + 1

        return aux

    def _update_N_bar(self, N):
        N_bar_updated = self._calculate_initial_N_bar(N)

        return N_bar_updated

    def _update_E(self, E, N_bar, p):
        aux = np.zeros(self.K)
        for k in range(self.K):
            if self._is_owner(k, p, N_bar):
                aux[k] = min(E[k] + self.delta, self.omega_max)
            else:
                aux[k] = max(E[k] - self.delta, self.omega_min)

        return aux

    def _update_S(self, E):
        aux = np.zeros(self.K)
        for k in range(self.K):
            if E[k] == self.omega_min:
                aux[k] = 1
            else:
                aux[k] = 0

        return aux

    def _is_owner(self, k, p, N_bar):
        if np.argmax(N_bar[int(p[k]), :]) == k:
            return True
        else:
            return False
