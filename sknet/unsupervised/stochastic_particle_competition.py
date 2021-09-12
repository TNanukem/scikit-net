import numpy as np
import networkx as nx


class StochasticParticleCompetition():

    def __init__(self, constructor, K, lambda_, delta,
                 epsilon, omega_max, omega_min):
        self.constructor = constructor
        self.K = K
        self.lambda_ = lambda_
        self.delta = delta
        self.epsilon = epsilon
        self.omega_max = omega_max
        self.omega_min = omega_min

    def fit(self, X=None, y=None, G=None):

        if X is None and G is None:
            raise Exception('X or G must be defined')

        if X is None and G is not None:
            self.G = G
        else:
            self.G = self.constructor.fit_transform(X, y)

        A = nx.to_numpy_array(self.G)
        self.V = A.shape[0]

        P_pref = np.zeros((self.V, self.V, self.K))
        P_pref = P_pref[:, :, :, None]
        print(f'P_pref shape: {P_pref.shape}')

        P_rean = np.zeros((self.V, self.V, self.K))
        P_rean = P_rean[:, :, :, None]
        print(f'P_rean shape: {P_rean.shape}')

        P_rand = self._create_p_rand(A)
        print(f'P_rand shape: {P_rand.shape}')
        print(P_rand)
        p = np.zeros((1, self.K))

        # Set the initial random position of the particles
        node_list = np.array(list(self.G))
        p[0] = np.random.choice(node_list, self.K, False)

        # Calculate initial N
        print('p: ', p)
        N = self._calculate_initial_N(p[0])
        N = N[:, :, None]
        print(f'N Shape: {N.shape}')

        N_bar = self._calculate_initial_N_bar(N[:, :, 0])
        N_bar = N_bar[:, :, None]
        print(f'N_bar Shape: {N_bar.shape}')

        # Calculate initial E
        initial_energy = self.omega_min + (
            (self.omega_max - self.omega_min) / self.K
        )
        E = np.array([initial_energy] * self.K)
        E = E[:, None]
        print('E shape: ', E.shape)
        print('E: ', E)

        # Calculate initial S
        S = np.zeros(self.K)
        S = S[:, None]
        print('S shape: ', S.shape)
        print('S: ', S)

        P_tran = np.zeros((self.V, self.V, self.K))
        P_tran = P_tran[:, :, :, None]
        print('P_tran shape: ', P_tran.shape)

        convergence = False
        t = 0

        print('Getting into Loop')
        while not convergence:

            # Updates the movement matrices
            P_pref = self._calculate_P_pref(A, N_bar[:, :, -1], P_pref)
            P_rean = self._calculate_P_rean(P_rean, N_bar[:, :, -1])
            P_tran = self._calculate_P_tran(P_tran, P_rand,
                                            P_pref, P_rean, S, -1)
            p = self._choose_next_vertices(P_tran[:, :, :, -1], p)

            N = self._update_N(p[-1], N)
            N_bar = self._update_N_bar(N_bar, N[:, :, -1])
            E = self._update_E(E, N_bar, p)
            S = self._update_S(S, E[:, -1])

            # Update time and verify convergence
            t += 1
            convergence = self._verify_convergence(N_bar)

        return N_bar

    def _verify_convergence(self, N_bar):
        diff = np.sum(np.abs(N_bar[:, :, -1] - N_bar[:, :, -2]))
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

    def _calculate_P_pref(self, A, N_bar, P_pref):
        #  This first implementation is not optimized
        #  We should transform it into a list comprehension later
        aux = np.zeros((self.V, self.V, self.K))
        for i in range(self.V):
            for j in range(self.V):
                for k in range(self.K):
                    num = A[i, j] * N_bar[j, k]

                    den = np.sum(
                        [A[i, l_]*N_bar[l_, k] for l_ in range(self.V)]
                    )

                    aux[i, j, k] = num / den

        aux = aux[:, :, :, None]

        P_pref = np.append(P_pref, aux, axis=-1)

        return P_pref

    def _calculate_P_rean(self, P_rean, N_bar):
        #  This first implementation is not optimized
        #  We should transform it into a list comprehension later
        aux = np.zeros((self.V, self.V, self.K))
        for k in range(self.K):
            den = np.sum(
                    [np.argmax(N_bar[u, :]) == k for u in range(self.V)]
                )
            for j in range(self.V):
                num = 0
                if np.argmax(N_bar[j, :]) == k:
                    num = 1

                aux[:, j, k] = [num/den for i in range(self.V)]
        aux = aux[:, :, :, None]
        P_rean = np.append(P_rean, aux, axis=-1)
        return P_rean

    def _calculate_P_tran(self, P_tran, P_rand, P_pref, P_rean, S, t):
        aux = np.zeros((self.V, self.V, self.K))
        for k in range(self.K):

            non_exhausted = (
                1 - S[k, t]) * (
                    self.lambda_ * P_pref[:, :, k, t] + (
                        1 - self.lambda_) * P_rand
                    )

            exhausted = S[k, t] * P_rean[:, :, k, t]
            aux[:, :, k] = non_exhausted + exhausted

        aux = aux[:, :, :, None]
        P_tran = np.append(P_tran, aux, axis=-1)

        return P_tran

    def _choose_next_vertices(self, P_tran, p):
        aux = np.zeros((1, self.K))
        for k in range(self.K):
            aux[0, k] = np.random.choice(
                [i for i in range(self.V)],
                p=P_tran[int(p[-1, k]), :, k]
            )

        p = np.append(p, aux, axis=0)

        return p

    def _update_N(self, p, N):
        aux = N.copy()
        for k, i in enumerate(p):
            aux[int(i), k, -1] += 1

        N = np.append(N, aux, axis=-1)
        return N

    def _update_N_bar(self, N_bar, N):
        N_bar_updated = self._calculate_initial_N_bar(N)
        N_bar_updated = N_bar_updated[:, :, None]

        N_bar = np.append(N_bar, N_bar_updated, axis=-1)
        return N_bar

    def _update_E(self, E, N_bar, p):
        aux = np.zeros(self.K)
        for k in range(self.K):
            if self._is_owner(k, p[-1], N_bar):
                aux[k] = min(E[k, -1] + self.delta, self.omega_max)
            else:
                aux[k] = max(E[k, -1] - self.delta, self.omega_min)
        aux = aux[:, None]
        E = np.append(E, aux, axis=-1)
        return E

    def _update_S(self, S, E):
        aux = np.zeros(self.K)
        for k in range(self.K):
            if E[k] == self.omega_min:
                aux[k] = 1
            else:
                aux[k] = 0
        aux = aux[:, None]
        S = np.append(S, aux, axis=-1)
        return S

    def _is_owner(self, k, p, N_bar):
        if np.argmax(N_bar[int(p[k]), :, -1]) == k:
            return True
        else:
            return False
