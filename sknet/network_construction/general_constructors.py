import networkx as nx 

from abc import ABCMeta, abstractmethod
from sknet.utils import NetworkTypesHandler


class GeneralConstructor(metaclass=ABCMeta):
    def __init__(self, net_type):
        self.net_type = net_type
        self.network_type_handler = NetworkTypesHandler()

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def transform(self):
        """Returns the networkX graph after the constructor is fitted

        Returns
        -----
        G : NetworkX graph
            The network version of the inserted data
        """
        try:
            return self.G
        except AttributeError:
            raise Exception("Transformer is not fitted")

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.G

    def get_network(self):
        """Retrieves the network generated in the constructor class
        """
        return self.G


class EdgeListConstructor():
    def __init__(self, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G = nx.read_edgelist(X)

        self.G = network_type(self.G)


class AdjacencyListConstructor():
    def __init__(self, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G = nx.read_adjlist(X)

        self.G = network_type(self.G)


class YAMLConstructor():
    def __init__(self, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G = nx.read_yaml(X)

        self.G = network_type(self.G)


class PajekConstructor():
    def __init__(self, path, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G = nx.read_pajek(X)

        self.G = network_type(self.G)
