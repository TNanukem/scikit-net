import networkx as nx

from abc import ABCMeta, abstractmethod
from sknet.utils import NetworkTypesHandler


class GeneralConstructor(metaclass=ABCMeta):
    def __init__(self, net_type):
        self.net_type = net_type
        self.network_type_handler = NetworkTypesHandler()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"net_type": self.net_type}

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
            return self.G_
        except AttributeError:
            raise Exception("Transformer is not fitted")

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.G_

    def get_network(self):
        """Retrieves the network generated in the constructor class
        """
        return self.G_


class EdgeListConstructor():
    def __init__(self, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G_ = nx.read_edgelist(X)

        self.G_ = network_type(self.G_)
        return self


class AdjacencyListConstructor():
    def __init__(self, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G_ = nx.read_adjlist(X)

        self.G_ = network_type(self.G_)
        return self


class YAMLConstructor():
    def __init__(self, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G_ = nx.read_yaml(X)

        self.G_ = network_type(self.G_)
        return self


class PajekConstructor():
    def __init__(self, path, net_type='graph'):
        super().__init__(net_type)

    def fit(self, X, y=None):
        network_type = self.network_type_handler.get_net(self.net_type)
        self.G_ = nx.read_pajek(X)

        self.G_ = network_type(self.G_)
        return self
