import networkx as nx

from sknet.utils import NetworkTypesHandler


def test_handler_mapper():
    handler = NetworkTypesHandler()
    assert handler.mapper == {'graph': nx.Graph,
                              'digraph': nx.DiGraph,
                              'multi_graph': nx.MultiGraph,
                              'multi_digraph': nx.MultiDiGraph}


def test_get_net():
    handler = NetworkTypesHandler()
    assert handler.get_net('graph') == nx.Graph
    assert handler.get_net('digraph') == nx.DiGraph
    assert handler.get_net('multi_graph') == nx.MultiGraph
    assert handler.get_net('multi_digraph') == nx.MultiDiGraph
