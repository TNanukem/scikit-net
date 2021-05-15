import networkx as nx


class NetworkTypesHandler():
    """"""

    def __init__(self):
        self.mapper = self._generate_types_mapper()

    def _generate_types_mapper(self):
        mapper = {
            'graph': nx.Graph,
            'digraph': nx.DiGraph,
            'multi_graph': nx.MultiGraph,
            'multi_digraph': nx.MultiDiGraph,
        }
        return mapper

    def get_net(self, metric):
        return self.mapper[metric]
