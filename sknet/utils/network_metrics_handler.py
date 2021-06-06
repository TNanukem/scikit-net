import networkx as nx


class NetworkMetricsHandler():
    """"""

    def __init__(self):
        self.mapper = self._generate_metrics_mapper()
        self.default_values = self._default_values_mapper()

    def _generate_metrics_mapper(self):
        mapper = {
            'assortativity': nx.degree_assortativity_coefficient,
            'clustering_coefficient': nx.average_clustering,
            'average_degree': nx.average_degree_connectivity,
            'transitivity': nx.transitivity,
            'connectivity': nx.average_node_connectivity,
        }
        return mapper

    def _default_values_mapper(self):
        mapper = {
            'assortativity': 2,
            'clustering_coefficient': 1,
            'average_degree': 0,
            'transitivity': 0,
            'connectivity': 0,
        }
        return mapper

    def get_metric(self, metric):
        return self.mapper[metric]

    def get_default_value(self, metric):
        return self.default_values[metric]
