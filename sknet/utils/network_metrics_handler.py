import networkx as nx


class NetworkMetricsHandler():
    """"""

    def __init__(self):
        self.mapper = self._generate_metrics_mapper()

    def _generate_metrics_mapper(self):
        mapper = {
            'betweenness': {},
            'assortativity': {},
            'clustering_coefficient': {},
            'connectivity': {},
        }
        return mapper

    def get_metric(self, metric):
        return self.mapper[metric]
