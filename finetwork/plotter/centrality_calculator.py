from finetwork.optimiser._centrality_metrics import _CentralityMetrics
class Centrality:
    
    def __init__(self, G, metrics='degree_centrality'):
        self.G = G
        self.metrics = metrics    
    
    def calculate_centrality(self):
        metrics = self.metrics 
        G = self.G
        cm = _CentralityMetrics(G, metrics)._compute_metrics()
        return cm
            