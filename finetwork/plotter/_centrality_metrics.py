import networkx as nx

class _CentralityMetrics:
    
    def __init__(self, G, metrics):
        self.G = G
        self.metrics = metrics
        
    def _compute_metrics(self):
        metrics = self.metrics
        if metrics == 'degree_centrality':
            c = self.degree_centrality()
        elif metrics == 'betweenness_centrality':
            c = self.betweenness_centrality()
        elif metrics == 'closeness_centrality':
            c = self.closeness_centrality()
        elif metrics == 'eigenvector_centrality':
            c = self.bonachi_eigenvector_centrality()
            
        return c
    
    def degree_centrality(self):
        centrality = nx.degree_centrality(self.G, weight='weight')
        return centrality
    
    def betweenness_centrality(self):
        centrality = nx.betweenness_centrality(self.G, weight='weight')
        return centrality
    
    def closeness_centrality(self):
        centrality = nx.closeness_centrality(self.G, weight='weight')
        return centrality
    
    def bonachi_eigenvector_centrality(self):
        centrality = nx.eigenvector_centrality(self.G, weight='weight')
        return centrality
