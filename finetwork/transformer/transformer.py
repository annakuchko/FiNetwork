import numpy as np
import networkx as nx
from finetwork.transformer import _filtering

def to_graph(data):
    new_dict = {}
    for i, date in enumerate(data.keys()):
            
        if data[date].sum().sum()==0:
            pass
        else:
            stocks = data[date].index.values
            dist_matrix = np.asmatrix(data[date])
            G = nx.from_numpy_matrix(dist_matrix)
            G = nx.relabel_nodes(G,lambda x: stocks[x])
            new_dict[date] = G
    return new_dict
    
class NetTransformer:
    def __init__(self, method='mst'):
        self.method = method
        

    def fit(self, data):
        graph_dict = to_graph(data)
        return graph_dict
    
    def transform(self, graph_dict):
        filtered_dict = {}
        for i, date in enumerate(graph_dict.keys()):
            G_filtered = _filtering._FilterNetwork(
                graph_dict[date], 
                method=self.method
                )._filter()
            filtered_dict[date] = G_filtered
        return filtered_dict
    
    def fit_transform(self, data):
        graph_dict = self.fit(data)
        filtered_dict = self.transform(graph_dict)
        return filtered_dict
        

        
    





