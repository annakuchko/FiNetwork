from finetwork.clusterer._clustering_methods import _ClusteringMethods
import numpy as np
from prettytable import PrettyTable
from collections import defaultdict

class NetClusterer:
    def __init__(self, method, data_dict, return_validation_scores=True):
        self.method = method
        self.data_dict = data_dict
        if method=='None':
            return_validation_scores=False
        else:
            pass
        self.return_validation_scores = return_validation_scores
        self.score_table = None
    
    def fit(self):
        data_dict = self.data_dict
        method = self.method
        metrics_list = ['calinski_harabasz_index',
                        'sillhouette_score',
                        'davies_bouldin_score']
        scores = {}
        for m in metrics_list:
            scores[m] = []
        partition = {}
        
        for cycle in data_dict.keys():
            if data_dict[cycle].sum().sum()==0:
                pass
            else:
                cm = _ClusteringMethods(method=method, 
                                        return_validation_scores=self.return_validation_scores)
                partition[cycle] = cm._fit(data_dict[cycle])
                if self.return_validation_scores:
                    d = defaultdict(list)
                    for k, v in scores.items():
                        d[k].append(cm._get_metrics(k))
                    
        
        if self.return_validation_scores:
                myTable = PrettyTable(["Metrics", "Value"])
                for k, v in d.items():
                    myTable.add_row([f'Mean {k}', round(np.mean(v), 4)])
                    myTable.align = 'l'
                self.score_table = myTable
        return partition
    
    def print_scores(self):
        print(f'{self.method} clustering results')
        print(self.score_table)
    
        
            
    
if __name__ == '__main__':
    import numpy as np
    import networkx as nx
    import string
    import random
    
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    
    G = nx.powerlaw_cluster_graph(100, 99, 1)
   
    for (u,v,w) in G.edges(data=True):
        w['weight'] = abs(np.random.power(1))*abs(np.random.power(1))

    lbl = {i: str(id_generator(5)) for i in G.nodes()}
    G = nx.relabel_nodes(G, lbl)
    data_dict = {'2000-2005':nx.convert_matrix.to_pandas_adjacency(G)}
    G = nx.powerlaw_cluster_graph(100, 99, 1)
   
    for (u,v,w) in G.edges(data=True):
        w['weight'] = abs(np.random.power(1))*abs(np.random.power(1))

    lbl = {i: str(id_generator(5)) for i in G.nodes()}
    G = nx.relabel_nodes(G, lbl)
    data_dict['2005-2010'] = nx.convert_matrix.to_pandas_adjacency(G)
    nc = NetClusterer('SpectralKmeans', data_dict)
    nc._get_clusters()
    nc.print_scores()
    nc = NetClusterer('Kmeans', data_dict)
    nc._get_clusters()
    nc.print_scores()
    nc = NetClusterer('Spectral', data_dict)
    nc._get_clusters()
    nc.print_scores()
    # NetClusterer('Kmedoids', data_dict)._get_clusters()
    nc = NetClusterer('SpectralGaussianMixture', data_dict)
    nc._get_clusters()
    nc.print_scores()
    nc = NetClusterer('GaussianMixture', data_dict)
    nc._get_clusters()
    nc.print_scores()
    nc = NetClusterer('Hierarchical', data_dict)
    nc._get_clusters()
    nc.print_scores()
         