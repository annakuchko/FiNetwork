from finetwork.clusterer._clustering_methods import _ClusteringMethods
import numpy as np
from prettytable import PrettyTable
from collections import defaultdict

class NetClusterer:
    def __init__(self, 
                 method, 
                 data_dict, 
                 return_validation_scores=True,
                 normalized=False,
                 n_clusters=None, 
                 min_clusters=2):
        self.method = method
        self.data_dict = data_dict
        if method=='None':
            return_validation_scores=False
        else:
            pass
        self.return_validation_scores = return_validation_scores
        self.normalized = normalized 
        self.n_clusters = n_clusters 
        self.min_clusters = min_clusters
        self.score_table = None
    
    def fit(self):
        data_dict = self.data_dict
        metrics_list = ['calinski_harabasz_index',
                        'sillhouette_score',
                        'davies_bouldin_score']
        scores = {}
        for m in metrics_list:
            scores[m] = []
        partition = {}
        
        for cycle in data_dict.keys():
            if data_dict[cycle].sum().sum()==0:
                G = list(data_dict[cycle].keys())
                partition[cycle] = {G[i][1]:'Cluster -1' for i in range(len(G))}
            else:
                cm = _ClusteringMethods(
                    method=self.method, 
                    normalized=self.normalized, 
                    n_clusters=self.n_clusters,
                    return_validation_scores=self.return_validation_scores,
                    min_clusters=self.min_clusters
                    )
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