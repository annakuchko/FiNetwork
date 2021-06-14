import networkx as nx
import numpy as np

class _Laplacian:
    def __init__(self, dist_matrix, normalized=False):
        self.dist_matrix = dist_matrix
        self.normalized = normalized
        
    def _get_graph(self):
        G = nx.from_numpy_matrix(self.dist_matrix.values)
        labels = self.dist_matrix.columns.values
        G = nx.relabel_nodes(G, dict(zip(range(len(labels)), labels)))
        return G
        
    def _calculate_laplacian(self):
        G = self._get_graph()
        
        if self.normalized:
            L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(
                G, 
                weight='weight'
                )
        else:
            L = nx.laplacian_matrix(
                G, 
                weight='weight'
                )
        L = np.nan_to_num(L.toarray().astype(np.float32))
        return L
            
    def _calculate_adjacency(self):
        G = self._get_graph()
        A = nx.linalg.graphmatrix.adjacency_matrix(
            G, 
            weight='weight'
            )
        A = np.nan_to_num(A.toarray().astype(float))
        return A
    
    def _get_spectrum(self):
        L = self._calculate_laplacian()
        e, v = np.linalg.eig(L)
        e = e.real
        v = v.real
        return e, v
        