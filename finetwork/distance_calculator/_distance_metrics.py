import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import squareform

class _Metrics:
    def __init__(self, 
                 data_matrix, 
                 method='pearson', 
                 scaled=False,
                 sigma=None,
                 k=2):
        self.data_matrix = data_matrix.astype(float)
        self.method = method
        self.scaled = scaled
        self.sigma = sigma
        if k > data_matrix.shape[1]:
            k = data_matrix.shape[1]
        else:
            pass
        self.k = k
        
    def _calculate(self):
        method = self.method
        scaled = self.scaled
        if method == 'pearson':
            distance_matrix = self._pearson_based()
            
        elif method == 'dtw':
            distance_matrix = self._dtw_based()  
            
        elif method == 'theil_index':
            distance_matrix = self.theil_based()
        
        elif method == 'atkinson_index':
            distance_matrix = self._theil_based(atkinson=True)
            
        if scaled:
            distance_matrix = self._calculate_scaled_affinity(distance_matrix)
        else:
            pass
        distance_matrix = distance_matrix.fillna(0)
        return distance_matrix
    
    def _dtw_based(self):
        data_matrix = self.data_matrix
        for col in data_matrix.columns:
            data_matrix[col] = (data_matrix-data_matrix[col].mean())/data_matrix[col].std()
        data_matrix = data_matrix.fillna(0)
        distance_matrix = np.empty((data_matrix.shape[1], data_matrix.shape[1]), dtype=float)
        for i, col1 in enumerate(data_matrix.columns):
            for j, col2 in enumerate(data_matrix.columns):
                distance_matrix[i,j] = fastdtw(data_matrix[col1].values, 
                                          data_matrix[col2].values, 
                                          dist=cityblock)[0]
        distance_matrix = pd.DataFrame(
            data=distance_matrix, 
            index = data_matrix.columns, 
            columns = data_matrix.columns
            )
        return distance_matrix
    
    def _pearson_based(self):
        data_matrix = self.data_matrix
        corr_matrix = data_matrix.corr()
        distance_matrix = np.sqrt(2*(1-corr_matrix))
        
        return distance_matrix
    
    
    def _theil_based(self, atkinson = False):
        data_matrix = self.data_matrix
        # Theil index based Manhattan Distance
        # ThMD
        data_mean = data_matrix.mean()
        mean_fraq = data_matrix / data_mean
        log_mean_fraq = mean_fraq.applymap(lambda x: np.log(x) if x>0 else 0)
    
        Th = (mean_fraq * log_mean_fraq).sum() / len(mean_fraq)
        index_metric = pd.DataFrame(Th).T
        if atkinson:
            index_metric = 1-np.exp(-index_metric) # Atkinson index
        else:
            pass
        distance_matrix = self.manhattan_distance(index_metric)
        
        return distance_matrix
    
    def _manhattan_distance(self, pd_data):
        distance = lambda col1, col2: np.abs(col1 - col2).sum() / len(col1)
        result = pd_data.apply(
            lambda col1: pd_data.apply(
                lambda col2: distance(col1, col2)
                )
            )
        return result
    
    def  _calculate_scaled_affinity(self, distance_matrix):
        # https://papers.nips.cc/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf
        sigma = self.sigma
        if sigma:
            sigma = sigma**2
        else:
            knn_dist = np.sort(squareform(distance_matrix), axis=0)[self.k]
            knn_dist = knn_dist[np.newaxis].T
            sigma = knn_dist.dot(knn_dist.T)
        distance_matrix = np.exp((distance_matrix**2) / sigma)
        np.fill_diagonal(distance_matrix.values, 0)
        return distance_matrix
    