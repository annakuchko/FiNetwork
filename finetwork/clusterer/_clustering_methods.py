import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering, DBSCAN 
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from finetwork.clusterer._laplacian import _Laplacian
from finetwork.clusterer._validation_metrics import _InternalEvaluation

class _ClusteringMethods:
    def __init__(self, method='Kmeans', normalized=False, n_clusters=None, params={},
                 return_validation_scores=False,
                 min_clusters=2):
        self.e = None
        self.v = None
        self.A = None
        self.U = None
        if method=='None':
            return_validation_scores = False
        else:
            pass
        self.return_validation_scores = return_validation_scores
        self.n_clusters = n_clusters
        self.normalized = normalized
        self.method = method
        self.params=params
        self.score_ = None
        self.min_clusters = min_clusters
        self.clustering_results = None
        
    def _get_metrics(self, validation_metrics):
        return _InternalEvaluation(self.A, 
                                  self.clustering_results, 
                                  validation_metrics)._get_metrics()
        
    def _fit(self, G):
        self._get_spectrum(G)
        method = self.method
        if method == 'Kmeans':
            clustering_results = self.Kmeans()
        if method == 'Spectral':
            clustering_results = self.Spectral()
        elif method == 'SpectralKmeans':
            clustering_results = self.SpectralKmeans()
        elif method == 'Kmedoids':
            clustering_results = self.Kmedoids()
        elif method == 'SpectralGaussianMixture':
            clustering_results = self.SpectralGaussianMixture()
        elif method == 'GaussianMixture':
            clustering_results = self.GaussianMixture()
        elif method == 'Hierarchical':
            clustering_results = self.Hierarchical()
        elif method=='None':
            clustering_results = self.no_partition()
        
        clustering_results = {
            list(G.columns)[i][1]:f'Cluster {clustering_results[i]}'\
                for i in range(len(G.columns))
                }
        self.clustering_results = clustering_results
        return clustering_results
            
        
    def _calculate_e_order(self):
        e = self.e
        sorted_e_idx = sorted(range(len(e)),
                              key=lambda k: e[k], reverse=True)
        sorted_e = e[sorted_e_idx]
        abs_diff = np.abs(np.diff(sorted_e))[self.min_clusters:10]
        e_order = self.min_clusters + np.argmax(abs_diff)
        return e_order
        
    def _get_spectrum(self, G):
        lapl = _Laplacian(G, normalized=self.normalized)
        e, v = lapl._get_spectrum()
        A = lapl._calculate_adjacency()
        A[A == -np.inf] = 0
        A[A == np.inf] = 0
        A[A == -np.nan] = 0
        A = A.astype(np.float64)
        self.A = A
        self.e = e
        self.v = v

        if not self.n_clusters:
             self.n_clusters = self._calculate_e_order()

        i = np.argsort(e)[1]
        U = np.array(v[:, i]).reshape(-1, 1)
        U[U == -np.inf] = 0
        U[U == np.inf] = 0
        U[U == -np.nan] = 0
        U = U.astype(np.float64)
        self.U = U
        
    def Spectral(self):
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed'
            ).set_params(**self.params).fit(self.A)
        return sc.labels_
    
    def SpectralKmeans(self):
        skm = KMeans(
            init='k-means++', 
            n_clusters=self.n_clusters,
            ).set_params(**self.params).fit(self.U)
        return skm.labels_
    
    def Kmeans(self):
        km = KMeans(
            init='k-means++', 
            n_clusters=self.n_clusters
            ).set_params(**self.params).fit(self.A)
        return km.labels_
            
    def Kmedoids(self):
        kmd = KMedoids(
            n_clusters=self.n_clusters,
            metric='precomputed'
            ).set_params(**self.params).fit(self.A)
        return kmd.labels_
    
    def DBSCAN(self):
        dbscan = DBSCAN(
            metric='precomputed'
            ).set_params(**self.params).fit(self.A)
        return dbscan.labels_
        
    def SpectralGaussianMixture(self):
        sgm = GaussianMixture(
            n_components=self.n_clusters
            ).set_params(**self.params).fit_predict(self.U)
        return sgm
    
    def GaussianMixture(self):
        gm = GaussianMixture(
            n_components=self.n_clusters
            ).set_params(**self.params).fit_predict(self.A)
        return gm
        
    def Hierarchical(self):
        aglo = AgglomerativeClustering(
            affinity='precomputed',
            linkage='complete',
            n_clusters=self.n_clusters
            ).set_params(**self.params).fit(self.A)
        return aglo.labels_
    
    def no_partition(self):
        return [int(0) for i in range(self.A.shape[0])]
