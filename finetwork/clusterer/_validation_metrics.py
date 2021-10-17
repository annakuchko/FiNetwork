from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score


class _InternalEvaluation:
    def __init__(self, X, labels, metrics='calinski_harabasz_index'):
        self.metrics = metrics
        self.X = X
        self.labels = list(labels.values())
    
    def _get_metrics(self):
        metrics = self.metrics
        if metrics == 'calinski_harabasz_index':
            score = self._calinski_harabasz_index()
        elif metrics == 'sillhouette_score':
            score = self._sillhouette_score()
        elif metrics == 'davies_bouldin_score':
            score = self._davies_bouldin_score()
        else:
            pass
        return score
    # TODO: implement more clustering metrics 
    # 1
    def _rmsstd():
        pass
    # 2
    def _r_squared():
        pass
    # 3
    def modified_hubert():
        pass
    # 4
    def _calinski_harabasz_index(self):
        return calinski_harabasz_score(self.X, self.labels)

    # 5
    def _i_index():
        pass
    # 6
    def _dunn_index():
        pass
    # 7
    def _sillhouette_score(self):
        return silhouette_score(self.X, self.labels)
    # 8
    def _davies_bouldin_score(self):
        return davies_bouldin_score(self.X, self.labels)
    # 9
    def _xie_beni_index():
        pass
    # 10
    def _sd_index():
        pass
    # 11
    def _dens_bw():
        pass
    