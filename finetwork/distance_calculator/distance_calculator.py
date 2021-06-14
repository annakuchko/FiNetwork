from finetwork.distance_calculator import _distance_metrics
import pandas as pd

class CalculateDistance:
    def __init__(self, data, method='pearson', scaled=False, sigma = 0.5):
        self.data = data
        self.method = method
        self.scaled = scaled
        self.sigma = sigma 
        
        
    def calculate_distance(self):
        data = self.data
        dist_dict = {}
        for i in data.keys():
            tmp = pd.DataFrame.from_dict({(v,k): data[i][v][k]['log_return'] 
                               for v in data[i].keys() 
                               for k in data[i][v].keys()},
                           orient='index')
            
            tmp.index = pd.MultiIndex.from_arrays(
                [
                    [tmp.index[i][0] for i in range(len(tmp.index))], 
                    [tmp.index[i][1] for i in range(len(tmp.index))]
                    ]
                )
            
            tmp = tmp.reset_index().pivot('level_1', 'level_0')
            
            distance_matrix = _distance_metrics._Metrics(
                tmp, 
                method = self.method,
                scaled=self.scaled, sigma=self.sigma
                )._calculate() 
            distance_matrix.index = distance_matrix.index.get_level_values(
                'level_0'
                )

            dist_dict[i] = distance_matrix
            
        return dist_dict
