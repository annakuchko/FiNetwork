import random
import numpy as np
import math
# из каждой отраслевой группы выбирается число акций, равное 
# размеру портфеля (в акциях), деленному на число отраслей 
# (в случае деления с остатком выбирается дополнительное число
# отраслей на основе равномерного распределения, из которых 
# выбираются акции). Акции внутри отраслей также выбираются 
# на основе равномерного распределения.

class Optimiser:
    def __init__(self, portfolio_size=12, distribution=None):
        self.portfolio_size = portfolio_size
        self.distribution = distribution
        
    def _calculate_n_stocks(self, partition_dict):
        portfolio_size = self.portfolio_size
        n_groups = len(np.unique(list(partition_dict.values())))
        return portfolio_size / n_groups
    
    def select(self, cluster_dict):
        select_dict = {}
        for cycle in list(cluster_dict.keys()):
            n = math.floor(self._calculate_n_stocks(cluster_dict[cycle]))
            tmp_dict = {}
            cd = cluster_dict[cycle]
            for cl in np.unique(list(cd.values())):
                # if (n-int(n))==0:
                cluster_stocks = [k for k,v in cd.items() if v == cl]
                if n <= len(cluster_stocks):
                    tmp_dict[cl] = random.sample(
                        cluster_stocks,
                        n
                        )
                else:
                    tmp_dict[cl] = cluster_stocks
                    
            select_dict[cycle] = tmp_dict
        return select_dict 
