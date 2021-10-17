import pandas as pd
import numpy as np
from finetwork.optimiser._portfolio_performance_metrics import calmar_ratio, volatility,\
    sharpe_ratio, sortino_ratio, max_drawdown

class Validator:
    def __init__(self, data_dict, selected_stocks):
        self.data_dict = data_dict
        self.selected_stocks = selected_stocks
        self.ret = None
    
    def fit(self):
        data_dict = self.data_dict
        selected_stocks = self.selected_stocks
        r = []
        for i in range(len(list(selected_stocks.keys()))-1):
            buy_in = list(selected_stocks.keys())[i]
            sell_in = list(selected_stocks.keys())[i+1]
            s = [j for i in list(selected_stocks[buy_in].values()) for j in i]
            data = {key: data_dict[sell_in][key] for key in s}
            start = []
            stop = []
            if not data:
                p_returns = np.nan
            else:
                for stock in data.keys():
                    sub = {k:v for k,v in {k:v['close_adj'] for k,v in data[stock].items()}.items() if v!='NaN'}
                    start.append(list(sub.values())[0])
                    stop.append(list(sub.values())[-1])
                    p_returns = (sum(stop) / sum(start))-1
            r.append(p_returns)
            
        self.ret = pd.Series(r)
    
    def returns(self):
        return self.ret

    
    def performance_metrics(self):
        returns = self.ret
        performance_metrics = {
             'volatility': volatility(returns), 
             'calmar_ratio': calmar_ratio(returns), 
             'sharpe_ratio': sharpe_ratio(returns),
             'sortino_ratio': sortino_ratio(returns),
             'max_drawdown': max_drawdown(returns)}   
        return performance_metrics
    # TODO: plot valdation aka backtesting
    def _plot_validation(self):
        pass