import pandas as pd
import numpy as np
from finetwork.data_collector.import_data import FinData
from finetwork.transformer.splitter import Split
import finetwork.plotter._plot_market_cycles as pmc

class MarketCycle:
    def __init__(self, 
                 benchmark_index,
                 from_date, 
                 to_date, 
                 country,
                 market,
                 criterion='trading_day',
                 window=6,
                 theta_min=0.45,
                 theta_max=0.55):
        
        self.from_date = from_date
        self.to_date = to_date
        self.country = country
        self.benchmark_index = benchmark_index
        self.market = market
        self.window = window
        self.criterion = criterion
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.index_data = None
        
    def _import_index(self):
        from_date = self.from_date
        to_date = self.to_date
        benchmark_index = self.benchmark_index
        country = self.country
        market = self.market
        
        data = FinData(
            from_date=from_date, 
                to_date=to_date,
                tickers=benchmark_index,
                country=country,
                market=market,
                index=True).get_data()
        return data
    
    def _split_data(self):
        window = self.window
        from_date = self.from_date
        to_date = self.to_date
        dat = self._import_index()
        data = dat[0]
        original_data = dat[1]
        dates_list = pd.to_datetime(list(data.keys()))
        list_of_windows = [dates_list[0]]
        for i in np.arange(0, len(dates_list)-window*22, window*22):
            start_dt = dates_list[i]
            end_dt = start_dt + pd.DateOffset(months=window)
            list_of_windows.append(end_dt)
        
        data_split = Split(
            data_dict=data,
            from_date=from_date,
            to_date=to_date,
            window_size=window
            ).split_index()       
        original_data_split = Split(
            data_dict=original_data,
            from_date=from_date,
            to_date=to_date,
            window_size=window
            ).split_index()       
        
        return data_split, original_data_split
    
    def fit(self, return_criterion_vals=False):
        
        theta_min = self.theta_min
        theta_max = self.theta_max
        
        dat = self._split_data()
        data = dat[0]
        orig_data = dat[1]
        self.index_data = data
        criterion = self.criterion
        criterion_dict = {}
        rd_dict = {}
        rf_dict = {}
        
        for i in data.keys():
            period_data = np.array(list(data[i].values()))
            period_data = period_data[~np.isnan(period_data)]
            
            rd = (period_data>0).sum() / len(period_data)
            rd_dict[i] = rd
            
            rf = np.abs(
                period_data[period_data>0]
                ).sum() / np.abs(
                    period_data
                    ).sum()
            rf_dict[i] = rf
            if criterion == 'trading_day':    
                if rd > theta_max:
                    criterion_val='markup'
                elif rd < theta_min:
                    criterion_val='markdown'
                else:
                    criterion_val='flat'
            
            elif criterion == 'amplitude':
                if rf > theta_max:
                    criterion_val='markup' 
                elif rf < theta_min:
                    criterion_val='markdown'
                else:
                    criterion_val='flat'
            
            elif criterion == 'and':
                if rd > theta_max and rf > theta_max:
                    criterion_val = 'markup'
                elif rd < theta_min and rf < theta_min:
                    criterion_val = 'markdown'
                else:
                    criterion_val = 'flat'

            elif criterion == 'or':
                if rd > theta_max or rf > theta_max:
                    criterion_val = 'markup'
                elif rd < theta_min or rf < theta_min:
                    criterion_val = 'markdown'
                else:
                    criterion_val = 'flat'
            
            criterion_dict[i] = criterion_val
        list_of_periods = list(criterion_dict.keys())
        
        for i in np.arange(0, len(list_of_periods)):
            period = list_of_periods[i]
            
            if i == 0 and criterion_dict[period]=='flat':
                criterion_dict[period] = 'accumulation'
            else:
                prev_period = list_of_periods[i-1]
            
                if criterion_dict[period] == 'flat' and\
                    criterion_dict[prev_period]=='markup':
                    criterion_dict[period] = 'accumulation'
                elif criterion_dict[period] == 'flat' and\
                    criterion_dict[prev_period]=='accumulation':
                    criterion_dict[period] = 'accumulation'
                elif criterion_dict[period] == 'flat' and\
                    criterion_dict[prev_period]=='markdown':
                    criterion_dict[period] = 'distribution'
                elif criterion_dict[period] == 'flat' and\
                    criterion_dict[prev_period]=='distribution':
                    criterion_dict[period] = 'distribution'
        if return_criterion_vals:
            return rd_dict, rf_dict
        else:
            return criterion_dict, orig_data

    def plot_cycles(self):
        data = self.fit(return_criterion_vals=True)
        pmc._plot_cycles(data,
                         self.theta_min, 
                     self.theta_max, 
                     self.window, 
                     self.benchmark_index)
        
    def plot_index_with_cycles(self, name=None, save=False):
        data = self.fit(return_criterion_vals=False)
        pmc._plot_index_with_market_cycles(data[0],
                                           data[1], 
                                           self.benchmark_index,
                                           self.criterion,
                                           self.window,
                                           self.from_date,
                                           self.to_date,
                                           name,
                                           save)
