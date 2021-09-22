import pandas as pd
import datetime
import numpy as np

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta
        
        
class Split:
    def __init__(self,
                 data_dict,
                 from_date,
                 to_date,
                 window_size=6):
        
        self.window_size = window_size
        self.data_dict = data_dict
        self.from_date = from_date
        self.to_date = to_date
    
    
    def transform(self):
        data_dict = self.data_dict
        window_size = self.window_size
        
        min_dt = pd.to_datetime(datetime.datetime.strptime(self.from_date,'%d/%m/%Y').date())
        max_dt = pd.to_datetime(datetime.datetime.strptime(self.to_date, "%d/%m/%Y").date())
        
        list_of_windows = [str(i.date()) for i in perdelta(
            min_dt,
            max_dt,
            np.timedelta64(window_size, 'M')
            )]
        if max_dt not in list_of_windows:
            list_of_windows.append(str(max_dt.date()))
        n = len(list_of_windows)
        new_dict = {}
        for i in range(0, n-1):
            l = {}
            for stock in data_dict.keys():
                sub_dict = {}
                sub_dict[stock] = {
                    k: v for k,v in data_dict[stock].items() if\
                        k >= list_of_windows[i] and\
                            k < list_of_windows[i+1]
                            }
                l.update(sub_dict)
            
            new_dict[f'{list_of_windows[i]}-{list_of_windows[i+1]}'] = l
            
        for i, dtr in enumerate(new_dict.keys()):
            dates_list = [str(j) for j in perdelta(
                pd.to_datetime(list_of_windows[i]),
                pd.to_datetime(list_of_windows[i+1]),
                np.timedelta64(1, 'D')
                )]
            empty_dates_dict = {}
            for dt in dates_list:
                empty_dates_dict[dt] = ''
            m = {}
            for stock in new_dict[dtr].keys():
                sub_dict = {}
                sub_dict[stock] = {
                    k: new_dict[dtr][stock].get(
                        k, {
                            'close_adj': 'NaN', 
                            'log_return':'NaN'
                            }
                        ) for k in empty_dates_dict
                    }
                m.update(sub_dict)
            new_dict[dtr] = m
        
        return new_dict
    
    def split_index(self):
        data_dict = self.data_dict
        min_dt = pd.to_datetime(datetime.datetime.strptime(self.from_date,'%d/%m/%Y').date())
        max_dt = pd.to_datetime(datetime.datetime.strptime(self.to_date, "%d/%m/%Y").date())
        
        list_of_windows = [str(i.date()) for i in perdelta(
            min_dt,
            max_dt,
            np.timedelta64(self.window_size, 'M')
            )]
        if max_dt not in list_of_windows:
            list_of_windows.append(str(max_dt.date()))
        n = len(list_of_windows)
        new_dict = {}
        for i in range(0, n-1):
            l = {k: v for k,v in data_dict.items() if\
                 pd.to_datetime(k) >= pd.to_datetime(list_of_windows[i]) and\
                     pd.to_datetime(k) < pd.to_datetime(list_of_windows[i+1])}
            new_dict[f'["{list_of_windows[i]}","{list_of_windows[i+1]}"]'] = l
        return new_dict
    