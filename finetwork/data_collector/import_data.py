import investpy
import pandas as pd
from lxml import html  
import requests
import os
from finetwork.utils._utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import json
import numpy as np
import datetime
import trading_calendars as tc

class FinData:
    def __init__(self,
                 from_date,
                 to_date,
                 tickers,
                 country,
                 market=None,
                 index=False):
        self.from_date = from_date
        self.to_date = to_date
        self.country = country
        self.tickers = tickers
        self.market = market
        self.index = index
        
    def get_data(self,
                 dividends_correction = True,
                   save = True,
                   save_format = 'csv',
                   save_path = 'data/', 
                   trading_day_thresh = 0.55):
        
        index = self.index
        from_date = self.from_date
        to_date = self.to_date
        market = self.market
        tickers = self.tickers
        country = self.country
        
        if index:
            tickers=[tickers]
        data_dict = {}
        sectors_dict = {}
        if market == 'world_indices':
            cal = tc.get_calendar('NYSE')
        else:
            cal = tc.get_calendar(market)
        
        cal_trading = cal.sessions_in_range(pd.to_datetime(from_date),
                                            pd.to_datetime(to_date))
        cal_trading = [
            datetime.date.strftime(i, "%Y-%m-%d") for i in cal_trading
            ]

        from_date =  datetime.datetime.strptime(
            min(cal_trading),
            '%Y-%m-%d'
            ).strftime('%d/%m/%Y')
        to_date =  datetime.datetime.strptime(
            max(cal_trading),
            '%Y-%m-%d'
            ).strftime('%d/%m/%Y')
        n_trading = len(cal_trading)
        for i, code in (enumerate(tqdm(tickers))):
            try:
                if index:
                    data = investpy.indices.get_index_historical_data(
                        index=code,
                        country=country,
                        from_date=from_date,
                        to_date=to_date
                        )
                else:
                    data = investpy.get_stock_historical_data(
                        stock=code, 
                        country=country,
                        from_date=from_date,
                        to_date=to_date
                        )
               
                if datetime.datetime.strptime(
                        str(data.index.min().date()), '%Y-%m-%d'
                        ).strftime('%d/%m/%Y')==(from_date)\
                    and datetime.datetime.strptime(
                        str(data.index.max().date()), '%Y-%m-%d'
                        ).strftime('%d/%m/%Y')==(to_date)\
                        and pd.to_datetime(data.index).isin(
                            cal_trading
                            ).sum()>=trading_day_thresh*n_trading:
                    data = data.reset_index(level = 'Date')
                    if dividends_correction and not index:
                        dividends = investpy.stocks.get_stock_dividends(
                            stock=code,
                            country=country)
                        data = pd.merge(data, 
                                     dividends.drop('Date', axis = 1), 
                                     left_on='Date', 
                                     right_on=['Payment Date'],
                                     how='left')
                        data['factor'] = (data.Close-data.Dividend)/data.Close
                        data.loc[data.factor.isna(), 'factor'] = 1.0
                        data['close_adj'] = data.Close
                        for j in range(len(data.factor)):
                            if data.factor[j]!=1.0:
                                data.loc[:j,'close_adj'] = data.loc[:j, 'close_adj']\
                                    * data.factor[j]
                        data['log_return'] = np.log(data.close_adj) -\
                            np.log(data.close_adj.shift(1))
                    else:
                        data['prices_diff'] = data.Close - data.Close.shift(1)
                        
                    
                    dates_dict = {}
                    dates_orig_dict = {}
                    if not index:
                        url = investpy.get_stock_company_profile(
                            stock=code,
                            country=country
                            )['url']
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64;'
                            ' rv:81.0) Gecko/20100101 Firefox/81.0'
                            }
                        page=requests.get(url, headers=headers)
                        tree = html.fromstring(page.content)
                        sector = tree.xpath(
                            '//*[contains(text(), "Sector")]/a/text()'
                            )[0]
                        
                        for dt in data.Date:
                            close_adj = data[data.Date==dt]['close_adj'].values[0]
                            log_return = data[data.Date==dt]['log_return'].values[0]
                            dates_dict[str(pd.to_datetime(dt))] = \
                                {'close_adj': close_adj,
                                 'log_return': log_return}
                        data_dict[str(code)] = dates_dict
                        sectors_dict[str(code)] =  sector
                        
                    else:
                        for dt in data.Date:
                            dates_dict[str(pd.to_datetime(dt))] = \
                                data[data.Date==dt]['prices_diff'].values[0]
                            dates_orig_dict[str(pd.to_datetime(dt))] = \
                                data[data.Date==dt]['Close'].values[0]
                        data_dict = dates_dict
                        
            except:
                pass
            
        if index:
            result_dict = data_dict, dates_orig_dict
        else:
            result_dict = {'data': data_dict, 'sectors': sectors_dict}
        if save:
            outdir = './data'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            json_fullname = os.path.join(outdir, f'data_dict_{str(datetime.date.today())}.json')
            with open(json_fullname, "w") as wf:
                json.dump(result_dict, wf, indent=4)
        
        return result_dict
     
