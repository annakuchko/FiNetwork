import matplotlib.pyplot as plt
import pandas as pd
import json
from matplotlib.lines import Line2D
import datetime

def plot_cycles(data,
                 theta_min, theta_max, window, benchmark_index):
        
        rd_data = data[0]
        rf_data = data[1]
        date_range = [
            pd.to_datetime(json.loads(i))[0] for i in rf_data.keys()
            ]
        theta_min = [theta_min] * len(date_range)
        theta_max = [theta_max] * len(date_range)
        
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(f'Criterion values for {benchmark_index} index \n'
                     f'(window = {window} months)', fontsize=14)             
        ax1.plot(date_range, rd_data.values())
        ax1.axhline(theta_min[0], ls='--', color='r')
        ax1.axhline(theta_max[0], ls='--', color='r')
        ax1.fill_between(date_range,
                         theta_min,
                         theta_max, 
                         color='red',
                         alpha=.2,
                         linewidth=0)
        ax1.text(date_range[0], 
                 theta_max[0], 
                 r'$\theta_+$'+f' = {theta_max[0]}',
                 va='center',
                 bbox=dict(facecolor='white',
                           edgecolor='r',
                           pad=0.75),
                 color='r',
                 fontsize=8)        
        ax1.text(date_range[0], 
                 theta_min[0], 
                 r'$\theta_-$'+f' = {theta_min[0]}',
                 va='center',
                 bbox=dict(facecolor='white',
                           edgecolor='r',
                           pad=0.75),
                 color='r',
                 fontsize=8)    
        
        ax1.set_ylim([max([min(rd_data.values())-0.075,0]), 
                      min([max(rd_data.values())+0.075,1])])
        ax1.set_xlim([min(date_range),max(date_range)])
        ax1.set_title('Trading day criterion', fontsize=11)

        ax2.plot(date_range, rf_data.values())
        ax2.fill_between(date_range,
                         theta_min,
                         theta_max, 
                         color='red',
                         alpha=.2,
                         linewidth=0)
        ax2.axhline(theta_min[0], ls='--', color='r')
        ax2.axhline(theta_max[0], ls='--', color='r')
        ax2.text(date_range[0], 
                 theta_max[0], 
                 r'$\theta_+$'+f' = {theta_max[0]}',
                 va='center',
                 bbox=dict(facecolor='white',
                           edgecolor='r',
                           pad=0.75),
                 color='r',
                 fontsize=8)        
        ax2.text(date_range[0], 
                 theta_min[0], 
                 r'$\theta_-$'+f' = {theta_min[0]}',
                 va='center',
                 bbox=dict(facecolor='white',
                           edgecolor='r',
                           pad=0.75),
                 color='r',
                 fontsize=8)    
        ax2.set_ylim([max([min(rf_data.values())-0.075,0]), 
                      min([max(rf_data.values())+0.075,1])])
        ax2.set_xlim([min(date_range),max(date_range)])
        ax2.set_title('Amplitude criterion', fontsize=11)

        fig.tight_layout()
        

def _get_dates_range(dates_range_list):
    dates_range = []
    for i in range(len(dates_range_list)):
        
        d1 = datetime.datetime.strptime(eval(dates_range_list[i])[0],
                                        '%Y-%m-%d %H:%M:%S')
        d2 = datetime.datetime.strptime(eval(dates_range_list[i])[1],
                                        '%Y-%m-%d %H:%M:%S')
        diff = d2 - d1
        for i in range(diff.days + 1):
            dates_range.append(d1 + datetime.timedelta(i))
        
    return dates_range

def plot_index_with_market_cycles(data, 
                                   orig_data, 
                                   benchmark_index,
                                   criterion,
                                   window,
                                   from_date,
                                   to_date,
                                   name='market_cycles',
                                   save=False):
    
    markup = [key for key, value in data.items() if 'markup' in value]
    markdown = [key for key, value in data.items() if 'markdown' in value]
    accumulation = [key for key, value in data.items() if 'accumulation' in value]
    distribution = [key for key, value in data.items() if 'distribution' in value]
    
    
    markup=[orig_data[key] for key in [el for el in orig_data.keys() if el in markup]]
    markdown = [orig_data[key] for key in [el for el in orig_data.keys() if el in markdown]]
    accumulation = [orig_data[key] for key in [el for el in orig_data.keys() if el in accumulation]]
    distribution = [orig_data[key] for key in [el for el in orig_data.keys() if el in distribution]]

    markup_data = [markup[i].values() for i in range(len(markup))]
    markup_dates = [pd.to_datetime(list(markup[i].keys())) for i in range(len(markup))]
    
    markdown_data = [markdown[i].values() for i in range(len(markdown))]
    markdown_dates = [pd.to_datetime(list(markdown[i].keys())) for i in range(len(markdown))]
    accumulation_data = [accumulation[i].values() for i in range(
                len(accumulation)
                )]
    accumulation_dates = [pd.to_datetime(list(accumulation[i].keys())) for i in range(
                len(accumulation)
                )]
    distribution_data = [distribution[i].values() for i in range(
                len(distribution)
                )]
    distribution_dates = [pd.to_datetime(list(distribution[i].keys())) for i in range(
                len(distribution)
                )]
            
    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(len(markup_dates)):
        ax.plot(markup_dates[i], markup_data[i], 'green', lw=1)
    for i in range(len(markdown_dates)):
        ax.plot(markdown_dates[i], markdown_data[i], 'red', lw=1)
    for i in range(len(accumulation_dates)):
        ax.plot(accumulation_dates[i], accumulation_data[i], 'blue', lw=1)
    for i in range(len(distribution_dates)):
        ax.plot(distribution_dates[i], distribution_data[i], 'orange', lw=1)

    legend_elements = [Line2D([0], [0], color='green', lw=2, label='markup'),
                       Line2D([0], [0], color='red', lw=2, label='markdown'),
                       Line2D([0], [0], color='blue', lw=2, label='accumulation'),
                       Line2D([0], [0], color='orange', lw=2, label='distribution')]
    plt.legend(handles=legend_elements)
    plt.title(f'Market cycles based on "{criterion}" criterion for {benchmark_index} index\n'
              f'for period {from_date}-{to_date}, window = {window} months', fontsize=14)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-30)
    if save:
        plt.savefig(name+'.png')
    plt.show()

    
    
    
    
    
    
    
    
    