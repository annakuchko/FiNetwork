import matplotlib.pyplot as plt
from finetwork.plotter._plot_circus import _CircosPlot
from finetwork.transformer import NetTransformer, to_graph
import datetime
from pathlib import Path

class Plotter:
    def __init__(self, 
                 data, 
                 plot_type, 
                 params={}, 
                 partition=None, 
                 name=None, 
                 path=None):
        self.data = data
        self.plot_type = plot_type
        self.params = params
        self.partition = partition
        self.path = path
        self._count = 1
        self._pos = None
        self._lim = None
        self._lm = None
        self._order = None
        
        
        if name != None:
            self.name = name
        else:
            self.name = str(datetime.datetime.now().time()).replace(":", "_")
    
    def plot(self):
        plot_type = self.plot_type
        params = self.params
        data = self.data
        for period in data.keys():
            if plot_type == 'circus':
                self.plot_circus(data[period], period, params)
            
            elif plot_type == 'adjacency':
                self.plot_adjacency(data[period], params, self._count)
            
            elif plot_type == 'spectrum':
                self.plot_spectrum(data[period], params, self._count)
            
            elif plot_type == 'degree':
                self.plot_degree(data[period], params, self._count)
            self._count += 1
            
    def plot_circus(self, data, period, params=None):
        params = self.params
        partition = self.partition[period]
        path = self.path
        if self._count==1:
            cp = _CircosPlot(
                data,
                node_labels=True,
                partition=partition
                )
            self._pos = cp._compute_positions()
            self._lim = cp._get_lim_orig()
            self._lm = cp._get_label_meta()
            self._order = cp._get_label_order()
        else:
            cp = _CircosPlot(
                data, 
                node_labels=True,
                nodes_pos=self._pos,
                limit_orig=self._lim,
                label_meta=self._lm,
                partition=partition,
                label_order=self._order
                )
                    
        figname=f'test_figure_{self.name}_10{self._count}'
        if path != None:
            figname = str(path) + '/' + figname
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            figname = 'tmp/' + figname
            Path("tmp/").mkdir(parents=True, exist_ok=True)
            
        cp._draw(
            save=True,
            figname=figname,
            title=f'Graph for period {period}'
            )