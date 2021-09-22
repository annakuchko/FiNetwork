import networkx as nx
# from matplotlib import patches
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

def _draw_adjacency_matrix(G, title=None, cmap='YlGnBu', save=True):
    fig, ax = plt.subplots(figsize=(15,15))
    node_order = list(G.nodes())
    adjacency_matrix = nx.to_numpy_matrix(G, nodelist=node_order)
    
    sns.heatmap(adjacency_matrix,
                cmap=cmap,
                yticklabels=node_order, xticklabels=node_order, ax=ax)
    ax.set_xticklabels(labels=node_order,rotation=90, fontsize=10)
    ax.set_yticklabels(labels=node_order,fontsize=10)
    ax.set_title(title, fontsize=14)
    if save:
        figname = f'test_figure_{str(datetime.datetime.now().time()).replace(":", "_")}'
        plt.savefig(f'{figname}.png',
                     bbox_inches="tight", 
                     pad_inches=0)
        plt.close()
    else:
        pass
    # plt.show()
    