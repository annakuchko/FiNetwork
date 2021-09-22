import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
# TODO: add title

def plot_degree_distribution(G, plot_type='bar', scale=None, color='b'):
   degrees = list(dict(G.degree(weight='weight')).values())
   # degrees = [i[0] for i in degrees]
   # print(degrees)
   h = np.histogram(degrees, bins=len(degrees)) 
   freq = h[0]
   # print(freq)
   # print(np.log(freq))
   bins = [(h[1][i]+h[1][i+1])/2 for i in range(0,len(h[1])-1)]
   # print(np.log(bins))
   if plot_type == 'bar':
       plt.bar(bins, freq, color=color, alpha=0.3)
        
   elif plot_type == 'scatter':
       
       fig, ax1 = plt.subplots()
       if scale == 'log':
           ax1.set(xscale="log", yscale="log", xlim=min(bins))
       ax1.plot(bins, freq, 'x', color=color)
       # sns.scatterplot(bins, freq, ax=ax1, color=color)
       ax2 = ax1.twinx()
       ax2.set(xscale="linear", yscale="linear", xlim=min(bins))
       sns.kdeplot(degrees, ax=ax2, fill=True, alpha=.2, color=color)
   
   elif plot_type == 'connected':
       plt.plot(bins, freq, '-o', color=color)
       
   elif plot_type == 'density':
       plt.hist(degrees, density=True, bins=bins, alpha=0.3, log=True)
        
   if scale == 'log':
       plt.xlabel('Log degree')
        # print(min(np.log(bins)))
        # print(max(np.log(bins)))
        # plt.xlim((min(np.log(bins)), max(np.log(bins))))
       plt.ylabel('Log fraction of nodes')
       # plt.xscale('log')
       # plt.yscale('log')
   else:
       plt.xlabel('Degree')
       plt.ylabel('Fraction of nodes')
        
   # plt.show()
    
if __name__ == '__main__':

    G = nx.powerlaw_cluster_graph(100,9, 1)
    for (u,v,w) in G.edges(data=True):
        w['weight'] = abs(np.random.power(1))
    plot_degree_distribution(G, plot_type='scatter', scale='log') 
    # plot_degree_distribution(G, plot_type='density') 
    # plot_degree_distribution(G, plot_type='bar', scale='log') 
    
    
    # G = nx.powerlaw_cluster_graph(100,9, 1)
    # for (u,v,w) in G.edges(data=True):
    #     w['weight'] = abs(np.random.normal(0,1,1))+np.random.normal(0,1,1)
    # plot_degree_distribution(G, plot_type='scatter', scale='log', color='r') 
    
   
    # plot_degree_distribution(G, plot_type='bar')
    # plot_degree_distribution(G, plot_type='bar', scale='log')
    # plot_degree_distribution(G, plot_type='connected')
    