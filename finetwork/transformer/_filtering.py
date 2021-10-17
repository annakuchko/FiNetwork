import networkx as nx
import planarity

class _FilterNetwork:
    def __init__(self,
                 G,
                 method='mst'):
        self.G = G
        self.method = method
        
    def _filter(self):
        G = self.G
        method = self.method
        
        if method == 'mst':
            result_G = self.compute_mst(G)
        
        elif method == 'pmfg':
            result_G = self.compute_PMFG(G)
            
        return result_G
        
    def sort_graph_edges(self, G):
        sorted_edges = []
        for source, dest, data in sorted(G.edges(data=True),
                                         key=lambda x: x[2]['weight']):
            sorted_edges.append({'source': source,
                                 'dest': dest,
                                 'weight': data['weight']})
        
        return sorted_edges
        
    def compute_mst(self, G):
        mst = nx.minimum_spanning_tree(G, weight='weight')
        return mst
        
    
    def compute_PMFG(self, G):
        sorted_edges = self.sort_graph_edges(G)
        nb_nodes =  len(G.nodes)
        PMFG = nx.Graph()
        for edge in sorted_edges:
            PMFG.add_edge(edge['source'], edge['dest'])
            if not planarity.is_planar(PMFG):
                PMFG.remove_edge(edge['source'], edge['dest'])
                
            if len(PMFG.edges()) == 3*(nb_nodes-2):
                break
        H = nx.Graph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(PMFG.edges(data=True))
    
        return H

