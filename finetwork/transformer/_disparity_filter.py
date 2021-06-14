from scipy.stats import percentileofscore
import networkx as nx
import numpy as np
import pandas as pd

## disparity filter for extracting the multiscale backbone of
## complex weighted networks

def get_nes(G, label):
    """
    find the neighborhood attention set (NES) for the given label
    """
    for node_id in G.nodes():
        node = G.nodes[node_id]

        if node["label"].lower() == label:
            return set([node_id]).union(set([id for id in G.neighbors(node_id)]))


def disparity_integral(x, k):
    """
    calculate the definite integral for the PDF in the disparity filter
    """
    assert x != 1.0, "x == 1.0"
    assert k != 1.0, "k == 1.0"
    return ((1.0 - x)**k) / ((k - 1.0) * (x - 1.0))


def get_disparity_significance(norm_weight, degree):
    """
    calculate the significance (alpha) for the disparity filter
    """
    return 1.0 - ((degree - 1.0) * (disparity_integral(norm_weight, degree) - disparity_integral(0.0, degree)))


def disparity_filter(G):
    """
    implements a disparity filter, based on multiscale backbone networks
    https://arxiv.org/pdf/0904.2389.pdf
    """
    alpha_measures = []
    
    for node_id in G.nodes():
        node = G.nodes[node_id]
        degree = G.degree(node_id)
        strength = 0.0

        for id0, id1 in G.edges(nbunch=[node_id]):
            edge = G[id0][id1]
            strength += edge["weight"]

        node["strength"] = strength

        for id0, id1 in G.edges(nbunch=[node_id]):
            edge = G[id0][id1]

            norm_weight = edge["weight"] / strength
            edge["norm_weight"] = norm_weight

            if degree > 1:
                try:
                    if norm_weight == 1.0:
                        norm_weight -= 0.0001

                    alpha = get_disparity_significance(norm_weight, degree)
                except AssertionError:
                    continue

                edge["alpha"] = alpha
                alpha_measures.append(alpha)
            else:
                edge["alpha"] = 0.0

    for id0, id1 in G.edges():
        edge = G[id0][id1]
        edge["alpha_ptile"] = percentileofscore(alpha_measures, edge["alpha"]) / 100.0

    return alpha_measures



def calc_centrality(G, min_degree=1):
    """
    to conserve compute costs, ignore centrality for nodes below `min_degree`
    """
    sub_graph = G.copy()
    sub_graph.remove_nodes_from([ n for n, d in list(G.degree) if d < min_degree ])

    centrality = nx.betweenness_centrality(sub_graph, weight="weight")

    return centrality


def calc_quantiles(metrics, num):
    """
    calculate `num` quantiles for the given list
    """

    bins = np.linspace(0, 1, num=num, endpoint=True)
    s = pd.Series(metrics)
    q = s.quantile(bins, interpolation="nearest")
    quantiles = []

    for idx, q_hi in q.iteritems():
        quantiles.append(q_hi)

    return quantiles


def calc_alpha_ptile (alpha_measures):
    """
    calculate the quantiles used to define a threshold alpha cutoff
    """
    quantiles = calc_quantiles(alpha_measures, num=10)
    num_quant = len(quantiles)


    return quantiles, num_quant


def cut_graph (G, min_alpha_ptile=0.5, min_degree=2):
    """
    apply the disparity filter to cut the given graph
    """
    filtered_set = set([])

    for id0, id1 in G.edges():
        edge = G[id0][id1]

        if edge["alpha_ptile"] < min_alpha_ptile:
            filtered_set.add((id0, id1))

    for id0, id1 in filtered_set:
        G.remove_edge(id0, id1)

    filtered_set = set([])

    for node_id in G.nodes():
        if G.degree(node_id) < min_degree:
            filtered_set.add(node_id)

    for node_id in filtered_set:
        G.remove_node(node_id)
        
    return G


def apply_disparity_filter(G, min_alpha_ptile=0.5, min_degree=2):
    
    alpha_measures = disparity_filter(G)
    quantiles, num_quant = calc_alpha_ptile(alpha_measures)
    G = cut_graph(G, min_alpha_ptile, min_degree)
    
    return G

