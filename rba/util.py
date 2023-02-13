from scipy.special import rel_entr
import networkx as nx


def jenson_shannon_divergence(distribution1, distribution2):
    average = [(distribution1[i] + distribution2[i])/2 for i in range(distribution1)]


def copy_adjacency(graph):
    """Copies adjacency information from a graph but not attribute data.
    """
    copy_graph = nx.Graph()
    for node in graph.nodes():
        copy_graph.add_node(node)
    for u, v in graph.edges:
        copy_graph.add_edge(u, v)
    return copy_graph