"""
Given graph of precincts/blocks, create similarity metric and apply persistent-homology based method to 
generate supercommunities.
"""

import json
import networkx as nx
from scipy.spatial import distance


def create_similarity_metric(graph):
    """
    Generates similarities between precincts.
    """
    for node1, node2 in graph.edges:
        print(node1, node2)
        data1 = graph.nodes(node1)
        data2 = graph.nodes(node2)
        race1 = [data1[race] for race in ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]]
        race2 = [data2[race] for race in ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]]
        race_distance = distance(race1, race2, 2)
        


def create_communities(graph_file, num_thresholds, verbose=False):
    """
    Generates supercommunities from precincts
    """
    if verbose:
        print("Loading geodata... ", end="")

    with open(graph_file, "r") as f:
        data = json.load(f)
    geodata = nx.readwrite.json_graph.adjacency_graph(data)

    if verbose:
        print("done!")

    