"""Miscellaneous utilities.
"""

import random

import networkx as nx

from . import constants


def copy_adjacency(graph):
    """Copies adjacency information from a graph but not attribute data.
    """
    copy_graph = nx.Graph()
    for node in graph.nodes():
        copy_graph.add_node(node)
    for u, v in graph.edges:
        copy_graph.add_edge(u, v)
    return copy_graph


def get_num_vra_districts(partition, minority, threshold):
    """Returns the number of minority-opportunity distrcts for a given minority and threshold.

    Parameters
    ----------
    partition : gerrychain.Parition
        Proposed district plan.
    minority : str
        Node data key that returns the population of that minority.
    threshold : float
        Value between 0 and 1 indicating the percent population required for a district to be
        considered minority opportunity.
    """
    num_vra_districts = 0
    for part in partition.parts:
        total_pop = 0
        minority_pop = 0
        for node in partition.parts[part]:
            total_pop += node["total_pop"]
            minority_pop += node[minority]
        if minority_pop / total_pop >= threshold:
            num_vra_districts += 1
    return num_vra_districts


def get_county_weighted_random_spanning_tree(graph):
    """Applies random edge weights to a graph, then multiplies those weights depending on whether or
    not the edge crosses a county border. Then returns the maximum spanning tree for the graph."""
    for u, v in graph.edges:
        weight = random.random()
        if graph[u]["COUNTYFP10"] == graph[v]["COUNTYFP10"]:
            weight *= constants.SAME_COUNTY_PENALTY
        graph[u][v]["random_weight"] = weight

    spanning_tree = nx.tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )
    return spanning_tree