"""Miscellaneous utilities.
"""

import json
import random

import networkx as nx

from . import constants
from .district_quantification import quantify_gerrymandering


def copy_adjacency(graph):
    """Copies adjacency information from a graph but not attribute data.
    """
    copy_graph = nx.Graph()
    for node in graph.nodes():
        copy_graph.add_node(node)
    for u, v in graph.edges:
        copy_graph.add_edge(u, v)
    return copy_graph


def get_num_vra_districts(partition, label, threshold):
    """Returns the number of minority-opportunity distrcts for a given minority and threshold.

    Parameters
    ----------
    partition : gerrychain.Parition
        Proposed district plan.
    label : str
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
            total_pop += partition.graph.nodes[node]["total_pop"]
            if label == "total_combined":
                for minority in constants.MINORITY_NAMES:
                    minority_pop += partition.graph.nodes[node][f"total_{minority}"]
            else:
                minority_pop += partition.graph.nodes[node][label]
        if minority_pop / total_pop >= threshold:
            num_vra_districts += 1
    return num_vra_districts


def get_county_weighted_random_spanning_tree(graph):
    """Applies random edge weights to a graph, then multiplies those weights depending on whether or
    not the edge crosses a county border. Then returns the maximum spanning tree for the graph."""
    for u, v in graph.edges:
        weight = random.random()
        if graph.nodes[u]["COUNTYFP10"] == graph.nodes[v]["COUNTYFP10"]:
            weight *= constants.SAME_COUNTY_PENALTY
        graph[u][v]["random_weight"] = weight

    spanning_tree = nx.tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="random_weight"
    )
    return spanning_tree


def save_assignment(partition, fpath):
    """Saves a partition's node assignment data to a file.
    """
    assignment = {}
    for u in partition.graph.nodes:
        assignment[u] = partition.assignment[u]
    with open(fpath, "w+") as f:
        json.dump(assignment, f)