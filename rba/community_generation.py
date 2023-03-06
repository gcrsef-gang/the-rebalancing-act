"""
Given graph of precincts/blocks, create similarity metric and apply persistent-homology based method to 
generate supercommunities.
"""

from itertools import combinations
import json
import math
import sys

from scipy.spatial import distance
from tqdm import tqdm
import networkx as nx
import numpy as np
import shapely
import matplotlib.pyplot as plt

import warnings
# warnings.filterwarnings("ignore")

from . import constants
from .util import copy_adjacency


def compute_precinct_similarities(graph, name=None, verbose=False):
    """Generates similarities between precincts. Edits graph in-place.

    Parameters
    ----------
    graph : nx.Graph
        Precinct dual graph with data for geometry, race, population, and county.
    name : str, default=None
        Used in title of histogram saved to the current working directory.
    """
    if verbose:
        print("Calculating log population densities...", end="")
        sys.stdout.flush()

    min_log_pop_density = None
    max_log_pop_density = None
    for node, data in graph.nodes(data=True):
        if data["total_pop"] == 0:
            log_pop_density = constants.MIN_POP_DENSITY
        else:
            if isinstance(data["geometry"]["coordinates"][0][0][0], float):
                geometry = shapely.geometry.Polygon(data["geometry"]["coordinates"][0])
            else:
                geomlist = [shapely.geometry.Polygon(data["geometry"]["coordinates"][i][0])
                            for i in range(len(data["geometry"]["coordinates"]))]
                geometry = shapely.ops.unary_union(geomlist)
            log_pop_density = math.log(data["total_pop"] / geometry.area)
        if min_log_pop_density is None or log_pop_density < min_log_pop_density:
            min_log_pop_density = log_pop_density
        elif max_log_pop_density is None or log_pop_density > max_log_pop_density:
            max_log_pop_density = log_pop_density
        graph.nodes[node]["log_pop_density"] = log_pop_density

    log_pop_density_range = max_log_pop_density - min_log_pop_density

    if verbose:
        print("done!")
        edges_iter = tqdm(graph.edges)
    else:
        edges_iter = graph.edges

    all_edges_distances = {}  # "metric": [dist for each edge]
    similarities = []
    for nodes in edges_iter:
        node1, node2 = nodes

        data1 = graph.nodes[node1]
        data2 = graph.nodes[node2]

        distances = {}

        race1 = np.array([data1[race] for race in constants.RACE_KEYS])
        race2 = np.array([data2[race] for race in constants.RACE_KEYS])
        distances["race"] = distance.jensenshannon(race1 / np.sum(race1), race2 / np.sum(race2), 2)

        votes1 = [data1[party] for party in constants.VOTE_KEYS]
        if data1["total_votes"] - sum(votes1) < 0:
            warnings.warn("total_dem + total rep > total_votes for", node1)
            votes1.append(0)  # total_other
        else:
            votes1.append(data1["total_votes"] - sum(votes1))  # total_other

        votes2 = [data2[party] for party in constants.VOTE_KEYS]
        if data2["total_votes"] - sum(votes2) < 0:
            warnings.warn("total_dem + total rep > total_votes for", node2)
            votes2.append(0)  # total_other
        else:
            votes2.append(data2["total_votes"] - sum(votes2))  # total_other
            
        # NOTE: votes are not included in the similarity score.
        # distances["votes"] = distance.jensenshannon(votes1 / np.sum(votes1), votes2 / np.sum(votes2), 2)

        # Put log pop densities on a 0-1 scale
        transformed_log_pop_1 = (data1["log_pop_density"] - min_log_pop_density) / log_pop_density_range
        transformed_log_pop_2 = (data2["log_pop_density"] - min_log_pop_density) / log_pop_density_range
        distances["pop_density"] = abs(transformed_log_pop_1 - transformed_log_pop_2)

        distances["county"] = 1 if data1["COUNTYFP10"] == data2["COUNTYFP10"] else 0

        distance_list = []
        weight_list = []
        for metric in distances:
            distance_list.append(distances[metric])
            weight_list.append(constants.SIMILARITY_WEIGHTS[metric])
        similarity = 1 - np.average(distance_list, weights=weight_list)

        graph.edges[node1, node2]["similarity"] = similarity
        for metric, dist in distances.items():
            if metric in all_edges_distances:
                all_edges_distances["metric"].append(dist)
            else:
                all_edges_distances["metric"] = [dist]
        similarities.append(similarity)

        if verbose:
            edges_iter.set_description(
                "\t".join([f"{distances[metric]} dist: {round(dist, 3)}" for metric, dist in distances.items()])
                + f"\tsimilarity: {round(similarity, 3)}                "
            )

    for metric, distances in all_edges_distances.items():
        plt.hist(distances, bins=50, label=f"{metric.capitalize()} Distance")
    plt.hist(similarities, bins=50, label="Similarity")
    plt.title(f"{name} Similarities")
    plt.legend()
    plt.savefig("similarities.png")
    plt.clf()

    if verbose:
        print()


def create_communities(graph_file, num_thresholds, output_file, use_similarities=False,
                       verbose=False):
    """Generates supercommunity map from geodata. Outputs a JSON dictionary mapping pairs of
    precincts to one minus the threshold at which they became part of the same community.

    Parameters
    ----------
    graph_file : str
        Path to geoJSON file with precinct geodata and information for similarity metrics.
    num_thresholds : int or NoneType
        The number of thresholds to examine between zero and one (number of iterations of the
        algorithm.) If set to None, the algorithm will have one iteration for every community merge
        it encounters, always merging the most similar pair. TODO: implement num_thresholds=None
    output_file : str
        Path to JSON file which will contain "difference scores" for each pair of precincts.
    use_similarities : bool, default=False
        Whether or not to use existing similarity values in the state geodata.
    """
    if verbose:
        print("Loading geodata... ", end="")
        sys.stdout.flush()

    with open(graph_file, "r") as f:
        data = json.load(f)
    graph = nx.readwrite.json_graph.adjacency_graph(data)

    if verbose:
        print("done!")

    if not use_similarities:
        if verbose:
            print("Calculating precinct similarities...", end="")
            sys.stdout.flush()

        compute_precinct_similarities(graph, verbose)
        new_data = nx.readwrite.json_graph.adjacency_data(graph)
        with open(graph_file, "w") as f:
            json.dump(new_data, f)

        if verbose:
            print("done!")
    elif verbose:
        print("Using pre-existing similarity values from geodata file.")

    # Contains the threshold at which each pair of nodes became part of the same community.
    node_differences = {frozenset((u, v)): None for u, v in combinations(graph.nodes, 2)}
    communities = copy_adjacency(graph)  # Community dual graph
    for c1, c2 in communities.edges:
        communities.edges[c1, c2]["constituent_edges"] = {frozenset((c1, c2))}
    for community in communities.nodes:
        communities.nodes[community]["constituent_nodes"] = {community}

    if num_thresholds is None:
        raise NotImplementedError("Infinite thresholds not yet implemented.")

    # After each iteration, the current community map contains only borders whose average similarity
    # is less than or equal to than the threshold. This means it is possible for a single community
    # to be involved in multiple contractions during a single iteration.
    if verbose:
        thresholds_iter = tqdm(range(1, num_thresholds + 1))
    else:
        thresholds_iter = range(1, num_thresholds + 1)
    for t in thresholds_iter:
        threshold = 1 - (t / num_thresholds)
        thresholds_iter.set_description(f"Current threshold: {round(threshold, 4)}   \r", end="")
        # Implemented with nested loops because we don't want to iterate over communities.edges
        # while contractions are occurring. The next iteration of this loop is reached whenever a
        # contraction occurs.
        explored_edges = set()
        while explored_edges.intersection(communities_edges := {frozenset(edge) for edge in communities.edges}) != communities_edges:
            for c1, c2 in communities.edges:
                if frozenset((c1, c2)) not in explored_edges:
                    explored_edges.add(frozenset((c1, c2)))
                    total_similarity = sum([graph.edges[u, v]["similarity"] for u, v
                                            in communities.edges[c1, c2]["constituent_edges"]])
                    num_constituent_edges = len(communities.edges[c1, c2]["constituent_edges"])
                    if total_similarity / num_constituent_edges > threshold:  # Perform contraction
                        # Add edges to c1 and update constituent_edges sets.
                        for neighbor in communities[c2]:
                            if neighbor == c1:
                                continue
                            if neighbor in communities[c1]:
                                c_edges = communities.edges[c1, neighbor]["constituent_edges"].union(
                                    communities.edges[c2, neighbor]["constituent_edges"])
                            else:
                                c_edges = communities.edges[c2, neighbor]["constituent_edges"]
                            communities.add_edge(c1, neighbor, constituent_edges=c_edges)

                        # Update node_differences, update node set of c1, and remove c2.
                        c1_nodes = communities.nodes[c1]["constituent_nodes"]
                        c2_nodes = communities.nodes[c2]["constituent_nodes"]
                        for u in c1_nodes:
                            for v in c2_nodes:
                                node_differences[frozenset((u, v))] = 1 - threshold
                        communities.nodes[c1]["constituent_nodes"] = c1_nodes.union(c2_nodes)
                        communities.remove_node(c2)

                        break  # communities.edges has changed. Continue to next iteration.
    
    for pair, lifetime in node_differences.items():
        if lifetime is None:
            raise ValueError(f"Something must have gone wrong. Pair {pair} was never merged.")

    with open(output_file, 'w+') as f:
        json.dump({str(tuple(pair)): t for pair, t in node_differences.items()}, f)