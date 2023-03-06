"""
Given graph of precincts/blocks, create similarity metric and apply persistent-homology based method to 
generate supercommunities.
"""

from dataclasses import dataclass
from itertools import combinations
import json
import math

from scipy.spatial import distance
from tqdm import tqdm
import networkx as nx
import numpy as np
import shapely
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from .constants import SIMILARITY_WEIGHTS
from .util import copy_adjacency


def compute_precinct_similarities(graph, verbose=False):
    """
    Generates similarities between precincts. Edits graph in-place.
    """
    min_pop_density = 1e20
    max_pop_density = -1e20
    population_densities = {}
    for node, data in graph.nodes(data=True):
        if data["total_pop"] == 0:
            population_density = 0
        else:
            # population_density = math.log(data["total_pop"]/data["ALAND10"])
            # print(data["geometry"]["coordinates"])
            if isinstance(data["geometry"]["coordinates"][0][0][0], float):
                geometry = shapely.geometry.Polygon(data["geometry"]["coordinates"][0])
            else:
                geomlist = [shapely.geometry.Polygon(data["geometry"]["coordinates"][i][0]) for i in range(len(data["geometry"]["coordinates"]))]
                geometry = shapely.ops.unary_union(geomlist)
            # print(geometry.type)
            population_density = math.log(data["total_pop"]/geometry.area)
        min_pop_density = min(min_pop_density, population_density)
        max_pop_density = max(max_pop_density, population_density)
        population_densities[node] = population_density

    race_distances = []
    votes_distances = []
    pop_density_distances = []
    similarities = []
    for i, nodes in enumerate(graph.edges):
        node1, node2 = nodes
        if verbose:
            # TODO: write stats
            print(f"Computing similarity #: {i+1}/{len(graph.edges)}\tCurrent edge: ({node1}, {node2})" + " " * 10 + "\r", end="")

        data1 = graph.nodes[node1]
        data2 = graph.nodes[node2]
        
        race1 = np.array([data1[race] for race in ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]])
        race2 = np.array([data2[race] for race in ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]])
        race_distance = distance.jensenshannon(race1 / np.sum(race1), race2 / np.sum(race2), 2)

        votes1 = [data1[party] for party in ["total_rep", "total_dem"]]
        if data1["total_votes"] - sum(votes1) < 0:
            print("SUS other votes:", data1["total_votes"] - sum(votes1))
            votes1.append(0)  # total_other
        else:
            votes1.append(data1["total_votes"] - sum(votes1))  # total_other

        votes2 = [data2[party] for party in ["total_rep", "total_dem"]]
        if data2["total_votes"] - sum(votes2) < 0:
            print("SUS other votes:", data2["total_votes"] - sum(votes2))
            votes2.append(0)  # total_other
        else:
            votes2.append(data2["total_votes"] - sum(votes2))  # total_other
            
        votes_distance = distance.jensenshannon(votes1 / np.sum(votes1), votes2 / np.sum(votes2), 2)
            # print(votes1, votes2, votes_distances, "SUS VOTES")
        pop1 = (population_densities[node1]-min_pop_density)/(max_pop_density-min_pop_density)
        pop2 = (population_densities[node2]-min_pop_density)/(max_pop_density-min_pop_density)
        pop_density_distance = abs(pop1 - pop2)

        similarity = 1 - np.average(
            [race_distance, pop_density_distance],
            weights=[SIMILARITY_WEIGHTS["race"], SIMILARITY_WEIGHTS["pop_density"]])
        # similarity = 1 - np.average(
        #     [race_distance, votes_distance, pop_density_distance],
        #     weights=[SIMILARITY_WEIGHTS["race"], SIMILARITY_WEIGHTS["votes"], SIMILARITY_WEIGHTS["pop_density"]])
        # print(similarity, race_distance, votes_distance, pop_density_distance)
        graph.edges[node1, node2]["similarity"] = similarity
        race_distances.append(race_distance)
        votes_distances.append(votes_distance)
        pop_density_distances.append(pop_density_distance)
        similarities.append(similarity)
    plt.hist(race_distances, label="race", bins=50)
    plt.hist(votes_distances, label="votes", bins=50)
    plt.hist(pop_density_distances, label="pop_density", bins=50)
    plt.hist(similarities, bins=50)
    plt.legend()
    plt.savefig("maryland_similarities.png")
    # plt.savefig("race_distances_distribution.png")
    # plt.clear()
    if verbose:
        print()


def create_communities(graph_file, num_thresholds, output_file, verbose=False):
    """Generates supercommunity map from geodata. Outputs a JSON dictionary mapping pairs of
    precincts to the threshold at which they joined the same community.
    """
    if verbose:
        print("Loading geodata... ", end="")

    with open(graph_file, "r") as f:
        data = json.load(f)
    geodata = nx.readwrite.json_graph.adjacency_graph(data)
    if verbose:
        print("done!")
        print("Calculating precinct similarities...")

    compute_precinct_similarities(geodata, verbose)
    new_data = nx.readwrite.json_graph.adjacency_data(geodata)
    with open(graph_file, "w") as f:
        json.dump(new_data, f)

    if verbose:
        print("done!")

    # Contains the threshold at which each pair of nodes became part of the same community.
    node_differences = {frozenset((u, v)): None for u, v in combinations(geodata.nodes, 2)}
    communities = copy_adjacency(geodata)  # Community dual graph
    for c1, c2 in communities.edges:
        communities.edges[c1, c2]["constituent_edges"] = {(c1, c2, geodata.edges[c1, c2]["similarity"])}
    for community in communities.nodes:
        communities.nodes[community]["constituent_nodes"] = {community}

    # After each iteration, the current community map contains only borders constituted whose
    # average similarity is less than or equal to than the threshold. This means it is possible for
    # a single community to be involved in multiple contractions during a single iteration.
    for t in range(num_thresholds + 1):
        print(f"Current threshold: {t}/{num_thresholds+1}\r", end="")
        threshold = 1 - (t / num_thresholds)
        # Implemented with nested loops because we don't want to iterate over communities.edges
        # while contractions are occurring. The next iteration of this loop is reached whenever a
        # contraction occurs.
        explored_edges = set()
        while explored_edges.intersection(communities_edges := {frozenset(edge) for edge in communities.edges}) != communities_edges:
            for c1, c2 in communities.edges:
                if frozenset((c1, c2)) not in explored_edges:
                    explored_edges.add(frozenset((c1, c2)))
                    total_similarity = sum([similarity for _, _, similarity
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
            raise ValueError(f"Something must have gone wrong. Pair {pair} was never .")

    with open(output_file, 'w+') as f:
        json.dump({str(tuple(pair)): t for pair, t in node_differences.items()}, f)