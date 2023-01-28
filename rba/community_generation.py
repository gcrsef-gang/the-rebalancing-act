"""
Given graph of precincts/blocks, create similarity metric and apply persistent-homology based method to 
generate supercommunities.
"""

from dataclasses import dataclass
import json
import math

import networkx as nx
import numpy as np
from gerrychain import Graph
from scipy.spatial import distance

import warnings
warnings.filterwarnings("ignore")



SIMILARITY_WEIGHTS = {
    "race": 1.0,
    "votes": 1.0,
    "pop_density": 1.0
}


@dataclass
class Community:
    precincts: frozenset
    birth_time: float
    death_time: float = -1


def compute_precinct_similarities(graph, verbose=False):
    """
    Generates similarities between precincts.
    """
    min_pop_density = 1e20
    max_pop_density = -1e20
    population_densities = {}
    for node, data in graph.nodes(data=True):
        if data["total_pop"] == 0:
            population_density = 0
        else:
            population_density = math.log(data["total_pop"]/data["ALAND10"])
        min_pop_density = min(min_pop_density, population_density)
        max_pop_density = max(max_pop_density, population_density)
        population_densities[node] = population_density

    for i, nodes in enumerate(graph.edges):
        node1, node2 = nodes

        if verbose:
            # TODO: write stats
            print(f"Computing similarity #: {i+1}/{len(graph.edges)}\tCurrent edge: ({node1}, {node2})\r", end="")

        data1 = graph.nodes[node1]
        data2 = graph.nodes[node2]
        
        race1 = np.array([data1[race] for race in ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]])
        race2 = np.array([data2[race] for race in ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]])
        race_distance = distance.jensenshannon(race1 / np.sum(race1), race2 / np.sum(race2), 2)

        votes1 = [data1[party] for party in ["total_rep", "total_dem"]]
        votes1.append(data1["total_votes"] - sum(votes1))  # total_other
        votes2 = [data2[party] for party in ["total_rep", "total_dem"]]
        votes2.append(data2["total_votes"] - sum(votes2))  # total_other
        votes_distance = distance.jensenshannon(votes1 / np.sum(votes1), votes2 / np.sum(votes2), 2)
        
        pop1 = (population_densities[node1]-min_pop_density)/(max_pop_density-min_pop_density)
        pop2 = (population_densities[node2]-min_pop_density)/(max_pop_density-min_pop_density)
        pop_density_distance = abs(pop1 - pop2)
        similarity = 1 - np.average(
            [race_distance, votes_distance, pop_density_distance],
            weights=[SIMILARITY_WEIGHTS["race"], SIMILARITY_WEIGHTS["votes"], SIMILARITY_WEIGHTS["pop_density"]])
        # print(similarity, race_distance, votes_distance, pop_density_distance)
        graph.edges[node1, node2]["similarity"] = similarity
    
    if verbose:
        print()

    return graph
        


def create_communities(graph_file, num_thresholds, verbose=False):
    """
    Generates supercommunities from precincts
    """
    if verbose:
        print("Loading geodata... ", end="")

    # geodata = Graph.from_json(graph_file)
    with open(graph_file, "r") as f:
        data = json.load(f)
    geodata = nx.readwrite.json_graph.adjacency_graph(data)

    if verbose:
        print("done!")
        print("Calculating precinct similarities...")

    graph = compute_precinct_similarities(geodata, verbose)

    if verbose:
        print("done!")

    # communities = set()
    # for t in num_thresholds:
    #     threshold = t / num_thresholds

    #     for node1, node2 in geodata.edges:
    #         if geodata.edges[node1, node2]["similarity"] < threshold:
    #             geodata.remove_edge(node1, node2)
        
    #     for component in nx.connected_components(geodata):
    #         communities.add(frozenset(component))
