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


def create_communities(graph_file, num_thresholds, output_file, verbose=False):
    """
    Generates supercommunity map from geodata. Outputs a list of edges and their lifetimes.
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

    edge_lifetimes = {tuple(edge): None for edge in geodata.edges}
    communities = geodata.copy()  # Community dual graph
    for c1, c2 in communities.edges:
        communities.edges[c1, c2]["constituent_edges"] = {(c1, c2, geodata.edges[c1, c2]["similarity"])}

    # After each iteration, the current community map contains only borders constituted entirely of
    # edges with lower similarity than the threshold. This means it is possible for a single
    # community to be involved in multiple contractions during a single iteration.
    contractions = []  # Contains lists: [c1, c2, time], where time = 1 - threshold
    for t in range(num_thresholds + 1):
        threshold = 1 - (t / num_thresholds)
        # print(threshold)
        # print(len(communities.edges))
        # Implemented with nested loops because we don't want to iterate over communities.edges
        # while contractions are occurring. The next iteration of this loop is reached whenever a
        # contraction occurs.
        explored_edges = set()
        while len(explored_edges) < communities.number_of_edges():
            for c1, c2 in communities.edges:
                if frozenset((c1, c2)) not in explored_edges:
                    explored_edges.add(frozenset((c1, c2)))
                    contract = False
                    for _, _, similarity in communities.edges[c1, c2]["constituent_edges"]:
                        if similarity > threshold:
                            contract = True
                            break
                    if contract:
                        for edge in communities.edges[c1, c2]["constituent_edges"]:
                            edge_lifetimes[tuple(edge[:2])] = 1 - threshold

                        # Delete c2, add its edges to c1, and update constituent_edges sets.
                        for neighbor in communities[c2]:
                            if neighbor == c1:
                                continue
                            if neighbor in communities[c1]:
                                c_edges = communities.edges[c1, neighbor]["constituent_edges"].union(communities.edges[c2, neighbor]["constituent_edges"])
                            else:
                                c_edges = communities.edges[c2, neighbor]["constituent_edges"]
                            communities.add_edge(c1, neighbor, constituent_edges=c_edges)
                        contractions.append([c1, c2, 1 - threshold])
                        communities.remove_node(c2)
                        break  # communities.edges has changed. Continue to next iteration.
    # print(edge_lifetimes)
    for edge, lifetime in edge_lifetimes.items():
        if lifetime is None:
            print(geodata.edges[edge]["similarity"])
            raise ValueError(f"Something must have gone wrong. Edge {edge} never got removed.")

    output = {
        "contractions": contractions,
        "edge_lifetimes": {str(edge): lifetime for edge, lifetime in edge_lifetimes.items()}
    }

    with open(output_file, 'w+') as f:
        json.dump(output, f)