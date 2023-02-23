"""
Uses a recombination Markov Chain Monte Carlo method to generate an ensemble of districting plans
for a state. Enforces population equality, VRA compliance, and biases towards following county lines.
"""

from functools import partial
import json
import random
from collections import defaultdict
import statistics

import matplotlib.pyplot as plt
import networkx as nx
from gerrychain import Partition, Graph, MarkovChain, updaters, constraints, accept
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.random import random
import pandas as pd

from rba import constants
from rba.district_quantification import quantify_gerrymandering, quantify_districts
from rba.util import get_num_vra_districts, get_county_weighted_random_spanning_tree
from rba.visualization import visualize_partition_geopandas, visualize_metric

# CONSTANTS

# random.seed(2023)
# GEODATA_FILE = "../rba/data/2010/new_hampshire_geodata_merged.json"
# COMMUNITY_OUTPUT_FILE = "../rba/data/2010/new_hampshire_communities.json"
# VRA_CONFIG_FILE = "../rba/data/2010/vra_nh.json"
# NUM_DISTRICTS = 2



# UPDATERS
def create_updaters(edge_lifetimes, vra_config, vra_threshold):
    rba_updaters = {
        "population": updaters.Tally("total_pop", alias="population"),
        "gerry_scores": lambda partition: quantify_gerrymandering(
            partition.graph,
            {dist: subgraph for dist, subgraph in partition.subgraphs.items()},
            edge_lifetimes
        )
    }

    vra_updaters = {f"num_{minority}_vra_districts": partial(get_num_vra_districts,
                                                            label=f"total_{minority}",
                                                            threshold=vra_threshold)
                    for minority in vra_config.keys()}

    rba_updaters.update(vra_updaters)

    return rba_updaters


def create_constraints(initial_partition, vra_config):
    # CONSTRAINTS

    # NOTE: we said we wouldn't have a compactness constraint but GerryChain uses one in their example
    # showing that maybe it's necessary even for ReCom. This keeps the proposals within 2x the number of
    # cut edges in the starting one.
    # compactness_bound = constraints.UpperBound(
    #     lambda p: len(p["cut_edges"]),
    #     2 * len(initial_partition["cut_edges"])
    # )
    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition,
                                                                    constants.POP_EQUALITY_THRESHOLD)

    vra_constraints = [
        constraints.LowerBound(
            lambda p: p[f"num_{minority}_vra_districts"],
            num_districts
        )
        for minority, num_districts in vra_config.items()]
    all_constraints = [
        pop_constraint,
        # compactness_bound
    ] + vra_constraints
    return all_constraints


def markov_chain(graph_file, community_file, vra_config_file, initial_assignment, num_districts, verbose=False):
    """
    Conduct the ensemble analysis for a state.
    """
    # LOADING DATA

    with open(graph_file, "r") as f:
        data = json.load(f)
    nx_graph = nx.readwrite.json_graph.adjacency_graph(data)
    graph = Graph.from_networkx(nx_graph)
    del nx_graph

    with open(community_file, "r") as f:
        community_data = json.load(f)

    edge_lifetimes = {}
    for edge, lifetime in community_data["edge_lifetimes"].items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        edge_lifetimes[(u, v)] = lifetime

    with open(vra_config_file, "r") as f:
        vra_config = json.load(f)

    vra_threshold = vra_config["opportunity_threshold"]
    del vra_config["opportunity_threshold"]

    rba_updaters = create_updaters(edge_lifetimes, vra_config, vra_threshold)

    # INITIAL STATE

    state_population = 0
    for node in graph:
        state_population += graph.nodes[node]["total_pop"]
    ideal_population = state_population / num_districts

    # initial_assignment = recursive_tree_part(
    #     graph, range(num_districts),
    #     pop_target=ideal_population,
    #     pop_col="total_pop",
    #     epsilon=constants.POP_EQUALITY_THRESHOLD)

    initial_partition = Partition(graph, initial_assignment, rba_updaters)
    for part in initial_partition.parts:
        pop_sum = 0
        for node in initial_partition.parts[part]:
            pop_sum += graph.nodes[node]["total_pop"]
        print(part, pop_sum)
    visualize_partition_geopandas(initial_partition)

    # PROPOSAL METHOD

    weighted_recom_proposal = partial(
        recom,
        pop_col="total_pop",
        pop_target=ideal_population,
        epsilon=constants.POP_EQUALITY_THRESHOLD,
        node_repeats=2,
        method=partial(
            bipartition_tree,
            spanning_tree_fn=get_county_weighted_random_spanning_tree)
    )

    # recom_proposal = partial(recom,
    #     pop_col="total_pop",
    #     pop_target=ideal_population,
    #     epsilon=constants.POP_EQUALITY_THRESHOLD,
    #     node_repeats=2
    # )

    all_constraints = create_constraints(initial_partition, vra_config)

    chain = MarkovChain(
        # proposal=recom_proposal,
        proposal=weighted_recom_proposal,
        constraints=all_constraints,
        # accept=lambda p: random.random() < get_county_border_proportion(p),
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=15
    )

    df = pd.DataFrame(columns=[f"district{i}" for i in range(1, num_districts + 1)] + ["state_gerry_score"], dtype=float)

    saved_partitions = []
    precinct_scores = {precinct : [] for precinct in initial_partition.graph.nodes}
    for i, partition in enumerate(chain.with_progress_bar()):
        district_scores, state_score = partition["gerry_scores"]
        assignment = partition.assignment
        # print(district_scores)
        # print(assignment)
        for j, node in enumerate(partition.graph.nodes):
            # if j < 3:
                # print(node, assignment[node], district_scores[assignment[node]])
            precinct_scores[node].append(district_scores[assignment[node]])
            if i == 0:
                graph.nodes[node]["precinct_scores"] = [district_scores[assignment[node]]]
            else:
                graph.nodes[node]["precinct_scores"].append(district_scores[assignment[node]])
        df.loc[len(df.index)] = sorted(list(district_scores.keys())) + [state_score]
        if i % 5 == 0:
            saved_partitions.append(partition)

            for i, partition in enumerate(saved_partitions):
                visualize_partition_geopandas(partition, i=i)  # TODO: add titles for what index in the chain each image is from.
    # print(precinct_scores)
    plt.hist(df["state_gerry_score"], bins=10)
    plt.show()
    plt.savefig("ensemble_analysis_results.png")

    for i, partition in enumerate(saved_partitions):    
        visualize_partition_geopandas(partition)  # TODO: add titles for what index in the chain each image is from.
    return graph


def ensemble_analysis(graph_file, community_file, vra_config_file, district_file, verbose=False):
    """Conducts a geographic ensemble analysis of a state's gerrymandering.
    """
    districts, district_scores, state_score = quantify_districts(graph_file, district_file, community_file, verbose)
    assignment = {}
    for district, node_list in districts.items():
        for node in node_list:
            assignment[node] = district
    graph = markov_chain(graph_file, community_file, vra_config_file, assignment, len(districts), verbose)
    for precinct in graph.nodes:
        scores = graph.nodes[precinct]["precinct_scores"]
        mean_score = sum(scores)/len(scores) 
        stdev = statistics.stdev(scores)
        real_score = district_scores[assignment[precinct]]
        z_score = (real_score-mean_score)/stdev
        print(scores, mean_score, stdev, real_score, z_score)
        graph.nodes[precinct]["z_score"] = z_score
        graph.nodes[precinct]["distribution_score"] = mean_score
        graph.nodes[precinct]["real_score"] = real_score
    # print(graph.nodes[graph.nodes[0]])
    visualize_metric("geographic_ensemble_analysis.png", graph, "z_score")
