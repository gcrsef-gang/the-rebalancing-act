"""
Uses a recombination Markov Chain Monte Carlo method to generate an ensemble of districting plans
for a state. Enforces population equality, VRA compliance, and biases towards following county lines.
"""

from dataclasses import dataclass
from functools import partial
import json
import math
import os
import pickle
import random
import sys
import statistics
import time
import warnings

from gerrychain import Partition, Graph, MarkovChain, updaters, constraints, accept
from gerrychain.constraints import Validator
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part, bipartition_tree
from tqdm import tqdm
from welford import Welford
import gerrychain.random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from rba import constants
from rba.district_quantification import quantify_gerrymandering, quantify_districts
from rba.util import (get_num_vra_districts, get_county_weighted_random_spanning_tree,
                      get_county_spanning_forest, choose_cut, create_folder, load_districts)
from rba.visualization import (visualize_gradient_geopandas, visualize_partition_geopandas,
                               visualize_metric)


class RBAMarkovChain(MarkovChain):
    """Markov Chain for The Rebalancing Act. Changes to gerrychain.MarkovChain:
    - Ignores validity of initial plan.
    - Stops chain if first or second partitions have taken more than 10 seconds.
    """

    ALLOWED_SECOND_STEP_DURATION = 10


    def __init__(self, proposal, constraints, accept, initial_state, total_steps):
        try:
            super().__init__(proposal, constraints, accept, initial_state, total_steps)
        except ValueError as e:
            if "initial_state" in repr(e):
                warnings.warn("GerryChain error was ignored: " + repr(e))
                # Initialize with no constraints.
                super().__init__(proposal, [], accept, initial_state, total_steps)
                if callable(constraints):
                    self.is_valid = constraints
                else:
                    self.is_valid = Validator(constraints)
            else:
                raise e
            
    def get_proposal_and_acceptance(self):
        """Produces a proposal and returns it along with whether or not it is valid and accepted.
        """
        proposed_next_state = self.proposal(self.state)
        # Erase the parent of the parent, to avoid memory leak
        if self.state is not None:
            self.state.parent = None

        acceptance = self.is_valid(proposed_next_state) and self.accept(proposed_next_state)
        return proposed_next_state, acceptance

    def __next__(self):
        if self.counter == 0:
            self.counter += 1
            return self.state

        start_time = time.time()
        if self.counter == 1:
            keep_going = lambda: time.time() - start_time < RBAMarkovChain.ALLOWED_SECOND_STEP_DURATION
        else:
            keep_going = lambda: True
        while self.counter < self.total_steps and keep_going():
            proposal, accepted = self.get_proposal_and_acceptance()
            if accepted:
                self.state = proposal
                self.counter += 1
                return self.state

        if self.counter == 1 and not keep_going():
            raise TimeoutError("Waited more than 10 seconds for a valid and accepted first proposal.")

        raise StopIteration


@dataclass
class SimplePartition:
    """Only stores parts and assignment for easy pickling.
    """
    parts: dict
    assignment: dict


# UPDATERS
def create_updaters(differences, vra_config, vra_threshold):
    rba_updaters = {
        "population": updaters.Tally("total_pop", alias="population"),
        "gerry_scores": lambda partition: quantify_gerrymandering(
            partition.graph,
            {dist: subgraph for dist, subgraph in partition.subgraphs.items()},
            differences
        )
    }

    vra_updaters = {f"num_{minority}_vra_districts": partial(get_num_vra_districts,
                                                            label=f"total_{minority}",
                                                            threshold=vra_threshold)
                    for minority in vra_config.keys()}

    rba_updaters.update(vra_updaters)

    return rba_updaters


def create_constraints(initial_partition, vra_config):
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


def generate_ensemble(graph, node_differences, num_vra_districts, vra_threshold,
                      pop_equality_threshold, num_steps, num_districts, initial_assignment=None,
                      output_dir=None, verbose=False):
    """Conduct the ensemble analysis for a state. Data is returned, but all partitions are saved
    to output_dir (not wasting memory).

    Parameters
    ----------
    graph : gerrychain.Graph
        The state graph of precincts.
    node_differences : dict
        Maps edges (tuples of precinct IDs)
    num_vra_districts : dict
        Maps the name of each minority to the minimum number of VRA districts required for it.
    vra_threshold : float
        Between 0 and 1. The minimum percentage required to consider a district
        "minority opportunity."
    pop_equality_threshold : float
        Between 0 and 1. The allowed percent deviation allowed between the population of any two
        districts.
    num_steps : int
        The number of iterations to run the markov chain for.
    num_districts : int
        The number of districts to partition the state into.
    initial_assignment : dict, default=None
        Maps nodes to districts for the initial plan in the Markov chain. If set to None, a random
        partition is used.
    output_dir : str
        Path to directory where every partition produced will be saved (as pickle files). In each
        pickle, a tuple is stored containing the gerrychain.Partition object as well as the order
        of the districts based on gerrymandering score (for matching up with the dataframe).
    verbose : boolean, default=False
        Controls verbosity.

    Returns
    -------
    df : pandas.DataFrame
        Contains gerrymandering scores of the state and all the districts for each step in the
        Markov Chain.
    """
    rba_updaters = create_updaters(node_differences, num_vra_districts, vra_threshold)

    state_population = 0
    for node in graph:
        state_population += graph.nodes[node]["total_pop"]
    ideal_population = state_population / num_districts

    # weighted_recom_proposal = partial(
    #     recom,
    #     pop_col="total_pop",
    #     pop_target=ideal_population,
    #     epsilon=pop_equality_threshold,
    #     node_repeats=6,
    #     # method=partial(
    #     #     bipartition_tree,
    #     #     spanning_tree_fn=get_county_weighted_random_spanning_tree)
    #     method=partial(
    #         bipartition_tree,
    #         spanning_tree_fn=get_county_spanning_forest,
    #         choice=partial(choose_cut, graph=graph))
    # )

    recom_proposal = partial(recom,
        pop_col="total_pop",
        pop_target=ideal_population,
        epsilon=pop_equality_threshold,
        node_repeats=2
    )

    restarted = False
    while True:
        try:

            seed = int(time.time()) % 1000
            if verbose:
                print(f"Setting seed to {seed}")
            gerrychain.random.random.seed(seed)
            random.seed(seed)

            if initial_assignment is None or restarted:
                if verbose:
                    print("Creating random initial partition...", end="")
                    sys.stdout.flush()
                initial_assignment = recursive_tree_part(
                    graph, range(num_districts),
                    pop_target=ideal_population,
                    pop_col="total_pop",
                    epsilon=constants.POP_EQUALITY_THRESHOLD)
                if verbose:
                    print("done!")

            initial_partition = Partition(graph, initial_assignment, rba_updaters)

            all_constraints = create_constraints(initial_partition, num_vra_districts)

            chain = RBAMarkovChain(
                proposal=recom_proposal,
                # proposal=weighted_recom_proposal,
                constraints=all_constraints,
                # accept=lambda p: random.random() < get_county_border_proportion(p),
                accept=accept.always_accept,
                initial_state=initial_partition,
                total_steps=num_steps
            )

            scores_df = pd.DataFrame(columns=[f"district {i}" for i in range(1, num_districts + 1)] + ["state_gerry_score"], dtype=float)

            if output_dir is not None:
                create_folder(output_dir)
                create_folder(os.path.join(output_dir, "plans"))

            if verbose:
                print("Running Markov chain...")
                sys.stdout.flush()
                chain_iter = chain.with_progress_bar()
            else:
                chain_iter = chain
            for i, partition in enumerate(chain_iter, start=1):
                district_scores, state_score = partition["gerry_scores"]
                districts_order = sorted(list(district_scores.keys()), key=lambda d: district_scores[d])
                scores_df.loc[len(scores_df.index)] = [district_scores[d] for d in districts_order] + [state_score]
                if output_dir is not None:
                    with open(os.path.join(output_dir, "plans", f"{i}.pickle"), "wb+") as f:
                        pickle.dump((SimplePartition(partition.parts, partition.assignment), districts_order), f)
        
            break

        except TimeoutError:
            print("This initial partition wasn't going anywhere... trying a new one...")
            restarted = True

    return scores_df


def ensemble_analysis(graph_file, difference_file, vra_config_file, num_steps, num_districts,
                      initial_plan_file, district_file, output_dir, optimize_vis=False, vis_dir=None, verbose=False):
    """Conducts a geographic ensemble analysis of a state's gerrymandering.
    """

    if verbose:
        print("Loading precinct graph...", end="")
        sys.stdout.flush()

    with open(graph_file, "r") as f:
        data = json.load(f)
    nx_graph = nx.readwrite.json_graph.adjacency_graph(data)
    graph = Graph.from_networkx(nx_graph)
    del nx_graph

    if verbose:
        print("done!")
        print("Loading community algorithm output...", end="")
        sys.stdout.flush()

    with open(difference_file, "r") as f:
        difference_data = json.load(f)

    node_differences = {}
    for edge, lifetime in difference_data.items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        node_differences[(u, v)] = lifetime

    if verbose:
        print("done!")
        print("Loading VRA requirements...", end="")
        sys.stdout.flush()

    with open(vra_config_file, "r") as f:
        vra_config = json.load(f)
    vra_threshold = vra_config["opportunity_threshold"]
    del vra_config["opportunity_threshold"]

    if verbose:
        print("done!")

    if initial_plan_file is not None:
        if verbose:
            print("Loading starting map...", end="")
            sys.stdout.flush()

        initial_plan_node_sets = load_districts(graph, initial_plan_file, verbose)
        initial_assignment = {}
        for district, nodes in initial_plan_node_sets.items():
            for node in nodes:
                initial_assignment[node] = district
        
        if verbose:
            print("done!")
    else:
        if verbose:
            print("No starting map provided. Will generate a random one later.")
        initial_assignment = None

    if not optimize_vis:
        scores_df = generate_ensemble(graph, node_differences, vra_config, vra_threshold,
                                      constants.POP_EQUALITY_THRESHOLD, num_steps, num_districts,
                                      initial_assignment, output_dir, verbose)
        scores_df.to_csv(os.path.join(output_dir, "scores.csv"))
    else:
        # IN CASE THIS HAS ALREADY BEEN RUN AND WE WANT TO REGENERATE MAPS AND PLOTS, UNCOMMENT THE
        # BLOCK OF CODE BELOW AND COMMENT OUT THE BLOCK OF CODE ABOVE.
        scores_df = pd.DataFrame(columns=[f"district {i}" for i in range(1, num_districts + 1)] + ["state_gerry_score"], dtype=float)
        print("Re-calculating scores from existing ensemble")
        for step in tqdm(range(num_steps)):
            with open(os.path.join(output_dir, "plans", f"{step + 1}.pickle"), "rb") as f:
                partition, _ = pickle.load(f)
            district_scores, state_score = quantify_gerrymandering(
                graph,
                partition.parts,
                node_differences
            )
            districts_order = sorted(list(district_scores.keys()), key=lambda d: district_scores[d])
            scores_df.loc[len(scores_df.index)] = [district_scores[d] for d in districts_order] + [state_score]
            with open(os.path.join(output_dir, "plans", f"{step + 1}.pickle"), "wb") as f:
                partition = pickle.dump((partition, districts_order), f)
        scores_df.to_csv(os.path.join(output_dir, "scores.csv"))

        # IN CASE THIS HAS ALREADY BEEN RUN AND WE JUST WANT TO GENERATE THE FINAL THREE VISUALS FOR 
        # A GIVEN DISTRICT MAP, UNCOMMENT THE BLOCK OF CODE BELOW, AND COMMENT OUT THE SAME BLOCK OF
        # CODE AS MENTIONED ABOVE.
        print("Using existing partitions and scores.")
        scores_df = pd.read_csv(os.path.join(output_dir, "scores.csv"))

    create_folder(os.path.join(output_dir, "visuals"))

    if verbose:
        print("Calculating precinct-level statistics and visualizing partitions...")

    sorted_node_names = sorted(list(graph.nodes))
    # "score" is referring to the "goodness" score and "homogeneity" is referring to the homogeneity
    # metric (in this case standard deviation of republican vote share). This uses Welford's
    # algorithm to calculate mean and variance one at a time instead of saving all the values to
    # memory (there will be num_steps * len(graph.nodes) values. that is a lot.)
    score_accumulator = Welford()
    homogeneity_accumulator = Welford()
    if verbose:
        step_iter = tqdm(range(num_steps))
    else:
        step_iter = range(num_steps)
    for i in step_iter:
        with open(os.path.join(output_dir, "plans", f"{i + 1}.pickle"), "rb") as f:
            partition, district_order = pickle.load(f)

        part_values = {}  # part: (score, homogeneity)
        for part in partition.parts:
            score = scores_df.loc[i, f"district {district_order.index(part) + 1}"]
            homogeneity = statistics.stdev(
                [graph.nodes[node]["total_rep"] / graph.nodes[node]["total_votes"]
                 for node in partition.parts[part]]
            )
            part_values[part] = (score, homogeneity)
        score_sample = np.zeros((len(sorted_node_names),))
        homogeneity_sample = np.zeros((len(sorted_node_names),))
        for j, precinct in enumerate(sorted_node_names):
            score_sample[j] = part_values[partition.assignment[precinct]][0]
            homogeneity_sample[j] = part_values[partition.assignment[precinct]][1]
        score_accumulator.add(score_sample)
        homogeneity_accumulator.add(homogeneity_sample)

        # Visualize 100 partitions, or however many there are if there are less than 100.
        if num_steps >= 100:
            visualize = i % (num_steps // 100) == 0
        else:
            visualize = True
        if visualize:
            visualize_partition_geopandas(
                partition, graph=graph, img_path=os.path.join(output_dir, "visuals", f"{i + 1}.png"))

    if verbose:
        print("Evaluating inputted district map...", end="")
        sys.stdout.flush()

    precinct_df = pd.DataFrame(columns=["avg_score", "stdev_score", "avg_homogeneity",
                                        "stdev_homogeneity"],
                               index=sorted_node_names)
    for i, precinct in enumerate(sorted_node_names):
        precinct_df.loc[precinct] = [
            score_accumulator.mean[i],
            math.sqrt(score_accumulator.var_s[i]),
            homogeneity_accumulator.mean[i],
            math.sqrt(homogeneity_accumulator.var_s[i])
        ]

    districts_precinct_df = pd.DataFrame(columns=["score", "homogeneity"], index=sorted_node_names)
    if isinstance(district_file, str):
        district_node_sets = load_districts(graph, district_file, verbose)
    else:
        district_node_sets = district_file
    district_scores, state_score = quantify_gerrymandering(graph, district_node_sets, node_differences, verbose)
    for district, precincts in district_node_sets.items():
        homogeneity = statistics.stdev(
            [graph.nodes[node]["total_rep"] / graph.nodes[node]["total_votes"]
             for node in precincts]
        )
        for precinct in precincts:
            districts_precinct_df.loc[precinct] = [district_scores[district], homogeneity]

    if optimize_vis:
        output_dir = vis_dir
    # Save a histogram of statewide scores.
    plt.hist(scores_df["state_gerry_score"], bins=30)
    plt.axvline(scores_df["state_gerry_score"].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(state_score, color='red', linestyle='solid', linewidth=1)
    plt.savefig(os.path.join(output_dir, "score_distribution.png"))

    # Create gerrymandering and packing/cracking heatmaps for the inputted districting plan.

    def get_z_score(precinct, metric):
        mean = precinct_df.loc[precinct, f"avg_{metric}"]
        stdev = precinct_df.loc[precinct, f"stdev_{metric}"]
        flag = districts_precinct_df.loc[precinct, metric]
        return (flag - mean) / stdev

    # Needed for drawing district boundaries
    districts_assignment = {}
    for district, nodes in district_node_sets.items():
        for node in nodes:
            districts_assignment[node] = district
    districts_partition = Partition(graph, assignment=districts_assignment)

    # TODO: this doesn't work with Maryland for some reason
    _, ax = plt.subplots(figsize=(12.8, 9.6))
    visualize_gradient_geopandas(
        sorted_node_names,
        get_value=partial(get_z_score, metric="score"),
        get_geometry=lambda p: graph.nodes[p]["geometry"],
        clear=False,
        ax=ax,
        legend=True,
        # img_path=os.path.join(output_dir, "gradient_score.png")
    )
    visualize_partition_geopandas(
        districts_partition,
        union=True,
        img_path=os.path.join(output_dir, "gerry_scores.png"),
        clear=True,
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.5
    )

    _, ax = plt.subplots(figsize=(12.8, 9.6))
    visualize_gradient_geopandas(
        sorted_node_names,
        get_value=partial(get_z_score, metric="homogeneity"),
        get_geometry=lambda p: graph.nodes[p]["geometry"],
        clear=False,
        ax=ax,
        legend=True
    )
    visualize_partition_geopandas(
        districts_partition,
        union=True,
        img_path=os.path.join(output_dir, "packing_cracking.png"),
        clear=True,
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.5
    )

    if verbose:
        print("done!")