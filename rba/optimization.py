"""
Given supercommunity edge lifetimes, uses simulated annealing to generate a map that minimizes
the average border edge lifetime while conforming to redistricting requirements.
"""

from dataclasses import dataclass, field
from functools import partial
import heapq
import json
import os
import random
import sys
import warnings

from gerrychain import Partition, Graph, MarkovChain, updaters, constraints, accept
from gerrychain.constraints import Validator
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part, bipartition_tree
import gerrychain.random
import networkx as nx
import pandas as pd

from . import constants
from .district_quantification import quantify_gerrymandering
from .util import (get_num_vra_districts, load_districts, get_county_spanning_forest,
                   save_assignment, choose_cut)


class SimulatedAnnealingChain(MarkovChain):
    """Simulated annealing Markov Chain. Major changes to gerrychain.MarkovChain:
    - `get_temperature` is now the first positional argument, and it must take the current iteration 
       and return the current temperature.
    - `accept` must take the current partition, the proposed next partition, and the current 
       temperature.
    - Contains static methods for temperature cooling schedules.
    """

    @staticmethod
    def get_temperature_linear(i, num_steps):
        return 1 - (i / num_steps)

    COOLING_SCHEDULES = {
        "linear": get_temperature_linear
    }

    def __init__(self, get_temperature, proposal, constraints, accept, initial_state, total_steps):
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

        self.get_temperature = get_temperature

    def __iter__(self):
        super().__iter__()
        self.temperature = self.get_temperature(self.counter)
        return self
        
    def __next__(self):
        if self.counter == 0:
            self.counter += 1
            return self.state

        while self.counter < self.total_steps:
            proposed_next_state = self.proposal(self.state)
            # Erase the parent of the parent, to avoid memory leak
            if self.state is not None:
                self.state.parent = None

            if self.is_valid(proposed_next_state):
                if self.accept(self.state, proposed_next_state, self.get_temperature(self.counter)):
                    self.state = proposed_next_state
                self.counter += 1
                self.temperature = self.get_temperature(self.counter)
                return self.state
        raise StopIteration


@dataclass(order=True)
class ScoredPartition:
    """A comparable class for storing partitions and they gerrymandering scores."""
    score: float
    partition: Partition=field(compare=False)


def sa_accept_proposal(current_state, proposed_next_state, temperature):
    """Simple simulated-annealing acceptance function. NOTE: this is for maximizing energy.
    """
    current_energy = current_state["gerry_scores"][1]
    proposed_energy = proposed_next_state["gerry_scores"][1]
    if (current_energy < proposed_energy or random.random() < temperature):
        return True
    return False


def generate_districts_simulated_annealing(graph, edge_lifetimes, num_vra_districts, vra_threshold,
                                           pop_equality_threshold, num_steps, num_districts,
                                           cooling_schedule="linear", initial_assignment=None,
                                           verbose=False):
    """Runs a simulated annealing Markov Chain that maximizes the "RBA goodness score."

    Parameters
    ----------
    graph : gerrychain.Graph
        The state graph of precincts.
    edge_lifetimes : dict
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
        The number of iterations to run simulated annealing for.
    num_districts : int
        The number of districts to partition the state into.
    cooling_schedule : str, default="linear"
        Determines how the temperature decreases during simulated annealing.
        Options: "linear"
    initial_assignment : dict, default=None
        Maps nodes to districts for the initial plan in the Markov chain. If set to None, a random
        partition is used.
    verbose : boolean, default=False
        Controls verbosity.

    Returns
    -------
    good_partitions : list of gerrychain.Partition
        Contains the 10 best districting plans obtained by the algorithm.
    df : pandas.DataFrame
        Contains gerrymandering scores of the state and all the districts as well as the temperature
        for each iteration of the algorithm.
    """

    sa_updaters = {
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
                    for minority in num_vra_districts.keys()}

    sa_updaters.update(vra_updaters)

    state_population = 0
    for node in graph:
        state_population += graph.nodes[node]["total_pop"]
    ideal_population = state_population / num_districts

    if initial_assignment is None:
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

    initial_partition = Partition(graph, initial_assignment, sa_updaters)

    county_recom_proposal = partial(
        recom,
        pop_col="total_pop",
        pop_target=ideal_population,
        epsilon=constants.POP_EQUALITY_THRESHOLD,
        node_repeats=2,
        method=partial(
            bipartition_tree,
            spanning_tree_fn=get_county_spanning_forest,
            choice=partial(choose_cut, graph=graph))
    )

    # recom_proposal = partial(recom,
    #     pop_col="total_pop",
    #     pop_target=ideal_population,
    #     epsilon=constants.POP_EQUALITY_THRESHOLD,
    #     node_repeats=2
    # )

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition,
                                                                    pop_equality_threshold)
    
    # cut edges in the starting one.
    # compactness_bound = constraints.UpperBound(
    #     lambda p: len(p["cut_edges"]),
    #     2 * len(initial_partition["cut_edges"])
    # )

    vra_constraints = [
        constraints.LowerBound(
            lambda p: p[f"num_{minority}_vra_districts"],
            num_districts
        )
        for minority, num_districts in num_vra_districts.items()]

    # acceptance_func = lambda curr, next_, t: sa_accept_proposal(curr, next_, t) \
    #                                      and random.random() < get_county_border_proportion(next_)

    chain = SimulatedAnnealingChain(
        get_temperature=partial(
            SimulatedAnnealingChain.COOLING_SCHEDULES[cooling_schedule],
            num_steps=num_steps),
        proposal=county_recom_proposal,
        # proposal=recom_proposal,
        constraints=[
            pop_constraint,
            # compactness_bound
        ] + vra_constraints,
        accept=sa_accept_proposal,
        initial_state=initial_partition,
        total_steps=num_steps
    )

    df = pd.DataFrame(
        columns=[f"district_{i}_score" for i in range(1, num_districts + 1)] \
            + ["state_gerry_score"] + ["temperature"],
        dtype=float
    )

    if verbose:
        print("Running Markov chain...")

    good_partitions = []  # min heap based on "goodness" score
    if verbose:
        chain_iter = chain.with_progress_bar()
    else:
        chain_iter = chain
    for i, partition in enumerate(chain_iter):
        district_scores, state_score = partition["gerry_scores"]
        df.loc[len(df.index)] = sorted(list(district_scores.values())) + [state_score] \
            + [chain.get_temperature(i)]

        # First 10 partitions will be the best 10 so far.
        if i < 10:
            heapq.heappush(
                good_partitions,
                ScoredPartition(score=state_score, partition=partition)
            )
        elif state_score > good_partitions[0].score:  # better than the worst good score.
            heapq.heapreplace(
                good_partitions,
                ScoredPartition(score=state_score, partition=partition)
            )

        if verbose:
            chain_iter.set_description(f"State score: {round(state_score, 4)}")
    
    good_partitions = [obj.partition for obj in good_partitions]
    return good_partitions, df


def optimize(graph_file, communitygen_out_file, vra_config_file, num_steps, num_districts,
             initial_plan_file, output_dir, verbose):
    """Wrapper function for command-line usage.
    """
    gerrychain.random.random.seed(2023)
    random.seed(2023)

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

    with open(communitygen_out_file, "r") as f:
        community_data = json.load(f)
    edge_lifetimes = {}
    for edge, lifetime in community_data["edge_lifetimes"].items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        edge_lifetimes[(u, v)] = lifetime

    if verbose:
        print("done!")
        print("VRA requirements...", end="")
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

        initial_subgraphs = load_districts(graph, initial_plan_file, verbose=verbose)
        initial_assignment = {}
        for district, nodes in initial_subgraphs.items():
            for node in nodes:
                initial_assignment[node] = district
        
        if verbose:
            print("done!")
    else:
        if verbose:
            print("No starting map provided. Will generate a random one later.")
        initial_assignment = None

    plans, df = generate_districts_simulated_annealing(
        graph, edge_lifetimes, vra_config, vra_threshold, constants.POP_EQUALITY_THRESHOLD,
        num_steps, num_districts, initial_assignment=initial_assignment, verbose=verbose)

    if verbose:
        print("Saving data from optimization...", end="")
        sys.stdout.flush()

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # Save districts in order of decreasing goodness.
    for i, partition in enumerate(sorted(plans, key=lambda p: p["gerry_scores"][1], reverse=True)):
        save_assignment(partition, os.path.join(output_dir, f"Plan_{i + 1}.json"))
    
    df.to_csv(os.path.join(output_dir, "optimization_stats.csv"))

    if verbose:
        print("done!")