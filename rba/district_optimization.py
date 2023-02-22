"""
Given supercommunity edge lifetimes, uses simulated annealing to generate a map that minimizes
the average border edge lifetime while conforming to redistricting requirements.
"""

from functools import partial
import random

from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from networkx import tree

from .util import get_num_vra_districts, get_county_weighted_random_spanning_tree


class SimulatedAnnealingChain(MarkovChain):
    """Augments gerrychain.MarkovChain to take both the current state and proposal in the `accept`
    function.
    """
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
                if self.accept(self.state, proposed_next_state):
                    self.state = proposed_next_state
                self.counter += 1
                return self.state
        raise StopIteration


def accept_proposal(temperature, current_energy, proposed_energy):
    """Simple simulated-annealing acceptance function.
    """
    if current_energy > proposed_energy or random.random() < temperature:
        return True
    return False


def generate_districts_simulated_annealing(graph, edge_lifetimes, num_vra_districts, vra_threshold,
                                           pop_equality_threshold):
    """Returns the 10 best maps and a dataframe of statistics for the entire chain.
    """

    weighted_recom_proposal = partial(
        recom,
        method=partial(
            bipartition_tree,
            spanning_tree_fn=get_county_weighted_random_spanning_tree)
    )


def optimize():
    """
    """