"""Arbitrarily decided parameters for all algorithms.
"""


# Edges between nodes in the same county should be weighted less than those that cross, because
# the maximum spanning tree should be made more likely to choose an edge crossing county lines.
SAME_COUNTY_PENALTY = 0.1

POP_EQUALITY_THRESHOLD = 0.005

MINORITY_NAMES = ["black", "hispanic", "asian", "native", "islander"]