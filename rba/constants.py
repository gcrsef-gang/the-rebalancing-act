"""Arbitrarily decided parameters for all algorithms.
"""


# Edges between nodes in the same county should be weighted less than those that cross, because
# the maximum spanning tree should be made more likely to choose an edge crossing county lines.
# SAME_COUNTY_PENALTY = 0.1
SAME_COUNTY_PENALTY = 10

POP_EQUALITY_THRESHOLD = 0.005

MINORITY_NAMES = ["black", "hispanic", "asian", "native", "islander"]

# SIMILARITY_WEIGHTS = {
#     "race": 1.5,
#     "votes": 1.0,
#     "pop_density": 1.0,
#     "county": 0.1
# }
SIMILARITY_WEIGHTS = {
    "race": 1.0,
    "votes": 1.0,
    "pop_density": 1.0,
    "county": 0.3
}

# Due logarithm domain redistriction
MIN_POP_DENSITY = 1e-20

RACE_KEYS = ["total_white", "total_black", "total_hispanic", "total_asian", "total_islander", "total_native"]

VOTE_KEYS = ["total_rep", "total_dem"]