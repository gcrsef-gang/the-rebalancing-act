"""
Given a community file, evaluates a district map based on how well it represents communities
"""
import geopandas as gpd
import maup


def load_districts(graph_file, district_file):
    """
    Given a path to the district boundaries of a state, creates a list of districts and their composition.
    """
    district_boundaries = gpd.read_file(district_file)
    if "GEOID10" in district_boundaries.columns:
        district_boundaries["GEOID10"].type = str
    else: 
        district_boundaries["GEOID20"].type = str
    graph = gpd.read_file(graph_file)
    district_assignment = maup.assign(graph, district_boundaries)
    districts = {}
    for i, district in district_assignment.iteritems():
        if district in districts:
            districts[district].append(i)
        else:
            districts[district] = [i]
    district_graphs = {district : graph_file.subgraph(districts[district]) for district in districts}
    # Two lookups 
    crossdistrict_edges = {district : [] for district in districts}
    for edge in graph.edges():
        first_community = district_assignment[edge[0]]
        second_community = district_assignment[edge[1]]
        if first_community != second_community:
            crossdistrict_edges[first_community].append(edge[1])
            crossdistrict_edges[second_community].append(edge[0])
    return graph, district_graphs, crossdistrict_edges

def quantify_districts(graph_file, district_file, community_lifespan):
    """
    Given a list of district graphs as well as dictionary of community boundary lifespan, calculates
    gerrymandering scores for each district and the state.
    """