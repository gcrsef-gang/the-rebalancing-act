"""
Given a community file, evaluates a district map based on how well it represents communities
"""
import json
import geopandas as gpd
import pandas as pd
import maup
import shapely
import networkx as nx

from pyproj import CRS


def load_districts(graph_file, district_file, verbose=False):
    """
    Given a path to the district boundaries of a state, creates a list of districts and their composition.
    """
    district_boundaries = gpd.read_file(district_file)
    cc = CRS('esri:102008')
    district_boundaries = district_boundaries.to_crs(cc)
    if "GEOID10" in district_boundaries.columns:
        district_boundaries["GEOID10"].type = str
        district_boundaries.set_index("GEOID10", inplace=True)
    else: 
        district_boundaries["GEOID20"].type = str
        district_boundaries.set_index("GEOID20", inplace=True)
    with open(graph_file, "r") as f:
        graph_json = json.load(f)
    graph = nx.readwrite.json_graph.adjacency_graph(graph_json)
    geodata_dict = {}
    for node, data in graph.nodes(data=True):
        data["geoid"] = node
        data["geometry"] = shapely.geometry.shape(data["geometry"])
        geodata_dict[node] = data
    geodata_dataframe = pd.DataFrame.from_dict(geodata_dict, orient='index')
    geodata = gpd.GeoDataFrame(geodata_dataframe, geometry=geodata_dataframe.geometry, crs='esri:102008')
    district_assignment = maup.assign(geodata, district_boundaries)
    district_assignment = district_assignment.astype(str)
    district_assignment = district_assignment.str.split('.').str[0]
    district_assignment.to_csv("district_assignment.csv")
    districts = {}
    for i, district in district_assignment.iteritems():
        if district in districts:
            districts[district].append(i)
        else:
            districts[district] = [i]
    district_graphs = {district : graph.subgraph(districts[district]).copy() for district in districts}
    return graph, district_graphs

def quantify_gerrymandering(state_graph, district_graphs, community_lifespan, verbose=False):
    """
    Given a dictionary of district graphs/a state graph as well as dictionary of community boundary lifespan, calculates
    gerrymandering scores for each district and the state.
    """
    # Two lookups 
    crossdistrict_edges = {district : [] for district in district_graphs}
    for district, graph in district_graphs.items():
        for edge in graph.edges():
            state_graph.remove_edge(edge[0], edge[1])
        # state_graph.remove_edges_from(graph.edges)
        for node in graph:
            state_graph.nodes[node]["district"] = district
    for edge in state_graph.edges():
        first_community = state_graph.nodes[edge[0]]["district"]
        second_community = state_graph.nodes[edge[1]]["district"]
        if first_community != second_community:
            crossdistrict_edges[state_graph.nodes[edge[0]]["district"]].append((edge[0], edge[1]))
            crossdistrict_edges[state_graph.nodes[edge[1]]["district"]].append((edge[1], edge[0]))
    state_gerrymandering = 0
    district_gerrymanderings = {}
    for district, district_graph in district_graphs.items():
        district_gerrymandering = 0
        # for edge in district_graph.edges():
            # try:
                # district_gerrymandering += community_lifespan[edge]
                # state_gerrymandering += community_lifespan[edge]
            # except:
                # district_gerrymandering += (community_lifespan[(edge[1], edge[0])])
                # state_gerrymandering += community_lifespan[(edge[1], edge[0])]
        total_crossedge_num = len(crossdistrict_edges[district])
        for crossedge in crossdistrict_edges[district]:
            try:
                district_gerrymandering += (community_lifespan[crossedge])/total_crossedge_num
                # district_gerrymandering -= (community_lifespan[crossedge])/2
                # state_gerrymandering -= community_lifespan[crossedge]/2
            except:
                district_gerrymandering += (community_lifespan[(crossedge[1], crossedge[0])])/total_crossedge_num
                # district_gerrymandering -= (community_lifespan[(crossedge[1], crossedge[0])])/2
                # state_gerrymandering -= community_lifespan[(crossedge[1], crossedge[0])]/2
        district_gerrymanderings[district] = district_gerrymandering
    return district_gerrymanderings, state_gerrymandering
    
def quantify_districts(graph_file, district_file, community_file, verbose=False):
    """
    Wraps both functions into a single function for direct use from main.py
    """
    state_graph, district_graphs = load_districts(graph_file, district_file)

    with open(community_file, "r") as f:
        supercommunity_output = json.load(f)  # Contains strings as keys.

    community_lifespan = {}
    for edge, lifetime in supercommunity_output["edge_lifetimes"].items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        community_lifespan[(u, v)] = lifetime

    district_gerrymanderings, state_gerrymandering = quantify_gerrymandering(state_graph, district_graphs, community_lifespan)
    print(district_gerrymanderings, state_gerrymandering)
    return district_gerrymanderings, state_gerrymandering