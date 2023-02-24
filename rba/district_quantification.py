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

from .util import load_districts

def quantify_gerrymandering(state_graph, districts, community_lifespan, verbose=False):
    """
    Given a dictionary of districts to node lists/a state graph as well as dictionary of community boundary lifespan, calculates
    gerrymandering scores for each district and the state.
    """
    # Two lookups 
    crossdistrict_edges = {district : [] for district in districts}
    for district, graph in districts.items():
            # state_graph.remove_edge(edge[0], edge[1])
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
    num_crossedges = sum([len(edge_list) for edge_list in crossdistrict_edges.values()])
    for district, node_list in districts.items():
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
                state_gerrymandering += community_lifespan[crossedge]/(num_crossedges)
            except:
                district_gerrymandering += (community_lifespan[(crossedge[1], crossedge[0])])/total_crossedge_num
                # district_gerrymandering -= (community_lifespan[(crossedge[1], crossedge[0])])/2
                # state_gerrymandering -= community_lifespan[(crossedge[1], crossedge[0])]/2
                state_gerrymandering += community_lifespan[(crossedge[1], crossedge[0])]/(num_crossedges)
        district_gerrymanderings[district] = district_gerrymandering
    return district_gerrymanderings, state_gerrymandering
    
def quantify_districts(graph_file, district_file, community_file, verbose=False):
    """
    Wraps both functions into a single function for direct use from main.py
    """
    with open(graph_file, "r") as f:
        graph_json = json.load(f)
    graph = nx.readwrite.json_graph.adjacency_graph(graph_json)
    districts = load_districts(graph, district_file)

    with open(community_file, "r") as f:
        supercommunity_output = json.load(f)  # Contains strings as keys.

    community_lifespan = {}
    for edge, lifetime in supercommunity_output["edge_lifetimes"].items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        community_lifespan[(u, v)] = lifetime

    district_gerrymanderings, state_gerrymandering = quantify_gerrymandering(graph, districts, community_lifespan)
    print(district_gerrymanderings, state_gerrymandering)
    return districts, district_gerrymanderings, state_gerrymandering