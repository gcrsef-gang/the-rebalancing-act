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

def quantify_gerrymandering(state_graph, districts, difference_scores, verbose=False):
    """
    Given a dictionary of districts to node lists/a state graph as well as dictionary of community boundary lifespan, calculates
    gerrymandering scores for each district and the state.
    """
    # print("gerrymandering being quantified!")
    # Two lookups 
    crossdistrict_edges = {district : [] for district in districts}
    for district, graph in districts.items():
        # print(district, graph.nodes)
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
    # num_crossedges = sum([len(edge_list) for edge_list in crossdistrict_edges.values()])
    for district, node_list in districts.items():
        # print(district, "district being quantified!")
        district_gerrymandering = 0
        # for edge in district_graph.edges():
        for node1 in node_list:
            for node2 in node_list:
                if node1 == node2:
                    continue
                try:
                    district_gerrymandering += difference_scores[(node1, node2)]
                    state_gerrymandering += difference_scores[(node1, node2)]
                except:
                    district_gerrymandering += difference_scores[(node2, node1)]
                    state_gerrymandering += difference_scores[(node2, node1)]
            # try:
            #     district_gerrymandering += difference_scores[edge]
            #     state_gerrymandering += difference_scores[edge]
            # except:
            #     district_gerrymandering += (difference_scores[(edge[1], edge[0])])
            #     state_gerrymandering += difference_scores[(edge[1], edge[0])]
        # total_crossedge_num = len(crossdistrict_edges[district])
        # for crossedge in crossdistrict_edges[district]:
        #     try:
        #         district_gerrymandering += (difference_scores[crossedge])/total_crossedge_num
        #         # district_gerrymandering -= (difference_scores[crossedge])/2
        #         # state_gerrymandering -= difference_scores[crossedge]/2
        #         state_gerrymandering += difference_scores[crossedge]/(num_crossedges)
        #     except:
        #         district_gerrymandering += (difference_scores[(crossedge[1], crossedge[0])])/total_crossedge_num
        #         # district_gerrymandering -= (difference_scores[(crossedge[1], crossedge[0])])/2
        #         # state_gerrymandering -= difference_scores[(crossedge[1], crossedge[0])]/2
        #         state_gerrymandering += difference_scores[(crossedge[1], crossedge[0])]/(num_crossedges)
        district_gerrymanderings[district] = district_gerrymandering/(len(node_list)*(len(node_list)-1))
    state_gerrymandering = sum(district_gerrymanderings.values())/len(district_gerrymanderings)
    return district_gerrymanderings, state_gerrymandering
    
def quantify_districts(graph_file, district_file, difference_file, verbose=False):
    """
    Wraps both functions into a single function for direct use from main.py
    """
    with open(graph_file, "r") as f:
        graph_json = json.load(f)
    graph = nx.readwrite.json_graph.adjacency_graph(graph_json)
    districts = load_districts(graph, district_file)
    # with open(district_file, "r") as f:
    #     districts = json.load(f)

    with open(difference_file, "r") as f:
        supercommunity_output = json.load(f)  # Contains strings as keys.

    difference_scores = {}
    for edge, lifetime in supercommunity_output.items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        difference_scores[(u, v)] = lifetime
    print('Differences loaded')
    district_gerrymanderings, state_gerrymandering = quantify_gerrymandering(graph, districts, difference_scores)
    print(sorted(district_gerrymanderings.items()), state_gerrymandering)
    return districts, district_gerrymanderings, state_gerrymandering