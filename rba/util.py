"""Miscellaneous utilities.
"""

import json
import random
import time

from collections import defaultdict

from gerrychain import Partition
from pyproj import CRS
import networkx as nx
import geopandas as gpd
import shapely
import pandas as pd
import maup

from . import constants
from . import visualization
# from .district_quantification import quantify_gerrymandering


def copy_adjacency(graph):
    """Copies adjacency information from a graph but not attribute data.
    """
    copy_graph = nx.Graph()
    for node in graph.nodes():
        copy_graph.add_node(node)
    for u, v in graph.edges:
        copy_graph.add_edge(u, v)
    return copy_graph


def get_num_vra_districts(partition, label, threshold):
    """Returns the number of minority-opportunity distrcts for a given minority and threshold.

    Parameters
    ----------
    partition : gerrychain.Parition
        Proposed district plan.
    label : str
        Node data key that returns the population of that minority.
    threshold : float
        Value between 0 and 1 indicating the percent population required for a district to be
        considered minority opportunity.
    """
    num_vra_districts = 0
    for part in partition.parts:
        total_pop = 0
        minority_pop = 0
        for node in partition.parts[part]:
            total_pop += partition.graph.nodes[node]["total_pop"]
            if label == "total_combined":
                for minority in constants.MINORITY_NAMES:
                    minority_pop += partition.graph.nodes[node][f"total_{minority}"]
            else:
                minority_pop += partition.graph.nodes[node][label]
        if minority_pop / total_pop >= threshold:
            num_vra_districts += 1
    return num_vra_districts


def get_county_weighted_random_spanning_tree(graph):
    """Applies random edge weights to a graph, then multiplies those weights depending on whether or
    not the edge crosses a county border. Then returns the maximum spanning tree for the graph."""
    # start_time = time.time()
    county_assignments = defaultdict(list)
    for u in graph.nodes:
        county_assignments[graph.nodes[u]["COUNTYFP10"]].append(u)
    county_graph = nx.Graph()
    county_graph.add_nodes_from(county_assignments.keys())
    superedges = defaultdict(list)
    # print(graph.nodes)
    # print(type(graph.node_indices), "NODE INDICIES")
    for edge in graph.edges():
        weight = random.random()
        graph.edges[edge]["random_weight"] = weight
        if graph.nodes[edge[0]]["COUNTYFP10"] != graph.nodes[edge[1]]["COUNTYFP10"]:
            superedges[frozenset((graph.nodes[edge[0]]["COUNTYFP10"],graph.nodes[edge[1]]["COUNTYFP10"]))].append(edge)
    for edge in superedges:
        county_graph.add_edge(tuple(edge)[0], tuple(edge)[1], random_weight=random.random())
    supercounty_spanning_tree = nx.tree.maximum_spanning_tree(
            county_graph, algorithm="kruskal", weight="random_weight"
        )

    # county_graph.add_edges_from(superedges.keys())
    # county_spanning_trees = {}
    full_spanning_tree = nx.Graph()
    i = 0
    for county, node_list in county_assignments.items():
        county_subgraph = graph.subgraph(node_list)
        county_spanning_tree = nx.tree.maximum_spanning_tree(
            county_subgraph, algorithm="kruskal", weight="random_weight"
        )

        full_spanning_tree = nx.compose(full_spanning_tree, county_spanning_tree)
        # print(full_spsanning_tree.node_indices())
        # county_spanning_trees[county] = county_spanning_tree
    # edge_colors = def ()
    # visualization.visualize_graph(full_spanning_tree, "before_spanning_tree.png", lambda node: shapely.geometry.mapping(shapely.geometry.shape(graph.nodes[node]['geometry']).centroid)["coordinates"], show=True)
    for edge in supercounty_spanning_tree.edges():
        edge_list = superedges[frozenset(edge)]
        # print(edge, superedges.keys())
        chosen_edge = random.choice(edge_list)
        full_spanning_tree.add_edge(chosen_edge[0], chosen_edge[1])
        # weight = random.random()
        # full_spanning_tree.edges[chosen_edge]["random_weight"] = weight*constants.SAME_COUNTY_PENALTY
    full_spanning_tree.node_indices = graph.node_indices
    # visualization.visualize_graph(full_spanning_tree, f"spanning_tree_{random.randint(0, 10000)}.png", lambda node: shapely.geometry.mapping(shapely.geometry.shape(graph.nodes[node]['geometry']).centroid)["coordinates"], show=True)
    # print(len(full_spanning_tree.nodes()), len(graph.nodes()))
    # print(len(full_spanning_tree.edges()), len(graph.edges()))
    # print(time.time()-start_time)
    return full_spanning_tree
    
    # Old spanning tree code which does not work
    # for u, v in graph.edges:
    #     weight = random.random()
    #     if graph.nodes[u]["COUNTYFP10"] == graph.nodes[v]["COUNTYFP10"]:
    #         weight *= constants.SAME_COUNTY_PENALTY
    #     graph[u][v]["random_weight"] = weight

    # spanning_tree = nx.tree.maximum_spanning_tree(
    #     graph, algorithm="kruskal", weight="random_weight"
    # )
    # return spanning_tree


def save_assignment(partition, fpath):
    """Saves a partition's node assignment data to a file.
    """
    assignment = {}
    for u in partition.graph.nodes:
        assignment[u] = partition.assignment[u]
    with open(fpath, "w+") as f:
        json.dump(assignment, f)


def partition_by_county(graph):
    """Returns a partition which splits the graph by county.
    """
    assignment = {}
    for u in graph.nodes:
        assignment[u] = graph.nodes[u]["COUNTYFP10"]
    return Partition(graph, assignment)


def get_county_border_proportion(partition):
    """Returns the proportion of cross-district edges that are also cross-county edges.
    """
    num_cross_county_edges = 0
    for u, v in partition["cut_edges"]:
        if partition.graph.nodes[u]["COUNTYFP10"] != partition.graph.nodes[v]["COUNTYFP10"]:
            num_cross_county_edges += 1
    return num_cross_county_edges / len(partition["cut_edges"])


def load_districts(graph, district_file, verbose=False):
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

    # graph = nx.readwrite.json_graph.adjacency_graph(graph_json)
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
    # districts = {district : graph.subgraph(districts[district]).copy() for district in districts}
    return districts
