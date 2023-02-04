import sys
import os
import subprocess
import math
from collections import defaultdict, deque
from itertools import combinations
import json
import time
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

import fiona  
import pandas as pd
import geopandas as gpd
from pyproj import CRS

import maup
import networkx as nx

# The link to the data directory in James' computer
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))+ "/hte-data-new/raw"
final_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))+ "/hte-data-new/graphs"
# final_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+ "/data"

def compress_all_data(type):
    """
    This function automatically compresses all data files in the data directory to .7z files
    """
    if type == "data":
        dir = data_dir
    else:
        dir = final_dir
    for year in ["2010", "2020"]:
        for root, dirs, files in os.walk(dir+"/"+year):
            print(root, dirs, files)
            # To hold only files that need to be compressed so that there is no double compression. 
            files_to_compress = []    
            for file in files:
                # Files that don't need to be compressed.
                if file in ["desktop.ini", "demographics.csv", "election_data.csv", "block_group_demographics.csv", "block_group_election_data.csv", "README.md"]:
                    continue
                full_path = os.path.join(root, file)  
                corresponding_file = file[:file.find(".")] + ".7z"
                full_7z_path = os.path.join(final_dir+"/"+year, corresponding_file)
                if not corresponding_file in files:
                    files_to_compress.append([full_7z_path, full_path])
            for file in files_to_compress:
                    subprocess.call(["7z", "a", file[0], file[1]])

def compress_simplified_graphs():
    """
    This function automatically compresses simplified (no geometry) graphs to .7z files. 
    """
    for year in ["2010", "2020"]:
        for file in os.listdir(final_dir+"/"+year):
            if "simplified" in file:
                corresponding_file = file[:file.find(".")] + ".7z"
                full_7z_path = os.path.join(final_dir+"/"+year, corresponding_file)
                if not os.path.isfile(full_7z_path):
                    full_path = final_dir + "/" + year + "/" + file
                    subprocess.call(["7z", "a", full_7z_path, full_path])

def merge_graphs():
    """
    This function automatically decompresses, merges empty precincts/blocks, and then recompresses.
    """
    for year in ["2010", "2020"]:
        for file in os.listdir(final_dir+"/"+year):
            if "merged" in file or "simplified" in file:
                continue
            elif os.path.isfile(os.path.join(final_dir+"/"+year, file[:file.find(".")] + "_merged.json")):
                continue
            print(file)
            corresponding_file = file[:file.find(".")] + ".json"
            full_json_path = os.path.join(final_dir+"/"+year, corresponding_file)
            if not os.path.isfile(full_json_path):
                # print(["7z", "e", os.path.join(final_dir+"/"+year, file), "-o"+final_dir+"/"+year,])
                subprocess.call(["7z", "e", os.path.join(final_dir+"/"+year, file), "-o"+final_dir+"/"+year,])
            with open(full_json_path, "r") as f:
                data = json.load(f)
            graph = nx.readwrite.adjacency_graph(data)
            print(f"Total number of nodes: {len(graph.nodes)}")
            merged_graph = merge_empty(graph)
            merged_data = nx.readwrite.adjacency_data(merged_graph)
            # for id, data in merged_graph.nodes(data=True):
                # print(data)
                # merged_data["geometry"] = shapely.geometry.mapping(data['geometry'])
            merged_corresponding_file = file[:file.find(".")] + "_merged.json"  
            with open(os.path.join(final_dir+"/"+year, merged_corresponding_file), "w") as f:
                # for i in range(len(merged_data["nodes"])):
                    # print(merged_data["nodes"][i])
                    # json.dump(merged_data["nodes"][i], f)
                json.dump(merged_data, f)
                # data_str = json.dumps(merged_data["nodes"][0])
                # f.write(data_str)
            if os.path.isfile(os.path.join(final_dir+"/"+year, file[:file.find(".")] + ".7z")):
                os.remove(full_json_path)

def split_multipolygons(geodata, assignment=None, block_data=None):
    """
    This function takes a GeoDataFrame and splits all the multipolygons so that every polygon is contiguous
    Arguments:
        geodata - GeoPandas GeoDataFrame. 
        assignment - Pandas Series
        block_data - GeoPandas GeoDataFrame
    Returns:
        modified_geodata - split GeoDataFrame
    """
    row_geoids_to_remove = []
    rows_to_add = {}
    for geoid, row in geodata.iterrows():
        if type(row["geometry"]) == shapely.geometry.MultiPolygon:
            polygons = row["geometry"].geoms
            if type(assignment) == pd.Series:
                # Precinct geodata, which means the values can be aggregated upwards for each precinct. 
                blocks = block_data.loc[assignment[assignment == geoid].index]
                used = []
                id = 1
                for polygon in polygons:
                    new_geoid = geoid + "s" + str(id)
                    new_row = {}
                    # new_row = pd.Series(index=[row.index])
                    # new_row.index.names = [new_geoid]
                    id += 1
                    for block_geoid, block in blocks.iterrows():
                        if not block_geoid in used:
                            if polygon.intersection(block["geometry"]).area > 0.5*block["geometry"].area:
                                for column, value in row.iteritems():
                                    # Obviously use precinct polygon geometry, not block geometry
                                    if column == "geometry":
                                        new_row[column] = polygon
                                    elif column == "geoid":
                                        pass
                                    elif column in block_data.columns:
                                        if type(block[column]) in [int, float]:
                                            try:
                                                new_row[column] += block[column]
                                            except:
                                                new_row[column] = block[column]
                                        else:
                                            new_row[column] = block[column]
                                    else:
                                        new_row[column] = value
                                used += block_geoid
                    # The precinct is so small that it didn't match up to any blocks - 
                    # Add a "fake" row with 0 values inside so that the hole matchings don't create problems
                    if len(new_row) == 0:
                        for column, _ in row.iteritems():
                            # Obviously use precinct polygon geometry, not block geometry
                            if column == "geometry":
                                new_row[column] = polygon
                            elif column == "geoid":
                                pass
                            elif column in block_data.columns:
                                if type(block_data[column].iloc[0]) in [int, float]:
                                    new_row[column] = 0
                                else:
                                    new_row[column] = None
                            else:
                                new_row[column] = None
                            # print("PROBLEM: No Blocks added in precinct", new_geoid, geoid, blocks, polygon)
                    rows_to_add[new_geoid] = new_row
                row_geoids_to_remove.append(geoid)
            else:
                # Block geodata, which means that area has to be used.
                total_area = row["geometry"].area
                id = 1
                for polygon in polygons:
                    new_geoid = str(geoid) + "p" + str(id)
                    new_row = {}
                    # new_row.index.names = [new_geoid]
                    id += 1
                    percentage = polygon.area/total_area
                    for column, value in row.iteritems():
                        # Obviously use precinct polygon geometry, not block geometry
                        if column == "geometry":
                            new_row[column] = polygon
                        elif type(value) in [int, float]:
                            try:
                                new_row[column] += value * percentage
                            except:
                                new_row[column] = value * percentage
                        else:
                            new_row[column] = value                    
                    rows_to_add[new_geoid] = new_row
                row_geoids_to_remove.append(geoid)
    modified_geodata = geodata.drop(row_geoids_to_remove)
    concated_geodata = gpd.GeoDataFrame(pd.concat([modified_geodata, pd.DataFrame.from_dict(rows_to_add, orient='index')]))
    concated_geodata.set_crs(modified_geodata.crs, allow_override=True, inplace=True) 
    concated_geodata.index.names = ["geoid"]
    print(f"{len(row_geoids_to_remove)} multipolygons removed and {len(rows_to_add)} polygons added")
    return concated_geodata


def combine_holypolygons(geodata, assignment=None):
    """
    This function takes a GeoDataFrame and combines all the polygons that are encased within another polygon.
    Arguments:
        geodata - GeoPandas GeoDataFrame with integrated precinct/block geodata.
    Returns:
        modified_geodata - non-holy GeoDataFrame
    """
    # Use minimum x coordinate to idenfiy holes
    # min_xes = {}
    bounds = geodata.bounds.astype(float)
    # min_xes_dict = {}
    hole_geoids = []
    # Generate minimum x coordinates
    # for geoid, value in min_xes.iteritems():
    #     if value == -72.125922:
    #         print("caught!", geoid)
    #     min_xes_dict[value] = geoid
    source_geometry = []
    # target_geometry = {}
    target_geometry = {"geoid":[], "geometry":[]}
    min_x_list = []
    i = 0
    for geoid, row in geodata.iterrows():
        interiors = row["geometry"].interiors
        # Check if there's actually a hole that needs to be fixed
        if len(interiors) > 0:
            # target_geometry.append(shapely.geometry.Polygon(row["geometry"].exterior))
            # target_geometry.append(geoid)
            # target_geometry["geoid"].append(geoid)
            # target_geometry["geometry"].append(shapely.geometry.Polygon(row["geometry"].exterior))
            for interior in interiors:
                target_geometry["geoid"].append(geoid)
                polygon_interior = shapely.geometry.Polygon(interior)
                target_geometry["geometry"].append(polygon_interior)
                if isinstance(polygon_interior, shapely.prepared.PreparedGeometry):
                    print(interior, "YOOOOOOOOOOOOOOOOOOOOO", polygon_interior, geoid)
                min_x_list.append([polygon_interior.bounds[0], i])
                i += 1
    min_x_list = sorted(min_x_list,  key = lambda x: x[0])
    # No holes! I'm certainly not complaining...
    if len(min_x_list) == 0:
        return geodata
    holes_in_holes = set()
    active_holes = [min_x_list[0][1]]
    # Test to see if this hole is a hole of another hole
    for i, [min_x, index] in enumerate(min_x_list):
        geoid = target_geometry["geoid"][index]
        _, min_y, max_x, max_y = bounds.loc[geoid].astype(float)
        possible = []
        # Add holes with the same minimum x as well to cover cases were the hole/hole container have same min x
        j = i
        while j < len(min_x_list)-1 and abs(min_x_list[j+1][0]-min_x) <= 1e-7:
            # Will lead to duplication, but it shouldn't matter
            possible.append(min_x_list[i+1][1])
            j += 1
        for hole_index in active_holes:
            hole_geoid = target_geometry["geoid"][hole_index]
            hole_max_x = bounds.loc[hole_geoid][2]
            if hole_max_x >= min_x:
                possible.append(hole_index)
        active_holes = possible
        active_holes.append(index)
        # Testing for polygons that could fit around the polygon in question
        for hole_index in possible:
            if hole_index == index:
                continue
            hole_geoid = target_geometry["geoid"][hole_index]
            _, hole_min_y, hole_max_x, hole_max_y = bounds.loc[hole_geoid].astype(float)
            if hole_max_x >= float(max_x) and hole_min_y <= float(min_y) and hole_max_y >= float(max_y):
                hole_geometry = target_geometry["geometry"][hole_index]
                geometry = target_geometry["geometry"][index]
                if geometry.difference(hole_geometry).area < geometry.area * 0.01:
                    holes_in_holes.add(index)
    # print([target_geometry["geoid"][hole] for hole in holes_in_holes], "holes in holes")
    target_geometry = gpd.GeoDataFrame(target_geometry, crs=geodata.crs)
    target_geometry.to_file("target_geometry.json", driver="GeoJSON")
    # target_geometry.set_index("geoid", inplace=True)
    target_geometry = target_geometry.drop(list(holes_in_holes))
    target_geometry.to_file("post_target_geometry.json", driver="GeoJSON")
    # source_geometry = gpd.GeoSeries(source_geometry, crs=geodata.crs)
    source_geometry = geodata
    hole_assignment = maup.assign(source_geometry, target_geometry)
    # print(hole_assignment.loc["0514509p1"], "0514509 check!")
    # print(hole_assignment.loc["0514509p2"], "0514509 check!")
    # for hole_geoid, geoid in hole_assignment.groupby[geoid].iteritems():
    for id, row in target_geometry.iterrows():
        geoid = row["geoid"]
        geodata.loc[geoid, "geometry"] = shapely.geometry.Polygon(geodata.loc[geoid, "geometry"].exterior)
        target_area = row["geometry"].area
        confirmed_area = 0
        current_hole_geoids = hole_assignment[hole_assignment == id].index
        for hole_geoid in current_hole_geoids:
            if hole_geoid == geoid:
                continue
            hole_geoids.append(hole_geoid)
            hole_row = geodata.loc[hole_geoid]
            confirmed_area += hole_row["geometry"].area
            geodata.loc[geoid, "total_pop"] += hole_row["total_pop"]
            geodata.loc[geoid, "total_white"] += hole_row["total_white"]
            geodata.loc[geoid, "total_hispanic"] += hole_row["total_hispanic"]
            geodata.loc[geoid, "total_black"] += hole_row["total_black"]
            geodata.loc[geoid, "total_asian"] += hole_row["total_asian"]
            geodata.loc[geoid, "total_native"] += hole_row["total_native"]
            geodata.loc[geoid, "total_islander"] += hole_row["total_islander"]
            geodata.loc[geoid, "total_vap"] += hole_row["total_vap"]
            geodata.loc[geoid, "total_votes"] += hole_row["total_votes"]
            geodata.loc[geoid, "total_dem"] += hole_row["total_dem"]
            geodata.loc[geoid, "total_rep"] += hole_row["total_rep"]
        if abs(target_area-confirmed_area) > 1e-7:
            print(target_area, confirmed_area, "AREA MISMATCH", geoid, current_hole_geoids)
    # print(hole_geoids, "hole geoids")
    # Delete the holes
    modified_geodata = geodata.drop(hole_geoids)
    print(f"{len(hole_geoids)} holes removed")
    return modified_geodata



def convert_to_graph(geodata):
    """
    This function converts a GeoDataFrame into a networkx Graph. 
    Arguments:
        geodata - GeoPandas GeoDataFrame with integrated precinct/block geodata.
    Returns:
        graph - Networkx Graph representing the integrated precinct/block geodata. 
    """
    # geometry_to_id = {}
    graph = nx.Graph()
    for geoid, row in geodata.iterrows():
        graph.add_node(geoid, **row.to_dict())
        # geometry_to_id[row["geometry"]] = geoid
        # min_xes_dict[row["geometry"].min_x] = geoid
    print(f"{len(geodata)} Nodes created")
    bounds = geodata.bounds.astype(float)

    start_time = time.time()
    min_yes = bounds["miny"]
    # Can be changed!
    dividers = round(math.pow(len(geodata),1/3))
    print(f"Number of dividers: {dividers}")
    min_yes_list = []
    for geoid, min_y in min_yes.items():
        min_yes_list.append([min_y, geoid])
    min_yes_list = sorted(min_yes_list, key=lambda x: x[0])
    divider_indexes = [round(len(min_yes_list)*(i/dividers)) for i in range(0, dividers)]
    divider_min_ys = [min_yes_list[index][0] for index in divider_indexes]
    y_groupings = [set() for i in range(0, len(divider_indexes))]
    current_grouping = 0
    for i, row in enumerate(min_yes_list):
        if current_grouping < dividers-1:
            min_y = bounds.loc[row[1]][1]
            max_y = bounds.loc[row[1]][3]
            to_add = 1
            while current_grouping+to_add < dividers and max_y >= divider_min_ys[current_grouping+to_add]:
                y_groupings[current_grouping+to_add].add(row[1])
                to_add += 1
            if min_y >= divider_min_ys[current_grouping+1]:
                current_grouping += 1
        y_groupings[current_grouping].add(row[1])
    edges = set()
    edges_added = 0
    attempts = 0
    for i, grouping in enumerate(y_groupings):
        # Iterate over the groupings individually
        min_xes = bounds["minx"]
        min_xes.index = geodata.index
        min_xes = min_xes.loc[list(grouping)]
        min_xes_list = []
        for geoid, min_x in min_xes.iteritems():
            # if min_x in min_xes_dict.keys():
                # print(min_xes_dict[min_x], geoid, "ALREADY CAUGHT")
            # min_xes_dict[min_x] = geoid
            min_xes_list.append([min_x, geoid])
        min_xes_list = sorted(min_xes_list, key = lambda x: x[0])
        
        # This will keep only precincts/blocks which have a chance of bordering the first (current) precinct/block
        # active_geoids = [min_xes_dict[min_xes_list[0]]]
        # active_geoids = [min_xes_list[0][1]]
        active_geoids = []
        for j, [current_min_x, current_geoid] in enumerate(min_xes_list):
            # start_time = time.time() 
            print(f"\rBlock:  {j}/{len(min_xes_list)}, Grouping: {i+1}/{dividers}", end="")
            # current_min_x = min_xes_list[i]
            # current_geoid = min_xes_dict[current_min_x]
            _, current_min_y, current_max_x, current_max_y = bounds.loc[current_geoid]
            current_geometry = geodata.loc[current_geoid, "geometry"]
            possible_borders = []
            # Surprisingly, popping is faster for some reason
            # to_remove = []
            for k, other_geoid in enumerate(active_geoids):
                other_max_x = bounds.loc[other_geoid][2]
                if other_max_x >= current_min_x:
                    possible_borders.append(other_geoid)
                else:
                    # to_remove.append(j)
                    pass
            # to_remove.sort(reverse=True)  
            # for index in to_remove:
                # _ = active_geoids.pop(index)
            active_geoids = possible_borders
            # print("\n", time.time()-start_time, "time taken before possible border iteration", len(possible_borders))
            # print("COMPARISON", len(active_geoids), len(to_remove), len(possible_borders))
            for other_geoid in possible_borders:
                _, other_min_y, _, other_max_y = bounds.loc[other_geoid]
                if other_min_y > current_max_y or other_max_y < current_min_y:
                    continue
                other_geometry = geodata.loc[other_geoid, "geometry"]
                intersection = current_geometry.intersection(other_geometry)
                attempts += 1
                if intersection.is_empty or isinstance(intersection, shapely.geometry.Point) or isinstance(intersection, shapely.geometry.MultiPoint):
                    # print("FAILED", current_geoid, other_geoid)
                    continue
                elif isinstance(intersection, shapely.geometry.collection.GeometryCollection):
                    edge_exists = False
                    no_polygons = True
                    for element in intersection.geoms:
                        if isinstance(element, shapely.geometry.LineString) or isinstance(element, shapely.geometry.MultiLineString):
                            # print("edge added!", [current_geoid, other_geoid])
                            edge_exists = True
                        elif isinstance(element, shapely.geometry.Polygon) or isinstance(element, shapely.geometry.MultiPolygon):
                            no_polygons = False
                            break
                    if edge_exists and no_polygons:
                        edges.add((current_geoid, other_geoid))
                        edges_added += 1
                elif isinstance(intersection, shapely.geometry.LineString) or isinstance(intersection, shapely.geometry.MultiLineString):
                    # print("edge added!", [current_geoid, other_geoid])
                    edges.add((current_geoid, other_geoid))
                    edges_added += 1
                else:
                    print("INTERSECTION PROBLEM", current_geoid, other_geoid, type(intersection))
                # print("WOO time:", time.time()-start_time, edges_added/attempts, edges_added, attempts)
            active_geoids.append(current_geoid)
            # print(time.time()-start_time, "total cycle time")
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    print(f"\n{len(edges)} edges added")
    print(f"Edges creation time: {time.time()-start_time}")
    return graph

def connect_islands(graph):
    """ 
    This function takes a GeoDataFrame and connects blocks/precincts wout connections to their nearest block/precinct.
    Arguments:
        graph - Networkx Graph of the integrated precinct/block geodata.
    Returns:
        graph - A single contiguous Networkx Graph of the integrated precinct/block geodata.
    """
    graph_components_list = list(nx.algorithms.connected_components(graph))
    graph_components_num = len(graph_components_list)
    graph_components_dict = {i : graph_components_list[i] for i in range(0, graph_components_num)}
    # Compute centroids
    centroid_dict = {geoid : graph.nodes[geoid]["geometry"].centroid for geoid in graph.nodes}
    if len(graph_components_list) == 1:
        return graph
    # Maps the minimum distance between two components to their indexes
    graph_distance_to_connections = {}
    # Maps component indexes to the minimum distance between them
    graph_connections_to_distance = {}
    # Maps component indexes to node geoid tuple
    components_to_nodes = {}
    # Iterate through all combinations and find the smallest distances
    for i, combo in enumerate(combinations([num for num in range(0, graph_components_num)], 2)):
        print(f"\rDistance combination: {i+1}/{int(graph_components_num*(graph_components_num-1)/2)}", end="")
        # Combo is an int index for its position in the graph components list
        component = graph_components_list[combo[0]]
        other_component = graph_components_list[combo[1]]
        min_distance = 1e10
        for node in component:
            for other_node in other_component:
                distance = centroid_dict[node].distance(centroid_dict[other_node])
                if distance < min_distance:
                    min_connection = (node, other_node)
                    min_distance = distance
        components_to_nodes[(combo[0], combo[1])] = min_connection

        graph_distance_to_connections[min_distance] = (combo[0], combo[1])
    
        graph_connections_to_distance[(combo[0], combo[1])] = min_distance
    for i in range(0, graph_components_num-1):
        min_distance = min(graph_distance_to_connections.keys())
        overall_min_connection = graph_distance_to_connections[min_distance]
        graph.add_edge(components_to_nodes[overall_min_connection][0], components_to_nodes[overall_min_connection][1])
        print(f"Edge added: {components_to_nodes[overall_min_connection][0]}, {components_to_nodes[overall_min_connection][1]}, {i}/{graph_components_num}")
        # Now delete the added connection from the lists 
        del graph_distance_to_connections[min_distance]
        del graph_connections_to_distance[overall_min_connection]
        del components_to_nodes[overall_min_connection]
        # Compare the other possibilities and their distances to both newly connected components to find which one
        # is better to keep
        for index in graph_components_dict:
            if index in overall_min_connection:
                continue
            if index < overall_min_connection[0]:
                first_distance = graph_connections_to_distance[(index, overall_min_connection[0])]
            else:
                first_distance = graph_connections_to_distance[(overall_min_connection[0], index)]
            if index < overall_min_connection[1]:
                second_distance = graph_connections_to_distance[(index, overall_min_connection[1])]
            else:
                second_distance = graph_connections_to_distance[(overall_min_connection[1], index)]
            if first_distance <= second_distance:
                # Nothing needs to be changed, just the second component needs to be deleted
                if index < overall_min_connection[1]:
                    # If the graph component in question 
                    del graph_connections_to_distance[(index, overall_min_connection[1])]
                    del components_to_nodes[(index, overall_min_connection[1])]
                else:
                    del graph_connections_to_distance[(overall_min_connection[1], index)]
                    del components_to_nodes[(overall_min_connection[1], index)]
                del graph_distance_to_connections[second_distance]
            else:
                # The first distance needs to be changed and component to nodes dict needs to be changed
                if index < overall_min_connection[0]:
                    graph_connections_to_distance[(index, overall_min_connection[0])] = second_distance
                    graph_distance_to_connections[second_distance] = (index, overall_min_connection[0])
                    if index < overall_min_connection[1]:
                        components_to_nodes[index, overall_min_connection[0]] = components_to_nodes[index, overall_min_connection[1]]
                    else:
                        components_to_nodes[(overall_min_connection[0], index)] = components_to_nodes[(overall_min_connection[1], index)]
                else:
                    graph_connections_to_distance[(overall_min_connection[0], index)] = second_distance
                    graph_distance_to_connections[second_distance] = (overall_min_connection[0], index)
                    if index < overall_min_connection[1]:
                        components_to_nodes[index, overall_min_connection[0]] = components_to_nodes[index, overall_min_connection[1]]
                    else:
                        components_to_nodes[(overall_min_connection[0], index)] = components_to_nodes[(overall_min_connection[1], index)]
                # Then the second distance needs to be deleted
                if index < overall_min_connection[1]:
                    del graph_connections_to_distance[index, overall_min_connection[1]]
                    del components_to_nodes[index, overall_min_connection[1]]
                else:
                    del graph_connections_to_distance[(overall_min_connection[1], index)]
                    del components_to_nodes[(overall_min_connection[1], index)]
                del graph_distance_to_connections[first_distance]
        del graph_components_dict[overall_min_connection[1]]
    return graph

def merge_empty(graph):
    """
    This function takes a graph and merges precincts/blocks with zero people with other precincts/blocks
    """
    empty_nodes = []
    for node in graph.nodes(data=True):
        node_data = node[1]
        # CUTOFF TO MERGE: 20 PEOPLE
        if node_data["total_pop"] < 20:
            empty_nodes.append(node[0])
    print(f"Nodes below population cutoff to merge: {len(empty_nodes)}")
    empty_graph = graph.subgraph(empty_nodes)
    empty_groups = list(nx.algorithms.connected_components(empty_graph))
    for group in empty_groups:
        bordering = set()
        for node in group:
            for other_node in graph.neighbors(node):
                bordering.add(other_node)
        bordering = bordering.difference(set(group))
        substituted_node = list(bordering)[0]
        geometry = [shapely.geometry.shape(graph.nodes[node]["geometry"]) for node in group]
        geometry.append(shapely.geometry.shape(graph.nodes[substituted_node]["geometry"]))
        geometry_union = shapely.ops.unary_union(geometry)
        graph.nodes[substituted_node]["geometry"] = shapely.geometry.mapping(geometry_union)
        graph.add_edges_from([(substituted_node, border_node) for border_node in bordering])
        graph.remove_nodes_from(group)
    return graph

def extract_state(year, state):
    """
    This function takes a year and state and returns all the uncompressed data files in that state's folder. 
    Arguments: 
        year - Either 2010 or 2020. (string)
        state - State name (string) 
    Returns:
        block_demographics - Pandas dataframe for the block_demographics.csv file in the state folder. 
        block_geodata - GeoPandas dataframe for the block_geodata.json file in the state folder.
        demographics - Pandas dataframe for the demographics.csv (or block_group_demographics.csv) file in the state folder.
            Note: demographics will return None for Maine 2020. 
        geodata - GeoPandas dataframe for the geodata.csv (or block_group_geodata.csv) file in the state folder.
        election_data - Pandas dataframe for the election_data.csv (or block_group_geodata.csv) file in the state folder. 
            Note: election_data will return None for Maine 2020. 
    """
    path = data_dir+"/"+str(year)+"/"+str(state) + "/"
    if year == 2010 and state in ["california", "montana", "rhode_island", "oregon"]:
        prefix = "block_group_"
    elif year == 2020 and state in ["california", "hawaii", "oregon", "west_virginia"]:
        prefix = "block_group_"
    else:
        prefix = ""
    if year == 2010:
        geoid_name = "GEOID10"
    else:
        geoid_name = "GEOID20"
    try:
        demographics = pd.read_csv(path+prefix+"demographics.csv", dtype={geoid_name:"string"})
    except:
        subprocess.call(["7z", "e", path+prefix+"demographics.7z", "-o"+path])
        # 2020 Maine has no demographic data.
        if str(state) == "maine" and str(year) == "2020":
            demographics = None
        else:
            demographics = pd.read_csv(path+prefix+"demographics.csv", dtype={geoid_name:"string"})
    print("Demographic data loaded")

    try:
        election_data = pd.read_csv(path+prefix+"election_data.csv", dtype={geoid_name:"string"})
    # Shouldn't be needed but just in case
    except:
        subprocess.call(["7z", "e", path+prefix+"election_data.7z", "-o"+path])
        # 2020 Maine has no election data.
        if str(state) == "maine" and str(year) == "2020":
            election_data = None
        else:
            election_data = pd.read_csv(path+prefix+"election_data.csv", dtype={geoid_name:"string"})
    print("Election data loaded")
    try:
        geodata = gpd.read_file(path+prefix+"geodata.json", dtype={geoid_name:"string"})
    except:
        subprocess.call(["7z", "e", path+prefix+"geodata.7z", "-o"+path])
        geodata = gpd.read_file(path+prefix+"geodata.json", dtype={geoid_name:"string"})
    print(f"Initial number of precincts/block groups: {len(geodata)}")
    print("Geodata loaded")

    try:
        block_demographics = pd.read_csv(path+"block_demographics.csv", skiprows=1)
    except:
        subprocess.call(["7z", "e", path+"block_demographics.7z", "-o"+path])
        block_demographics = pd.read_csv(path+"block_demographics.csv", skiprows=1)
    print("Block demographic data loaded")

    try:
        block_vap_data = pd.read_csv(path+"block_vap_data.csv")
    except:
        subprocess.call(["7z", "e", path+"block_vap_data.7z", "-o"+path])
        block_vap_data = pd.read_csv(path+"block_vap_data.csv")
    print("Block VAP data loaded")

    try:
        block_geodata = gpd.read_file(path+"block_geodata.json")
    except:
        subprocess.call(["7z", "e", path+"block_geodata.7z", "-o"+path])
        block_geodata = gpd.read_file(path+"block_geodata.json")
    print(f"Initial number of blocks: {len(block_geodata)}")
    print("Block geodata data loaded")

    return block_demographics, block_vap_data, block_geodata, demographics, geodata, election_data

def serialize(year, state, checkpoint="beginning"):
    """
    This function links all the data into two adjacency graph json files at the block and precinct levels.
    Arguments: 
        year - Either 2010 or 2020. (string)
        state - State name (string) 
    """
    if checkpoint == "beginning":
        block_demographics, block_vap_data, block_geodata, demographics, geodata, election_data = extract_state(year, state)
        # Project the geometry to North America Albers Equal Area Conic projection
        # NOTE: NOT IN USE at the moment due to introducing geometry errors
        cc = CRS('esri:102008')
        geodata = geodata.to_crs(cc)
        block_geodata = block_geodata.to_crs(cc)
        # Join precinct demographics and geodata
        if year == 2020 and state == "maine":
            # Generate fake GEOIDs for Maine 2020
            geoid_name = "GEOID20"
            county_counters = defaultdict(int)
            geoid_column = {}
            for index, row in geodata.iterrows():
                county = row["COUNTY20"]
                if "/" in county:
                    county = county[:county.find("/")]
                fips_codes = pd.read_csv(data_dir + "/2020/maine/" + "FIPS_to_name.csv")
                try:

                    fake_geoid = str(fips_codes[fips_codes["county_name"] == county]["code"].item()) + str(county_counters[county]+1)
                except:
                    raise Exception(county)
                county_counters[county] += 1
                geoid_column[index] = fake_geoid
            geodata[geoid_name] = pd.Series(geoid_column)
            geodata.set_index(geoid_name, inplace=True)
            geodata.index.names = ['geoid']
        else:
            if year == 2010:
                geoid_name = "GEOID10"
                demographics = demographics[["GEOID10", "Tot_2010_tot", "Wh_2010_tot", "Hisp_2010_tot", "Bl+_2010_tot", "Nat+_2010_tot","Asn+_2010_tot", "Pac+_2010_tot","Tot_2010_vap"]]
            else:
                geoid_name = "GEOID20"
                demographics = demographics[["GEOID20", "Tot_2020_tot", "Wh_2020_tot", "His_2020_tot", "BlC_2020_tot", "NatC_2020_tot","AsnC_2020_tot", "PacC_2010_tot","Tot_2020_vap"]]
            # NOTE: Categories will not add up to 100%, but each percentage of the total will be accurate for how many people
            # in the population are of some race, either alone or in combination with another race
            demographics.columns = ["geoid", "total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander", "total_vap"]
            # Convert geoid column to str
            demographics["geoid"] = demographics["geoid"].astype(str)
            demographics.set_index("geoid", inplace=True)
            demographics.index.names = ['geoid']
            geodata.set_index(geoid_name, inplace=True)
            geodata.index.names = ['geoid']
            geodata = geodata.join(demographics)
        print("Precinct demographics/geodata joined")

        # Join block demographics and geodata
        if year == 2010:
            try:
                block_demographics["id"] = block_demographics["id"].astype(str)
                block_demographics.set_index("id", inplace=True)
            except:
                block_demographics["Geography"] = block_demographics["Geography"].astype(str)
                block_demographics.set_index("Geography", inplace=True)
            try:
                block_demographics["total_pop"] = block_demographics["Total"].astype(int)
            except:
                block_demographics["total_pop"] = block_demographics["Total"].astype(str).str.replace(r"\([A-Za-z0-9]+\)", "").astype(int)
            # White is treated differently to match up with the precinct racial data
            try:
                block_demographics["total_white"] = block_demographics["Total!!Not Hispanic or Latino!!Population of one race!!White alone"].astype(int)
            except:
                block_demographics["total_white"] = block_demographics["Total!!Not Hispanic or Latino!!Population of one race!!White alone"].astype(str).str.replace(r"\([A-Za-z0-9]+\)", "").astype(int)
            try:
                block_demographics["total_hispanic"] = block_demographics["Total!!Hispanic or Latino"].astype(int)
            except:
                block_demographics["total_hispanic"] = block_demographics["Total!!Hispanic or Latino"].astype(str).str.replace(r"\([A-Za-z0-9]+\)", "").astype(int)
            block_demographics["total_black"] = 0
            block_demographics["total_native"] = 0
            block_demographics["total_asian"] = 0
            block_demographics["total_islander"] = 0
            # NOTE: Precinct data includes Hispanic-other race combos in the other race as well, block data does not
            for column in block_demographics.columns:
                if "Annotation" in column:
                    continue
                if "Black or African American" in column:
                    block_demographics["total_black"] += block_demographics[column]
                if "American Indian and Alaska Native" in column:
                    block_demographics["total_native"] += block_demographics[column]
                if "Asian" in column:
                    block_demographics["total_asian"] += block_demographics[column]
                if "Native Hawaiian and Other Pacific Islander" in column:
                    block_demographics["total_islander"] += block_demographics[column]
        else:
            block_demographics["Geography"] = block_demographics["Geography"].astype(str)
            block_demographics.set_index("Geography", inplace=True)
            try:
                block_demographics["total_pop"] = block_demographics[" !!Total:"].astype(int)
            except:
                block_demographics["total_pop"] = block_demographics[" !!Total:"].astype(str).str.replace(r"\([A-Za-z0-9]+\)", "").astype(int)
            try:
                block_demographics["total_white"] = block_demographics[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone"].astype(int)
            except:
                block_demographics["total_white"] = block_demographics[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone"].astype(str).str.replace(r"\([A-Za-z0-9]+\)", "").astype(int)
            try:
                block_demographics["total_hispanic"] = block_demographics[" !!Total:!!Hispanic or Latino"].astype(int)
            except:
                block_demographics["total_hispanic"] = block_demographics[" !!Total:!!Hispanic or Latino"].astype(str).str.replace(r"\([A-Za-z0-9]+\)", "").astype(int)
            block_demographics["total_black"] = 0
            block_demographics["total_native"] = 0
            block_demographics["total_asian"] = 0
            block_demographics["total_islander"] = 0
            for column in block_demographics.columns:
                if "Annotation" in column:
                    continue
                if "Black or African American" in column:
                    block_demographics["total_black"] += block_demographics[column]
                if "American Indian and Alaska Native" in column:
                    block_demographics["total_native"] += block_demographics[column]
                if "Asian" in column:
                    block_demographics["total_asian"] += block_demographics[column]
                if "Native Hawaiian and Other Pacific Islander" in column:
                    block_demographics["total_islander"] += block_demographics[column]
        block_demographics.index.names = ['geoid']
        block_demographics = block_demographics[["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander"]]
        
        # Add VAP data
        block_vap_data["Geography"] = block_vap_data["Geography"].astype(str)
        block_vap_data.set_index("Geography", inplace=True)
        block_vap_data.index.names = ['geoid']
        block_vap_data.columns = ["total_vap"]
        block_demographics = block_demographics.join(block_vap_data)
        # Drop the 1000000US part from the demographic geoids to conform to the geodata geoids
        block_demographics.set_index(block_demographics.index.str[9:], inplace=True)
        print("Block demographics/VAP joined")


        block_geodata.set_index(geoid_name, inplace=True)
        block_geodata.index.names = ['geoid']
        block_geodata = block_geodata.join(block_demographics)
        print("Block demographics/geodata joined")

        # CHECK: VAP amount is less than TOT amount for all precincts/blocks
        if state == "maine" and year == 2020:
            pass
        else:
            if len(geodata) != len(geodata[geodata["total_pop"] >= geodata["total_vap"]]):
                print(geodata[["total_pop", "total_vap"]])
                print(geodata[geodata["total_pop"] < geodata["total_vap"]])
                print(len(geodata), len(geodata[geodata["total_pop"] >= geodata["total_vap"]]))
                raise Exception("DATA ISSUE: Higher VAP than TOTAL for precincts")
            if len(block_geodata) != len(block_geodata[block_geodata["total_pop"] >= block_geodata["total_vap"]]):
                print(block_geodata[["total_pop", "total_vap"]])
                print(block_geodata[block_geodata["total_pop"] < block_geodata["total_vap"]])
                raise Exception("DATA ISSUE: Higher VAP than TOTAL for blocks")

        # Add election data
        if year == 2020 and state == "maine":  
            pass
        else:
            election_data[geoid_name] = election_data[geoid_name].astype(str)
            election_data.set_index(geoid_name, inplace=True)
            election_data.index.names = ["geoid"]
            if year == 2010:
                election_data = election_data[["Tot_2008_pres","Dem_2008_pres","Rep_2008_pres"]]
            else:            
                election_data = election_data[["Tot_2020_pres","D_2020_pres","R_2020_pres"]]
            election_data.columns = ["total_votes", "total_dem", "total_rep"]
            geodata = geodata.join(election_data)
            # CHECK: Election votes is less than VAP amount for all precincts/blocks
            if len(geodata) != len(geodata[geodata["total_vap"] >= geodata["total_votes"]]):
                print(geodata[["total_vap", "total_votes"]])
                print(geodata[geodata["total_vap"] < geodata["total_votes"]])
                print("POSSIBLE DATA ISSUE: Higher votes than VAP for precincts")

        duplicated_geodata = geodata[geodata.duplicated()]
        duplicated_block_geodata = block_geodata[block_geodata.duplicated()]
        # CHECK: There are no repeats in the data that are duplicated. 
        if len(duplicated_geodata) > 0:
            print(duplicated_geodata, "DATA ISSUE: DUPLICATIONS IN INTEGRATED GEODATA")
        if len(duplicated_block_geodata) > 0:
            print(duplicated_block_geodata, "DATA ISSUE: DUPLICATIONS IN INTEGRATED BLOCK GEODATA")

        # Delete duplicates if they exist
        geodata.drop_duplicates(inplace=True)
        block_geodata.drop_duplicates(inplace=True)

        # Drop water precincts
        # print(geodata.index.str.contains("ZZZZZZ"))
        geodata = geodata[~geodata.index.str.contains("ZZZZZZ")]
        # Now that both levels are unified as much as possible, we need to relate them to each other to join them.
        assignment = maup.assign(block_geodata, geodata)
        assignment.columns = ["precinct"]
        assignment = assignment.astype(str)
        # This essentially converts the precincts from floats to GEOID strings
        assignment = assignment.str.split('.').str[0]
        # Aggregate demographic data to precinct level
        if year == 2020 and state == "maine":
            variables = ["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander", "total_vap"]
            geodata[variables] = block_demographics[variables].groupby(assignment).sum()
            geodata.rename(columns={"G20PREDBID":"total_dem", "G20PRERTRU": "total_rep"}, inplace=True)
            geodata["total_votes"] = geodata["total_dem"] + geodata["total_rep"] + geodata["G20PRELJOR"] + geodata["G20PREGHAW"] + geodata["G20PREAFUE"]
        # Prorate election data from precinct to block level    
        weights = block_geodata.total_vap / assignment.map(geodata.total_vap)
        prorated = maup.prorate(assignment, geodata[["total_votes", "total_dem", "total_rep"]], weights)
        block_geodata[["total_votes", "total_dem", "total_rep"]] = prorated.round(3)
        block_geodata[["total_votes", "total_dem", "total_rep"]] = block_geodata[["total_votes","total_dem", "total_rep"]].fillna(0)
        
        geodata.to_file("testing_geodata.json", driver="GeoJSON")
        block_geodata.to_file("testing_block_geodata.json", driver="GeoJSON")
        # geodata["geoid"] = geodata["geoid"].astype(str)
        # block_geodata["geoid"] = block_geodata["geoid"].astype(str)
        # geodata.set_index("geoid", inplace=True)
        # block_geodata.set_index("geoid", inplace=True)
        assignment.to_csv("testing_assignment.csv")
    # For incredibly mysterious reasons this may be necessary to prevent problems
    elif checkpoint == "beginning" or checkpoint == "integration":
        geodata = gpd.read_file("testing_geodata.json", driver="GeoJSON")
        print("Integrated Geodata loaded")
        block_geodata = gpd.read_file("testing_block_geodata.json", driver="GeoJSON")
        geodata["geoid"] = geodata["geoid"].astype(str)
        block_geodata["geoid"] = block_geodata["geoid"].astype(str)
        print("Integrated Block Geodata loaded")
        geodata.set_index("geoid", inplace=True)
        block_geodata.set_index("geoid", inplace=True)
        assignment = pd.read_csv("testing_assignment.csv")

    if checkpoint == "beginning" or checkpoint == "integration":
        # print("23013220363!!", geodata.loc["23013220363"])
        split_geodata = split_multipolygons(geodata, assignment, block_geodata)
        split_geodata.to_file("testing_split_geodata.json", driver="GeoJSON")
        split_block_geodata = split_multipolygons(block_geodata)
        print("Multipolygons split")

        combined_geodata = combine_holypolygons(split_geodata)
        # split_assignment = maup.assign(split_block_geodata, combined_geodata)
        # split_assignment.columns = ["precinct"]
        # split_assignment = split_assignment.astype(str)
        # split_assignment = split_assignment.str[:-2]
        # print(split_assignment)
        # split_assignment.to_csv("testing_split_assignment.csv")
        # print(combined_geodata, len(combined_geodata))
        # combined_block_geodata = combine_holypolygons(split_block_geodata, split_assignment)
        combined_block_geodata = combine_holypolygons(split_block_geodata)
        print("Holes removed")
        combined_geodata.to_file("testing_combined_geodata.json", driver="GeoJSON")
        combined_block_geodata.to_file("testing_combined_block_geodata.json", driver="GeoJSON")

    elif checkpoint == "geometry":
        combined_geodata = gpd.read_file("testing_combined_geodata.json", driver="GeoJSON")
        print("Cleaned Geodata loaded")
        combined_block_geodata = gpd.read_file("testing_combined_block_geodata.json", driver="GeoJSON")
        print("Cleaned Block Geodata loaded")
        combined_geodata["geoid"] = combined_geodata["geoid"].astype(str)
        combined_block_geodata["geoid"] = combined_block_geodata["geoid"].astype(str)
        combined_geodata.set_index("geoid", inplace=True)
        combined_block_geodata.set_index("geoid", inplace=True)

        print(combined_geodata)
        # print(combined_geodata.loc["23003743"])

    if checkpoint in ["beginning", "integration", "geometry"]:
        geodata_graph = convert_to_graph(combined_geodata)
        block_geodata_graph = convert_to_graph(combined_block_geodata)
        print("Graph created")
        nx.write_gpickle(geodata_graph, "test_geodata_graph.gpickle")
        nx.write_gpickle(block_geodata_graph, "test_block_geodata_graph.gpickle")
    else:
        geodata_graph = nx.read_gpickle("test_geodata_graph.gpickle")    
        block_geodata_graph = nx.read_gpickle("test_block_geodata_graph.gpickle")    
        print(len(geodata_graph))
        print(len(block_geodata_graph))
    # Drop water-only precincts and blocks NOTE: not necessary 
    # if year == 2010:
    #     geodata.drop(geodata[geodata["ALAND10"] == 0].index, inplace=True)
    #     block_geodata.drop(block_geodata[block_geodata["ALAND10"] == 0].index, inplace=True)
    # else:
    #     geodata.drop(geodata[geodata["ALAND20"] == 0].index, inplace=True)
    #     block_geodata.drop(block_geodata[block_geodata["ALAND20"] == 0].index, inplace=True)

    # Convert shapely geometry to json
    # for id, data in geodata_graph.nodes(data=True):
    #     data["geometry"] = shapely.geometry.mapping(data['geometry'])
    # for id, data in block_geodata_graph.nodes(data=True):
    #     data["geometry"] = shapely.geometry.mapping(data['geometry'])

    # data = nx.readwrite.json_graph.adjacency_data(geodata_graph)
    # block_data = nx.readwrite.json_graph.adjacency_data(block_geodata_graph)

    # with open(final_dir + f"/{year}/{state}_before_connected_geodata.json", "w") as f:
    #     json.dump(data, f)
    # with open(final_dir + f"/{year}/{state}_before_connected_block_geodata.json", "w") as f:
    #     json.dump(block_data, f)

    connected_geodata_graph = connect_islands(geodata_graph)
    connected_block_geodata_graph = connect_islands(block_geodata_graph)
    for id, data in connected_geodata_graph.nodes(data=True):
        data["geometry"] = shapely.geometry.mapping(data['geometry'])
    for id, data in connected_block_geodata_graph.nodes(data=True):
        data["geometry"] = shapely.geometry.mapping(data['geometry'])
    data = nx.readwrite.json_graph.adjacency_data(connected_geodata_graph)
    block_data = nx.readwrite.json_graph.adjacency_data(connected_block_geodata_graph)

    with open(final_dir + f"/{year}/{state}_geodata.json", "w") as f:
        json.dump(data, f)
    with open(final_dir + f"/{year}/{state}_block_geodata.json", "w") as f:
        json.dump(block_data, f)

    # Create a version with merged precincts/blocks under a certain threshold
    merged_geodata_graph = merge_empty(connected_geodata_graph)
    merged_block_geodata_graph = merge_empty(connected_block_geodata_graph)
    data = nx.readwrite.json_graph.adjacency_data(merged_geodata_graph)
    block_data = nx.readwrite.json_graph.adjacency_data(merged_block_geodata_graph)

    with open(final_dir + f"/{year}/{state}_geodata_merged.json", "w") as f:
        json.dump(data, f)
    with open(final_dir + f"/{year}/{state}_block_geodata_merged.json", "w") as f:
        json.dump(block_data, f)

    # Drop the geometry to create a simplified version (Currently not being used)
    # for id, data in connected_geodata_graph.nodes(data=True):
    #     del data["geometry"]
    # for id, data in connected_block_geodata_graph.nodes(data=True):
    #     del data["geometry"]
    # data = nx.readwrite.json_graph.adjacency_data(connected_geodata_graph)
    # block_data = nx.readwrite.json_graph.adjacency_data(connected_block_geodata_graph)

    # with open(final_dir + f"/{year}/{state}_simplified_geodata.json", "w") as f:
    #     json.dump(data, f)
    # with open(final_dir + f"/{year}/{state}_simplified_block_geodata.json", "w") as f:
    #     json.dump(block_data, f)
    print("Islands connected")


def serialize_all():
    """
    This function automatically serializes all data files in the data directory to json files
    """
    for year in ["2010", "2020"]:
        existing_files = os.listdir(final_dir+"/"+year)
        for root, _, _ in os.walk(data_dir+"/"+year):
            year_pos = root.find(year)
            state = root[year_pos+5:]
            print(year, state)
            if state:
                exists = False
                for file in existing_files:
                    if file in [state+"_geodata.json", state+"_geodata.7z", state+"_geodata_merged.json"]:
                        exists = True
                        break
                if not exists:
                    serialize(int(year), state, checkpoint="beginning")

if __name__ == "__main__":
    # compress_all_data("final")
    # merge_graphs()
    serialize_all()
    # serialize(2010, "north_dakota", checkpoint="geometry")
    # serialize(2010, "missouri", checkpoint="graph")       