import sys
import os
import subprocess
from collections import defaultdict, deque
from itertools import combinations
import json
import time
import shapely
import pandas as pd
import geopandas as gpd

import maup
import networkx as nx

# The link to the data directory in James' computer
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+ "/hte-data-new/raw"
final_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+ "/hte-data-new/graphs"

def compress_all():
    """
    This function automatically compresses all data files in the data directory to .7z files
    """
    for year in ["2010", "2020"]:
        for root, dirs, files in os.walk(data_dir+"/"+year):
            print(root, dirs, files)
            # To hold only files that need to be compressed so that there is no double compression. 
            files_to_compress = []    
            for file in files:
                # Files that don't need to be compressed.
                if file in ["desktop.ini", "demographics.csv", "election_data.csv", "block_group_demographics.csv", "block_group_election_data.csv", "README.md"]:
                    continue
                full_path = os.path.join(root, file)  
                corresponding_file = file[:file.find(".")] + ".7z"
                print(corresponding_file)
                if not corresponding_file in files:
                    files_to_compress.append([os.path.join(root, corresponding_file), full_path])
            for file in files_to_compress:
                    subprocess.call(["7z", "a", file[0], file[1]])

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
                    new_geoid = geoid + str(id)
                    new_row = {}
                    # new_row = pd.Series(index=[row.index])
                    # new_row.index.names = [new_geoid]
                    id += 1
                    for block_geoid, block in blocks.iterrows():
                        if not block_geoid in used:
                            if polygon.intersection(block["geometry"]).area > 0.7*block["geometry"].area:
                                # for column, value in block_data.loc[block_geoid].iteritems():
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
                    rows_to_add[new_geoid] = new_row
                row_geoids_to_remove.append(geoid)
            else:
                # Block geodata, which means that area has to be used.
                total_area = row["geometry"].area
                id = 1
                for polygon in polygons:
                    new_geoid = str(geoid) + str(id)
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
    # bounds = geodata.bounds.astype(float)
    # min_xes_dict = {}
    hole_geoids = []
    # Generate minimum x coordinates
    # for geoid, value in min_xes.iteritems():
    #     if value == -72.125922:
    #         print("caught!", geoid)
    #     min_xes_dict[value] = geoid
    # source_geometry = []
    target_geometry = {"geoid":[], "geometry":[]}
    for geoid, row in geodata.iterrows():
        interiors = row["geometry"].interiors
        # Check if there's actually a hole that needs to be fixed
        if len(interiors) > 0:
            # target_geometry.append(shapely.geometry.Polygon(row["geometry"].exterior))
            # target_geometry.append(geoid)
            for interior in interiors:
                target_geometry["geoid"].append(geoid)
                target_geometry["geometry"].append(shapely.geometry.Polygon(interior))
    # source_geometry = gpd.GeoSeries(source_geometry, crs=geodata.crs)
    source_geometry = geodata
    target_geometry = gpd.GeoDataFrame(target_geometry, crs=geodata.crs)
    hole_assignment = maup.assign(source_geometry, target_geometry)
    # for hole_geoid, geoid in hole_assignment.groupby[geoid].iteritems():
    for id, row in target_geometry.iterrows():
        geoid = row["geoid"]
        geodata.loc[geoid, "geometry"] = shapely.geometry.Polygon(geodata.loc[geoid, "geometry"].exterior)
        target_area = row["geometry"].area
        confirmed_area = 0
        current_hole_geoids = hole_assignment[hole_assignment == id].index
        for hole_geoid in current_hole_geoids:
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
        if abs(target_area-confirmed_area) > 1e-10:
            print(target_area, confirmed_area, "AREA MISMATCH", geoid)
    # for geoid, row in geodata.iterrows():
    #     interiors = row["geometry"].interiors
    #     # Check if there's actually a hole that needs to be fixed
    #     if len(interiors) > 0:
    #         print(geoid, len(interiors))
    #         for interior in interiors:
    #             min_x, min_y, max_x, max_y = interior.bounds
    #             target_area = shapely.geometry.Polygon(interior).area
    #             confirmed_area = 0
    #             # min_x = interior.min_x
    #             # max_x = interior.max_x
    #             # min_y = interior.min_y
    #             # max_y = interior.max_y
    #             # Get the id of the hole NOTE: PROBABLY THE PROBLEM for debugging
    #             # print("min_x to be checked", geoid, min_x)
    #             # hole_id = min_xes_dict[min_x]
    #             # hole_row = geodata.loc[hole_id]
    #             if type(assignment) == pd.Series:
    #                 geodata_domain = geodata.loc[assignment[assignment == assignment[geoid]].index]
    #                 # print(geodata_domain, geoid)  
    #             else:
    #                 geodata_domain = geodata
    #             for hole_id, hole_row in geodata_domain.iterrows():
    #                 # start = time.time()
    #                 hole_min_x, hole_min_y, hole_max_x, hole_max_y = bounds.loc[hole_id]
    #                 # print(min_x, hole_id)
    #                 # print(time.time()-start)
    #                 # Check to make sure the hole actually fits
    #                 # print(shapely.geometry.Point(hole_row["geometry"].exterior.coords[0]).distance(interior))
    #                 if float(min_x) <= float(hole_min_x) and float(max_x) >= float(hole_max_x) and float(max_y) >= float(hole_max_y) and float(min_y) <= float(hole_min_y):
    #                     # if hole_row["geometry"].within(interior):
    #                     # start = time.time()

    #                     point = shapely.geometry.Point(hole_row["geometry"].exterior.coords[0])
    #                     # Either point is in the hole or on the boundary
    #                     if point.within(shapely.geometry.Polygon(interior)) or point.distance(interior) < 1e-10:
    #                         # print(time.time()-start, "geometry within")
    #                         confirmed_area += hole_row["geometry"].area
    #                         # Add the data from the hole to the surrounding precinct/block
    #                         geodata.loc[geoid, "total_pop"] += hole_row["total_pop"]
    #                         geodata.loc[geoid, "total_white"] += hole_row["total_white"]
    #                         geodata.loc[geoid, "total_hispanic"] += hole_row["total_hispanic"]
    #                         geodata.loc[geoid, "total_black"] += hole_row["total_black"]
    #                         geodata.loc[geoid, "total_asian"] += hole_row["total_asian"]
    #                         geodata.loc[geoid, "total_native"] += hole_row["total_native"]
    #                         geodata.loc[geoid, "total_islander"] += hole_row["total_islander"]
    #                         geodata.loc[geoid, "total_vap"] += hole_row["total_vap"]
    #                         geodata.loc[geoid, "total_votes"] += hole_row["total_votes"]
    #                         geodata.loc[geoid, "total_dem"] += hole_row["total_dem"]
    #                         geodata.loc[geoid, "total_rep"] += hole_row["total_rep"]
    #                         hole_geoids.append(hole_id)
    #             if abs(target_area-confirmed_area) > 1e-10:
    #                 print(target_area, confirmed_area, "AREA MISMATCH", geoid)
    #         geodata.loc[geoid, "geometry"] = shapely.geometry.Polygon(row["geometry"].exterior)
    print(hole_geoids)
    # Delete the holes
    modified_geodata = geodata.drop(hole_geoids)
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
    print(len(geodata))
    for geoid, row in geodata.iterrows():
        graph.add_node(geoid, **row.to_dict())
        # geometry_to_id[row["geometry"]] = geoid
        # min_xes_dict[row["geometry"].min_x] = geoid
    bounds = geodata.bounds.astype(float)
    min_xes = bounds["minx"]
    print(len(min_xes), "len min xes")

    min_xes.index = geodata.index
    min_xes_list = []
    for geoid, min_x in min_xes.iteritems():
        # if min_x in min_xes_dict.keys():
            # print(min_xes_dict[min_x], geoid, "ALREADY CAUGHT")
        # min_xes_dict[min_x] = geoid
        min_xes_list.append([min_x, geoid])
    min_xes_list = sorted(min_xes_list, key = lambda x: x[0])
    
    edges = []
    # This will keep only precincts/blocks which have a chance of bordering the first (current) precinct/block
    # active_geoids = [min_xes_dict[min_xes_list[0]]]
    active_geoids = [min_xes_list[0][1]]
    edges_added = 0
    attempts = 0
    for i, [current_min_x, current_geoid] in enumerate(min_xes_list):
        print(f"Block:  {i}/{len(min_xes_list)}")
        # current_min_x = min_xes_list[i]
        # current_geoid = min_xes_dict[current_min_x]
        print(current_geoid)
        _, current_min_y, current_max_x, current_max_y = bounds.loc[current_geoid]
        current_geometry = geodata.loc[current_geoid, "geometry"]
        possible_borders = []
        # Surprisingly, popping is faster for some reason
        # to_remove = []
        for j, other_geoid in enumerate(active_geoids):
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
        if current_geoid == "50011501105":
            print(active_geoids, "active_geoids")
        # print("COMPARISON", len(active_geoids), len(to_remove), len(possible_borders))
        for other_geoid in possible_borders:
            start_time = time.time() 
            _, other_min_y, _, other_max_y = bounds.loc[other_geoid]
            if other_min_y > current_max_y or other_max_y < current_min_y:
                continue
            other_geometry = geodata.loc[other_geoid, "geometry"]
            intersection = current_geometry.intersection(other_geometry)
            if current_geoid == "50011501105":
                print(intersection, other_geoid)
            attempts += 1
            if intersection.is_empty or isinstance(intersection, shapely.geometry.Point) or isinstance(intersection, shapely.geometry.MultiPoint):
                # print("FAILED", current_geoid, other_geoid)
                continue
            elif isinstance(intersection, shapely.geometry.collection.GeometryCollection):
                for element in intersection:
                    if isinstance(element, shapely.geometry.LineString) or isinstance(element, shapely.geometry.MultiLineString):
                        # print("edge added!", [current_geoid, other_geoid])
                        edges.append([current_geoid, other_geoid])
                        continue
                    edges_added += 1
            elif isinstance(intersection, shapely.geometry.LineString) or isinstance(intersection, shapely.geometry.MultiLineString):
                # print("edge added!", [current_geoid, other_geoid])
                edges.append([current_geoid, other_geoid])
                edges_added += 1
            else:
                print("INTERSECTION PROBLEM", current_geoid, other_geoid, type(intersection))
            # print("WOO time:", time.time()-start_time, edges_added/attempts, edges_added, attempts)
        active_geoids.append(current_geoid)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
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
    # print(graph_components_list, "graph components list")
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
        print(f"\r{i}/{int(graph_components_num*(graph_components_num-1)/2)}", sep="")
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
    # print(graph_connections_to_distance)
    # print(components_to_nodes)
    # print(components_to_nodes, graph_connections_to_distance)
    # index_list = [i for i in range(0, graph_components_num)]
    for i in range(0, graph_components_num-1):
        min_distance = min(graph_distance_to_connections.keys())
        overall_min_connection = graph_distance_to_connections[min_distance]
        graph.add_edge(components_to_nodes[overall_min_connection][0], components_to_nodes[overall_min_connection][1])
        # Now delete the added connection from the lists 
        del graph_distance_to_connections[min_distance]
        del graph_connections_to_distance[overall_min_connection]
        del components_to_nodes[overall_min_connection]
        # Compare the other possibilities and their distances to both newly connected components to find which one
        # is better to keep
        # print(overall_min_connection, "overall_min_connection")
        # print(f"Component connections: {i}/{graph_components_num-1}")
        for index in graph_components_dict:
            # print(index)
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
            # print(first_distance, second_distance, "first second comparison")
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
                # print(graph_connections_to_distance, "first tripped")
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
                # print(graph_connections_to_distance, "second tripped")
        # index_list.remove(overall_min_connection[1])
        del graph_components_dict[overall_min_connection[1]]
        # del graph_components_dict[overall_min_connection[1]]
    # nx.algorithms.number_connected_components(graph)
    # graph_components_list = list(nx.algorithms.connected_components(graph))
    # print(graph_components_list, "graph components")
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

    try:
        demographics = pd.read_csv(path+prefix+"demographics.csv")
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+prefix+"demographics.7z"])
        # 2020 Maine has no demographic data.
        if str(state) == "maine" and str(year) == 2020:
            demographics = None
        demographics = pd.read_csv(path+prefix+"demographics.csv")
    print("Demographic data loaded")

    try:
        election_data = pd.read_csv(path+prefix+"election_data.csv")
    # Shouldn't be needed but just in case
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+prefix+"election_data.7z"])
        # 2020 Maine has no election data.
        if str(state) == "maine" and str(year) == 2020:
            election_data = None
        else:
            election_data = pd.read_csv(path+prefix+"election_data.csv")
    print("Election data loaded")

    try:
        geodata = gpd.read_file(path+prefix+"geodata.json")
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+"demographics.7z"])
        geodata = gpd.read_file(path+prefix+"geodata.json")
    print("Geodata loaded")

    try:
        block_demographics = pd.read_csv(path+"block_demographics.csv", skiprows=1)
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+"block_demographics.7z"])
        block_demographics = pd.read_csv(path+"block_demographics.csv", skiprows=1)
    print("Block demographic data loaded")

    try:
        block_vap_data = pd.read_csv(path+"block_vap_data.csv")
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+"block_vap_data.7z"])
        block_vap_data = pd.read_csv(path+"block_vap_data.csv")
    print("Block VAP data loaded")

    try:
        block_geodata = gpd.read_file(path+"block_geodata.json")
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+"block_geodata.7z"])
        block_geodata = gpd.read_file(path+"block_geodata.json")
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
        # Join precinct demographics and geodata
        if year == 2020 and state == "maine":
            # Generate fake GEOIDs for Maine 2020
            geoid_name = "GEOID20"
            county_counters = defaultdict(int)
            geoid_column = []
            for index, row in geodata.iterrows():
                county = row["COUNTY20"]
                fips_codes = pd.read_csv(data_dir + "/2020/maine/" + "FIPS_to_name.csv")
                fake_geoid = str(fips_codes[county]) + str(county_counters[county]+1)
                county_counters[county] += 1
                geoid_column.append(fake_geoid)
            geodata[geoid_name] = geoid_column
            geodata.index.names = ['geoid']

        else:
            if year == 2010:
                geoid_name = "GEOID10"
                demographics = demographics[["GEOID10", "Tot_2010_tot", "Wh_2010_tot", "Hisp_2010_tot", "Bl+_2010_tot", "Nat+_2010_tot","Asn+_2010_tot", "Pac+_2010_tot","Tot_2010_vap"]]
            else:
                geoid_name = "GEOID20"
                demographics = demographics[["GEOID20", "Tot_2020_tot", "Wh_2020_tot", "His_2020_tot", "BlC_2020_tot", "NatC_2020_tot","AsnC_2020_tot", "PacC_2010_tot","Tot_2020_vap"]]
            # NOTE: Categories will not add up to 100%, but each percentage of the total will be accurate for how many poeple
            # in the population are of some race, either alone or in combination with another race
            demographics.columns = ["geoid", "total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander", "total_vap"]
            # Convert geoid column to str
            demographics["geoid"] = demographics["geoid"].astype(str)
            demographics.set_index("geoid", inplace=True)
            demographics.index.names = ['geoid']
            geodata.set_index(geoid_name, inplace=True)
            geodata.index.names = ['geoid']
            print(demographics)
            print(geodata)
            geodata = geodata.join(demographics)
            print(geodata)
        print("Precinct demographics/geodata joined")

        # Join block demographics and geodata
        if year == 2010:
            block_demographics["id"] = block_demographics["id"].astype(str)
            block_demographics.set_index("id", inplace=True)
            block_demographics["total_pop"] = block_demographics["Total"]
            # White is treated differently to match up with the precinct racial data
            block_demographics["total_white"] = block_demographics["Total!!Not Hispanic or Latino!!Population of one race!!White alone"]
            block_demographics["total_hispanic"] = block_demographics["Total!!Hispanic or Latino"]
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
            block_demographics["total_pop"] = block_demographics[" !!Total:"]
            block_demographics["total_white"] = block_demographics[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone"]
            block_demographics["total_hispanic"] = block_demographics[" !!Total:!!Hispanic or Latino"]
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
        if len(geodata) != len(geodata[geodata["total_pop"] >= geodata["total_vap"]]):
            raise Exception("DATA ISSUE: Higher VAP than TOTAL for precincts")
        if len(block_geodata) != len(block_geodata[block_geodata["total_pop"] >= block_geodata["total_vap"]]):
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
            print(geodata)
            print(election_data)
            geodata = geodata.join(election_data)
            print(geodata)
            # CHECK: Election votes is less than VAP amount for all precincts/blocks
            if len(geodata) != len(geodata[geodata["total_vap"] >= geodata["total_votes"]]):
                raise Exception("DATA ISSUE: Higher votes than VAP for precincts")

        # Drop water-only precincts and blocks
        # if year == 2010:
        #     geodata.drop(geodata[geodata["ALAND10"] == 0].index, inplace=True)
        #     block_geodata.drop(block_geodata[block_geodata["ALAND10"] == 0].index, inplace=True)
        # else:
        #     geodata.drop(geodata[geodata["ALAND20"] == 0].index, inplace=True)
        #     block_geodata.drop(block_geodata[block_geodata["ALAND20"] == 0].index, inplace=True)

        # Now that both levels are unified as much as possible, we need to relate them to each other to join them.
        assignment = maup.assign(block_geodata, geodata)
        assignment.columns = ["precinct"]
        assignment = assignment.astype(str)
        assignment = assignment.str[:-2]
        # Prorate election data from precinct to block level    
        weights = block_geodata.total_vap / assignment.map(geodata.total_vap)
        print(weights)
        prorated = maup.prorate(assignment, geodata[["total_votes", "total_dem", "total_rep"]], weights)
        print(prorated)
        block_geodata[["total_votes", "total_dem", "total_rep"]] = prorated.round(3)
        block_geodata[["total_votes", "total_dem", "total_rep"]] = block_geodata[["total_votes","total_dem", "total_rep"]].fillna(0)
        # Aggregate demographic data to precinct level
        if year == 2020 and state == "maine":
            variables = ["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander"]
            geodata[variables] = block_demographics[variables].groupby(assignment).sum()
        
        geodata.to_file("testing_geodata.json", driver="GeoJSON")
        block_geodata.to_file("testing_block_geodata.json", driver="GeoJSON")
        assignment.to_csv("testing_assignment.csv")
    elif checkpoint == "integration":
        geodata = gpd.read_file("testing_geodata.json", driver="GeoJSON")
        print("Geodata loaded")
        block_geodata = gpd.read_file("testing_block_geodata.json", driver="GeoJSON")
        print("Block Geodata loaded")
        geodata["geoid"] = geodata["geoid"].astype(str)
        block_geodata["geoid"] = block_geodata["geoid"].astype(str)
        geodata.set_index("geoid", inplace=True)
        block_geodata.set_index("geoid", inplace=True)
        assignment = pd.read_csv("testing_assignment.csv")

    if checkpoint == "beginning" or checkpoint == "integration":
        print(assignment)
        print(geodata, len(geodata), geodata.index)
        split_geodata = split_multipolygons(geodata, assignment, block_geodata)
        print(split_geodata, len(split_geodata))
        split_geodata.to_file("testing_split_geodata.json", driver="GeoJSON")
        split_block_geodata = split_multipolygons(block_geodata)
        print("Multipolygons split")
        print(split_geodata, len(split_geodata))

        combined_geodata = combine_holypolygons(split_geodata)
        split_assignment = maup.assign(split_block_geodata, combined_geodata)
        split_assignment.columns = ["precinct"]
        split_assignment = split_assignment.astype(str)
        split_assignment = split_assignment.str[:-2]
        print(split_assignment)
        split_assignment.to_csv("testing_split_assignment.csv")
        print(combined_geodata, len(combined_geodata))
        combined_block_geodata = combine_holypolygons(split_block_geodata, split_assignment)
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

    if checkpoint in ["beginning", "integration", "geometry"]:
        geodata_graph = convert_to_graph(combined_geodata)
        block_geodata_graph = convert_to_graph(combined_block_geodata)
        print("Graph created")
        nx.write_gpickle(geodata_graph, "test_geodata_graph.gpickle")
        nx.write_gpickle(block_geodata_graph, "test_block_geodata_graph.gpickle")
    else:
        geodata_graph = nx.read_gpickle("test_geodata_graph.gpickle")    
        block_geodata_graph = nx.read_gpickle("test_block_geodata_graph.gpickle")    

    connected_geodata = connect_islands(geodata_graph)
    connected_block_geodata = connect_islands(block_geodata_graph)
    print("Islands connected")
    for id, data in geodata_graph.nodes(data=True):
        data["geometry"] = shapely.geometry.mapping(data['geometry'])
    for id, data in block_geodata_graph.nodes(data=True):
        data["geometry"] = shapely.geometry.mapping(data['geometry'])
    data = nx.readwrite.json_graph.adjacency_data(connected_geodata)
    # adjacency_geodata = nx.readwrite.json_graph.adjacency_graph(data)

    block_data = nx.readwrite.json_graph.adjacency_data(connected_block_geodata)
    # adjacency_block_geodata = nx.readwrite.json_graph.adjacency_graph(data)

    with open(final_dir + f"/{year}/{state}_geodata.json", "w") as f:
        json.dump(data, f)
    with open(final_dir + f"/{year}/{state}_block_geodata.json", "w") as f:
        json.dump(block_data, f)

if __name__ == "__main__":
    # compress_all()
    # serialize(2020, "vermont", checkpoint="beginning")
    # serialize(2020, "vermont", checkpoint="integration")
    serialize(2020, "vermont", checkpoint="geometry")
    # serialize(2020, "vermont", checkpoint="graph")