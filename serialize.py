import sys
import os
import subprocess
from collections import defaultdict, deque

import shapely
import pandas as pd
import geopandas as gpd

import maup
import networkx as nx

# The link to the data directory in James' computer
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+ "/hte-data-new/raw"


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
    Returns:
        modified_geodata - split GeoDataFrame
    """
    row_geoids_to_remove = []
    rows_to_add = []
    for geoid, row in geodata.iterrows():
        if type(row["geometry"]) == shapely.MultiPolygon:
            if assignment:
                # Precinct geodata, which means the values can be aggregated upwards for each precinct. 
                blocks= block_data.groupby(assignment)[geodata]
                used = []
                id = 1
                for polygon in polygons:
                    new_geoid = geoid + id
                    new_row = pd.Series(index=[row.columns])
                    new_row.index.names = [new_geoid]
                    id += 1
                    for block_geoid, block in blocks.iterrows():
                        if not block_geoid in used:
                            if polygon.intersection(block["geometry"]).area() > 0.7*block["geometry"].area():
                                for column, value in block_data[block_geoid].iteritems():
                                    # Obviously use precinct polygon geometry, not block geometry
                                    if column == "geometry":
                                        new_row[column] = polygon
                                    elif type(value) in [int, float]:
                                        try:
                                            new_row[column] += value
                                        except:
                                            new_row[column] = value
                                    else:
                                        new_row[column] = value
                                used += block_geoid
                    rows_to_add.append(new_row)
                row_geoids_to_remove.append(geoid)
            else:
                # Block geodata, which means that area has to be used.
                total_area = row["geometry"].area()
                polygons = list(row["geometry"])
                id = 1
                for polygon in polygons:
                    new_geoid = geoid + id
                    new_row = pd.Series(index=[row.columns])
                    new_row.index.names = [new_geoid]
                    id += 1
                    percentage = polygon.area()/total_area
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
                    rows_to_add.append(new_row)
                row_geoids_to_remove.append(geoid)
    modified_geodata = geodata.drop(row_geoids_to_remove)
    modified_geodata = modified_geodata.concat(rows_to_add)
    return modified_geodata


def combine_holypolygons(geodata):
    """
    This function takes a GeoDataFrame and combines all the polygons that are encased within another polygon.
    Arguments:
        geodata - GeoPandas GeoDataFrame. 
    Returns:
        modified_geodata - non-holy GeoDataFrame
    """
    min_xes = {}
    hole_geoids = []
    for geoid, row in geodata.iterrows():
        min_xes[row["geometry"].min_x] = geoid
    for geoid, row in geodata.iterrows():
        interiors = row["geometry"].interiors
        if len(interiors) > 0:
            for interior in interiors:
                min_x = interior.min_x
                max_x = interior.max_x
                min_y = interior.min_y
                max_y = interior.max_y
                hole_id = min_xes[min_x]
                hole_row = geodata[hole_id]
                hole_max_x = hole_row["geometry"].max_x
                hole_min_y = hole_row["geometry"].min_y
                hole_max_y = hole_row["geometry"].max_y
                if max_x == hole_max_x and max_y == hole_max_y and min_y == hole_min_y:
                    row["total_pop"] += hole_row["total_pop"]
                    row["total_white"] += hole_row["total_white"]
                    row["total_hispanic"] += hole_row["total_hispanic"]
                    row["total_black"] += hole_row["total_black"]
                    row["total_asian"] += hole_row["total_asian"]
                    row["total_native"] += hole_row["total_native"]
                    row["total_islander"] += hole_row["total_islander"]
                    row["total_vap"] += hole_row["total_vap"]
                    row["total_votes"] += hole_row["total_votes"]
                    row["total_dem"] += hole_row["total_dem"]
                    row["total_rep"] += hole_row["total_rep"]
                    hole_geoids.append(hole_id)
    modified_geodata = geodata.drop(hole_geoids)
    return modified_geodata


def connect_islands(geodata):
    """ 
    This function takes a GeoDataFrame and connects blocks/precincts wout connections to their nearest block/precinct. 
    """

def convert_to_graph(geodata):
    """
    This function converts a GeoDataFrame into a networkx Graph. 
    """
    geometry_to_id = {}
    min_xes = {}
    graph = nx.Graph()
    for geoid, row in geodata.iteritems():
        graph.add_node(geoid, row.to_dict())
        geometry_to_id[row["geometry"]] = geoid
        min_xes[row["geometry"].min_x] = geoid
    min_xes.sort()
    min_xes_list = min_xes.keys()
    edges = []
    # This will keep only precincts/blocks which have a chance of bordering the first (current) precinct/block
    sliding_scale = deque([min_xes[0]])
    i = 1
    while len(sliding_scale) > 0:
        current = sliding_scale.popleft()
        current_geoid = min_xes[current]
        current_geometry = geodata[current_geoid]["geometry"]
        current_max = current_geometry.max_x
        other_geoids = []
        while min_xes[i] < current_max:
            sliding_scale.append(min_xes_list[i])
            other_geoids.append(min_xes[min_xes_list[i]])
        for other_geoid in other_geoids:
            other_geometry = geodata[current_geoid]["geometry"]
            intersection = current_geometry.intersection(other_geometry)
            if intersection.is_empty or isinstance(intersection, shapely.geometry.Point):
                continue
            elif isinstance(intersection, shapely.geometry.LineString):
                edges.append([current_geoid, other_geoid])
            else:
                print("INTERSECTION PROBLEM", current_geoid, other_geoid, type(intersection))
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
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
    files_list = ["demographics.csv", "geodata.json", "election_geodata.csv", "block_demographics.csv", "block_geodata.json"]
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

def serialize(year, state):
    """
    This function links all the data into two adjacency graph json files at the block and precinct levels.
    Arguments: 
        year - Either 2010 or 2020. (string)
        state - State name (string) 
    """
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
        demographics.rename(["geoid", "total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander", "total_vap"], inplace=True)
        demographics.set_index("geoid", inplace=True)
        demographics.index.names = ['geoid']
        # geodata.rename(columns={geoid_name : "geoid"}, inplace=True)
        geodata.set_index(geoid_name, inplace=True)
        geodata.index.names = ['geoid']
        geodata = geodata.join(block_demographics)

    # Join block demographics and geodata
    if year == 2010:
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
            if "Black or African American" in column:
                block_demographics["total_black"] += block_demographics[column]
        for column in block_demographics.columns:
            if "American Indian and Alaska Native" in column:
                block_demographics["total_native"] += block_demographics[column]
        for column in block_demographics.columns:
            if "Asian" in column:
                block_demographics["total_asian"] += block_demographics[column]
        for column in block_demographics.columns:
            if "Native Hawaiian and Other Pacific Islander" in column:
                block_demographics["total_islander"] += block_demographics[column]
            # block_demographics["Total!!Not Hispanic or Latino!!Population of one race!!Black or African American alone"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of two races!!White; Black or African American"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of two races!!Black or African American; American Indian and Alaska Native"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of two races!!Black or African American; Asian"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of two races!!Black or African American; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of two races!!Black or African American; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!White; Black or African American; American Indian and Alaska Native"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!White; Black or African American; Asian"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!White; Black or African American; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!White; Black or African American; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!Black or African American; American Indian and Alaska Native; Asian"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!Black or African American; American Indian and Alaska Native; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!Black or African American; Asian; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of three races!!Black or African American; Asian; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!White; Black or African American; American Indian and Alaska Native; Asian"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!White; Black or African American; American Indian and Alaska Native; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!White; Black or African American; Asian; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!White; Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!Black or African American; American Indian and Alaska Native; Asian; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of four races!!Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of five races!!White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of five races!!White; Black or African American; American Indian and Alaska Native; Asian; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of five races!!White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of five races!!White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
            # block_demographics["Total!!Not Hispanic or Latino!!Two or More Races!!Population of five races!!Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race"] + \
    else:
        block_demographics.set_index("Geography", inplace=True)
        block_demographics["total_pop"] = block_demographics[" !!Total:"]
        block_demographics["total_white"] = block_demographics[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone"]
        block_demographics["total_hispanic"] = block_demographics[" !!Total:!!Hispanic or Latino"]
        block_demographics["total_black"] = 0
        block_demographics["total_native"] = 0
        block_demographics["total_asian"] = 0
        block_demographics["total_islander"] = 0
        for column in block_demographics.columns:
            if "Black or African American" in column:
                block_demographics["total_black"] += block_demographics[column]
        for column in block_demographics.columns:
            if "American Indian and Alaska Native" in column:
                block_demographics["total_native"] += block_demographics[column]
        for column in block_demographics.columns:
            if "Asian" in column:
                block_demographics["total_asian"] += block_demographics[column]
        for column in block_demographics.columns:
            if "Native Hawaiian and Other Pacific Islander" in column:
                block_demographics["total_islander"] += block_demographics[column]
    block_demographics.index.names = ['geoid']
    block_demographics = block_demographics[["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander"]]
    
    # Add VAP data
    block_vap_data.set_index("Geography")
    block_vap_data.index.names = ['geoid']
    block_vap_data.rename(["total_vap"])
    block_demographics.join(block_vap_data)

    block_geodata.set_index(geoid_name, inplace=True)
    block_geodata.index.names = ['geoid']
    block_geodata = block_geodata.join(block_demographics)
    
    # Add election data
    if year == 2020 and state == "maine":  
        pass
    else:
        election_data.set_index(geoid_name, inplace=True)
        election_data.index.names = "geoid"
        if year == 2010:
            election_data = election_data[["Tot_2008_pres","Dem_2008_pres","Rep_2008_pres"]]
        else:            
            election_data = election_data[["Tot_2020_pres","D_2020_pres","R_2020_pres"]]
        election_data.rename(["total_votes", "total_dem", "total_rep"], inplace=True)
        geodata.join(election_data)
    if year == 2010:
        geodata.drop(geodata[geodata["ALAND10"] == 0])
    else:
        geodata.drop(geodata[geodata["ALAND20"] == 0])

    # Now that both levels are unified as much as possible, we need to relate them to each other to join them.
    assignment = maup.assign(block_geodata, geodata)
    # Prorate election data from precinct to block level
    # TODO: Switch this out for CVAP data
    weights = block_geodata.total_vap / assignment.map(geodata.total_vap)
    prorated = maup.prorate(assignment, geodata[["total_votes", "total_dem", "total_rep"]], weights)
    block_geodata[["total_votes", "total_dem", "total_rep"]] = prorated.round(3)

    # Aggregate demographic data to precinct level
    if year == 2020 and state == "maine":
        variables = ["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander"]
        geodata[variables] = block_demographics.groupby(assignment).sum()
    
    split_geodata = split_multipolygons(geodata, assignment, block_geodata)
    split_block_geodata = split_multipolygons(block_geodata)

    combined_geodata = combine_holypolygons(split_geodata)
    combined_block_geodata = combine_holypolygons(split_block_geodata)

    geodata_graph = convert_to_graph(combined_geodata)
    block_geodata_graph = convert_to_graph(combined_block_geodata)
    connect_islands()


    data = nx.readwrite.json_graph.adjacency_data(G)
    H = nx.readwrite.json_graph.adjacency_graph(data)

if __name__ == "__main__":
    compress_all()
    # serialize(2020, "vermont")