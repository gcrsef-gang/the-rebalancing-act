import sys
import os
import subprocess

import pandas as pd
import geopandas as gpd

import maup

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
                if file in ["demographics.csv", "election_data.csv", "block_group_demographics.csv", "block_group_election_data.csv", "README.md"]:
                    continue
                full_path = os.path.join(root, file)  
                corresponding_file = file[:file.find(".")] + ".7z"
                print(corresponding_file)
                if not corresponding_file in files:
                    files_to_compress.append([os.path.join(root, corresponding_file), full_path])
            for file in files_to_compress:
                    subprocess.call(["7z", "a", file[0], file[1]])

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
        block_geodata = gpd.read_file(path+"block_geodata.json")
    except FileNotFoundError:
        subprocess.call(["7z", "e", path+"block_geodata.7z"])
        block_geodata = gpd.read_file(path+"block_geodata.json")
    print("Block geodata data loaded")

    return block_demographics, block_geodata, demographics, geodata, election_data

def serialize(year, state):
    """
    This function links all the data into two adjacency graph json files at the block and precinct levels.
    Arguments: 
        year - Either 2010 or 2020. (string)
        state - State name (string) 
    """
    block_demographics, block_geodata, demographics, geodata, election_data = extract_state(year, state)
    print(block_demographics, block_geodata)
    # Join precinct demographics and geodata
    if year == 2010:
        geoid_name = "GEOID10"
        demographics = demographics[["GEOID10", "Tot_2010_tot", "Wh_2010_tot", "Hisp_2010_tot", "Bl+_2010_tot", "Nat+_2010_tot","Asn+_2010_tot", "Pac+_2010_tot","Tot_2010_vap"]]
    else:
        geoid_name = "GEOID20"
        demographics = demographics[["GEOID20", "Tot_2020_tot", "Wh_2020_tot", "His_2020_tot", "BlC_2020_tot", "NatC_2020_tot","AsnC_2020_tot", "PacC_2010_tot","Tot_2020_vap"]]
    demographics.set_index(geoid_name)
    # NOTE: Categories will not add up to 100%, but each percentage of the total will be accurate for how many poeple
    # in the population are of some race, either alone or in combination with another race
    demographics.rename(["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander", "total_vap"])
    geodata.set_index(geoid_name)
    geodata = geodata.join(block_demographics)

    # Join block demographics and geodata
    if year == 2010:
        block_demographics.set_index("id")
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
        block_demographics.set_index("Geography")
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
    block_demographics = block_demographics[["total_pop", "total_white", "total_hispanic", "total_black", "total_native", "total_asian", "total_islander"]]
    block_geodata.set_index(geoid_name)
    block_geodata = block_geodata.join(block_demographics)
    
    # Add election data
    if year == 2020 and state == "maine":  
        pass
    else:
        election_data.set_index(geoid_name)
        if year == 2010:
            election_data = election_data[["Tot_2008_pres","Dem_2008_pres","Rep_2008_pres"]]
        else:            
            election_data = election_data[["Tot_2020_pres","D_2020_pres","R_2020_pres"]]
        election_data.rename(["total_2008", "total_dem", "total_rep"])
        geodata.join(election_data)
    # Now that both levels are unified as much as possible, we need to relate them to each other to join them. 
if __name__ == "__main__":
    # compress_all()
    serialize(2020, "vermont")