import sys
import os
import subprocess

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

if __name__ == "__main__":
    compress_all()