"""
Converts the Census VAP files so that they only include the total VAP. 
"""
import os
import pandas as pd
import subprocess
# The link to the data directory in James' computer
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+ "/hte-data-new/raw"
for year in ["2010", "2020"]:
    print(data_dir+"/"+year)
    for root, dirs, files in os.walk(data_dir+"/"+year):
        for dir in dirs:
            path = root + "/" + dir + "/block_vap.csv"
            try:
                vap = pd.read_csv(path, skiprows=1)
            except FileNotFoundError:
                continue
            vap.set_index("Geography", inplace=True)
            if year == "2010":
                vap = vap["Total"]
            else:
                vap = vap[" !!Total:"]
            vap.index.names = ['Geography']
            vap.to_csv(root + "/" + dir + "/block_vap_data.csv")
            os.remove(root + "/" + dir + "/block_vap.csv")
            print("state done!")