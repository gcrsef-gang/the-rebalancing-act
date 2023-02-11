"""
Fix florida's messed up (actually too detailed) 2020 precinct demographics and election data.
"""
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

dem = pd.read_csv("../hte-data-new/raw/2020/florida/demographics.csv", index_col="GEOID20")
new_dem = {}
old_dem = pd.DataFrame(columns=dem.columns)
for row, values in dem.iterrows():
    # print(dem)
    if len(str(row)) == 12:
        if str(row)[:-1] in new_dem:
            for i, value in enumerate(values):
                new_dem[str(row)[:-1]][i] += value
        else:
            new_row = values
            new_dem[str(row)[:-1]] = new_row
        old_dem = old_dem.append(dem.loc[row])
new_dem = pd.DataFrame.from_dict(new_dem, orient='columns').transpose()
print(new_dem)
print(old_dem, dem)
final = dem.drop(old_dem.index, axis=0)
final = pd.concat([final, new_dem])
final.to_csv("better_florida_demographics.csv")

elec = pd.read_csv("../hte-data-new/raw/2020/florida/election_data.csv", index_col="GEOID20")
new_elec = {}
old_elec = pd.DataFrame(columns=elec.columns)
for row, values in elec.iterrows():
    # print(elec)
    if len(str(row)) == 12:
        if str(row)[:-1] in new_elec:
            for i, value in enumerate(values):
                new_elec[str(row)[:-1]][i] += value
        else:
            new_row = values
            new_elec[str(row)[:-1]] = new_row
        old_elec = old_elec.append(elec.loc[row])
new_elec = pd.DataFrame.from_dict(new_elec, orient='columns').transpose()
print(new_elec)
print(old_elec, elec)
final = elec.drop(old_elec.index, axis=0)
final = pd.concat([final, new_elec])
final.to_csv("better_florida_election_data.csv")