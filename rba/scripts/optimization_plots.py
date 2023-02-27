"""Creates a graph of community representation score over time, visualize all 10 best maps.

usage:
python3 optimization_plots.py <state> <num_steps> <num_districts> <ensemble_dir> <optimize_dir>

state - name of the state
num_steps - number of steps of ensemble analysis to use in creating heatmap
num_distrcts - number of districts
ensemble_dir - directory holding output from ensemble analysis
optimize_dir - directory holding output from optimization

THINGS TO NOTE BEFORE RUNNING:
- ensemble.py MUST have the final scores_df block of code uncommented and the rest commented.
- Save previous ensemble visualizations (the 3 main ones), AS THOSE WILL BE OVERWRITTEN.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rba.ensemble import ensemble_analysis
from rba.util import create_folder


if __name__ == "__main__":
    package_dir = os.path.dirname(os.path.realpath(__file__))

    state = sys.argv[1]
    num_steps = int(sys.argv[2])
    num_districts = int(sys.argv[3])
    ensemble_dir = sys.argv[4]
    optimize_dir = sys.argv[5]

    graph_file = os.path.join(package_dir, f"../data/2010/{state}_geodata_merged.json")
    difference_file = os.path.join(package_dir, f"../data/2010/{state}_communities.json")
    vra_config_file = os.path.join(package_dir, f"../data/2010/vra_{state}.json")

    # State gerry scores over time.
    df = pd.read_csv(os.path.join(optimize_dir, "optimization_stats.csv"))

    plt.plot(np.arange(len(df.index)), df["state_gerry_score"])
    plt.title("Statewide RBA Score")
    plt.xlabel("Iteration")
    plt.savefig(os.path.join(optimize_dir, "simulated_annealing.png"))
    plt.clf()

    # Gerrymandering heatmap for the best map
    with open(os.path.join(optimize_dir, f"Plan_1.json"), "r") as f:
        parts = json.load(f)

    ensemble_analysis(graph_file, difference_file, vra_config_file, num_steps, num_districts,
                      initial_plan_file=None, district_file=parts, output_dir=ensemble_dir,optimize_vis=True,
                      vis_dir=optimize_dir,verbose=True)