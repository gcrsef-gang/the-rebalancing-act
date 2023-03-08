from gerrychain import Partition
import json
import networkx as nx
import sys
import rba

graph = sys.argv[1]
with open(graph, "r") as f:
    data = json.load(f)
graph = nx.readwrite.json_graph.adjacency_graph(data)

difference_file = sys.argv[2]
output_path = sys.argv[3]
num_frames = int(sys.argv[4])

partition = Partition(graph=graph, assignment={node: graph.nodes[node]["COUNTYFP10"] for node in graph.nodes})

rba.visualization.visualize_community_generation(difference_file, output_path, graph, num_frames, partition)