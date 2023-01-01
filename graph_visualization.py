"""Draws graphs (data structure)

Usage:
python3 graph_visualization.py <pickled_graph> (output_path | "None")
"""

import pickle
import sys
import json

from PIL import Image, ImageDraw
import networkx as nx
import shapely.geometry
import numpy as np

def modify_coords(coords, bounds):
    """Squishes coords into a bounding box.

    :param coords: The points to squish.
    :type coords: list of list of float (coordinate pairs)

    :param bounds: The box in which the inputted coords must be squished into. In format `[max_x, max_y]` (mins are 0).
    :type bounds: list of float

    :return: A list of coords that are squished into the bounds.
    :rtype: list of list of float
    """

    n_points = len(coords)

    X = np.zeros(n_points)
    Y = np.zeros(n_points)

    for i, point in enumerate(coords):
        X[i] = point.x
        Y[i] = point.y

    # Move to first quadrant.
    min_x = min(X)
    min_y = min(Y)

    for p in range(n_points):
        X[p] += -(min_x)
        Y[p] += -(min_y)

    # Get the bounding box dimensions
    bounding_box_width = max(X) - min(X)
    bounding_box_length = max(Y) - min(Y)

    # Dilate to fit within canvas
    dilation_factor = max([bounding_box_width / bounds[0], bounding_box_length / bounds[1]])
    dilation_factor = (1 / dilation_factor) * 0.95
    for i in range(n_points):
        X[i] *= dilation_factor
        Y[i] *= dilation_factor

    # Reflect because y is flipped in Pillow
    for i in range(n_points):
        Y[i] = bounds[1] - Y[i]

    # Center
    max_x = max(X)
    min_y = min(Y)
    for i in range(n_points):
        X[i] += (bounds[0] - max_x) / 2
        Y[i] -= min_y / 2

    # Translate down a bit (quick fix)
    # TODO: Actually solve the problem of coords being too high up.
    while min(Y) < 0:
        for i in range(n_points):
            Y[i] += 10
    
    new_coords = [[float(X[i]), float(Y[i])] for i in range(n_points)]

    return new_coords

def visualize_graph(graph, output_path, coords, colors=None, edge_colors=None, sizes=None, show=False):
    """Creates an image of a graph and saves it to a file.

    :param graph: The graph you want to visualize.
    :type graph: `networkx.Graph`

    :param output_path: Path to where image should be saved. None if you don't want it to be saved.
    :type output_path: str or NoneType

    :param coords: A function that outputs coordinates of a node.
    :type coords: function that outputs list of 2 float

    :param colors: A function that outputs an rgb code for each node, defaults to None
    :type colors: (function that outputs tuple of 3 float) or NoneType

    :param sizes: A function that outputs the radius for each node.
    :type sizes: (function that outputs float) or NoneType

    :param show: whether or not to show the image once generated.
    :type show: bool
    """
    graph_image = Image.new("RGB", (2000,2000), "white")
    draw = ImageDraw.Draw(graph_image, "RGB")

    graph_nodes = list(graph.nodes())
    graph_edges = list(graph.edges())
    for node in graph_nodes:
        try: 
            _ = coords(node)
        except:
            print(graph.nodes[node]['geometry']['coordinates'])
            print(node, graph[node])
    modified_coords = modify_coords(
        [coords(node) for node in graph_nodes], [2000, 2000]
    )
    
    if colors is not None:
        node_colors = [colors(node) for node in graph_nodes]
    else:
        node_colors = [(235, 64, 52) for _ in graph_nodes]
        
    # if edge_colors is not None:
    #     edge_colors = [edge_colors(edge) for edge in graph_edges]
    # else:
    #     edge_colors = [(0,0,0) for _ in graph_edges]

    if sizes is not None:
        node_sizes = [sizes(node) for node in graph_nodes]
    else:
        node_sizes = [1 for _ in graph_nodes]

    for edge in graph_edges:

        indices = [graph_nodes.index(v) for v in edge]
        centers = [tuple(modified_coords[i]) for i in indices]
        if edge_colors is not None:
            edge_color = edge_colors(edge)
        else:
            edge_color = (0,0,0)
        draw.line(
            centers,
            fill=edge_color,
            width=1
    )

    for center, node, color, r in zip(modified_coords, graph_nodes,
                                      node_colors, node_sizes):
        draw.ellipse(
            [(center[0] - r, center[1] - r),
             (center[0] + r, center[1] + r)],
            fill=color
        )

    if output_path is not None:
        graph_image.save(output_path)
    if show:
        graph_image.show()


if __name__ == "__main__":

    # If called from the command line, it is
    # assumed that the node attributes are precincts.

    with open(sys.argv[1], "rb") as f:
        graph = nx.readwrite.json_graph.adjacency_graph(json.load(f))
        # graph = pickle.load(f)

    visualize_graph(graph, sys.argv[2], lambda node: shapely.geometry.shape(graph.nodes[node]['geometry']).centroid, show=True)