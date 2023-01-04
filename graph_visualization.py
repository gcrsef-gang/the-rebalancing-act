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
    # y_coords = [42.72685, 42.848564, 43.022452, 43.213817, 43.354144, 43.533341, 43.607655, 43.750837, 43.873348, 44.008176, 44.107845, 44.185469, 44.304375, 44.382911, 44.461759, 44.492504, 44.57663, 44.670759, 44.781578, 44.858507]
    # y_coords = [shapely.geometry.Point((-73, coord)) for coord in y_coords]
    # new_y_coords = modify_coords(y_coords, [2000,2000])
    # for coord in new_y_coords:
    #     print([coord, (coord[0]-100, coord[1])])
    #     draw.line([tuple(coord), (coord[0]-1000, coord[1])], width=1, fill=(0,255,255))
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
    
    # def colors(node):
    #     groupings = [{'50003VD30', '50025VD221', '50025VD208', '50025VD211', '50025VD2022', '50025VD2021', '50025VD220', '50003VD36', '50025VD207', '50003VD25-2', '50003VD25-1', '50003VD39', '50003VD31', '50025VD217'}, {'50025VD221', '50025VD211', '50003VD35', '50025VD2021', '50025VD213', '50025VD216', '50025VD218', '50025VD2192', '50003VD25-2', '50003VDBEN1', '50003VD39', '50003VD31', '50025VDWIN1', '50025VD203', '50003VD34', '50003VD25-1', '50025VD2023', '50025VD204', '50025VD215', '50025VD2022', '50025VD212', '50025VD205'}, {'50003VD26', '50025VD2191', '50003VD24', '50025VD210', '50003VD35', '50003VD33', '50025VD222', '50025VD201', '50003VD37', '50003VD28', '50025VD213', '50025VD216', '50025VD218', '50025VD2192', '50003VDBEN1', '50025VDWIN1', '50025VD203', '50003VD27', '50025VD206', '50025VD214', '50025VD215', '50025VD212', '50025VD209', '50003VD38', '50003VD29'}, {'50003VD26', '50027VD228', '50025VD210', '50027VD224', '50003VD33', '50027VD2401', '50025VD222', '50027VD232', '50027VD223', '50021VD158', '50027VD2402', '50027VD229', '50003VD28', '50003VD32-2', '50003VD32-1', '50021VD166', '50003VD27', '50025VD206', '50027VD243', '50027VD242', '50025VD214', '50003VD29', '50021VD167'}, {'50027VD228', '50021VD177', '50027VD224', '50021VD1612', '50021VD156', '50027VD232', '50027VD223', '50021VD158', '50027VD231', '50027VD245', '50021VD174', '50021VD165', '50021VD176', '50021VD170', '50021VD178', '50021VD166', '50027VD236', '50027VD234', '50027VD243', '50027VD242', '50021VD164', '50027VD244', '50021VD167'}, {'50027VD227', '50021VD1612', '50021VD156', '50021VD163', '50021VD155', '50021VD1724', '50027VD246', '50027VD231', '50021VD174', '50027VD2302', '50021VD180', '50021VD170', '50021VD1723', '50021VD162', '50021VD1722', '50021VD1731', '50021VD1732', '50027VD236', '50027VD234', '50021VD159', '50021VD164', '50027VD244', '50021VD1611', '50021VD179'}, {'50021VD171', '50021VD153', '50027VD2301', '50027VD227', '50021VD168', '50027VD225', '50021VD163', '50021VD155', '50021VD1724', '50021VD160', '50021VD175', '50027VD2302', '50027VD235', '50021VD169', '50021VD180', '50021VD1723', '50021VD162', '50021VD1722', '50021VD1731', '50027VD241', '50021VD1732', '50027VD239', '50021VD159', '50021VD1721', '50021VD1611', '50021VD179', '50027VD233', '50021VD157'}, {'50021VD153', '50027VD237', '50021VD168', '50027VD225', '50001VD18', '50021VD160', '50021VD175', '50017VD126', '50027VD235', '50021VD169', '50001VD9', '50027VD226', '50027VD241', '50017VD122', '50027VD239', '50027VD238', '50001VD22', '50017VD127', '50021VD154', '50017VD129', '50001VD8', '50001VD6', '50001VD14', '50027VD233', '50021VD157'}, {'50001VD7', '50027VD237', '50017VD118', '50001VD18', '50017VD125', '50001VD11', '50017VD117', '50017VD126', '50001VD2', '50017VD119', '50001VD4', '50017VD120', '50001VD9', '50001VD16', '50027VD226', '50017VD132', '50017VD130', '50017VD122', '50027VD238', '50001VD17', '50001VD22', '50017VD127', '50017VD129', '50017VD121', '50001VD8', '50001VD6'}, {'50001VD7', '50017VD124', '50001VD13', '50001VD20', '50017VD118', '50023VD195', '50001VD11', '50017VD117', '50017VD133', '50001VD2', '50017VD119', '50001VD4', '50017VD120', '50001VD16', '50017VD128', '50001VD10', '50017VD130', '50001VD21', '50023VD197', '50017VD123', '50017VD131', '50001VD1', '50017VD121', '50001VD3', '50023VD193'}, {'50005VD491', '50017VD124', '50001VD13', '50001VD20', '50001VD23', '50005VD43', '50001VD15', '50023VD195', '50017VD133', '50023VD188', '50023VD182', '50023VD192', '50017VD128', '50001VD12', '50001VD5', '50023VD196', '50001VD19', '50001VD10', '50023VD1811', '50007VDCHI1', '50023VD197', '50017VD123', '50001VD1', '50023VD183', '50005VD492', '50001VD3', '50023VD193'}, {'50005VD43', '50007VD59', '50023VD188', '50007VD62-21', '50023VD182', '50023VD192', '50023VD1813', '50023VD187', '50005VD48', '50001VD12', '50001VD5', '50023VD196', '50001VD19', '50007VD62-1', '50023VD1811', '50007VD63', '50005VD40', '50007VDCHI1', '50023VD190', '50023VD183', '50023VD189', '50023VD1812', '50023VD194', '50005VD492', '50023VD186', '50023VD191'}, {'50005VD55', '50007VD57', '50023VD184', '50023VD185', '50007VD59', '50009VD78', '50023VD187', '50005VD48', '50007VD62-1', '50007VD62-22', '50007VD63', '50005VD40', '50007VD66', '50007VD68-2', '50007VD67', '50007VD68-1', '50023VD190', '50005VD42', '50007VD72', '50023VD200', '50023VD189', '50023VD198', '50023VD186', '50023VD191'}, {'50005VD55', '50007VD57', '50015VD114', '50007VD69-1', '50009VD83', '50007VD69-3', '50023VD184', '50007VD58-5', '50005VD50', '50009VD78', '50023VD199', '50007VD64', '50015VD110', '50007VD69-2', '50005VT45', '50007VD69-4', '50007VD58-6', '50005VD54', '50007VD66', '50007VD68-2', '50007VD67', '50007VD68-1', '50005VD42', '50007VD72', '50023VD200', '50007VD70', '50023VD198'}, {'50015VD113', '50015VD114', '50007VD69-1', '50007VD58-7', '50009VD81', '50007VD58-4', '50009VD83', '50007VD69-3', '50007VD58-5', '50005VD50', '50007VD61-1', '50007VD64', '50015VD110', '50007VD58-1', '50009VD86', '50007VD69-2', '50005VT45', '50007VD69-4', '50007VD58-6', '50005VD54', '50005VD44', '50007VD61-3', '50007VD73', '50007VD58-3', '50007VD58-2', '50007VD70', '50007VD60-1', '50005VD46'}, {'50015VD113', '50015VD116', '50007VD58-7', '50009VD81', '50007VD58-4', '50007VD71', '50007VD60-2', '50015VD108', '50007VD65-1', '50007VD61-1', '50009VD86', '50005VD56', '50007VD61-2', '50009VD80', '50009VD84', '50005VD52', '50013VD106', '50007VD61-3', '50007VD73', '50007VD58-2', '50015VD111', '50005VD41', '50007VD60-1', '50019VD142', '50005VD46'}, {'50015VD116', '50009VDESX3', '50005VD51', '50007VD71', '50011VD90', '50015VD112', '50019VD141', '50015VD108', '50007VD65-1', '50019VD139', '50011VD94', '50007VD65-2', '50005VD56', '50009VD80', '50009VD79', '50009VD84', '50009VD76', '50015VD109', '50005VD52', '50013VD106', '50015VD111', '50005VD41', '50005VD47', '50005VD53', '50011VD92', '50019VD142'}, {'50011VD91', '50009VDESX3', '50019VD144', '50005VD51', '50011VD90', '50019VD141', '50019VD139', '50011VD94', '50007VD65-2', '50011VD87', '50015VD115', '50009VD75', '50009VD79', '50009VD76', '50019VD135', '50015VD109', '50011VD99', '50019VD134', '50019VD146', '50015VD107', '50013VD105', '50009VD74', '50005VD47', '50005VD53', '50011VD92', '50019VD152', '50013VD103'}, {'50009VDESX4', '50011VD91', '50019VD144', '50011VD101', '50019VD138', '50019VD147', '50019VD136', '50013VD104', '50011VD87', '50009VD75', '50009VD82', '50009VDESX2', '50019VD135', '50019VD137', '50011VD99', '50019VD134', '50019VD146', '50015VD107', '50011VD96', '50013VD105', '50013VD102', '50009VD74', '50011VD89', '50019VD152', '50019VD151', '50019VD150', '50019VD149', '50013VD103'}, {'50009VDESX4', '50019VD140', '50011VD101', '50011VD97', '50019VD138', '50009VD77', '50019VD147', '50019VD148', '50011VD95', '50019VD136', '50009VDESX1', '50011VD100', '50013VD104', '50009VDESX6', '50009VDESX5', '50011VD93', '50019VD145', '50009VD85', '50009VD82', '50009VDESX2', '50011VD88', '50019VD137', '50019VD143', '50011VD89', '50013VD102', '50019VD151', '50019VD150', '50019VD149'}]
    #     num = 0
    #     for i in range(0, len(groupings)):
    #         if node in groupings[i]:
    #             num += 1
    #     if node == "50013VD106":
    #         print("southerner detected!")
    #         print(num)
    #     if node == "50013VD103":
    #         print("northerner detected!")
    #         print(num)
    #     if num == 0:
    #         return (255, 255, 255)
    #     if num == 1:
    #         return (0, 0, 255)
    #     if num == 2:
    #         return (255, 0, 0)

    # visualize_graph(graph, sys.argv[2], lambda node: shapely.geometry.shape(graph.nodes[node]['geometry']).centroid, colors=colors, show=True)
    visualize_graph(graph, sys.argv[2], lambda node: shapely.geometry.shape(graph.nodes[node]['geometry']).centroid, show=True)