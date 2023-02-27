"""Draws graphs (data structure)

Usage:
python3 graph_visualization.py <pickled_graph> (output_path | "None")
"""

import os
import sys
import json
import math

from PIL import Image, ImageDraw, ImageFont
import geopandas
import networkx as nx
import shapely.geometry
import shapely.ops
import numpy as np
import random
import gerrychain
import matplotlib.pyplot as plt
from pyproj import CRS

from . import community_generation
from . import util

# IMAGE_DIMS = (2000, 2000)
IMAGE_DIMS = (5000, 5000)
IMAGE_BG = "white"
EDGE_WIDTH_FACTOR = 15


def get_eqn(p1, p2):
    """Returns the slope and y-intercept of the segment connecting two poitns"""
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return m, p1[1] - m * p1[0]


def get_partisanship_color(percent_rep) -> tuple:
    """Returns RGB color on a blue-purple-red scale according to partisanship (as a decimal)."""
    if percent_rep > .5:
        # Republican majority - purple to red.
        return (207, 27, int(-180 * percent_rep + 207))
    else:
        # Democratic majority - blue to purple.
        return (int(180 * percent_rep + 27), 27, 207)


def modify_coords(coords, bounds):
    """Squishes coords into a bounding box.

    :param coords: The points to squish.
    :type coords: list of list of float or shapely.coords.CoordinateSequence

    :param bounds: The box in which the inputted coords must be squished into. In format `[max_x, max_y]` (mins are 0).
    :type bounds: list of float

    :return: A list of coords that are squished into the bounds.
    :rtype: list of tuple of float
    """

    n_points = len(coords)

    X = np.zeros(n_points)
    Y = np.zeros(n_points)

    for i, point in enumerate(coords):
        X[i] = point[0]
        Y[i] = point[1]

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
    
    new_coords = [(float(X[i]), float(Y[i])) for i in range(n_points)]

    return new_coords


def visualize_gradient_geopandas(precincts, get_value, get_geometry, *args, img_path=None,
                                 show=False, clear=True, **kwargs):
    """Visualizes a variable on a gradient using geopandas.

    Parameters
    ----------
    precincts : container of precincts
        Contains a list of precinct names.
    get_value : callable
        Returns a metric value for a given precinct
    get_geometry : callable
        Returns the geometry (GeoJSON list form, not shapely) for a given precinct.
    img_path : str, default=None
        Optional path to save the image.
    show : bool, default=False
        Whether or not to call plt.show()
    clear : bool, default=True
        Whether or not to call plt.clf()
    Also takes any parameters taken by geopandas.GeoDataFrame.plot()
    """
    gdf = geopandas.GeoDataFrame(columns=["val", "geometry"])
    gdf.set_geometry("geometry", inplace=True)
    gdf.set_crs(CRS('esri:102008'), allow_override=True, inplace=True) 
    for precinct in precincts:
        gdf.loc[len(gdf.index)] = [get_value(precinct), shapely.geometry.shape(get_geometry(precinct))]
        # gdf.loc[len(gdf.index)] = [get_value(precinct), get_geometry(precinct)]
    gdf.plot(
        column="val",
        *args,
        **{key: arg for key, arg in kwargs.items() if key not in ["img_path", "show", "clear"]}
    )
    if img_path is not None:
        plt.savefig(img_path)
    if show:
        plt.show()
    if clear:
        plt.clf()


def visualize_partition_geopandas(partition, *args, graph=None, union=False, img_path=None,
                                  show=False, clear=True, **kwargs):
    """Visualizes a gerrychain.Partition object using geopandas.

    Parameters
    ----------
    partition : gerrychain.Partition
        Partition to visualize.
    graph : gerrychain.Graph, default=None
        State precinct graph. Only needs to be provided if partition is a SimplePartition.
    union : bool, default=False
        Whether or not to visualize the partitions as a single polygon, as opposed to just showing
        their assignment by coloring.
    img_path : str, default=None
        Optional path to save the image.
    show : bool, default=False
        Whether or not to call plt.show()
    clear : bool, default=True
        Whether or not to call plt.clf()
    Also takes any parameters taken by geopandas.GeoDataFrame.plot()
    """
    if graph is None:
        graph = partition.graph

    if union:
        data = {"assignment": [], "geometry": []}

        for part in partition.parts:
            data["assignment"].append(part)
            geoms = [shapely.geometry.shape(graph.nodes[node]["geometry"])
                     for node in partition.parts[part]]
            data["geometry"].append(shapely.ops.unary_union(geoms))

    else:
        data = {"assignment": [], "geometry": []}
        for node in graph:
            data["assignment"].append(partition.assignment[node])
            data["geometry"].append(shapely.geometry.shape(graph.nodes[node]["geometry"]))

    gdf = geopandas.GeoDataFrame(data)
    del data
    gdf.plot(
        column="assignment",
        *args,
        **{key: arg for key, arg in kwargs.items() if key not in ["union", "img_path", "show", "clear"]}
    )
    if img_path is not None:
        plt.savefig(img_path)
    if show:
        plt.show()
    if clear:
        plt.clf()


def visualize_map(graph, output_fpath, node_coords, edge_coords, node_colors=None, edge_colors=None,
        edge_widths=None, partition=None, node_list=None, additional_polygons=None, text=None, show=False):
    """Creates an image of a map and saves it to a file.

    Parameters
    ----------
    graph : nx.Graph
        Contains adjacency data to avoid redundancy while drawing edges
    output_fpath : str
        Path to output file (PNG)
    node_coords : function
        Returns shapely geometry for any given node.
    edge_coords : function, optional
        Returns coordinate representation of a MultiLineString for any given edge expressed as a
        tuple of nodes.
    node_colors : function, optional
        Returns node colors (RGB)
    edge_colors : function, optional
        Returns edge colors (RGB)
    edge_widths : function, optional
        Returns width (px) for any given edge expressed as a tuple of nodes
    node_list : list, optional
        Alternative list of nodes from those in the graph. This is allowed because nodes are only
        involved in filling in colors.
    additional_polygons : list of shapely.geometry.Polygon, optional
        List of additional polygons to draw (such as the overall border if it is omitted from the
        edge coords). These are drawn with no fill and a black border with width of 1px.
    text : string, optional
        Text to display in the top-right corner.
    show : bool, default False
        Whether or not to display the generated map
    """
    map_image = Image.new("RGB", IMAGE_DIMS, IMAGE_BG)
    draw = ImageDraw.Draw(map_image, "RGB")

    if edge_widths is None:
        edge_widths = lambda edge: 1
    if edge_colors is None:
        edge_colors = lambda edge: "black"
    if node_colors is None:
        node_colors = lambda node: "black"
    if node_list is None:
        node_list = list(graph.nodes)

    # Get flattened list of all coordinates so that they can be transformed into the same bounding box.
    all_flattened_coords = []
    edge_width_values = []  # each value corresponds with two values (a line segment) in the list above
    edge_color_values = []  # same deal as edge_width_values
    node_end_indices = []  # contains the index of the last coordinate of each node
    node_color_values = []  # same number of elements as node_end_indices
    additional_polygon_end_indices = []  # same deal as node_end_indices
    for u, v in graph.edges:
        # Just drop connections between islands and the mainland
        try:
            for line_string in edge_coords((u, v)):
                all_flattened_coords.append(line_string[0])
                all_flattened_coords.append(line_string[1])
                edge_width_values.append(edge_widths((u, v)))
                edge_color_values.append(edge_colors((u, v)))
        except KeyError:
            # print("THINGS BEING DROPPED", u, v)
            continue
    num_edge_line_strings = len(edge_color_values) - 1
    split_nodes_num = 0
    for u in node_list:
        u_coords = node_coords(u)
            # print(shapely.geometry.mapping(u_coords))
        # Could be multiline string if the coordinates are a multipolygon
        if isinstance(u_coords.boundary, shapely.geometry.MultiLineString):
            for ls in u_coords.boundary.geoms:
                all_flattened_coords += ls.coords
                node_end_indices.append(len(all_flattened_coords) - 1)
                node_color_values.append(node_colors(u))
                split_nodes_num += 1
            split_nodes_num -= 1
        else:
            all_flattened_coords += list(u_coords.boundary.coords)
            node_end_indices.append(len(all_flattened_coords) - 1)
            node_color_values.append(node_colors(u))
        # if u == "2403707-001s3":
            # print(u_coords, node_end_indices[-1], "SEEN!")
    for poly in additional_polygons:
        if isinstance(poly, shapely.geometry.MultiPolygon):
            for polygon in poly.geoms:
                if isinstance(polygon.boundary, shapely.geometry.MultiLineString):
                    for ls in polygon.boundary.geoms:
                        all_flattened_coords += list(ls.coords)
                        additional_polygon_end_indices.append(len(all_flattened_coords) - 1)
                else:
                    all_flattened_coords += list(polygon.boundary.coords)
                    additional_polygon_end_indices.append(len(all_flattened_coords) - 1)
        else:
            all_flattened_coords += list(poly.boundary.coords)
            additional_polygon_end_indices.append(len(all_flattened_coords) - 1)
    all_flattened_coords = modify_coords(all_flattened_coords, IMAGE_DIMS)

    # Fill in colors (in a lower layer than borders)
    for i in range(len(node_list)+split_nodes_num):
        if i == 0:
            # Because each edge line string corresponds to two points in the all flattened coords
            start_index = num_edge_line_strings*2 + 2
        else:
            start_index = node_end_indices[i - 1] + 1
        draw.polygon(all_flattened_coords[start_index : node_end_indices[i] + 1], fill=node_color_values[i])
    # Draw outlines
    for i in range(num_edge_line_strings):
        draw.line(all_flattened_coords[(i * 2) : (i * 2 + 2)], fill=edge_color_values[i], width=edge_width_values[i])
    for i in range(len(additional_polygons)):
        if i == 0:
            start_index = node_end_indices[-1] + 1
        else:
            start_index = additional_polygon_end_indices[i - 1] + 1
        draw.polygon(all_flattened_coords[start_index : additional_polygon_end_indices[i] + 1], fill=None, outline="black", width=2)

    # Text
    if text is not None:
        fnt = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "assets/UbuntuMono-R.ttf"), 80)
        draw.text([IMAGE_DIMS[0] - 500, 50], text, font=fnt, fill="black")

    if output_fpath is not None:
        map_image.save(output_fpath)

    if show:
        map_image.show()

def get_coords(graph):
    print("Getting node coordinates... ", end="")
    sys.stdout.flush()
    node_coords = {}
    for u in graph.nodes:
        coords = graph.nodes[u]["geometry"]["coordinates"]
        if isinstance(coords[0][0][0], list):
            polygons = []
            for i, polygon in enumerate(coords):
                for j, point in enumerate(polygon[0]):
                    coords[i][0][j][0] = round(point[0], 4)
                    coords[i][0][j][1] = round(point[1], 4)
                polygons.append(shapely.geometry.Polygon(coords[i][0]))
            node_coords[u] = shapely.geometry.MultiPolygon(polygons=polygons)
        else:
            for i, point in enumerate(coords[0]):
                coords[0][i][0] = round(point[0], 4)
                coords[0][i][1] = round(point[1], 4)
            node_coords[u] = shapely.geometry.Polygon(coords[0])
        if u == "2403707-001":
            print(node_coords[u])
    print("Done!")

    print("Getting edge coordinates... ", end="")
    sys.stdout.flush()
    edge_coords = {}
    i = 0
    for u, v in graph.edges:
        intersection = node_coords[u].intersection(node_coords[v])
        if not intersection.is_empty:
            if isinstance(intersection, shapely.geometry.LineString):
                edge_coords[frozenset((u, v))] = [intersection.coords]
                # if i == 0:
                    # i += 1
            elif isinstance(intersection, shapely.geometry.MultiLineString):
                edge_coords[frozenset((u, v))] = [ls.coords for ls in intersection.geoms]
                # print([ls.coords for ls in intersection.geoms], "intersection coords")
            elif isinstance(intersection, shapely.geometry.collection.GeometryCollection):
                edge_coords[frozenset((u,v))] = [ls.coords for ls in intersection.geoms if isinstance(ls, shapely.geometry.LineString)]
            else:
                raise ValueError(f"Intersection between {u} and {v} is not a LineString or"
                                + f" MultiLineString. It is of type {type(intersection)}")
        # else:
            # This is an edge between an island and another precinct, so just connect their centers
            # print([shapely.geometry.mapping(node_coords[u].centroid), shapely.geometry.mapping(node_coords[v].centroid)])
            # unwrapped = [shapely.geometry.mapping(node_coords[u].centroid)["coordinates"], shapely.geometry.mapping(node_coords[v].centroid)["coordinates"]]
            # edge_coords[frozenset((u,v))] = [shapely.geometry.LineString(unwrapped).coords]
            # raise ValueError(f"Overlap not found between {u} and {v}")
    print("Done!")
    print("Computing outer border of state... ", end="")
    sys.stdout.flush()
    overall_border = shapely.ops.unary_union([node_coords[u] for u in graph.nodes])
    print("Done!")

    return node_coords, edge_coords, overall_border

def visualize_community_generation(difference_fpath, output_fpath, graph, num_frames, partition=None):
    """Writes frames for an animated visual of the community generation algorithm to a folder.
    The animation depicts borders between communities as black and borders between precints as gray.
    It also uses edge width as an indicator of similarity, and color as an indicator of
    community partisanship.
    
    Parameters
    ----------
    difference_fpath : str
        Path to JSON file containing edge lifetimes (communitygen ouptut).
    output_fpath : str
        Path to directory where frames will be stored as PNG (will be created if necessary).
    graph : nx.Graph
        State precinct graph.
    num_frames : int
        Number of frames in animation.
    """
    print("Loading supercommunity output data... ", end="")
    sys.stdout.flush()
    with open(difference_fpath, "r") as f:
        supercommunity_output = json.load(f)  # Contains strings as keys.

    differences = {}
    for edge, lifetime in supercommunity_output.items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        # print(edge, (u, v), type(u), type(v))
        if u == '19001001022' and v =='19001001021':
            print("DETECTED", type((u,v)), (u,v))
        if ((u,v) == ('19001001022', '19001001021')):
            print("should be added!")
        differences[frozenset((u, v))] = lifetime
    print("Done!")

    max_lt = max(differences.values())
    min_lt = min(differences.values())
    edge_widths = {
        edge: int((lt - min_lt) / max_lt * EDGE_WIDTH_FACTOR) + 1 for edge, lt in differences.items()
    }

    # node_colors = {
    #     u: get_partisanship_color(graph.nodes[u]["total_rep"] / graph.nodes[u]["total_votes"])
    #     for u in graph.nodes
    # }
    node_colors = {node: (random.randint(20, 235), random.randint(20, 235), random.randint(20, 235)) for node in graph.nodes}
    random_node_colors = {node: (random.randint(20, 235), random.randint(20, 235), random.randint(20, 235)) for node in graph.nodes}

    node_coords, edge_coords, overall_border = get_coords(graph)

    try:
        os.mkdir(output_fpath)
    except FileExistsError:
        pass


    living_edges = set(frozenset(e) for e in graph.edges)
    # unrendered_contractions = [frozenset(supercommunity_output[e]) for e in graph.edges]  # Not a set because order must be preserved.
    unrendered_contractions = []  # Not a set because order must be preserved.
    for edge in graph.edges:
        # print(edge[0], edge[1], differences[edge], "edge")
        unrendered_contractions.append((edge[0], edge[1], differences[frozenset(edge)]))
    unrendered_contractions = sorted(unrendered_contractions, key=lambda x: x[2])
    community_graph = util.copy_adjacency(graph)
    for edge in community_graph.edges:
        community_graph.edges[edge]["constituent_edges"] = {edge}
    for node in community_graph.nodes:
        community_graph.nodes[node]["constituent_nodes"] = {node}

    if isinstance(partition, gerrychain.Partition):
        assignment = partition.assignment
        for node in graph.nodes:
            graph.nodes[node]["partition"] = assignment[node]
    elif isinstance(partition, str):
        districts = util.load_districts(graph, partition)
        for district, node_list in districts.items():
            for node in node_list:
                graph.nodes[node]["partition"] = district
    else:
        for node in graph.nodes:
            graph.nodes[node]["partition"] = 1
    print("Partition data integrated!")

    # Replay supercommunity algorithm
    for f in range(1, num_frames + 1):
        print(f"\rOn frame {f}/{num_frames}", end="")
        sys.stdout.flush()
        t = (f - 1) / (num_frames - 1)
        edge_colors = {}
        for u, v in living_edges:
            if differences[frozenset((u, v))] < t:
                if graph.nodes[u]["partition"] != graph.nodes[v]["partition"]:
                    edge_colors[frozenset((u, v))] = (156, 156, 255)
                else:
                    edge_colors[frozenset((u, v))] = (106, 106, 106)
            else:
                if graph.nodes[u]["partition"] != graph.nodes[v]["partition"]:
                    edge_colors[frozenset((u, v))] = (50, 50, 150)
                else:
                    edge_colors[frozenset((u, v))] = (0, 0, 0)

        this_iter_contractions = set()
        for c1, c2, time in unrendered_contractions:
            # if (c1, c2, time) in this_iter_contractions:
            #     continue
            # if (c2, c1, time) in this_iter_contractions:
            #     continue
            if time < t:
                # for neighbor in community_graph.neighbors(c2):
                # this_iter_contractions.add((neighbor, c2, differences[frozenset((neighbor, c2))]))
                # Update graph
                print(c1, c2, time, this_iter_contractions)
                for neighbor in community_graph[c2]:
                    if neighbor == c1:
                        continue
                    # this_iter_contractions.add((neighbor, c2, differences[frozenset((neighbor, c2))]))
                    if neighbor in community_graph[c1]:
                        c_edges = community_graph.edges[c1, neighbor]["constituent_edges"].union(community_graph.edges[c2, neighbor]["constituent_edges"])
                    else:
                        c_edges = community_graph.edges[c2, neighbor]["constituent_edges"]
                    community_graph.add_edge(c1, neighbor, constituent_edges=c_edges)
                c_nodes = community_graph.nodes[c1]["constituent_nodes"].union(community_graph.nodes[c2]["constituent_nodes"])
                community_graph.nodes[c1]["constituent_nodes"] = c_nodes

                # New coords
                # node_coords[c1] = node_coords[c1].union(node_coords[c2])
                # if not isinstance(node_coords[c1], shapely.geometry.Polygon):
                #     raise ValueError(f"This set of nodes union to form a {type(node_coords[c1])} (should be Polygon): {community_graph.nodes[c1]['constituent_nodes']}")

                # New color
                total_rep = sum([graph.nodes[node]["total_rep"] for node in c_nodes])
                total_votes = sum([graph.nodes[node]["total_votes"] for node in c_nodes])
                first_node = list(c_nodes)[0]
                for node in c_nodes:
                    node_colors[node] = random_node_colors[first_node]
                    # node_colors[node] = get_partisanship_color(total_rep / total_votes)

                # Delete c2
                this_iter_contractions.add((c1, c2, time))
                community_graph.remove_node(c2)
                # del node_coords[c2]
                # del node_colors[c2]
        for c1, c2, time in this_iter_contractions:
            unrendered_contractions.remove((c1, c2, time))

        visualize_map(
            graph,
            os.path.join(output_fpath, (3 - int(math.log10(f))) * "0" + f"{f}.png"),
            lambda u: node_coords[u],
            lambda e: edge_coords[frozenset(e)],
            lambda u: node_colors[u],
            lambda e: edge_colors[frozenset(e)],
            lambda e: edge_widths[frozenset(e)],
            additional_polygons=[overall_border],
            text=f"t={round(t, 4)}",
            show=False)
    print()

    # ffmpeg -framerate 1 -pattern_type glob -i "NH-communitygen/*.png" -c:v libx264 out.mp4

def visualize_metric(output_fpath, graph, metric_name):
    """Visualizes a nx.Graph object with some kind of metric
    """
    edge_colors = {frozenset(edge) : (255, 255, 255) for edge in graph.edges()}
    edge_widths = {frozenset(edge) : 1 for edge in graph.edges()}

    node_colors = {}
    if metric_name == "z_score":
        max_z_score = -1e10
        min_z_score = 1e10
        for node in graph.nodes():
            z_score = graph.nodes[node][metric_name]
            max_z_score = max(z_score, max_z_score)
            min_z_score = min(z_score, min_z_score)
        print(max_z_score, min_z_score)
        for node in graph.nodes():
            z_score = graph.nodes[node][metric_name]
            if z_score < 0:
                node_colors[node] = (round(255*(z_score)/max_z_score), 0, round(255*(z_score)/max_z_score))
            else:
                node_colors[node] = (0, round(255*(z_score)/min_z_score), 0)

    node_coords, edge_coords, overall_border = get_coords(graph)
    visualize_map(
        graph,
        output_fpath,
        lambda u: node_coords[u],
        lambda e: edge_coords[frozenset(e)],
        lambda u: node_colors[u],
        lambda e: edge_colors[frozenset(e)],
        lambda e: edge_widths[frozenset(e)],
        additional_polygons=[overall_border],
        text=f"{metric_name}",
        show=False)


def visualize_graph(graph, output_path, coords, colors=None, edge_colors=None, node_sizes=None, show=False):
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
    graph_image = Image.new("RGB", IMAGE_DIMS, IMAGE_BG)
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
        [coords(node) for node in graph_nodes], IMAGE_DIMS
    )

    print("Coordinates modified")
    
    if colors is not None:
        node_colors = [colors(node) for node in graph_nodes]
    else:
        node_colors = [(235, 64, 52) for _ in graph_nodes]
        
    # if edge_colors is not None:
    #     edge_colors = [edge_colors(edge) for edge in graph_edges]
    # else:
    #     edge_colors = [(0,0,0) for _ in graph_edges]

    if node_sizes is not None:
        node_sizes = [node_sizes(node) for node in graph_nodes]
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

    print("Edges drawn")

    for center, node, color, r in zip(modified_coords, graph_nodes,
                                      node_colors, node_sizes):
        draw.ellipse(
            [(center[0] - r, center[1] - r),
             (center[0] + r, center[1] + r)],
            fill=color
        )

    print("Nodes drawn")
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


def visualize(output_file, graph_file, difference_file, num_frames, partition_file, verbose):
    """General visualization function (figures out what to do based on inputs).

    TODO: right now this only works for supercommunity animations.
    """
    if difference_file is None:
        raise NotImplementedError("rba draw only supports supercommunity animations at the moment")
    
    with open(graph_file, "r") as f:
        data = json.load(f)
    geodata = nx.readwrite.json_graph.adjacency_graph(data)
    community_generation.compute_precinct_similarities(geodata)

    visualize_community_generation(difference_file, output_file, geodata, num_frames, partition_file)

# def graph_difference_distribution(difference_path):
#     with open(difference_path, "r") as f:
#         supercommunity_output = json.load(f)

if __name__ == "__main__":

    # If called from the command line, it is
    # assumed that the node attributes are precincts.

    with open(sys.argv[1], "rb") as f:
        graph = nx.readwrite.json_graph.adjacency_graph(json.load(f))

    visualize_graph(graph, sys.argv[2], lambda node: shapely.geometry.mapping(shapely.geometry.shape(graph.nodes[node]['geometry']).centroid)["coordinates"], show=True)