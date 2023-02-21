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

from . import community_generation
from . import util

IMAGE_DIMS = (2000, 2000)
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


def visualize_partition_geopandas(partition):
    """Visualizes a gerrychain.Partition object using geopandas.
    """
    data = {"assignment": [], "geometry": []}
    for node in partition.graph:
        data["assignment"].append(partition.assignment[node])
        data["geometry"].append(shapely.geometry.shape(partition.graph.nodes[node]['geometry']))

    gdf = geopandas.GeoDataFrame(data)
    del data
    gdf.plot(column="assignment")


def visualize_map(graph, output_fpath, node_coords, edge_coords, node_colors=None, edge_colors=None,
        edge_widths=None, node_list=None, additional_polygons=None, text=None, show=False):
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
            continue
    num_edge_line_strings = len(edge_color_values) - 1
    for u in node_list:
        u_coords = node_coords(u)
            # print(shapely.geometry.mapping(u_coords))
        # Could be multiline string if the coordinates are a multipolygon
        if isinstance(u_coords.boundary, shapely.geometry.MultiLineString):
            for ls in u_coords.boundary.geoms:
                all_flattened_coords += ls.coords
            node_end_indices.append(len(all_flattened_coords) - 1)
            node_color_values.append(node_colors(u))
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
    for i in range(len(node_list)):
        if i == 0:
            # Because each edge line string corresponds to two points in the all flattened coords
            start_index = num_edge_line_strings*2 + 2
        else:
            start_index = node_end_indices[i - 1] + 1
        if node_list[i] == "2403707-001s3":
            print(node_color_values[i], "BEING COLORED IN?")
            print(all_flattened_coords[start_index : node_end_indices[i] + 1])
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
        fnt = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "assets/UbuntuMono-R.ttf"), 40)
        draw.text([IMAGE_DIMS[0] - 500, 50], text, font=fnt, fill="black")

    if output_fpath is not None:
        map_image.save(output_fpath)

    if show:
        map_image.show()


def visualize_community_generation(edge_lifetime_fpath, output_fpath, graph, num_frames):
    """Writes frames for an animated visual of the community generation algorithm to a folder.
    The animation depicts borders between communities as black and borders between precints as gray.
    It also uses edge width as an indicator of similarity, and color as an indicator of
    community partisanship.
    
    Parameters
    ----------
    edge_lifetime_fpath : str
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
    with open(edge_lifetime_fpath, "r") as f:
        supercommunity_output = json.load(f)  # Contains strings as keys.

    edge_lifetimes = {}
    for edge, lifetime in supercommunity_output["edge_lifetimes"].items():
        u = edge.split(",")[0][2:-1]
        v = edge.split(",")[1][2:-2]
        edge_lifetimes[frozenset((u, v))] = lifetime
    print("Done!")

    max_lt = max(edge_lifetimes.values())
    min_lt = min(edge_lifetimes.values())
    edge_widths = {
        edge: int((lt - min_lt) / max_lt * EDGE_WIDTH_FACTOR) + 1 for edge, lt in edge_lifetimes.items()
    }

    node_colors = {
        u: get_partisanship_color(graph.nodes[u]["total_rep"] / graph.nodes[u]["total_votes"])
        for u in graph.nodes
    }

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
    for u, v in graph.edges:
        intersection = node_coords[u].intersection(node_coords[v])
        if not intersection.is_empty:
            if isinstance(intersection, shapely.geometry.LineString):
                edge_coords[frozenset((u, v))] = [intersection.coords]
            elif isinstance(intersection, shapely.geometry.MultiLineString):
                edge_coords[frozenset((u, v))] = [ls.coords for ls in intersection.geoms]
            elif isinstance(intersection, shapely.geometry.collection.GeometryCollection):
                edge_coords[frozenset((u,v))] = [ls.coords for ls in intersection.geoms if isinstance(ls, shapely.geometry.LineString)]
            else:
                raise ValueError(f"Intersection between {u} and {v} is not a LineString or"
                                + f" MultiLineString. It is of type {type(intersection)}")
        # else:
            # This is an edge between an island and another precinct, so just connect their centers
            # print([node_coords[u].centroid.coords, node_coords[v].centroid.coords])
            # edge_coords[frozenset((u,v))] = [node_coords[u].centroid.coords, node_coords[v].centroid.coords]
            # raise ValueError(f"Overlap not found between {u} and {v}")
    print("Done!")

    try:
        os.mkdir(output_fpath)
    except FileExistsError:
        pass

    print("Computing outer border of state... ", end="")
    sys.stdout.flush()
    overall_border = shapely.ops.unary_union([node_coords[u] for u in graph.nodes])
    print("Done!")

    living_edges = set(frozenset(e) for e in graph.edges)
    unrendered_contractions = [tuple(c) for c in supercommunity_output["contractions"]]  # Not a set because order must be preserved.
    community_graph = util.copy_adjacency(graph)
    for edge in community_graph.edges:
        community_graph.edges[edge]["constituent_edges"] = {edge}
    for node in community_graph.nodes:
        community_graph.nodes[node]["constituent_nodes"] = {node}

    # Replay supercommunity algorithm
    for f in range(1, num_frames + 1):
        print(f"\rOn frame {f}/{num_frames}", end="")
        sys.stdout.flush()
        t = (f - 1) / (num_frames - 1)
        edge_colors = {}
        for u, v in living_edges:
            if edge_lifetimes[frozenset((u, v))] < t:
                edge_colors[frozenset((u, v))] = (106, 106, 106)
            else:
                edge_colors[frozenset((u, v))] = (0, 0, 0)

        this_iter_contractions = set()
        for c1, c2, time in unrendered_contractions:
            if time < t:
                # Update graph
                for neighbor in community_graph[c2]:
                    if neighbor == c1:
                        continue
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
                for node in c_nodes:
                    node_colors[node] = get_partisanship_color(total_rep / total_votes)

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


def visualize(output_file, graph_file, edge_lifetime_file, num_frames, verbose):
    """General visualization function (figures out what to do based on inputs).

    TODO: right now this only works for supercommunity animations.
    """
    if edge_lifetime_file is None:
        raise NotImplementedError("rba draw only supports supercommunity animations at the moment")
    
    with open(graph_file, "r") as f:
        data = json.load(f)
    geodata = nx.readwrite.json_graph.adjacency_graph(data)
    community_generation.compute_precinct_similarities(geodata)

    visualize_community_generation(edge_lifetime_file, output_file, geodata, num_frames)


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
    # visualize_graph(graph, sys.argv[2], lambda node: shapely.geometry.shape(graph.nodes[node]['geometry']).centroid, show=True)
    visualize_graph(graph, sys.argv[2], lambda node: shapely.geometry.mapping(shapely.geometry.shape(graph.nodes[node]['geometry']).centroid)["coordinates"], show=True)