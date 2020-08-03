import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame
from networkx import MultiDiGraph
from shapely.geometry.point import Point
from typing import Tuple
'''Library to support geographical computations.'''


def compute_route(start_point: Point, end_point: Point, graph: MultiDiGraph,
                  nodes: GeoDataFrame) -> list:
    '''Returns the shortest path between a starting and end point. 

    Arguments:
      start_point(Point): The lat-lng point of the origin point.
      end_point(Point): The lat-lng point of the destination point.
      graph(MultiDiGraph): The directed graph class that can store multiedges.
      nodes(GeoDataFrame): The GeoDataFrame of graph nodes.
    Returns:
      A list of Points which construct the geometry of the path.

    '''

    # Get closest nodes to points
    orig = ox.get_nearest_node(graph, tuple_from_point(start_point))
    dest = ox.get_nearest_node(graph, tuple_from_point(end_point))

    # Get route
    route = nx.shortest_path(
        graph, orig, dest, weight='length')  # Shortest path
    route_nodes = nodes[nodes['osmid'].isin(route)]
    route_points = route_nodes['geometry']
    return route_points.tolist()


def tuple_from_point(point: Point) -> Tuple[float, float]:
    '''Convert a Point into a tuple, with latitude as first element, and longitude as second.

    Arguments:
      point(Point): A lat-lng point.
    Returns:
      A lat-lng Tuple[float, float].

    '''

    return (point.y, point.x)
