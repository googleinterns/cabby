# coding=utf-8
# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Library to support geographical computations.'''

from typing import Tuple, Sequence, Optional, Dict
from random import sample
import pandas as pd
from geopandas import GeoDataFrame
import networkx as nx

import osmnx as ox
import util
from shapely.geometry.point import Point

from cabby.geo.map_processing import map_structure
# from map_processing import map_structure


def compute_route(start_point: Point, end_point: Point, graph: nx.MultiDiGraph,
                  nodes: GeoDataFrame) -> Sequence:
    '''Returns the shortest path between a starting and end point.

    Arguments:
      start_point(Point): The lat-lng point of the origin point.
      end_point(Point): The lat-lng point of the destination point.
      graph(nx.MultiDiGraph): The directed graph class that stores multiedges.
      nodes(GeoDataFrame): The GeoDataFrame of graph nodes.
    Returns:
      A sequence of Points which construct the geometry of the path.
    '''

    # Get closest nodes to points.
    orig = ox.get_nearest_node(graph, tuple_from_point(start_point))
    dest = ox.get_nearest_node(graph, tuple_from_point(end_point))

    # Get shortest route.
    route = nx.shortest_path(graph, orig, dest, 'length')
    route_nodes = nodes[nodes['osmid'].isin(route)]
    return route_nodes


def tuple_from_point(point: Point) -> Tuple[float, float]:
    '''Convert a Point into a tuple, with latitude as first element, and longitude as second.

    Arguments:
      point(Point): A lat-lng point.
    Returns:
      A lat-lng Tuple[float, float].
    '''

    return (point.y, point.x)


def get_end_poi(map: map_structure.Map) -> Optional[Dict]:
    '''Returns the a random POI.
    Arguments:
      map: The map of a specific region.
    Returns:
      A single POI.
    '''

    # Filter large POI.
    small_poi = map.poi[map.poi['cellids'].str.len() <= 4]

    # Pick random POI.
    poi = small_poi.sample(1).to_dict('records')[0]

    return poi


def get_start_poi(map: map_structure.Map, end_point: GeoDataFrame) -> Optional[Dict]:
    '''Returns the a random POI within distance of a given POI.
    Arguments:
      map: The map of a specific region.
      end_point: The POI to which the picked POI should be within distance range.
    Returns:
      A single POI.
    '''

    dist = map.poi['centroid'].apply(
        lambda x: util.get_distance_km(end_point['centroid'], x))

    # Get POI within a distance range.
    within_distance = map.poi[(dist > 0.2) & (dist < 3)] # TODO change to path distance.


    # Filter large POI.
    small_poi = within_distance[within_distance['cellids'].str.len() <= 4] 

    # Pick random POI.
    start_point = small_poi.sample(1).to_dict('records')[0]

    return start_point


def landmark_pick(df_pivots: GeoDataFrame) -> Optional[GeoDataFrame]:
    '''Select a landmark from a set of landmarks by priority.
    Arguments:
      df_pivots: The set of landmarks.
    Returns:
      A single landmark.
    '''

    coulmns = df_pivots.columns
    if 'wikipedia' in coulmns:
        pivots = df_pivots[df_pivots['wikipedia'].notnull()]
        if pivots.shape[0] > 0:
            return pivots.sample(1)
    if 'wikidata' in coulmns:
        pivots = df_pivots[df_pivots['wikidata'].notnull()]
        if pivots.shape[0] > 0:
            return pivots.sample(1)
    if 'brand' in coulmns:
        pivots = df_pivots[(df_pivots['brand'].notnull()) &
                           (df_pivots['name'].notnull())]
        if pivots.shape[0] > 0:
            return pivots.sample(1)
    if 'tourism' in coulmns:
        pivots = df_pivots[(df_pivots['tourism'].notnull())
                           & (df_pivots['name'].notnull())]
        if pivots.shape[0] > 0:
            return pivots.sample(1)
    if 'amenity' in coulmns:
        pivots = df_pivots[(df_pivots['amenity'].notnull())
                           & (df_pivots['name'].notnull())]
        if pivots.shape[0] > 0:
            return pivots.sample(1)
    else:
        return None


def get_pivot(route: GeoDataFrame, map: map_structure.Map) -> Optional[Dict]:

    points_route = map.nodes[map.nodes['osmid'].isin(
        route['osmid'])]['geometry']

    df_pivots = pd.DataFrame()

    for point in points_route:
        pivot_found = ox.footprints_from_point(
            (point.y, point.x), dist=20, footprint_type='building', retain_invalid=False)
        df_pivots = df_pivots.append(pivot_found, ignore_index=True)

    pivot = landmark_pick(df_pivots)

    return pivot.to_dict('records')[0]


def get_points_and_route(map: map_structure.Map) -> Optional[Tuple[Dict, Dict, GeoDataFrame, Dict]]:

    # Select end point.
    end_point = get_end_poi(map)
    if end_point is None:
        return

    # Select start point.
    start_point = get_start_poi(map, end_point)
    if start_point is None:
        return

    # Compute route between start and end points.
    route = compute_route(
        start_point['centroid'], end_point['centroid'], map.nx_graph, map.nodes)
    if route is None:
        return

    # Select pivot.
    pivot = get_pivot(route, map)
    if pivot is None:
        return

    return end_point, start_point, route, pivot



