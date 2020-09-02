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
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon, LinearRing

from shapely import geometry

from cabby.geo import util
from cabby.geo.map_processing import map_structure


def compute_route(start_point: Point, end_point: Point, graph: nx.MultiDiGraph,
                  nodes: GeoDataFrame) -> Optional[GeoDataFrame]:
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
    try:
        route = nx.shortest_path(graph, orig, dest, 'length')
    except:
        print("no route found for the start and end points.")
        return
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
    # TODO change to path distance.
    within_distance = map.poi[(dist > 0.2) & (dist < 3)]

    # Filter large POI.
    small_poi = within_distance[within_distance['cellids'].str.len() <= 4]

    # Pick random POI.
    start_point = small_poi.sample(1).to_dict('records')[0]

    return start_point


def pick_prominent_pivot(df_pivots: GeoDataFrame) -> Optional[GeoDataFrame]:
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
            pivots['main_tag'] = pivots['name']
            return pivots.sample(1)
    if 'wikidata' in coulmns:
        pivots = df_pivots[df_pivots['wikidata'].notnull()]
        if pivots.shape[0] > 0:
            pivots['main_tag'] = pivots['name']
            return pivots.sample(1)
    if 'brand' in coulmns:
        pivots = df_pivots[(df_pivots['brand'].notnull()) &
                           (df_pivots['name'].notnull())]
        if pivots.shape[0] > 0:
            pivots['main_tag'] = pivots['brand']
            return pivots.sample(1)
    if 'tourism' in coulmns:
        pivots = df_pivots[(df_pivots['tourism'].notnull())
                           & (df_pivots['name'].notnull())]
        if pivots.shape[0] > 0:
            picked = pivots.sample(1)
            if picked['name'] is not None:
                picked['main_tag'] = picked['name']
            else:
                picked['main_tag'] = picked['tourism']
            return picked
    if 'amenity' in coulmns:
        pivots = df_pivots[(df_pivots['amenity'].notnull())
                           & (df_pivots['name'].notnull())]
        if pivots.shape[0] > 0:
            picked = pivots.sample(1)
            if picked['name'] is not None:
                picked['main_tag'] = picked['name']
            else:
                picked['main_tag'] = picked['amenity']
            return picked
    else:
        return None


def get_pivot_near_goal(map: map_structure.Map, end_point: GeoDataFrame) -> Optional[Dict]:
    '''Return a picked landmark near the end_point.
    Arguments:
      map: The map of a specific region.
      end_point: The goal location.
    Returns:
      A single landmark near the goal location.
    '''
    tags = {'name': True, 'amenity': True, 'shop': True, 'tourism': True}
    poi = ox.pois.pois_from_point(tuple_from_point(
        end_point['centroid']), tags=tags, dist=20)

    nearby_poi = poi[poi['osmid'] != end_point['osmid']]

    prominent_poi = pick_prominent_pivot(nearby_poi)

    if prominent_poi is None:
        return

    return prominent_poi.to_dict('records')[0]


def get_pivots(route: GeoDataFrame, map: map_structure.Map, end_point: GeoDataFrame) -> Optional[Tuple[Dict, Dict]]:
    '''Return a picked landmark on a given route.
    Arguments:
      route: Along the route a landmark will be chosen.
      map: The map of a specific region.
      end_point: The goal location.
    Returns:
      A single landmark.
    '''

    # Get POI along the route.
    points_route = map.nodes[map.nodes['osmid'].isin(
        route['osmid'])]['geometry']

    coords = [(p.x, p.y) for p in points_route.tolist()]
    r = LinearRing(coords)
    s = Polygon(r)
    poly = Polygon(s.buffer(0.1).exterior, [r])

    df_pivots = ox.footprints_from_polygon(
        poly, footprint_type='building', retain_invalid=False)

    main_pivot = pick_prominent_pivot(df_pivots)
    main_pivot = main_pivot.to_dict('records')[0]

    # Get pivot near the goal location.
    near_pivot = get_pivot_near_goal(map, end_point)
    if main_pivot is None or near_pivot is None:
        return

    return main_pivot, near_pivot


def get_points_and_route(map: map_structure.Map) -> Optional[Tuple[Dict, Dict, GeoDataFrame, Dict, Dict]]:
    '''Sample start and end point, a pivot landmark and route.
    Arguments:
      map: The map of a specific region.
    Returns:
      A start and end point, a pivot landmark and route.
    '''

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

    # Select pivots.
    result = get_pivots(route, map, end_point)
    if result is None:
        return
    main_pivot, near_pivot = result

    return end_point, start_point, route, main_pivot, near_pivot
