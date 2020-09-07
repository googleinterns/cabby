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

from typing import Tuple, Sequence, Optional, Dict, Text
from random import sample
import pandas as pd
from geopandas import GeoDataFrame
import networkx as nx
import json
from shapely.geometry import box
import os
import geopandas as gpd
import time

import osmnx as ox
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon, LinearRing

from shapely import geometry

from cabby.geo import util
from cabby.geo.map_processing import map_structure

import util
from map_processing import map_structure


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
    orig = ox.get_nearest_node(graph, util.tuple_from_point(start_point))
    dest = ox.get_nearest_node(graph, util.tuple_from_point(end_point))

    # Get shortest route.
    try:
        route = nx.shortest_path(graph, orig, dest, 'length')
    except:
        print("no route found for the start and end points.")
        return
    route_nodes = nodes[nodes['osmid'].isin(route)]
    return route_nodes


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
    within_distance = map.poi[(dist > 0.2) & (dist < 0.8)]

    # Filter large POI.
    small_poi = within_distance[within_distance['cellids'].str.len() <= 4]

    # Pick random POI.
    start_point = small_poi.sample(1).to_dict('records')[0]

    return start_point


def check_if_tag_exists_set_main_tag(gdf: GeoDataFrame, tag: Text, main_tag: Text, alt_main_tag: Text) -> GeoDataFrame:
    '''Check if tag exists, set main tag name and choose pivot.
    Arguments:
      gdf: The set of landmarks.
      tag: tag to check if exists.
      main_tag: the tag that should be set as the main tag.
    Returns:
      A single landmark.
    '''
    coulmns = gdf.columns
    if tag in coulmns:
        pivots = gdf[gdf[tag].notnull()]
        if pivots.shape[0] > 0:
            pivots = gdf[gdf[main_tag].notnull()]
            if main_tag in coulmns and pivots.shape[0] > 0:
                pivots = pivots.assign(main_tag=pivots[main_tag])
                return pivots.sample(1)
            pivots = gdf[gdf[alt_main_tag].notnull()]
            if alt_main_tag in coulmns and pivots.shape[0] > 0:
                pivots = pivots.assign(main_tag=pivots[alt_main_tag])
                return pivots.sample(1)
    return None


def pick_prominent_pivot(df_pivots: GeoDataFrame) -> Optional[GeoDataFrame]:
    '''Select a landmark from a set of landmarks by priority.
    Arguments:
      df_pivots: The set of landmarks.
    Returns:
      A single landmark.
    '''

    pivot = check_if_tag_exists_set_main_tag(
        df_pivots, 'wikipedia', 'name', 'amenity')
    if pivot is not None:
        return pivot

    pivot = check_if_tag_exists_set_main_tag(
        df_pivots, 'wikidata', 'name', 'amenity')
    if pivot is not None:
        return pivot

    pivot = check_if_tag_exists_set_main_tag(
        df_pivots, 'brand', 'name', 'brand')
    if pivot is not None:
        return pivot

    pivot = check_if_tag_exists_set_main_tag(
        df_pivots, 'tourism', 'name', 'tourism')
    if pivot is not None:
        return pivot

    pivot = check_if_tag_exists_set_main_tag(
        df_pivots, 'amenity', 'name', 'amenity')
    if pivot is not None:
        return pivot

    pivot = check_if_tag_exists_set_main_tag(df_pivots, 'shop', 'name', 'shop')
    if pivot is not None:
        return pivot

    return None


def get_pivot_near_goal(map: map_structure.Map, end_point: GeoDataFrame) -> Optional[Dict]:
    '''Return a picked landmark near the end_point.
    Arguments:
      map: The map of a specific region.
      end_point: The goal location.
    Returns:
      A single landmark near the goal location.
    '''
    try:
        tags = {'name': True, 'wikidata': True,
                'amenity': True, 'shop': True, 'tourism': True}
        poi = ox.pois.pois_from_point(util.tuple_from_point(
            end_point['centroid']), tags=tags, dist=40)

        # Remove streets and roads.
        poi = poi[poi['highway'].isnull()]

    except Exception as e:
        print(e)
        return None

    # Remove the endpoint.
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

    try:
        start_time = time.time()
        poly = Polygon(points_route.tolist()).buffer(0.0005)
        bounds = poly.bounds
        bounding_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        tags = {'wikipedia': True, 'wikidata': True, 'brand': True,
                'tourism': True, 'amenity': True, 'shop': True, 'name': True}
        df_pivots = ox.pois.pois_from_polygon(
            bounding_box, tags=tags)

    except Exception as e:
        print(e)
        return

    # Polygon along the route.
    df_pivots = df_pivots[df_pivots['geometry'].intersects(poly)]

    # Remove streets.
    df_pivots = df_pivots[(df_pivots['highway'].isnull())
                          & (df_pivots['railway'].isnull())]

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


def get_single_sample(map: map_structure.Map):
    '''Sample start and end point, a pivot landmark and route and save to file.
    Arguments:
      map: The map of a specific region.
    Returns:
      A start and end point, a pivot landmark and route.
    '''
    result = get_points_and_route(map)
    if result is None:
        return
    end_point, start_point, route, main_pivot, near_pivot = result
    instruction = "Starting at {0} walk past {1} and your goal is {2}, near {3}.".format(
        start_point['name'], main_pivot['main_tag'], end_point['name'], near_pivot['main_tag'])

    main_pivot['centroid'] = main_pivot['geometry'] if isinstance(
        main_pivot['geometry'], Point) else main_pivot['geometry'].centroid

    near_pivot['centroid'] = near_pivot['geometry'] if isinstance(
        near_pivot['geometry'], Point) else near_pivot['geometry'].centroid

    gdf = gpd.GeoDataFrame({'end': end_point['name'], 'start': start_point['name'], 'main_pivot': main_pivot['main_tag'],
                            'near_pivot': near_pivot['main_tag'], 'instruction': instruction}, index=[0])

    gdf['geometry'] = start_point['centroid']
    gdf_end = gpd.GeoDataFrame(geometry=[end_point['centroid']])
    gdf_main_pivot = gpd.GeoDataFrame(geometry=[main_pivot['centroid']])
    gdf_near_pivot = gpd.GeoDataFrame(geometry=[near_pivot['centroid']])
    gdf_route = gpd.GeoDataFrame(
        geometry=[Polygon(route['geometry'].tolist())])
    return gdf, gdf_end, gdf_main_pivot, gdf_near_pivot, gdf_route


def get_samples(path: Text, map: map_structure.Map, n_samples: int):
    '''Sample start and end point, a pivot landmark and route and save to file.
    Arguments:
      path: The path to which the data will be appended.
      map: The map of a specific region.
      n_samples: the max number of samples to generate.
    '''
    gdf_start_list = gpd.GeoDataFrame(
        columns=["start", "end", "main_pivot", "near_pivot", "instruction", "geometry"])

    gdf_end_list = gpd.GeoDataFrame(columns=["geometry"])
    gdf_route_list = gpd.GeoDataFrame(columns=["geometry"])
    gdf_main_list = gpd.GeoDataFrame(columns=["geometry"])
    gdf_near_list = gpd.GeoDataFrame(columns=["geometry"])

    for i in range(n_samples):
        result = get_single_sample(map)
        if result is None:
            continue
        gdf_start, gdf_end, gdf_main_pivot, gdf_near_pivot, gdf_route = result
        gdf_start_list = gdf_start_list.append(gdf_start, ignore_index=True)
        gdf_end_list = gdf_end_list.append(gdf_end, ignore_index=True)
        gdf_route_list = gdf_route_list.append(gdf_route, ignore_index=True)
        gdf_main_list = gdf_main_list.append(gdf_main_pivot, ignore_index=True)
        gdf_near_list = gdf_near_list.append(gdf_near_pivot, ignore_index=True)

    gdf_start_list.to_file(path, layer='start', driver="GPKG")
    gdf_end_list.to_file(path, layer='end', driver="GPKG")
    gdf_route_list.to_file(path, layer='route', driver="GPKG")
    gdf_main_list.to_file(path, layer='main', driver="GPKG")
    gdf_near_list.to_file(path, layer='near', driver="GPKG")


def print_instructions(path: Text):
    '''Read a geodata file and print instruction.'''
    if not os.path.exists(path):
        return
    start = gpd.read_file(path, layer='start')
    print('\n'.join(start['instruction'].values))
