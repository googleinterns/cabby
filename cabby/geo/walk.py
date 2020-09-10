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

'''Library to support sampling points, creating routes between them and pivots
along the path and near the goal.'''

import geopandas as gpd
from geopandas import GeoDataFrame
import json
import networkx as nx
import os
import osmnx as ox
import pandas as pd
from random import sample
from shapely import geometry
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon, LinearRing
from shapely.geometry import box
from typing import Tuple, Sequence, Optional, Dict, Text, Any

from cabby.geo import item
from cabby.geo import util
from cabby.geo import geo_item
from cabby.geo.map_processing import map_structure

_Geo_DataFrame_Driver = "GPKG"


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
    print("No route found for the start and end points.")
    return None
  route_nodes = nodes[nodes['osmid'].isin(route)]
  return route_nodes


def get_end_poi(map: map_structure.Map) -> Optional[Dict[Text, Any]]:
  '''Returns the a random POI.
  Arguments:
    map: The map of a specific region.
  Returns:
    A single POI.
  '''

  # Filter large POI.
  small_poi = map.poi[map.poi['cellids'].str.len() <= 4]

  if small_poi.shape[0] == 0:
    return None

  # Pick random POI.
  poi = small_poi.sample(1).to_dict('records')[0]

  return poi


def get_start_poi(map: map_structure.Map, end_point: GeoDataFrame) -> \
    Optional[Dict]:
  '''Returns the a random POI within distance of a given POI.
  Arguments:
    map: The map of a specific region.
    end_point: The POI to which the picked POI should be within distance
    range.
  Returns:
    A single POI.
  '''

  dist = map.poi['centroid'].apply(
    lambda x: util.get_distance_km(end_point['centroid'], x))

  # Get closest nodes to points.
  dest = ox.get_nearest_node(
    map.nx_graph, util.tuple_from_point(end_point['centroid']))

  # Find nodes whithin 2000 meter path distance.
  outer_circle_graph = ox.truncate.truncate_graph_dist(
    map.nx_graph, dest, max_dist=2000, weight='length')

  # Get graph that is too close (less than 200 meter path distance)
  inner_circle_graph = ox.truncate.truncate_graph_dist(
    map.nx_graph, dest, max_dist=200, weight='length')

  outer_circle_graph_osmid = [x for x in outer_circle_graph.nodes.keys()]
  inner_circle_graph_osmid = [x for x in inner_circle_graph.nodes.keys()]

  osmid_in_range = [
    osmid for osmid in outer_circle_graph_osmid if osmid not in inner_circle_graph_osmid]

  within_distance = map.poi[map.poi['node'].isin(osmid_in_range)]

  # Filter large POI.
  small_poi = within_distance[within_distance['cellids'].str.len() <= 4]

  if small_poi.shape[0] == 0:
    return None

  # Pick random POI.
  start_point = small_poi.sample(1).to_dict('records')[0]

  return start_point


def get_landmark_if_tag_exists(gdf: GeoDataFrame, tag: Text, main_tag:
                 Text, alt_main_tag: Text) -> GeoDataFrame:
  '''Check if tag exists, set main tag name and choose pivot.
  Arguments:
    gdf: The set of landmarks.
    tag: tag to check if exists.
    main_tag: the tag that should be set as the main tag.
  Returns:
    A single landmark.
  '''
  candidate_landmarks = gdf.columns
  if tag in candidate_landmarks:
    pivots = gdf[gdf[tag].notnull()]
    if pivots.shape[0]:
      pivots = gdf[gdf[main_tag].notnull()]
      if main_tag in candidate_landmarks and pivots.shape[0]:
        pivots = pivots.assign(main_tag=pivots[main_tag])
        return pivots.sample(1)
      pivots = gdf[gdf[alt_main_tag].notnull()]
      if alt_main_tag in candidate_landmarks and pivots.shape[0]:
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

  tag_pairs = [('wikipedia', 'amenity'), ('wikidata', 'amenity'),
         ('brand', 'brand'), ('tourism', 'tourism'), ('tourism', 'tourism'),
         ('amenity', 'amenity'), ('shop', 'shop')]

  pivot = None

  for main_tag, named_tag in tag_pairs:
    pivot = get_landmark_if_tag_exists(df_pivots, main_tag, 'name',
                       named_tag)
    if pivot is not None:
      return pivot.to_dict('records')[0]

  return pivot


def get_pivot_near_goal(map: map_structure.Map, end_point: GeoDataFrame) -> \
    Optional[Dict]:
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

  return prominent_poi


def get_pivot_along_route(
  route: GeoDataFrame, map: map_structure.Map) -> Optional[Dict]: 
  '''Return a picked landmark on a given route. 
  Arguments: 
    route: The route along which a landmark will be chosen. 
    map: The map of a specific region. 
  Returns: 
    A single landmark. ''' 
    
  # Get POI along the route. 
  points_route = map.nodes[map.nodes['osmid'].isin( route['osmid'])]['geometry']
  
  try:
    poly = Polygon(points_route.tolist()).buffer(0.0001) 
    bounds = poly.bounds
    bounding_box = box(bounds[0], bounds[1], bounds[2], bounds[3]) 
    tags = {'wikipedia': True, 'wikidata': True, 'brand': True, 'tourism': True, 'amenity': True, 'shop': True, 'name': True} 
    df_pivots = ox.pois.pois_from_polygon(bounding_box, tags=tags)
    
    # Polygon along the route. 
    df_pivots = df_pivots[df_pivots['geometry'].intersects(poly)]
    
    # Remove streets. 
    df_pivots = df_pivots[(df_pivots['highway'].isnull())] 
  except Exception as e: 
    print(e) 
    return None
    
  main_pivot = pick_prominent_pivot(df_pivots)
  return main_pivot


def get_pivots(route: GeoDataFrame, map: map_structure.Map, end_point:
         GeoDataFrame) -> Optional[Tuple[Dict, Dict]]:
  '''Return a picked landmark on a given route.
  Arguments:
    route: The route along which a landmark will be chosen.
    map: The map of a specific region.
    end_point: The goal location.
  Returns:
    A single landmark.
  '''

  # Get pivot along the goal location. 
  main_pivot = get_pivot_along_route(route, map)

  # Get pivot near the goal location.
  near_pivot = get_pivot_near_goal(map, end_point)
  if main_pivot is None or near_pivot is None:
    return None

  return main_pivot, near_pivot


def get_points_and_route(map: map_structure.Map) -> Optional[item.RVSPath]:
  '''Sample start and end point, a pivot landmark and route.
  Arguments:
    map: The map of a specific region.
  Returns:
    A start and end point, a pivot landmark and route.
  '''

  # Select end point.
  end_point = get_end_poi(map)
  if end_point is None:
    return None

  # Select start point.
  start_point = get_start_poi(map, end_point)
  if start_point is None:
    return None

  # Compute route between start and end points.
  route = compute_route(
    start_point['centroid'], end_point['centroid'], map.nx_graph, map.nodes)
  if route is None:
    return None

  # Select pivots.
  result = get_pivots(route, map, end_point)
  if result is None:
    return None
  main_pivot, near_pivot = result

  rvs_path_entity = item.RVSPath.from_points_route_pivots(start_point,
                              end_point, route, main_pivot, near_pivot)

  return rvs_path_entity


def get_single_sample(map: map_structure.Map) -> Optional[geo_item.
                              GeoEntity]:
  '''Sample start and end point, a pivot landmark and route and save to file.
  Arguments:
    map: The map of a specific region.
  Returns:
    A start and end point, a pivot landmark and route.
  '''
  rvs_path_entity = get_points_and_route(map)
  if rvs_path_entity is None:
    return None

  gdf_tags_start = gpd.GeoDataFrame({'end': rvs_path_entity.end_point['name'],
                     'start': rvs_path_entity.start_point['name'], 'main_pivot': rvs_path_entity.main_pivot
                     ['main_tag'], 'near_pivot': rvs_path_entity.near_pivot['main_tag'], 'instruction':
                     rvs_path_entity.instruction}, index=[0])

  gdf_tags_start['geometry'] = rvs_path_entity.start_point['centroid']

  gdf_end = gpd.GeoDataFrame(
    geometry=[rvs_path_entity.end_point['centroid']])

  gdf_main_pivot = gpd.GeoDataFrame(geometry=[rvs_path_entity.main_pivot
                        ['centroid']])

  gdf_near_pivot = gpd.GeoDataFrame(geometry=[rvs_path_entity.near_pivot
                        ['centroid']])

  gdf_route = gpd.GeoDataFrame(
    geometry=[Polygon(rvs_path_entity.route['geometry'].tolist())])

  geo_entity = geo_item.GeoEntity.from_points_route_pivots(
    gdf_tags_start, gdf_end, gdf_route, gdf_main_pivot, gdf_near_pivot)

  return geo_entity


def generate_and_save_rvs_routes(path: Text, map: map_structure.Map, n_samples:
                 int):
  '''Sample start and end point, a pivot landmark and route and save to file.
  Arguments:
    path: The path to which the data will be appended.
    map: The map of a specific region.
    n_samples: the max number of samples to generate.
  '''
  gdf_start_list = gpd.GeoDataFrame(
    columns=["start", "end", "main_pivot", "near_pivot", "instruction",
         "geometry"])

  gdf_end_list = gpd.GeoDataFrame(columns=["geometry"])
  gdf_route_list = gpd.GeoDataFrame(columns=["geometry"])
  gdf_main_list = gpd.GeoDataFrame(columns=["geometry"])
  gdf_near_list = gpd.GeoDataFrame(columns=["geometry"])

  counter = 0
  while counter < n_samples:
    entity = get_single_sample(map)
    if entity is None:
      continue
    counter += 1
    gdf_start_list = gdf_start_list.append(entity.tags_start,
                         ignore_index=True)
    gdf_end_list = gdf_end_list.append(entity.end, ignore_index=True)
    gdf_route_list = gdf_route_list.append(entity.route,
                         ignore_index=True)
    gdf_main_list = gdf_main_list.append(entity.main_pivot,
                       ignore_index=True)
    gdf_near_list = gdf_near_list.append(entity.near_pivot,
                       ignore_index=True)

  if gdf_start_list.shape[0] == 0:
    return None

  gdf_start_list.to_file(path, layer='start', driver=_Geo_DataFrame_Driver)
  gdf_end_list.to_file(path, layer='end', driver=_Geo_DataFrame_Driver)
  gdf_route_list.to_file(path, layer='route', driver=_Geo_DataFrame_Driver)
  gdf_main_list.to_file(path, layer='main', driver=_Geo_DataFrame_Driver)
  gdf_near_list.to_file(path, layer='near', driver=_Geo_DataFrame_Driver)


def print_instructions(path: Text):
  '''Read a geodata file and print instruction.'''
  if not os.path.exists(path):
    return None
  start = gpd.read_file(path, layer='start')
  print('\n'.join(start['instruction'].values))
