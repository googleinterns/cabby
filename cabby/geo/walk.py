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

from typing import Tuple, Sequence, Optional, Dict, Text, Any
from shapely import geometry
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon, LinearRing
from shapely.geometry import box, mapping, LineString
from random import sample
import os
import osmnx as ox
import networkx as nx
import pandas as pd
import json
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from cabby.geo import util
from cabby.geo.map_processing import map_structure
from cabby.rvs import item
import sys


_Geo_DataFrame_Driver = "GPKG"
OSM_CRS = 32633  # UTM Zones (North).


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
  except nx.exception.NetworkXNoPath:
    print("No route found for the start and end points.")
    return None
  route_nodes = nodes[nodes['osmid'].isin(route)]

  # Create the dictionary that defines the order for sorting according to
  # route order.
  sorterIndex = dict(zip(route, range(len(route))))

  # Generate a rank column that will be used to sort
  # the dataframe numerically
  route_nodes['sort'] = route_nodes['osmid'].map(sorterIndex)

  route_nodes = route_nodes.sort_values(['sort'])

  return route_nodes


def get_end_poi(map: map_structure.Map) -> Optional[GeoSeries]:
  '''Returns a random POI.
  Arguments:
    map: The map of a specific region.
  Returns:
    A single POI.
  '''

  # Filter large POI.
  small_poi = map.poi[map.poi['s2cellids'].str.len() <= 4]

  if small_poi.shape[0] == 0:
    return None

  # Pick random POI.
  poi = small_poi.sample(1, random_state=1).iloc[0]
  poi['geometry'] = poi.centroid

  return poi


def get_start_poi(
    map: map_structure.Map, end_point: Dict) -> Optional[GeoSeries]:
  '''Returns the a random POI within distance of a given POI.
  Arguments:
    map: The map of a specific region.
    end_point: The POI to which the picked POI should be within distance
    range.
  Returns:
    A single POI.
  '''

  dist = map.poi['centroid'].apply(
    lambda x: util.get_distance_km(end_point['geometry'], x))

  # Get closest nodes to points.
  dest_osmid = end_point['osmid']

  # Find nodes whithin 2000 meter path distance.
  outer_circle_graph = ox.truncate.truncate_graph_dist(
    map.nx_graph, dest_osmid, max_dist=2000, weight='length')

  outer_circle_graph_osmid = list(outer_circle_graph.nodes.keys())

  try:
    # Get graph that is too close (less than 400 meter path distance)
    inner_circle_graph = ox.truncate.truncate_graph_dist(
      map.nx_graph, dest_osmid, max_dist=400, weight='length')
    inner_circle_graph_osmid = list(inner_circle_graph.nodes.keys())

  except ValueError:  # GeoDataFrame returned empty
    inner_circle_graph_osmid = []

  osmid_in_range = [
    osmid for osmid in outer_circle_graph_osmid if osmid not in
    inner_circle_graph_osmid]

  poi_in_ring = map.poi[map.poi['node'].isin(osmid_in_range)]

  # Filter large POI.
  small_poi = poi_in_ring[poi_in_ring['s2cellids'].str.len() <= 4]

  if small_poi.shape[0] == 0:
    return None

  # Pick random POI.
  start_point = small_poi.sample(1, random_state=1).iloc[0]
  start_point['geometry'] = start_point.centroid
  return start_point


def get_landmark_if_tag_exists(gdf: GeoDataFrame, tag: Text, main_tag:
                 Text, alt_main_tag: Text) -> GeoSeries:
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
        return pivots.sample(1, random_state=1).iloc[0]
      pivots = gdf[gdf[alt_main_tag].notnull()]
      if alt_main_tag in candidate_landmarks and pivots.shape[0]:
        pivots = pivots.assign(main_tag=pivots[alt_main_tag])
        return pivots.sample(1, random_state=1).iloc[0]
  return None


def pick_prominent_pivot(df_pivots: GeoDataFrame) -> Optional[GeoSeries]:
  '''Select a landmark from a set of landmarks by priority.
  Arguments:
    df_pivots: The set of landmarks.
  Returns:
    A single landmark.
  '''

  tag_pairs = [('wikipedia', 'amenity'), ('wikidata', 'amenity'),
         ('brand', 'brand'), ('tourism', 'tourism'),
         ('tourism', 'tourism'), ('amenity', 'amenity'), ('shop', 'shop')
         ]

  pivot = None

  for main_tag, named_tag in tag_pairs:
    pivot = get_landmark_if_tag_exists(df_pivots, main_tag, 'name',
                       named_tag)
    if pivot is not None:
      if not isinstance(pivot['geometry'], Point):
        pivot['geometry'] = pivot['geometry'].centroid
      return pivot

  return pivot


def get_pivot_near_goal(map: map_structure.Map, end_point: GeoSeries) -> \
    Optional[GeoSeries]:
  '''Return a picked landmark near the end_point.
  Arguments:
  map: The map of a specific region.
  end_point: The goal location.
  Returns:
  A single landmark near the goal location.
  '''

  tags = {'name': True, 'wikidata': True,
      'amenity': True, 'shop': True, 'tourism': True}
  try:
    poi = ox.pois.pois_from_point(util.tuple_from_point(
      end_point['centroid']), tags=tags, dist=40)

  except ValueError:  # GeoDataFrame returned empty
    return None

  # Remove streets and roads.
  if 'highway' in poi.columns:
    poi = poi[poi['highway'].isnull()]

  # Remove the endpoint.
  nearby_poi = poi[poi['osmid'] != end_point['osmid']]

  prominent_poi = pick_prominent_pivot(nearby_poi)

  return prominent_poi


def get_pivot_along_route(
    route: GeoDataFrame, map: map_structure.Map) -> \
    Optional[GeoSeries]:
  '''Return a picked landmark on a given route.
  Arguments:
    route: The route along which a landmark will be chosen.
    map: The map of a specific region.
  Returns:
    A single landmark. '''

  # Get POI along the route.
  points_route = route['geometry'].tolist()

  poly = Polygon(points_route).buffer(0.0001)
  bounds = poly.bounds
  bounding_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
  tags = {'wikipedia': True, 'wikidata': True, 'brand': True,
      'tourism': True, 'amenity': True, 'shop': True, 'name': True}
  try:
    df_pivots = ox.pois.pois_from_polygon(bounding_box, tags=tags)
  except ValueError:  # GeoDataFrame returned empty
    return None

  # Polygon along the route.
  df_pivots = df_pivots[df_pivots['geometry'].intersects(poly)]

  # Remove streets.
  if 'highway' in df_pivots.columns:
    df_pivots = df_pivots[(df_pivots['highway'].isnull())]

  main_pivot = pick_prominent_pivot(df_pivots)
  return main_pivot


def get_pivot_beyond_goal(map: map_structure.Map, end_point: GeoSeries,
              route: GeoDataFrame) -> Optional[GeoSeries]:
  '''Return a picked landmark on a given route.
  Arguments:
    map: The map of a specific region.
    end_point: The goal location.
    route: The route along which a landmark will be chosen.
  Returns:
    A single landmark. '''

  if route.shape[0] < 2:
    # Return Empty.
    return GeoDataFrame(index=[0], columns=map.nodes.columns).iloc[0]

  last_node_in_route = route.iloc[-1]
  before_last_node_in_route = route.iloc[-2]

  street_beyond_route = map.edges[
    (map.edges['u'] == last_node_in_route['osmid'])
    & (map.edges['v'] == before_last_node_in_route['osmid'])
  ]
  if street_beyond_route.shape[0] == 0:
    # Return Empty.
    return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

  street_beyond_osmid = street_beyond_route['osmid'].iloc[0]

  # Change OSMID to key
  segment_beyond_path = ((last_node_in_route['osmid'] == map.edges['u'])
               & (before_last_node_in_route['osmid'] !=
                map.edges['v']))
  condition_street_id = map.edges['osmid'].apply(
    lambda x: x == street_beyond_osmid)
  last_line = map.edges[condition_street_id
              & segment_beyond_path]

  if last_line.shape[0] == 0:
    # Return Empty.
    return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

  last_line_max = last_line[last_line['length'] ==
                last_line['length'].max()].iloc[0]

  poly = last_line_max['geometry'].buffer(0.0001)
  bounds = poly.bounds
  bounding_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
  try:

    df_pivots = ox.pois.pois_from_polygon(
      bounding_box, tags={"name": True})

  except ValueError:  # GeoDataFrame returned empty
    # Return Empty.
    return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

  df_pivots = df_pivots.set_crs(epsg=OSM_CRS, allow_override=True)
  # Remove streets.
  if 'highway' in df_pivots.columns:
    df_pivots = df_pivots[(df_pivots['highway'].isnull())]

  # Remove invalid geometry.
  df_pivots = df_pivots[(df_pivots['geometry'].is_valid)]
  if df_pivots.shape[0] == 0:
    # Return Empty.
    return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

  # Remove the route area.

  points_route = route['geometry'].tolist()

  poly_route = Polygon(points_route).buffer(0.0001)

  route_endpoint_points = [last_node_in_route["geometry"],
               end_point['centroid'],
               last_node_in_route["geometry"]]
  route_to_endpoint = Polygon(route_endpoint_points).buffer(0.0001)

  poly_route_with_end = poly_route.union(route_to_endpoint)

  df_pivots = df_pivots[df_pivots.apply(lambda x:
                      not util.check_if_geometry_in_polygon(
                        x, poly_route_with_end),
                      axis=1)]

  if df_pivots.shape[0] == 0:
    # Return Empty.
    return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

  # Remove the end_point.
  df_pivots = df_pivots[df_pivots['geometry'] != end_point['geometry']]
  beyond_pivot = pick_prominent_pivot(df_pivots)

  if beyond_pivot is None:
    # Return Empty.
    return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

  return beyond_pivot


def get_pivots(route: GeoDataFrame, map: map_structure.Map, end_point:
         Dict) -> Optional[Tuple[GeoSeries, GeoSeries, GeoSeries]]:
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
  if main_pivot is None:
    return None

  # Get pivot near the goal location.
  near_pivot = get_pivot_near_goal(map, end_point)

  if near_pivot is None:
    return None

  # Get pivot located past the goal location and beyond the route.
  beyond_pivot = get_pivot_beyond_goal(map, end_point, route)

  return main_pivot, near_pivot, beyond_pivot


def get_cardinal_direction(start_point: Point, end_point: Point) -> Text:
  '''Calculate the cardinal direction between start and and points.
  Arguments:
    start_point: The starting point.
    end_point: The end point.
  Returns:
    A cardinal direction.
  '''
  azim = util.get_bearing(start_point, end_point)
  if azim < 10 or azim > 350:
    cardinal = 'North'
  elif azim < 80:
    cardinal = 'North-East'
  elif azim > 280:
    cardinal = 'North-West'
  elif azim < 100:
    cardinal = 'West'
  elif azim < 170:
    cardinal = 'South-East'
  elif azim < 190:
    cardinal = 'South'
  elif azim < 260:
    cardinal = 'South-West'
  else:  # azim < 280:
    cardinal = 'West'
  return cardinal

  return main_pivot, near_pivot, beyond_pivot


def get_number_intersections_past(main_pivot: GeoSeries, route: GeoDataFrame,
                  map: map_structure.Map, end_point: Point) -> int:
  '''Return the number of intersections between the main_pivot and goal. 
  Arguments:
    main_pivot: The pivot along the route.
    route: The route along which a landmark will be chosen.
    map: The map of a specific region.
    end_point: The goal location.
  Returns:
    The number of intersections between the main_pivot and goal. 
    If the main_pivot and goal are on different streets return -1.
  '''
  dist = route['geometry'].apply(
    lambda x: x.distance(main_pivot['geometry']))
  pivot_idx = dist.idxmin().tolist()
  pivot_goal_route = route.loc[pivot_idx:]

  if pivot_goal_route.shape[0] <= 1:
    return -1

  # Check if main pivot is in the pivot_goal_route segment.
  segment_pivot = LineString(pivot_goal_route['geometry'].iloc[0:2].values)
  number_intersection = -util.project_point_in_segment(
    segment_pivot, main_pivot['geometry'])

  # Check if main end point is in the pivot_goal_route segment.
  segment_goal = LineString(pivot_goal_route['geometry'].iloc[-2:].values)
  number_intersection -= util.project_point_in_segment(
    segment_goal, end_point['centroid'])

  # Check if goal and pivot are on the same street.
  edges_in_pivot_goal_route = pivot_goal_route['osmid'].apply(
    lambda x: set(map.edges[map.edges['u'] == x]['osmid'].tolist()))

  pivot_streets = edges_in_pivot_goal_route.iloc[0]
  goal_streets = edges_in_pivot_goal_route.iloc[-1]
  common_streets = pivot_streets & goal_streets
  if not common_streets:
    return -1

  number_intersection += edges_in_pivot_goal_route.apply(
    lambda x: len(x - common_streets) > 0).count()

  if number_intersection <= 0:
    return -1

  return number_intersection


def get_single_sample(map: map_structure.Map) -> Optional[item.RVSPath]:
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
    start_point['geometry'], end_point['geometry'], map.nx_graph, map.nodes)
  if route is None:
    return None

  # Select pivots.
  result = get_pivots(route, map, end_point)
  if result is None:
    return None
  main_pivot, near_pivot, beyond_pivot = result

  # Get cardinal direction.
  cardinal_direction = get_cardinal_direction(
    start_point['geometry'], end_point['geometry'])

  # Get number of intersections between main pivot and goal location.
  intersections = get_number_intersections_past(
    main_pivot, route, map, end_point)

  rvs_path_entity = item.RVSPath.from_points_route_pivots(start_point,
                              end_point,
                              route,
                              main_pivot,
                              near_pivot,
                              beyond_pivot,
                              cardinal_direction,
                              intersections)

  return rvs_path_entity


def generate_and_save_rvs_routes(path: Text, map: map_structure.Map, n_samples:
                 int):
  '''Sample start and end point, a pivot landmark and route and save to file.
  Arguments:
    path: The path to which the data will be appended.
    map: The map of a specific region.
    n_samples: the max number of samples to generate.
  '''
  gdf_start_list = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag'])
  gdf_end_list = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag'])
  gdf_route_list = gpd.GeoDataFrame(
    columns=['instructions', 'geometry', 'cardinal_direction', 'intersections'])
  gdf_main_list = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag'])
  gdf_near_list = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag'])
  gdf_beyond_list = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag'])

  counter = 0
  while counter < n_samples:
    entity = get_single_sample(map)
    if entity is None:
      continue
    counter += 1

    gdf_start_list = gdf_start_list.append(entity.start_point,
                         ignore_index=True)
    gdf_end_list = gdf_end_list.append(entity.end_point, ignore_index=True)

    gdf_route_list = gdf_route_list.append(entity.route,
                         ignore_index=True)
    gdf_main_list = gdf_main_list.append(entity.main_pivot,
                       ignore_index=True)
    gdf_near_list = gdf_near_list.append(entity.near_pivot,
                       ignore_index=True)
    gdf_beyond_list = gdf_beyond_list.append(entity.beyond_pivot,
                         ignore_index=True)

  if gdf_start_list.shape[0] == 0:
    return None

  gdf_start_list.to_file(path, layer='start', driver=_Geo_DataFrame_Driver)
  gdf_end_list.to_file(path, layer='end', driver=_Geo_DataFrame_Driver)
  gdf_route_list.to_file(path, layer='route', driver=_Geo_DataFrame_Driver)
  gdf_main_list.to_file(path, layer='main', driver=_Geo_DataFrame_Driver)
  gdf_near_list.to_file(path, layer='near', driver=_Geo_DataFrame_Driver)
  gdf_beyond_list.to_file(path, layer='beyond', driver=_Geo_DataFrame_Driver)


def get_path_entities(path: Text):
  '''Read a geodata file and print instruction.'''
  if not os.path.exists(path):
    return None
  start = gpd.read_file(path, layer='start')
  end = gpd.read_file(path, layer='end')
  route = gpd.read_file(path, layer='route')
  main = gpd.read_file(path, layer='main')
  near = gpd.read_file(path, layer='near')
  beyond = gpd.read_file(path, layer='beyond')

  entities = []
  for index in range(beyond.shape[0]):
    entity = item.RVSPath.from_file(
      start=start.iloc[index],
      end=end.iloc[index],
      route=route.iloc[index].geometry,
      main_pivot=main.iloc[index],
      near_pivot=near.iloc[index],
      beyond_pivot=beyond.iloc[index],
      cardinal_direction=route.iloc[index].cardinal_direction,
      intersections=route.iloc[index].intersections
    )
    entities.append(entity)

  return entities


def print_instructions(path: Text):
  '''Read a geodata file and print instruction.'''
  if not os.path.exists(path):
    sys.exit("The path to the RVS data was not found.")
  route = gpd.read_file(path, layer='route')
  print('\n'.join(route['instructions'].values))