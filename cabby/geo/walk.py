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


from typing import Tuple, Sequence, Optional, Dict, Any

from absl import logging
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import inflect
import multiprocessing
from multiprocessing import Semaphore
import numpy as np
import networkx as nx
import os
import osmnx as ox
import pandas as pd
import random 
from shapely.geometry.point import Point
from shapely.geometry import LineString
import sys

from cabby.geo import util
from cabby.geo.map_processing import map_structure
from cabby.geo import geo_item
from cabby.geo import osm

SMALL_POI = 4 # Less than 4 S2Cellids.
SEED = 0
SAVE_ENTITIES_EVERY = 1000
MAX_BATCH_GEN = 100
MAX_SEED = 2**32 - 1
MAX_PATH_DIST = 2000
MIN_PATH_DIST = 200
NEAR_PIVOT_DIST = 80
# The max number of failed tries to generate a single path entities.
MAX_NUM_GEN_FAILED = 10
PIVOT_ALONG_ROUTE_MAX_DIST = 0.0001
ADD_POI_DISTANCE = 5000
MAX_NUM_BEYOND_TRY = 50

LANDMARK_TYPES = [
  "end_point", "start_point", "main_pivot", "near_pivot", "beyond_pivot"]

FEATURES_TYPES = ["cardinal_direction",
                  "spatial_rel_goal",
                  "spatial_rel_pivot",
                  "intersections",
                  "goal_position"]

inflect_engine = inflect.engine()


class Walker:
  def __init__(self, map: map_structure.Map, rand_sample: bool = True):
    #whether to sample randomly.
    self.rand_sample = rand_sample
    self.map = map

  def compute_route_from_nodes(self,
                               origin_id: str,
                               goal_id: str,
                               graph: nx.MultiDiGraph,
                               nodes: GeoDataFrame) -> Optional[GeoDataFrame]:
    '''Returns the shortest path between a starting and end point.
    Arguments:
      origin_id: The node id of the origin point.
      goal_id(Point): The node id of the destination point.
      graph(nx.MultiDiGraph): The directed graph class that stores multiedges.
      nodes(GeoDataFrame): The GeoDataFrame of graph nodes.
    Returns:
      A sequence of Points which construct the geometry of the path.
    '''

    # Get shortest route.
    try:
      route = nx.shortest_path(graph, origin_id, goal_id, 'length')
    except nx.exception.NetworkXNoPath:
      logging.info("No route found for the start and end points.")
      return None
    route_nodes = nodes[nodes['osmid'].isin(route)]

    # Create the dictionary that defines the order for sorting according to
    # route order.
    sorterIndex = dict(zip(route, range(len(route))))

    # Generate a rank column that will be used to sort
    # the dataframe numerically
    sorted_nodes = route_nodes['osmid'].map(sorterIndex)
    route_nodes = route_nodes.assign(sort=sorted_nodes)

    route_nodes = route_nodes.sort_values(['sort'])

    return route_nodes

  def compute_route_from_points(self,
                                start_point: Point,
                                end_point: Point,
                                graph: nx.MultiDiGraph,
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
      logging.info("No route found for the start and end points.")
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

  def get_generic_tag(self, poi: pd.Series) -> Optional[str]:
    '''Selects a non-specific tag (e.g., museum instead of "Austin Museum of
    Popular Culture") instead of a POI.
    Arguments:
      poi: The POI to select a non-specific tag for.
    Returns:
      A non-specific tag.
    '''
    for tag, addition in osm.NON_SPECIFIC_TAGS.items():
      if tag not in poi or not isinstance(poi[tag], str):
        continue
      if addition == True:
        return tag
      tag_value = poi[tag]
      tag_value_clean = tag_value.replace("_", " ")
      if tag_value in osm.CORRECTIONS:
        tag_value_clean = osm.CORRECTIONS[tag_value]
      if tag_value_clean in ['yes', 'no']:
        continue
      if addition == 'after':
        new_tag = tag_value_clean + " " + tag
      elif addition == "before":
        new_tag = tag + " " + tag_value_clean
      elif addition == False:
        new_tag = tag_value_clean
      elif tag_value not in addition:
        continue
      else:
        new_tag = tag_value_clean
      if new_tag in osm.CORRECTIONS:
        new_tag = osm.CORRECTIONS[new_tag]
      if new_tag in osm.BLOCK_LIST:
        continue
      return new_tag
    return None

  def select_generic_unique_pois(self, pois: pd.DataFrame, is_unique: bool = False):
    '''Returns a non-specific POIs with main tag being the non-specific tag.
    Arguments:
      pois: all pois to select from.
      is_unique: if to filter unique tags.
    Returns:
      A number of non-specific POIs which are unique.
    '''
    # Assign main tag.
    main_tags = pois.apply(self.get_generic_tag, axis=1)
    new_pois = pois.assign(main_tag = main_tags)
    new_pois.dropna(subset=['main_tag'], inplace=True)

    # Get Unique main tags.
    if is_unique:
      # Randomly select whether the near by pivot would be
      # a single pivot (e.g., `a toy shop` or
      # a group of unique landmark (e.g, `3 toy shops`)
      is_group = self.randomize_boolean()
      if is_group:
        uniqueness = new_pois.duplicated(subset=['main_tag'], keep=False)==True
        new_pois_uniq = new_pois[uniqueness]
        if new_pois_uniq.shape[0]==0:
          is_group = False
        else:
          count_by_tag = new_pois_uniq.main_tag.value_counts()
          chosen_tag = random.choice(count_by_tag.keys())
          chosen_count = count_by_tag.loc[chosen_tag]
          by_word = self.randomize_boolean()
          if by_word:
            chosen_count = inflect_engine.number_to_words(chosen_count)
          new_pois_uniq = new_pois_uniq[new_pois_uniq['main_tag']==chosen_tag][:1]
          new_pois_uniq['main_tag'] = str(chosen_count) + \
                                 " " + inflect_engine.plural(chosen_tag)
      if not is_group:
        uniqueness = new_pois.duplicated(subset=['main_tag'], keep=False)==False
        new_pois_uniq = new_pois[uniqueness]
      return new_pois_uniq
    return new_pois

  def select_generic_poi(self, pois: pd.DataFrame):
    '''Returns a non-specific POI with main tag being the non-specific tag.
    Arguments:
      pois: all pois to select from.
    Returns:
      A single sample of a POI with main tag being the non-specific tag.
    '''

    pois_generic = self.select_generic_unique_pois(pois)

    if pois_generic.shape[0]==0:
      return None
    # Sample POI.
    poi = self.sample_point(pois_generic)
    poi['geometry'] = poi.centroid
    return poi


  def get_end_poi(self,) -> Optional[GeoSeries]:
    '''Returns a random POI.
    Returns:
      A single POI.
    '''

    # Filter large POI.
    small_poi = self.map.poi[self.map.poi['s2cellids'].str.len() <= SMALL_POI]

    if small_poi.shape[0]==0:
      return None

    # Filter non-specific tags.
    return self.select_generic_poi(small_poi)

  def randomize_boolean(self) -> bool:
    '''Returns a random\non random boolean value.
    '''
    if self.rand_sample:
      return bool(random.getrandbits(1))
    return True

  def sample_point(self,
      df: gpd.GeoDataFrame
  ) -> GeoSeries:
    '''Returns a random\non random 1 sample of a POI.
    Arguments:
      df: data to sample.
    Returns:
      A single sample of a POI.
    '''
    if self.rand_sample:
      return df.sample(1, random_state = random.randint(0, MAX_SEED)).iloc[0]
    return df.sample(1, random_state=SEED).iloc[0]


  def get_start_poi(self,
                    end_point: Dict
  ) -> Optional[GeoSeries]:
    '''Returns the a random POI within distance of a given POI.
    Arguments:
      end_point: The POI to which the picked POI should be within distance
      range.
    Returns:
      A single POI.
    '''

    # Get closest nodes to points.
    dest_osmid = end_point['osmid']

    try:
      # Find nodes within 2000 meter path distance.
      outer_circle_graph = ox.truncate.truncate_graph_dist(
        self.map.nx_graph, dest_osmid,
        max_dist=(MAX_PATH_DIST + 2 * ADD_POI_DISTANCE), weight='length')

      outer_circle_graph_osmid = list(outer_circle_graph.nodes.keys())
    except nx.exception.NetworkXPointlessConcept:  # GeoDataFrame returned empty
      return None

    try:
      # Get graph that is too close (less than 200 meter path distance)
      inner_circle_graph = ox.truncate.truncate_graph_dist(
        self.map.nx_graph, dest_osmid,
        max_dist=MIN_PATH_DIST + 2 * ADD_POI_DISTANCE, weight='length')
      inner_circle_graph_osmid = list(inner_circle_graph.nodes.keys())

    except nx.exception.NetworkXPointlessConcept:  # GeoDataFrame returned empty
      inner_circle_graph_osmid = []

    osmid_in_range = [
      osmid for osmid in outer_circle_graph_osmid if osmid not in
      inner_circle_graph_osmid]

    poi_in_ring = self.map.poi[self.map.poi['osmid'].isin(osmid_in_range)]

    # Filter large POI.
    small_poi = poi_in_ring[poi_in_ring['s2cellids'].str.len() <= SMALL_POI]

    # Filter non-specific tags.
    return self.select_generic_poi(small_poi)

  def get_landmark_if_tag_exists(self, 
                                gdf: GeoDataFrame, 
                                tag: str, main_tag: str,
                                alt_main_tag: str
  ) -> GeoSeries:
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
          if 'main_tag' not in pivots:
            pivots = pivots.assign(main_tag=pivots[main_tag])
          return self.sample_point(pivots)
        pivots = gdf[gdf[alt_main_tag].notnull()]
        if alt_main_tag in candidate_landmarks and pivots.shape[0]:
          if 'main_tag' not in pivots:
            pivots = pivots.assign(main_tag=pivots[alt_main_tag])
          return self.sample_point(pivots)
    return None


  def pick_prominent_pivot(self,
                           df_pivots: GeoDataFrame,
                           end_point: Dict[str, Any],
  ) -> Optional[GeoSeries]:
    '''Select a landmark from a set of landmarks by priority.
    Arguments:
      df_pivots: The set of landmarks.
      end_point: The goal location.
    Returns:
      A single landmark.
    '''

    # Remove goal location.
    try:
     df_pivots = df_pivots[df_pivots['osmid']!=end_point['osmid']]
    except:
      pass

    if df_pivots.shape[0]==0:
      return None

    tag_pairs = [('wikipedia', 'amenity'), ('wikidata', 'amenity'),
          ('brand', 'brand'), ('tourism', 'tourism'),
          ('tourism', 'tourism'), ('amenity', 'amenity'), ('shop', 'shop')
          ]

    pivot = None

    for main_tag, named_tag in tag_pairs:
      pivot = self.get_landmark_if_tag_exists(df_pivots,
                                              main_tag,
                                              'name',
                                              named_tag)
      if pivot is not None:
        if not isinstance(pivot['geometry'], Point):
          pivot['geometry'] = pivot['geometry'].centroid
        return pivot

    return pivot


  def get_pivot_near_goal(self,
                          end_point: GeoSeries
  ) -> Optional[GeoSeries]:
    '''Return a picked landmark near the end_point.
    Arguments:
      end_point: The goal location.
    Returns:
      A single landmark near the goal location.
    '''

    near_poi_con = self.map.poi.apply(
      lambda x: util.get_distance_between_geometries(
        x.geometry,
        end_point['centroid']) < NEAR_PIVOT_DIST, axis=1)

    poi = self.map.poi[near_poi_con]

    if poi.shape[0]==0:
      return None

    # Remove streets and roads.
    if 'highway' in poi.columns:
      poi = poi[poi['highway'].isnull()]

    # Remove the endpoint.
    nearby_poi = poi[poi['osmid'] != end_point['osmid']]

    # Filter non-specific tags.
    unique_poi = self.select_generic_unique_pois(
      nearby_poi, is_unique=True)
    if unique_poi.shape[0]==0:
      return None

    prominent_poi = self.pick_prominent_pivot(unique_poi, end_point)
    return prominent_poi


  def get_pivot_along_route(self,
                            route: GeoDataFrame,
                            end_point: Dict
  ) -> Optional[GeoSeries]:
    '''Return a picked landmark on a given route.
    Arguments:
      route: The route along which a landmark will be chosen.
      end_point: The goal location.
    Returns:
      A single landmark. '''

    # Get POI along the route.
    points_route = route['geometry'].tolist()

    poly = LineString(points_route).buffer(PIVOT_ALONG_ROUTE_MAX_DIST)

    df_pivots = self.map.poi[self.map.poi.apply(
      lambda x: poly.intersects(x['geometry']), axis=1)]
    if df_pivots.shape[0]==0:
      return None

    # Remove streets.
    if 'highway' in df_pivots.columns:
      df_pivots = df_pivots[(df_pivots['highway'].isnull())]

    # Remove POI near goal.
    far_poi_con = df_pivots.apply(
    lambda x: util.get_distance_between_geometries(
      x.geometry,
      end_point['centroid']) > NEAR_PIVOT_DIST, axis=1)
    far_poi = df_pivots[far_poi_con]

    main_pivot = self.pick_prominent_pivot(far_poi, end_point)
    return main_pivot

  def get_pivot_beyond_goal(self,
                            end_point: GeoSeries,
                            route: GeoDataFrame,
  ) -> Optional[GeoSeries]:
    '''Return a picked landmark on a given route.
    Arguments:
      end_point: The goal location.
      route: The route along which a landmark will be chosen.
    Returns:
      A single landmark. '''

    columns_empty = self.map.nodes.columns.tolist() + ['main_tag']

    if route.shape[0] < 2:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=columns_empty).iloc[0]

    final_node_in_route = route.iloc[-1]
    last_node_in_route = route.iloc[-2]
    before_last_node_in_route = route.iloc[-3]

    street_beyond_route = self.map.edges[
      (self.map.edges['u'] == last_node_in_route['osmid'])
      & (self.map.edges['v'] == before_last_node_in_route['osmid'])
    ]
    if street_beyond_route.shape[0] == 0:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=columns_empty).iloc[0]

    street_beyond_osmid = street_beyond_route['osmid'].iloc[0]
    condition_street_id = self.map.edges['osmid'].apply(
      lambda x: x == street_beyond_osmid)
    street_nodes = self.map.edges[condition_street_id]['u'].unique()
    street_nodes = np.random.choice(street_nodes, MAX_NUM_BEYOND_TRY)
    for i in range(MAX_NUM_BEYOND_TRY):
      length = nx.shortest_path_length(
        self.map.nx_graph,
        source=street_nodes[i],
        target=last_node_in_route['osmid'])

      # The beyond pivot should not be too close but also not too far away.
      if not (length>3 and length<10):
        continue

      # Check the path between the POI and the last node in the route taken.
      # If the path calculated does not pass through the route taken then it is
      # beyond the route.
      path = nx.shortest_path(self.map.nx_graph,
                              source=street_nodes[i],
                              target=last_node_in_route['osmid'])

      # Remove the nodes in the route taken from the path calculated so that it
      # will not be choosen as the pivot beyond.
      path.remove(last_node_in_route['osmid'])
      if final_node_in_route['osmid'] in path:
        path.remove(final_node_in_route['osmid'])

      intersections = set(route['osmid']).intersection(path)

      # Check if the path calculated overlaps the route taken,
      # if not then pick a POI to be the pivot beyond.
      if len(intersections)<1 and len(path)>2:
        beyond = self.select_pivot_from_path(route, end_point, path)
        if beyond is not None:
          return beyond
    return GeoDataFrame(index=[0], columns=columns_empty).iloc[0]


  def select_pivot_from_path(self,
                        route: GeoDataFrame,
                        end_point: Dict,
                        path: Sequence):

    path_nodes = self.map.nodes[self.map.nodes['osmid'].isin(path)]
    points_route = path_nodes['geometry'].tolist()
    path_beyond = LineString(points_route).buffer(PIVOT_ALONG_ROUTE_MAX_DIST)
    route_shape = LineString(
      route['geometry'].tolist()).buffer(PIVOT_ALONG_ROUTE_MAX_DIST)

    df_pivots = self.map.poi[self.map.poi.apply(
      lambda x: (path_beyond.intersects(x['geometry'])) &
                (route_shape.intersects(x['geometry'])==False), axis=1)]

    if df_pivots.shape[0] == 0:
      return None

    # Remove streets.
    if 'highway' in df_pivots.columns:
      df_pivots = df_pivots[(df_pivots['highway'].isnull())]

    # Remove invalid geometry.
    df_pivots = df_pivots[(df_pivots['geometry'].is_valid)]
    if df_pivots.shape[0] == 0:
      # Return Empty.
      return None

    beyond_pivot = self.pick_prominent_pivot(df_pivots, end_point)
    return beyond_pivot


  def get_position_goal(self,
                        end_point: GeoSeries,
                        route: GeoDataFrame
  ) -> Optional[str]:
    '''Return the position of the goal in the last block:
    middle of the block\ near the closest intersection\ near the farther intersection
    Arguments:
      end_point: The goal location.
      route: The route along which a landmark will be chosen.
    Returns:
      The position of the goal in last block. '''

    street = self.map.edges[self.map.edges['u'] == end_point['osmid']].iloc[0]['osmid']
    nodes_u = self.map.edges[self.map.edges['osmid']==street]['u']
    condition_intersection = self.map.edges['osmid'] != street
    condition_not_poi = self.map.edges['name'] != 'poi'
    intersections_nodes_osmid = self.map.edges[
      condition_not_poi & condition_intersection & self.map.edges['u'].isin(nodes_u)]['u']
    intersections_nodes = self.map.nodes[self.map.nodes['osmid'].isin(intersections_nodes_osmid)]
    distances = intersections_nodes.apply(
      lambda x: util.get_distance_between_geometries(x.geometry, end_point.centroid), axis=1)
    intersections_nodes.insert(0, "distances", distances, True)

    bearing = intersections_nodes.apply(
      lambda x: util.get_bearing(x.geometry.centroid, end_point.centroid), axis=1)
    intersections_nodes.insert(0, "bearing", bearing, True)
    min_distance_idx = intersections_nodes['distances'].idxmin()
    bearing = intersections_nodes['bearing'].loc[min_distance_idx]
    distance_closest = intersections_nodes['distances'].loc[min_distance_idx]

    # Get second bearing in opposite direction.
    opposite_bearing = (bearing+180)%360
    intersection_opposite = intersections_nodes[(intersections_nodes['bearing']-opposite_bearing)%360<30]
    if intersection_opposite.shape[0]==0:
      return None
    intersection_opposite_idx = intersection_opposite['distances'].idxmin()
    intersection_opposite_distance = intersection_opposite.loc[intersection_opposite_idx]['distances']

    # Check the proportions.
    total_distance = intersection_opposite_distance + distance_closest
    closest_propotion = distance_closest/total_distance
    if closest_propotion>0.4:
      return "in the middle of the block"

    if closest_propotion>0.3:
      return None
    # Check to which intersection it is closer.
    closest_inter_node_osmid = intersections_nodes.loc[min_distance_idx]['osmid']
    if closest_inter_node_osmid in route['osmid'].tolist():
      return "near the last intersection passed"

    return "near the next intersection"

  def get_pivots(self,
                route: GeoDataFrame,
                end_point: Dict,
  ) -> Optional[Tuple[GeoSeries, GeoSeries, GeoSeries]]:
    '''Return a picked landmark on a given route.
    Arguments:
      route: The route along which a landmark will be chosen.
      end_point: The goal location.
    Returns:
      A single landmark.
    '''

    # Get pivot along the goal location.
    main_pivot = self.get_pivot_along_route(route, end_point)
    if main_pivot is None:
      return None

    # Get pivot near the goal location.
    near_pivot = self.get_pivot_near_goal(end_point)

    if near_pivot is None:
      return None

    # Get pivot located past the goal location and beyond the route.
    beyond_pivot = self.get_pivot_beyond_goal(end_point, route)

    return main_pivot, near_pivot, beyond_pivot

  def get_egocentric_spatial_relation_pivot(self,
                                            ref_point: Point,
                                            route: GeoDataFrame
                                            ) -> str:
    line = LineString(route['geometry'].tolist())
    dist_projected = line.project(ref_point)
    cut_geometry = util.cut(line, dist_projected)
    first_segment = cut_geometry[0]
    coords = list(first_segment.coords)
    return self.calc_spatial_relation_for_line(
      ref_point, Point(coords[-1]), Point(coords[-2]))

  def calc_spatial_relation_for_line(self,
                                     ref_point: Point,
                                     line_point_last_part: Point,
                                     line_point_second_from_last: Point,
  ) -> str:

    # Calculate the angle of the last segment of the line_point_last_part.
    azim_route = util.get_bearing(
      line_point_second_from_last, line_point_last_part)

    # Calculate the angle between the last segment of the route and the goal.
    azim_ref_point = util.get_bearing(
      line_point_last_part, ref_point)

    diff_azim = (azim_ref_point-azim_route) % 360

    if diff_azim < 180:
      return "right"

    return "left"

  def get_egocentric_spatial_relation_goal(self,
                                           ref_point: Point,
                                           route: GeoDataFrame
                                           ) -> str:

    final_node_in_route = route.iloc[-2]['geometry'].centroid
    last_node_in_route = route.iloc[-3]['geometry'].centroid

    return self.calc_spatial_relation_for_line(
      ref_point, final_node_in_route, last_node_in_route)


  def get_cardinal_direction(self, start_point: Point, end_point: Point
  ) -> str:
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


  def get_number_intersections_past(self,
                                    main_pivot: GeoSeries,
                                    route: GeoDataFrame,
                                    end_point: Point
  ) -> Optional[int]:
    '''Return the number of intersections between the main_pivot and goal.
    Arguments:
      main_pivot: The pivot along the route.
      route: The route along which a landmark will be chosen.
      end_point: The goal location.
    Returns:
      The number of intersections between the main_pivot and goal.
      If the main_pivot and goal are on different streets return None.
    '''

    pivot_goal_route = self.compute_route_from_nodes(
            main_pivot['osmid'],
            end_point['osmid'],
            self.map.nx_graph,
            self.map.nodes)


    edges_in_pivot_goal_route = pivot_goal_route['osmid'].apply(
      lambda x: set(self.map.edges[self.map.edges['u'] == x]['osmid'].tolist()))

    pivot_streets = edges_in_pivot_goal_route.iloc[0]
    goal_streets = edges_in_pivot_goal_route.iloc[-1]
    common_streets = pivot_streets & goal_streets
    if not common_streets:
      return None

    number_intersection = edges_in_pivot_goal_route.apply(
      lambda x: len(x - common_streets) > 0).sum()

    if number_intersection <= 0:
      return 0

    return number_intersection

  def get_sample(self) -> Optional[geo_item.GeoEntity]:

    '''Sample start and end point, a pivot landmark and route.
    Returns:
      A start and end point, a pivot landmark and route.
    '''

    geo_landmarks = {}
    # Select end point.
    geo_landmarks['end_point'] = self.get_end_poi()
    if geo_landmarks['end_point'] is None:
      return None

    # Select start point.
    geo_landmarks['start_point'] = self.get_start_poi(geo_landmarks['end_point'])
    if geo_landmarks['start_point'] is None:
      return None

    # Compute route between start and end points.
    route = self.compute_route_from_nodes(
          geo_landmarks['start_point']['osmid'],
          geo_landmarks['end_point']['osmid'],
          self.map.nx_graph,
          self.map.nodes)
    if route is None:
      return None

    # Select pivots.
    result = self.get_pivots(route, geo_landmarks['end_point'])
    if result is None:
      return None

    geo_landmarks['main_pivot'], geo_landmarks['near_pivot'], \
     geo_landmarks['beyond_pivot'] = result

    geo_features = {}
    # Get cardinal direction.
    geo_features['cardinal_direction'] = self.get_cardinal_direction(
      geo_landmarks['start_point']['geometry'], geo_landmarks['end_point']['geometry'])

    # Get Egocentric spatial relation from goal.
    geo_features['spatial_rel_goal'] = self.get_egocentric_spatial_relation_goal(
      geo_landmarks['end_point']['geometry'].centroid, route)

    # Get Egocentric spatial relation from main pivot.
    geo_features['spatial_rel_pivot'] = self.get_egocentric_spatial_relation_pivot(
      geo_landmarks['main_pivot']['geometry'].centroid, route)


    # Get number of intersections between main pivot and goal location.
    geo_features['intersections'] = self.get_number_intersections_past(
      geo_landmarks['main_pivot'], route, geo_landmarks['end_point'])

    geo_features['goal_position'] = self.get_position_goal(
      geo_landmarks['end_point'], route)

    rvs_path_entity = geo_item.GeoEntity.add_entity(
      route=route,
      geo_features=geo_features,
      geo_landmarks=geo_landmarks)

    return rvs_path_entity

  def get_single_sample(
            self,
            index: int,
            sema: Any,
            n_samples: int,
            return_dict: Dict[int, geo_item.GeoEntity]):
    '''Sample exactly one RVS path sample.
    Arguments:
      index: index of sample.
      sema: Semaphore Object.
      n_samples: the total number of samples to generate.
      return_dict: The dictionary of samples generated.
    '''
    sema.acquire()
    entity = None
    attempt = 0
    while entity is None:
      if attempt >= MAX_NUM_GEN_FAILED:
        sys.exit("Reached max number of failed attempts.")
      entity = self.get_sample()
      attempt += 1

    logging.info(f"Created sample {index}/{n_samples}.")
    return_dict[index]=entity
    sema.release()


  def generate_and_save_rvs_routes(self,
                                  path_rvs_path: str,
                                  n_samples: int,
                                  n_cpu: int = multiprocessing.cpu_count()-1
                                  ):
    '''Sample start and end point, a pivot landmark and route and save to file.
    Arguments:
      path_rvs_path: The path to which the data will be appended.
      map: The map of a specific region.
      n_samples: the max number of samples to generate.
    '''

    manager = multiprocessing.Manager()
    
    sema = Semaphore(n_cpu)
    new_entities = [] 
    lst = list(range(n_samples))
    batches = [
      lst[i:i + MAX_BATCH_GEN] for i in range(0, len(lst), MAX_BATCH_GEN)]
    for batch in batches:
      return_dict = manager.dict()
      jobs = []
      for i in batch:
          p = multiprocessing.Process(
            target=self.get_single_sample, 
            args=(i+1, sema ,n_samples, 
            return_dict))
          jobs.append(p)
          p.start()
      
      for proc in jobs:
          proc.join()
      new_entities += [entity for idx_entity, entity in return_dict.items()]

      if len(new_entities)>SAVE_ENTITIES_EVERY:
        geo_item.save(new_entities, path_rvs_path)
        new_entities = []
    if len(new_entities)>0:
      geo_item.save(new_entities, path_rvs_path)


def load_entities(path: str) -> Sequence[geo_item.GeoEntity]:
  if not os.path.exists(path):
    return []
  geo_types_all = {}
  for landmark_type in LANDMARK_TYPES:
    geo_types_all[landmark_type] = gpd.read_file(path, layer=landmark_type)
  geo_types_all['route'] = gpd.read_file(path, layer='path_features')['geometry']
  geo_types_all['path_features'] = gpd.read_file(path, layer='path_features')
  geo_entities = []
  for row_idx in range(geo_types_all[LANDMARK_TYPES[0]].shape[0]):
    landmarks = {}
    for landmark_type in LANDMARK_TYPES:
      landmarks[landmark_type] = geo_types_all[landmark_type].iloc[row_idx]
    features = geo_types_all['path_features'].iloc[row_idx].to_dict()
    del features['geometry']
    route = geo_types_all['route'].iloc[row_idx]
    geo_item_cur = geo_item.GeoEntity.add_entity(
      geo_landmarks=landmarks,
      geo_features=features,
      route=LineString(route.exterior.coords[:-1])
      )
    geo_entities.append(geo_item_cur)

  logging.info(f"Loaded entities {len(geo_entities)} from <= {path}")
  return geo_entities

