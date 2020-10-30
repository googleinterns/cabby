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

from typing import Tuple, Sequence, Optional, Dict, Text, Any, List

from absl import logging
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import networkx as nx
import sys
import os
import osmnx as ox
import pandas as pd
import json
import multiprocessing 
from multiprocessing import Pool
from multiprocessing import Semaphore
import random 
from shapely import geometry
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon, LinearRing
from shapely.geometry import box, mapping, LineString
import sys
    

from cabby.geo import util
from cabby.geo.map_processing import map_structure
from cabby.geo import geo_item
from cabby.rvs import item
from cabby.geo import osm

SMALL_POI = 4 # Less than 4 S2Cellids.
SEED = 4
SAVE_ENTITIES_EVERY = 1000
MAX_BATCH_GEN = 100
MAX_SEED = 2**32 - 1
MAX_PATH_DIST = 2000
MIN_PATH_DIST = 200
NEAR_PIVOT_DIST = 80
_Geo_DataFrame_Driver = "GPKG"
# The max number of failed tries to generate a single path entities.
MAX_NUM_GEN_FAILED = 10


ADD_POI_DISTANCE = 5000


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

  def get_non_specific_tag(self, poi):
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
      if new_tag in osm.BLACK_LIST:
        continue
      return new_tag
    return None
  
  def select_non_specific_unique_pois(self, pois: pd.DataFrame):
    '''Returns a non-specific POIs with main tag being the non-specific tag.
    Arguments:
      pois: all pois to select from.
    Returns:
      A number of non-specific POIs which are unique.
    '''
    # Assign main tag. 
    main_tags = pois.apply(self.get_non_specific_tag, axis=1)
    pois = pois.assign(main_tag = main_tags)
    pois.dropna(subset=['main_tag'], inplace=True)

    # Get Unique main tags.
    uniqueness = pois.duplicated(subset=['main_tag'], keep=False)==False
    pois_unique = pois[uniqueness]

    return pois_unique

  def select_non_specific_poi(self, pois: pd.DataFrame):
    '''Returns a non-specific POI with main tag being the non-specific tag.
    Arguments:
      pois: all pois to select from.
    Returns:
      A single sample of a POI with main tag being the non-specific tag.
    '''
    
    pois_unique = self.select_non_specific_unique_pois(pois)
    
    if pois_unique.shape[0]==0:
      return None
    # Sample POI.
    return self.sample_point(pois_unique)

  def get_end_poi(self,
  ) -> Optional[GeoSeries]:
    '''Returns a random POI.
    Returns:
      A single POI.
    '''
    
    # Filter large POI.
    small_poi = self.map.poi[self.map.poi['s2cellids'].str.len() <= SMALL_POI]

    if small_poi.shape[0]==0:
      return None
      
    # Filter non-specific tags.
    poi = self.select_non_specific_poi(small_poi)
    
    if poi is None:
      return None

    poi['geometry'] = poi.centroid

    return poi


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
    start_point = self.select_non_specific_poi(small_poi)
    
    if start_point is None:
      return None

    start_point['geometry'] = start_point.centroid
    return start_point


  def get_landmark_if_tag_exists(self, 
                                gdf: GeoDataFrame, 
                                tag: Text, main_tag: Text, 
                                alt_main_tag: Text
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


  def pick_prominent_pivot(self, df_pivots: GeoDataFrame, end_point: Dict
  ) -> Optional[GeoSeries]:
    '''Select a landmark from a set of landmarks by priority.
    Arguments:
      df_pivots: The set of landmarks.
      end_point: The goal location.
    Returns:
      A single landmark.
    '''

    # Remove goal location.
    df_pivots = df_pivots[df_pivots['osmid']!=end_point['osmid']]
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
    unique_poi = self.select_non_specific_unique_pois(nearby_poi)
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

    poly = LineString(points_route).buffer(0.0001)

    df_pivots = self.map.poi[self.map.poi.apply(
      lambda x: poly.intersects(x['geometry']), axis=1)]
    if df_pivots.shape[0]==0:
      return None
      
    # Remove streets.
    if 'highway' in df_pivots.columns:
      df_pivots = df_pivots[(df_pivots['highway'].isnull())]

    main_pivot = self.pick_prominent_pivot(df_pivots, end_point)
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


    if route.shape[0] < 2:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=self.map.nodes.columns).iloc[0]

    last_node_in_route = route.iloc[-1]
    before_last_node_in_route = route.iloc[-2]

    street_beyond_route = self.map.edges[
      (self.map.edges['u'] == last_node_in_route['osmid'])
      & (self.map.edges['v'] == before_last_node_in_route['osmid'])
    ]
    if street_beyond_route.shape[0] == 0:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

    street_beyond_osmid = street_beyond_route['osmid'].iloc[0]

    # Change OSMID to key
    segment_beyond_path = ((last_node_in_route['osmid'] == self.map.edges['u'])
                & (before_last_node_in_route['osmid'] !=
                  self.map.edges['v']))
    condition_street_id = self.map.edges['osmid'].apply(
      lambda x: x == street_beyond_osmid)
    last_line = self.map.edges[condition_street_id
                & segment_beyond_path]

    if last_line.shape[0] == 0:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

    last_line_max = last_line[last_line['length'] ==
                  last_line['length'].max()].iloc[0]

    poly = last_line_max['geometry'].buffer(0.0001)

    df_pivots = self.map.poi[self.map.poi.apply(
      lambda x: poly.intersects(x['geometry']), axis=1)]

    if df_pivots.shape[0] == 0:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

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

    poly_route = LineString(points_route).buffer(0.0001)

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
    beyond_pivot = self.pick_prominent_pivot(df_pivots, end_point)

    if beyond_pivot is None:
      # Return Empty.
      return GeoDataFrame(index=[0], columns=route.columns).iloc[0]

    return beyond_pivot

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


  def get_cardinal_direction(self, start_point: Point, end_point: Point
  ) -> Text:
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
  ) -> int:
    '''Return the number of intersections between the main_pivot and goal. 
    Arguments:
      main_pivot: The pivot along the route.
      route: The route along which a landmark will be chosen.
      end_point: The goal location.
    Returns:
      The number of intersections between the main_pivot and goal. 
      If the main_pivot and goal are on different streets return -1.
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
      return -1

    number_intersection = edges_in_pivot_goal_route.apply(
      lambda x: len(x - common_streets) > 0).sum()

    if number_intersection <= 0:
      return -1

    return number_intersection

  def get_sample(self, 
  ) -> Optional[item.RVSPath]:
    '''Sample start and end point, a pivot landmark and route.
    Returns:
      A start and end point, a pivot landmark and route.
    '''

    # Select end point.
    end_point = self.get_end_poi()
    if end_point is None:
      return None

    # Select start point.
    start_point = self.get_start_poi(end_point)
    if start_point is None:
      return None

    # Compute route between start and end points.
    route = self.compute_route_from_nodes(
          start_point['osmid'], 
          end_point['osmid'], 
          self.map.nx_graph, 
          self.map.nodes)
    if route is None:
      return None
    
    # Select pivots.
    result = self.get_pivots(route, end_point)
    if result is None:
      return None
    main_pivot, near_pivot, beyond_pivot = result

    # Get cardinal direction.
    cardinal_direction = self.get_cardinal_direction(
      start_point['geometry'], end_point['geometry'])

    # Get number of intersections between main pivot and goal location.
    intersections = self.get_number_intersections_past(
      main_pivot, route, end_point)

    rvs_path_entity = item.RVSPath.from_points_route_pivots(start_point,
                                end_point,
                                route,
                                main_pivot,
                                near_pivot,
                                beyond_pivot,
                                cardinal_direction,
                                intersections)

    return rvs_path_entity

  def get_single_sample(
            self, 
            index: int, 
            sema: Any, 
            n_samples: int, 
            return_dict: Dict[int, item.RVSPath]):
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
      entity = self.get_sample()
      attempt += 1
      if attempt >= MAX_NUM_GEN_FAILED:
        sys.exit("Reached max number of failed attempts.")
    
    logging.info(f"Created sample {index}/{n_samples}.")
    return_dict[index]=entity
    sema.release()


  def generate_and_save_rvs_routes(self,
                                  path_rvs_path: Text, 
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

      if len(new_entities)%SAVE_ENTITIES_EVERY == 0:
        self.save_entities(new_entities, path_rvs_path)
        new_entities = []
    if len(new_entities)>0:
      self.save_entities(new_entities, path_rvs_path)

    
  def save_entities(
    self, entities: Sequence[item.RVSPath], path_rvs_path: Text
  ):
    '''Save entities to path. If the path already exists append.
    Arguments:
      entities: RVSPath entities to add to path
      path_rvs_path: the path to add the entities too.
    '''

    if os.path.exists(path_rvs_path):
      geo_file = load(path_rvs_path)
    else:
      geo_file = geo_item.GeoPath.empty()
    
    for entity in entities:

      geo_file.start_point = geo_file.start_point.append(
        entity.start_point, ignore_index=True)
      geo_file.end_point = geo_file.end_point.append(
        entity.end_point, ignore_index=True)
      geo_file.path_features = geo_file.path_features.append(
        entity.path_features, ignore_index=True)
      geo_file.main_pivot = geo_file.main_pivot.append(
        entity.main_pivot, ignore_index=True)
      geo_file.near_pivot = geo_file.near_pivot.append(
        entity.near_pivot, ignore_index=True)
      geo_file.beyond_pivot = geo_file.beyond_pivot.append(
        entity.beyond_pivot, ignore_index=True)

    if geo_file.start_point.shape[0] == 0:
      return
    path = os.path.abspath(path_rvs_path)
    geo_file.start_point.to_file(
      path, layer='start', driver=_Geo_DataFrame_Driver)
    geo_file.end_point.to_file(path, layer='end', driver=_Geo_DataFrame_Driver)
    geo_file.path_features.to_file(
      path, layer='route', driver=_Geo_DataFrame_Driver)
    geo_file.main_pivot.to_file(
      path, layer='main', driver=_Geo_DataFrame_Driver)
    geo_file.near_pivot.to_file(
      path, layer='near', driver=_Geo_DataFrame_Driver)
    geo_file.beyond_pivot.to_file(
      path, layer='beyond', driver=_Geo_DataFrame_Driver)

    geo_file.size = geo_file.beyond_pivot.shape[0]

    logging.info(f"Saved {geo_file.size} entities to => {path}")


def load(path: Text) -> geo_item.GeoPath:
  start = gpd.read_file(path, layer='start')
  end = gpd.read_file(path, layer='end')
  route = gpd.read_file(path, layer='route')
  main = gpd.read_file(path, layer='main')
  near = gpd.read_file(path, layer='near')
  beyond = gpd.read_file(path, layer='beyond')
  return geo_item.GeoPath.from_file(start, end, route, main, near, beyond)

def load_entities(path: Text) -> List[item.RVSPath]:
  if not os.path.exists(path):
    return []
  
  geo_file = load(path)

  entities = []
  for index in range(geo_file.beyond_pivot.shape[0]):
    entity = item.RVSPath.from_file(
      start=geo_file.start_point.iloc[index],
      end=geo_file.end_point.iloc[index],
      route=geo_file.path_features.iloc[index].geometry,
      main_pivot=geo_file.main_pivot.iloc[index],
      near_pivot=geo_file.near_pivot.iloc[index],
      beyond_pivot=geo_file.beyond_pivot.iloc[index],
      cardinal_direction=geo_file.path_features.iloc[index].cardinal_direction,
      intersections=geo_file.path_features.iloc[index].intersections
    )
    entities.append(entity)

  logging.info(f"Loaded entities {len(entities)} from <= {path}")
  return entities


def print_instructions(path: Text):
  '''Read a geodata file and print instruction.'''
  if not os.path.exists(path):
    sys.exit(f"The path to the RVS data was not found {path}.")
  route = gpd.read_file(path, layer='route')
  logging.info('\n'.join(route['instructions'].values))