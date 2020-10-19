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

from absl import logging
import copy
import geopandas as gpd
from geopandas import GeoSeries
import networkx as nx
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import box
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import box, mapping, LineString
from shapely import wkt
from shapely.ops import split

import swifter
import sys
from typing import Dict, Tuple, Sequence, Text, Optional, List, Any


from cabby.geo import util
from cabby.geo.map_processing import graph
from cabby.geo.map_processing import edge
from cabby.geo import regions


# Coordinate Reference Systems (CRS) - UTM Zones (North).
# This variable is used (:map_structure.py; cabby.geo.walk.py) to project the 
# geometries into this CRS for geo operations such as calculating the centroid.
OSM_CRS = 4326


class Map:

  def __init__(self, map_name: Text, level: int = 18, load_directory: Text = None):
    self.map_name = map_name
    self.s2_graph = None
    self.level = level
    self.polygon_area = regions.get_region(map_name)

    if load_directory is None:
      logging.info("Preparing map.")
      logging.info("Extracting POI.")
      self.poi, self.streets = self.get_poi()
      logging.info("Constructing graph.")
      self.nx_graph = ox.graph_from_polygon(
        self.polygon_area, network_type='walk')
      

      logging.info("Add POI to graph.")
      self.add_poi_to_graph()
      self.nodes, self.edges = ox.graph_to_gdfs(self.nx_graph)

    else:
      logging.info("Loading map from directory.")
      self.load_map(load_directory)

    logging.info("Create S2Graph.")
    self.create_S2Graph(level)

    self.process_param()

  def process_param(self):
    '''Helper function for processing the class data objects.'''

    # Set the coordinate system.
    self.poi = self.poi.set_crs(epsg=OSM_CRS, allow_override=True)
    self.nodes = self.nodes.set_crs(epsg=OSM_CRS, allow_override=True)
    self.edges = self.edges.set_crs(epsg=OSM_CRS, allow_override=True)

    # Drop columns with list type.
    self.edges.drop(self.edges.columns.difference(
      ['osmid', 'length', 'geometry', 'u', 'v', 'key']), 1, inplace=True)
    self.edges['osmid'] = self.edges['osmid'].apply(lambda x: str(x))

  def get_poi(self) -> Tuple[GeoSeries, GeoSeries]:
    '''Extract point of interests (POI) for the defined region. 
    Returns:
      (1) The POI that are not roads; and (2) the roads POI.
    '''

    tags = {'name': True,
            'amenity': True,
            'wikidata': True,
            'wikipedia': True,
            'shop': True,
            'brand': True,
            'tourism': True}

    osm_poi = ox.pois.pois_from_polygon(self.polygon_area, tags=tags)
    osm_poi = osm_poi.set_crs(epsg=OSM_CRS, allow_override=True)

    osm_highway = osm_poi['highway']
    osm_poi_no_streets = osm_poi[osm_highway.isnull()]
    osm_poi_streets = osm_poi[osm_highway.notnull()]

    # Get centroid for POI.
    osm_poi_no_streets['centroid'] = osm_poi_no_streets['geometry'].apply(
      lambda x: x if isinstance(x, Point) else x.centroid)

    return osm_poi_no_streets, osm_poi_streets

  def add_single_poi_to_graph(
      self, single_poi: pd.Series) -> Sequence[edge.Edge]:
    '''Add single POI to nx_graph.
    Arguments:
      single_poi: a single POI to be added to graph.
    Returns:
      The new osmid.
    '''
    # If the POI is already in the graph, do not add it.
    if single_poi['osmid'] in self.nx_graph:
      return None

    logging.info("processing new poi %d" % single_poi['osmid'])

    # Project POI on to the closest edge in graph.
    geometry = single_poi['geometry']
    if isinstance(geometry, Point):
      points = [single_poi['geometry']]
    elif isinstance(geometry, Polygon):
      coords = single_poi['geometry'].exterior.coords
      n_points = len(coords)

      # Sample maximum 4 points.
      sample_1 = Point(coords[0])
      sample_2 = Point(coords[round(n_points/4)])
      sample_3 = Point(coords[round(n_points/2)])
      sample_4 = Point(coords[round(3*n_points/4)])
      points = [sample_1, sample_2, sample_3, sample_4]
      points = points[0:4]
    else:
      return single_poi['osmid']

    poi_osmid = single_poi['osmid']
    poi_osmid = util.concat_numbers(999, poi_osmid)
    assert poi_osmid not in self.poi['osmid'].tolist(), poi_osmid
    self.poi.loc[self.poi['osmid'] ==
           single_poi['osmid'], 'osmid'] = poi_osmid

    list_edges_connected_ids = []
    edges_to_add = []
    for point in points:
      edges_to_add += self.add_single_point_edge(
        point, list_edges_connected_ids, poi_osmid)

    # Add node POI to graph.
    self.nx_graph.add_node(
      poi_osmid,
      highway="poi",
      osmid=poi_osmid,
      x=point.x,
      y=point.y,
      name="poi",
    )

    return edges_to_add


  def add_single_point_edge(self, point: Point,
                list_edges_connected_ids: List, 
                poi_osmid: int) -> Optional[Sequence[edge.Edge]]:
    '''Connect a POI to the closest edge: (1) claculate the projected point to 
    the nearest edge; (2) add the projected node to the graph; (3) create an 
    edge between the point of the POI and the projcted point (add to the list 
    to be returned); (4) create two edges from the closest edge (u-v) and the 
    projected point: (a) u-projected point; (b) v-projected point; (5) remove 
    closest edge u-v.
    Arguments:
      point: a point POI to be connected to the closest edge.
      list_edges_connected_ids: list of edges ids already connected to the POI. 
      If the current edge found is already connected it will avoid connecting 
      it again.
      poi_osmid: the POI  id to be connected to the edge.
    Returns:
      The edges between the POI and the closest edge found to be added to the 
      graph.
    '''

    try:
      
      near_edge_u, near_edge_v, near_edge_key, line = \
        ox.distance.get_nearest_edge(
          self.nx_graph, util.tuple_from_point(point), return_geom=True, 
          )

    except Exception as e:
      print(e)
      return []

    edge_id = (near_edge_u, near_edge_v, near_edge_key)

    if edge_id in list_edges_connected_ids:  # Edge already connected
      return []

    # Get nearest points - u and v.
    u_node = self.nx_graph.nodes[near_edge_u]
    u_point = Point(u_node['x'], u_node['y'])

    v_node = self.nx_graph.nodes[near_edge_v]
    v_point = Point(v_node['x'], v_node['y'])

    # Add to connected edges.
    list_edges_connected_ids.append(edge_id)

    near_edge = self.nx_graph.edges[edge_id]

    dist_projected = line.project(point)
    projected_point = line.interpolate(line.project(point))

    cut_geometry = util.cut(line,dist_projected)

    n_lines = len(cut_geometry) 
    
    line_1 = cut_geometry[0]
    dist_1 = util.get_line_length(line_1)

    if n_lines==2:
      assert projected_point==Point(line_1.coords[-1])
      line_2 = cut_geometry[1]
      dist_2 = util.get_line_length(line_2)

      projected_point_osmid = util.concat_numbers(
      len(list_edges_connected_ids), poi_osmid)

    else: # Projected point is exactly on the end of the line (U or V).
      dist_u_p = util.get_distance_between_points(u_point, projected_point)
      dist_v_p = util.get_distance_between_points(v_point, projected_point)
      if dist_u_p<dist_v_p:
        projected_point_osmid = near_edge_u
      else:
        projected_point_osmid = near_edge_v

    projected_line = LineString([projected_point,point])
    projected_line_dist = util.get_linestring_distance(projected_line)

    assert n_lines==1 or projected_point_osmid not in self.poi['osmid'].tolist(), (
      projected_point_osmid)

    if isinstance(near_edge['highway'], list):
      highway = ','.join(near_edge['highway'])
    else:
      highway = near_edge['highway']

    edges_list = []
    edge_to_add = edge.Edge.from_poi(
      u_for_edge=poi_osmid,
      v_for_edge=projected_point_osmid,
      osmid=near_edge['osmid'],
      geometry=projected_line,
      length = projected_line_dist
    )
    edges_list.append(edge_to_add)

    if n_lines==1:
      return edges_list
    
    self.nx_graph.add_node(
      projected_point_osmid,
      highway=highway,
      osmid=projected_point_osmid,
      x=projected_point.x,
      y=projected_point.y,
      name = 'projected-poi'
    )

    line_1_point_end = Point(line_1.coords[0])
    dist_u_1 = util.get_distance_between_points(u_point, line_1_point_end)
    dist_v_1 = util.get_distance_between_points(v_point, line_1_point_end)
    

    if dist_u_1 < dist_v_1:
      dist_u = dist_1
      line_u = line_1 
      dist_v = dist_2
      line_v = line_2
    else:
      dist_u = dist_2
      line_u = line_2 
      dist_v = dist_1
      line_v = line_1

    # Add edges between projected point and u and v, on the street segment.
    street_name = near_edge['name'] if 'name' in near_edge else ""

    if not projected_point_osmid == near_edge_u:
      edge_to_add = edge.Edge.from_projected(
        near_edge_u, projected_point_osmid, dist_u, near_edge['highway'],
        near_edge['osmid'], street_name, line_u)
      self.add_two_ways_edges(edge_to_add)

    if not projected_point_osmid == near_edge_v:
      edge_to_add = edge.Edge.from_projected(
        near_edge_v, projected_point_osmid, dist_v, near_edge['highway'],
        near_edge['osmid'], street_name, line_v)
      self.add_two_ways_edges(edge_to_add)

    # Remove u-v edge.
    self.nx_graph.remove_edge(near_edge_u, near_edge_v)
    self.nx_graph.remove_edge(near_edge_v, near_edge_u)
    return edges_list

  def add_two_ways_edges(self, edge_add: edge.Edge):
    '''Add edges to graph.'''

    assert edge_add.u_for_edge is not None
    assert edge_add.v_for_edge is not None

    self.nx_graph.add_edge(
      u_for_edge=edge_add.u_for_edge,
      v_for_edge=edge_add.v_for_edge,
      length=edge_add.length,
      osmid=edge_add.osmid,
      name=edge_add.name,
      highway=edge_add.highway,
      oneway=edge_add.oneway,
      geometry=edge_add.geometry 
    )

    self.nx_graph.add_edge(
      u_for_edge=edge_add.v_for_edge,
      v_for_edge=edge_add.u_for_edge,
      length=edge_add.length,
      osmid=edge_add.osmid,
      name=edge_add.name,
      highway=edge_add.highway,
      oneway=edge_add.oneway,
      geometry=edge_add.geometry 
    )

  def add_poi_to_graph(self):
    '''Add all POI to nx_graph(currently contains only the roads).'''
    edges_to_add_list = self.poi.apply(self.add_single_poi_to_graph, axis=1)
    edges_to_add_list = edges_to_add_list.dropna()
    edges_to_add_list.swifter.apply(
      lambda edges_list: [self.add_two_ways_edges(edge) for edge in edges_list])

    self.poi.set_index('osmid', inplace=True, drop=False)

  def get_s2cellids_for_poi(self, geometry: Any) -> Optional[Sequence[int]]:
    '''get cellids for POI. 
    Arguments:
      geometry: The geometry to which a cellids will be retrived.
    Returns:
      The the cellids.
    '''
    ""
    if isinstance(geometry, Point):
      return util.s2cellids_from_point(geometry, self.level)
    else:
      return util.s2cellids_from_polygon(geometry, self.level)

  def create_S2Graph(self, level: int):
    '''Helper funcion for creating S2Graph.'''

    # Get cellids for POI.
    self.poi['s2cellids'] = self.poi['geometry'].apply(
      self.get_s2cellids_for_poi)

    # Filter out entities that we didn't mange to get cellids covering.
    self.poi = self.poi[self.poi['s2cellids'].notnull()]

    # Create graph.
    self.s2_graph = graph.MapGraph()

    # Add POI to graph.
    self.poi[['s2cellids', 'osmid']].apply(
      lambda x: self.s2_graph.add_poi(x.s2cellids, x.osmid), axis=1)

  def get_valid_path(self, dir_name: Text, name_ending: Text,
             file_ending: Text) -> Optional[Text]:
    '''Creates the file path and checks validity.
    Arguments:
      dir_name: The directory of the path.
      name_ending: the end of the name  of the file
      (_graph or _node or_poi or_streets).
      file_ending: the type of the file.
    Returns:
      The valid path.
    '''

    base_filename = self.map_name.lower() + name_ending

    # Check if directory is valid.
    assert os.path.exists(dir_name), "Current directory is: {0}. The \
  directory {1} doesn't exist.".format(
      os.getcwd(), dir_name)

    # Create path.
    path = os.path.join(dir_name, base_filename + file_ending)

    return path

  def write_map(self, dir_name: Text):
    '''Save POI to disk.'''

    # Write POI.
    pd_poi = copy.deepcopy(self.poi)
    if 's2cellids' in pd_poi.columns:
      pd_poi['cellids'] = pd_poi['s2cellids'].apply(
        lambda x: util.cellids_from_s2cellids(x))
    pd_poi.drop(['s2cellids'], 1, inplace=True)

    path = self.get_valid_path(dir_name, '_poi', '.pkl')
    if not os.path.exists(path):
      pd_poi.to_pickle(path)
    else:
      logging.info("path {0} already exist.".format(path))

    # Write streets.
    pd_streets = copy.deepcopy(self.streets)

    path = self.get_valid_path(dir_name, '_streets', '.pkl')
    if not os.path.exists(path):
      pd_streets.to_pickle(path)
    else:
      logging.info("path {0} already exist.".format(path))

    # Write graph.
    base_filename = self.map_name.lower() + "_graph"
    path = os.path.join(dir_name, base_filename + ".gpickle")
    if not os.path.exists(path):
      nx.write_gpickle(self.nx_graph, path)
    else:
      logging.info("path {0} already exist.".format(path))

    # Write nodes.
    path = self.get_valid_path(dir_name, '_nodes', '.geojson')
    if not os.path.exists(path):
      self.nodes.to_file(path, driver='GeoJSON')
    else:
      logging.info("path {0} already exist.".format(path))

    # Write edges.
    path = self.get_valid_path(dir_name, '_edges', '.geojson')
    if not os.path.exists(path):
      self.edges.to_file(path, driver='GeoJSON')
    else:
      logging.info("path {0} already exist.".format(path))

  def load_map(self, dir_name: Text):
    '''Load POI from disk.'''

    # Load POI.
    path = self.get_valid_path(dir_name, '_poi', '.pkl')
    self.poi = load_poi(path)

    # Load streets.
    path = self.get_valid_path(dir_name, '_streets', '.pkl')
    self.streets = load_poi(path)

    # Load graph.
    path = self.get_valid_path(dir_name, '_graph', '.gpickle')
    assert os.path.exists(
      path), "path {0} doesn't exists".format(path)
    self.nx_graph = nx.read_gpickle(path)

    # Load nodes.
    path = self.get_valid_path(dir_name, '_nodes', '.geojson')
    assert os.path.exists(
      path), "path {0} doesn't exist.".format(path)
    self.nodes = gpd.read_file(path, driver='GeoJSON')

    # Load edges.
    path = self.get_valid_path(dir_name, '_edges', '.geojson')
    assert os.path.exists(
      path), "path {0} doesn't exist.".format(path)
    self.edges = gpd.read_file(path, driver='GeoJSON')
    self.edges['osmid_list'] = self.edges['osmid'].apply(
      lambda x: convert_string_to_list(x))

def load_poi(path: Text):
    '''Load POI from disk.'''
    assert os.path.exists(
      path), "Path {0} doesn't exist.".format(path)
    poi_pandas = pd.read_pickle(path)
    if 'cellids' in poi_pandas:
      poi_pandas['s2cellids'] = poi_pandas['cellids'].apply(
        lambda x: util.s2cellids_from_cellids(x))
      poi_pandas.drop(['cellids'], 1, inplace=True)

    return poi_pandas

def convert_string_to_list(string_list: Text) -> Sequence:
  '''Splitting a string into integers and creates a new list of the integers. 
  Arguments: 
  string_list: A string in the form of a list. E.g "[1,2,3]". 
  Returns: 
  A list of integers. 
  '''
  string_list = string_list.replace("[", "").replace("]", "")
  string_list = string_list.split(",")
  map_object = map(int, string_list)
  return list(map_object)