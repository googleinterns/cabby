 
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
from shapely import wkt
import swifter
import sys
from typing import Dict, Tuple, Sequence, Text, Optional, List, Any


from cabby import logger
from cabby.geo import util
from cabby.geo.map_processing import graph
from cabby.geo.map_processing import edge
from cabby import logger

map_logger = logger.create_logger("map.log", 'map')

OSM_CRS = 32633 # UTM Zones (North).

class Map:

    def __init__(self, map_name: Text, level: int, load_directory: Text = None):
        assert map_name == "Manhattan" or map_name == "Pittsburgh" or \
            map_name == "DC"
        self.map_name = map_name
        self.s2_graph = None
        self.level = level

        if map_name == "Manhattan":
            self.polygon_area = wkt.loads(
                'POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'
            )

        elif map_name == "Pittsburgh":
            self.polygon_area = box(
                miny=40.425, minx=-80.035, maxy=40.460, maxx=-79.930,
                ccw=True)
        else:  # Bologna.
            self.polygon_area = box(
                miny=38.90821, minx=-77.04053, maxy=38.90922, maxx=-77.03937,
                ccw=True)


        if load_directory is None:
            self.poi, self.streets = self.get_poi()
            self.nx_graph = ox.graph_from_polygon(
                self.polygon_area, network_type='walk')
            self.add_poi_to_graph()
            self.nodes, self.edges = ox.graph_to_gdfs(self.nx_graph)

            # Find closest nodes to POI.
            self.poi['node'] = self.poi['centroid'].apply(self.closest_nodes)

        else:
            self.load_map(load_directory)
        
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

    def closest_nodes(self, point: Point) -> int:
      '''Find closest nodes to POI. 
      Arguments:
        point: The point to which a closest node will be retrived.
      Returns:
        The Node id closest to point.
      '''
      ""
      point_xy = util.tuple_from_point(point)
      return ox.distance.get_nearest_node(self.nx_graph, point_xy)

    def get_poi(self) -> Tuple[GeoSeries, GeoSeries]:
        '''Helper funcion for extracting POI for the defined place.'''

        tags = {'name': True, 'building': True, 'amenity': True}

        osm_poi = ox.pois.pois_from_polygon(self.polygon_area, tags=tags)
        osm_poi = osm_poi.set_crs(epsg=OSM_CRS, allow_override=True)

        osm_poi_named_entities = osm_poi[osm_poi['name'].notnull()]
        osm_highway = osm_poi_named_entities['highway']
        osm_poi_no_streets = osm_poi_named_entities[osm_highway.isnull()]
        osm_poi_streets = osm_poi_named_entities[osm_highway.notnull()]

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

        
        # Project POI on to the closest edge in graph.
        geometry = single_poi['geometry']
        if isinstance(geometry, Point):
          points = [single_poi['geometry']]
        elif  isinstance(geometry, Polygon):
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
        poi_osmid =util.concat_numbers(9999, poi_osmid) 
        single_poi['osmid'] = poi_osmid
        assert poi_osmid not in self.poi['osmid'].tolist(), poi_osmid

        list_edges_connected_ids = []
        eddges_to_add = []
        for point in points:
          eddges_to_add += self.add_single_point_edge(
          point, list_edges_connected_ids, poi_osmid)

        # Add node POI to graph.
        self.nx_graph.add_node(
                              poi_osmid,
                              highway = "poi",
                              osmid = poi_osmid, 
                              x = point.x , 
                              y = point.y
                              )

        return eddges_to_add

    
    def add_single_point_edge(self, point: Point, 
    list_edges_connected_ids: List, poi_osmid: int) -> Optional[Sequence[edge.Edge]]:
        '''Connect a poi to the closest edge.
        Arguments:
          point: a point POI to be connected to the closest edge.
          list_edges_connected_ids: list of edges ids already connected to the POI. 
          If the current edge found is already connected it will avoid connecting it again.
          poi_osmid: the POI  id to be connected to the edge.
        Returns:
          The edges between the POI and the closest edge found to be added to the graph.
        '''
        
        try:
          near_edge_u, near_edge_v, near_edge_key, line = \
          ox.distance.get_nearest_edge(
          self.nx_graph, util.tuple_from_point(point), return_geom=True)

        except Exception as e:
          print (e)
          return []

        edge_id = (near_edge_u,near_edge_v, near_edge_key)

        if edge_id in list_edges_connected_ids: # Edge already connected
          return  []
        
        # Add to connected edges.
        list_edges_connected_ids.append(edge_id)

        near_edge = self.nx_graph.edges[edge_id]
        
        projected_point = line.interpolate(line.project(point))
        
        projected_point_osmid = util.concat_numbers(
        len(list_edges_connected_ids),poi_osmid)

        assert projected_point_osmid not in self.poi['osmid'].tolist(), (
        projected_point_osmid)
        


        self.nx_graph.add_node(
                              projected_point_osmid, 
                              highway = ','.join(near_edge['highway']), 
                              osmid = projected_point_osmid, 
                              x = projected_point.x, 
                              y = projected_point.y
                              ) 
        
        edges_list = []
        edge_to_add = edge.Edge.from_poi(u_for_edge = poi_osmid, 
                v_for_edge = projected_point_osmid, osmid = poi_osmid
                )
        edges_list.append(edge_to_add)

        edge_to_add = edge.Edge.from_poi(u_for_edge = projected_point_osmid, 
        v_for_edge = poi_osmid, osmid = poi_osmid)
        edges_list.append(edge_to_add)

        # Get nearest points - u and v.
        u_node = self.nx_graph.nodes[near_edge_u]
        u_point = Point(u_node['x'], u_node['y'])
        
        v_node = self.nx_graph.nodes[near_edge_v]
        v_point = Point(v_node['x'], v_node['y'])

        # Calculate distance between projected point and u and v.
        dist_u = util.get_distance_m(u_point, projected_point)
        dist_v = util.get_distance_m(v_point, projected_point)

        # Add edges between projected point and u and v, on the street segment. 

        street_name = near_edge['name'] if 'name' in near_edge else ""
      
        edge_to_add = edge.Edge.from_projected(
        near_edge_u, projected_point_osmid, dist_u,  near_edge['highway'], 
        near_edge['osmid'], street_name)
        self.add_two_ways_edges(edge_to_add)

        edge_to_add = edge.Edge.from_projected(
        near_edge_v, projected_point_osmid, dist_v, near_edge['highway'],
        near_edge['osmid'], street_name)
        self.add_two_ways_edges(edge_to_add)
      

        # Remove u-v edge.
        self.nx_graph.remove_edge(near_edge_u, near_edge_v)
        self.nx_graph.remove_edge(near_edge_v ,near_edge_u)
        return edges_list

    def add_two_ways_edges(self, edge_add: edge.Edge):  
        '''Add edges to graph.'''

        self.nx_graph.add_edge(
                        u_for_edge = edge_add.u_for_edge,
                        v_for_edge =  edge_add.v_for_edge, 
                        length = edge_add.length, 
                        osmid = edge_add.osmid, 
                        name = edge_add.name, 
                        highway = edge_add.highway, 
                        oneway = edge_add.oneway
                        )
        
        self.nx_graph.add_edge(
                        u_for_edge = edge_add.v_for_edge,
                        v_for_edge =  edge_add.u_for_edge, 
                        length = edge_add.length, 
                        osmid = edge_add.osmid, 
                        name = edge_add.name, 
                        highway = edge_add.highway, 
                        oneway = edge_add.oneway
                        )

    def add_poi_to_graph(self):
        '''Add all POI to nx_graph.'''
        eges_to_add_list = self.poi.swifter.apply(
        self.add_single_poi_to_graph, axis =1)

        eges_to_add_list.swifter.apply(
        lambda e_list: self.add_two_ways_edges(e_list[0]))

        eges_to_add_list.swifter.apply(
        lambda e_list: self.add_two_ways_edges(e_list[1]))

    def get_s2cellids_for_poi(self, geometry: Any) -> Optional[Sequence[int]]:
      '''get cellids for POI. 
      Arguments:
        geometry: The geometry to which a cellids will be retrived.
      Returns:
        The the cellids.
      '''
      ""
      if isinstance(geometry, Point):
        return util.cellid_from_point(geometry, self.level)
      else:
        return util.cellid_from_polygon(geometry, self.level)

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
          (_graph\_node\_poi\_streets).
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
            map_logger.info("path {0} already exist.".format(path))

        # Write streets.
        pd_streets = copy.deepcopy(self.streets)
        
        path = self.get_valid_path(dir_name, '_streets', '.pkl')
        if not os.path.exists(path):
            pd_streets.to_pickle(path)
        else:
            map_logger.info("path {0} already exist.".format(path))

        # Write graph.
        base_filename = self.map_name.lower() + "_graph"
        path = os.path.join(dir_name, base_filename + ".gpickle")
        if not os.path.exists(path):
            nx.write_gpickle(self.nx_graph, path)
        else:
            map_logger.info("path {0} already exist.".format(path))

        # Write nodes.
        path = self.get_valid_path(dir_name, '_nodes', '.geojson')
        if not os.path.exists(path):
            self.nodes.to_file(path, driver='GeoJSON')
        else:
            map_logger.info("path {0} already exist.".format(path))

        # Write edges.
        path = self.get_valid_path(dir_name, '_edges', '.geojson')
        if not os.path.exists(path):
            self.edges.to_file(path, driver='GeoJSON')
        else:
            map_logger.info("path {0} already exist.".format(path))

    def load_map(self, dir_name: Text):
        '''Load POI from disk.'''

        # Load POI.
        path = self.get_valid_path(dir_name, '_poi', '.pkl')
        assert os.path.exists(
            path), "path {0} doesn't exist.".format(path)
        poi_pandas = pd.read_pickle(path)
        if 'cellids' in poi_pandas.columns:
            poi_pandas['s2cellids'] = poi_pandas['cellids'].apply(
                lambda x: util.s2cellids_from_cellids(x))
        self.poi = poi_pandas

        # Load streets.
        path = self.get_valid_path(dir_name, '_streets', '.pkl')
        assert os.path.exists(
            path), "path {0} doesn't exist.".format(path)
        streets_pandas = pd.read_pickle(path)
        self.streets = streets_pandas

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