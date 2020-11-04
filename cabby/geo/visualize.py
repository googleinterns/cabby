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

'''Library to support geographical visualization.'''

import folium
import geopandas as gpd
import pandas as pd
import shapely.geometry as geom
from shapely.geometry import Polygon, Point, LineString
from typing import Tuple, Sequence, Optional, Dict, Text
import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd() )))
from cabby.geo import util
from cabby.geo import walk
from cabby.rvs import item

 
def get_osm_map(entity: item.RVSPath) -> Sequence[folium.Map]:
    '''Create the OSM maps.
    Arguments:
      gdf: the GeoDataFrame from which to create the OSM map.
    Returns:
      OSM maps from the GeoDataFrame.
    '''

    mid_point = util.midpoint(
      entity.end_point.geometry, entity.start_point.geometry)
    zoom_location = util.list_yx_from_point(mid_point)

    # create a map
    map_osm = folium.Map(location=zoom_location,
                         zoom_start=15, tiles='OpenStreetMap')

    # draw the points
    start_point_geom = util.list_yx_from_point(entity.start_point.geometry)
    end_point_geom = util.list_yx_from_point(entity.end_point.geometry)
    main_pivot_geom = util.list_yx_from_point(
      entity.main_pivot.geometry)
    near_pivot_geom = util.list_yx_from_point(
      entity.near_pivot.geometry)

    folium.Marker(
      start_point_geom, popup=f'start: {entity.start_point.main_tag}' , 
      icon=folium.Icon(color='pink')).add_to(map_osm)
    folium.Marker(end_point_geom, popup=f'end: {entity.end_point.main_tag}', 
      icon=folium.Icon(color='black')).add_to(map_osm)
    folium.Marker(main_pivot_geom, 
      popup=f'main pivot: {entity.main_pivot.main_tag}', icon=folium.Icon(
        color='orange', icon='info-sign')).add_to(map_osm)
    folium.Marker(
      near_pivot_geom, popup=f'near pivot: {entity.near_pivot.main_tag}', 
       icon=folium.Icon(color='red', icon='info-sign')).add_to(map_osm)
    
    if not entity.beyond_pivot.geometry is None: 
      beyond_pivot_geom = util.list_yx_from_point(entity.beyond_pivot.geometry)
      folium.Marker(
        beyond_pivot_geom, popup=f'beyond pivot: {entity.beyond_pivot.main_tag}', 
        icon=folium.Icon(
          color='green', icon='info-sign')).add_to(map_osm)

    lat_lng_list = []
    for coord in entity.route.coords:
        lat_lng_list.append([coord[1], coord[0]])

    for index, coord_lat_lng in enumerate(lat_lng_list):
        folium.Circle(location = coord_lat_lng,
                       radius = 5,
                      color='crimson',
                      ).add_to(map_osm)

    return map_osm


def get_maps_and_instructions(path: Text
) -> Sequence[Tuple[folium.Map, str]]:
    '''Create the OSM maps and instructions.
    Arguments:
      path: The path from the start point to the goal location.
    Returns:
      OSM maps from the GeoDataFrame.
    '''

    map_osms_instructions = []
    entities = walk.load_entities(path)
    for entity in entities:
      map_osm = get_osm_map(entity)
      map_osms_instructions.append((map_osm, entity.instructions))

    return map_osms_instructions

