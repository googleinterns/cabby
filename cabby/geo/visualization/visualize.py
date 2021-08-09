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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd() ))))
from cabby.geo import util
from cabby.geo import walk
from cabby.geo import geo_item

 
def get_osm_map(entity: geo_item.GeoEntity) -> Sequence[folium.Map]:
    '''Create the OSM maps.
    Arguments:
      gdf: the GeoDataFrame from which to create the OSM map.
    Returns:
      OSM maps from the GeoDataFrame.
    '''

    mid_point = util.midpoint(
      entity.geo_landmarks['end_point'].geometry,
      entity.geo_landmarks['start_point'].geometry)
    zoom_location = util.list_yx_from_point(mid_point)

    # create a map
    map_osm = folium.Map(location=zoom_location,
                         zoom_start=15, tiles='OpenStreetMap')

    # draw the points
    colors = [
      'red', 'green', 'pink', 'blue', 'black', 'purple', 'orange']
    for landmark_type, landmark in entity.geo_landmarks.items():
      if landmark.geometry is not None:
        landmark_geom = util.list_yx_from_point(landmark.geometry)
        folium.Marker(
          landmark_geom,
          popup=f'{landmark_type.replace("_", " ")}: {landmark.main_tag.replace("_", " ")}',
          icon=folium.Icon(color=colors.pop(0))).add_to(map_osm)
    #
    # lat_lng_list = []
    # for coord in entity.route.coords:
    #     lat_lng_list.append([coord[1], coord[0]])
    #
    # for index, coord_lat_lng in enumerate(lat_lng_list):
    #     folium.Circle(location = coord_lat_lng,
    #                    radius = 5,
    #                   color='crimson',
    #                   ).add_to(map_osm)
    #

    line = LineString(entity.route)
    folium.GeoJson(data=line, style_function=lambda feature: {
        'fillColor': 'crimson',
        'color' : 'crimson',
        'weight' : 5,
        'fillOpacity' : 1,
        }).add_to(map_osm)

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
      features_list = []
      for feature_type, feature in entity.geo_features.items():
        features_list.append(feature_type + ": " + str(feature))

      landmark_list = []
      for landmark_type, landmark in entity.geo_landmarks.items():
        landmark_list.append(landmark_type + ": " + str(landmark.main_tag))

      instruction = '; '.join(features_list) + '; '.join(landmark_list)
      map_osms_instructions.append((map_osm, instruction))

    return map_osms_instructions

