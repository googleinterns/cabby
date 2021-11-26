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

import util

NOT_PRIVIEW_TAGS = ['osmid', 'main_tag'] 
PIVOTS_COLORS = {"end_point":'red', "start_point":'green'} 


def get_osm_map(entity) -> Sequence[folium.Map]:
  '''Create the OSM maps.
  Arguments:
    gdf: the GeoDataFrame from which to create the OSM map.
  Returns:
    OSM maps from the GeoDataFrame.
  '''
  goal_point = entity.geo_landmarks['end_point'].geometry
  start_point = entity.geo_landmarks['start_point'].geometry

  mid_point = util.midpoint(goal_point, start_point)
  zoom_location = util.list_yx_from_point(mid_point)

  dist = util.get_distance_m(goal_point, start_point)
  if dist>2500:
    zoom_start = 12
  elif dist>1800:
    zoom_start = 13
  elif dist>400:
    zoom_start = 14
  else:
    zoom_start = 15
  # create a map
  map_osm = folium.Map(location=zoom_location,
                       zoom_start=zoom_start  , tiles='OpenStreetMap')

  # draw the points
  for landmark_type, landmark in entity.geo_landmarks.items():
    if landmark_type in PIVOTS_COLORS:
      continue
    if landmark.geometry is not None:  
      if 'pivot_view' in landmark.pivot_gdf:
        desc = landmark.pivot_gdf.pivot_view.replace(";", "<br>") 
        desc = "<b> " + desc.replace("_", "</b>", 1)
      else:
        desc = landmark.pivot_gdf.main_tag
      landmark_geom = util.list_yx_from_point(landmark.geometry)
      color = 'black'
      folium.Marker(
        landmark_geom,
        popup=desc,
        icon=folium.Icon(color=color)).add_to(map_osm)

  # add start point
  add_landmark_to_osm_map(
    landmark=entity.geo_landmarks['start_point'],
    map_osm=map_osm,
    color=PIVOTS_COLORS['start_point'])

  # add end point
  add_landmark_to_osm_map(
    landmark=entity.geo_landmarks['end_point'],
    map_osm=map_osm,
    color=PIVOTS_COLORS['end_point'])


  # line = LineString(entity.route)
  # folium.GeoJson(data=line, style_function=lambda feature: {
  #   'fillColor': 'crimson',
  #   'color': 'crimson',
  #   'weight': 5,
  #   'fillOpacity': 1,
  # }).add_to(map_osm)

  return map_osm

def add_landmark_to_osm_map(
  landmark, map_osm, color):
  if landmark.geometry is not None:  
    if 'pivot_view' in landmark.pivot_gdf:
      desc = landmark.pivot_gdf.pivot_view.replace(";", "<br>") 
      desc = "<b> " + desc.replace("_", "</b>", 1)
    else:
      desc = landmark.pivot_gdf.main_tag
    landmark_geom = util.list_yx_from_point(landmark.geometry)
    folium.Marker(
      landmark_geom,
      popup=desc,
      icon=folium.Icon(color=color)).add_to(map_osm)

def get_maps_and_instructions(path: Text
                              ) -> Sequence[Tuple[folium.Map, str]]:
  '''Create the OSM maps and instructions.
  Arguments:
    path: The path from the start point to the goal location.
  Returns:
    OSM maps from the GeoDataFrame.
  '''

  map_osms_instructions = []
  entities = util.load_entities(path)
  for entity in entities:
    map_osm = get_osm_map(entity)
    features_list = []
    for feature_type, feature in entity.geo_features.items():
      features_list.append(feature_type + ": " + str(feature))

    landmark_list = []
    for landmark_type, landmark in entity.geo_landmarks.items():
      landmark_list.append(str(landmark.main_tag))

    instruction = '; '.join(features_list) + '; '.join(landmark_list)
    map_osms_instructions.append((map_osm, instruction, landmark_list, entity))

  return map_osms_instructions
