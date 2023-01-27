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
import os
from shapely.geometry import LineString
from typing import Tuple, Sequence, Text, Optional


import util

NOT_PRIVIEW_TAGS = ['osmid', 'main_tag'] 
PIVOTS_COLORS = {"end_point":'red', "start_point":'green'} 
ICON_TAGS = ['tourism','amenity', 'shop', 'leisure']

dirname = os.path.dirname(__file__)
icon_dir = os.path.join(dirname, "./static/osm_icons")


onlyfiles = [
  f for f in os.listdir(icon_dir) if os.path.isfile(os.path.join(icon_dir, f))]

dict_icon_path = {x.replace('.svg', "").split('=')[1].replace('_', " "): x for x in onlyfiles}


def get_osm_map(
  entity, with_path, with_end_point, with_landmarks) -> Sequence[folium.Map]:
  '''Create the OSM maps.
  Arguments:
    gdf: the GeoDataFrame from which to create the OSM map.
    with_path: Add path to map.
    with_end_point: Add end point to map.
  Returns:
    OSM maps from the GeoDataFrame.
  '''
  goal_point = entity.geo_landmarks['end_point'].geometry
  start_point = entity.geo_landmarks['start_point'].geometry

  zoom_location = util.list_yx_from_point(goal_point)

  dist = util.get_distance_m(goal_point, start_point)
  if dist>1500:
    zoom_start = 13
  elif dist>1000:
    zoom_start = 14
  elif dist>800:
    zoom_start = 15
  else:
    zoom_start = 16

  # create a map
  map_osm = folium.Map(location=zoom_location,
                       zoom_start=zoom_start, tiles='OpenStreetMap')

  # draw the points
  for landmark_type, landmark in entity.geo_landmarks.items():
    if not with_end_point and landmark_type=='end_point':
      continue
    if not with_landmarks and 'pivot' in landmark_type:
      continue
    color = PIVOTS_COLORS[
      landmark_type] if landmark_type in PIVOTS_COLORS else 'black'
    add_landmark_to_osm_map(
      landmark=landmark,
      map_osm=map_osm,
      color=color,
      landmark_type=landmark_type
      )

  # add path between start and end point
  if with_path:
    line = LineString(entity.route)
    folium.GeoJson(data=line, style_function=lambda feature: {
      'fillColor': 'crimson',
      'color': 'crimson',
      'weight': 5,
      'fillOpacity': 1,
    }).add_to(map_osm)


  return map_osm

def get_landmark_desc_geo(landmark, landmark_type):
  if landmark.geometry is not None:  
    if 'pivot_view' in landmark.pivot_gdf:
      desc = landmark.pivot_gdf.pivot_view
      if landmark_type in ['end_point', 'start_point']:
        desc = landmark_type.replace("end_point", "Goal").replace("_", " ") + ": " + desc
      desc = "<b> " + desc.replace(";", "<br>").replace("_", "</b>", 1)
    else:
      desc = landmark.pivot_gdf.main_tag
    landmark_geom = util.list_yx_from_point(landmark.geometry)
    return landmark_geom, desc
  return None, "" 


def add_landmark_to_osm_map(landmark, map_osm, color, landmark_type):
  landmark_geom, desc = get_landmark_desc_geo(landmark, landmark_type)
  if landmark_geom:
    folium.Marker(
          landmark_geom,
          popup=desc,
          icon=folium.Icon(color=color)).add_to(map_osm)

def get_goal_icon(goal):
    for tag in ICON_TAGS:
      if tag not in goal.pivot_gdf:
        continue 
      value_goal = goal.pivot_gdf[tag]
      if value_goal is None:
        continue
      candidate_path = os.path.join(icon_dir, tag+"="+value_goal+".svg")
      if os.path.exists(candidate_path):
        return candidate_path
    return None
    

def get_maps_and_instructions(
  path: Text, with_path: bool = True, with_end_point: bool = True, 
  with_landmarks: bool = True, specific_sample: int = -1
) -> Sequence[Tuple[Sequence, str, Sequence[str], folium.Map, Optional[str]]]:
  '''Create the OSM maps and instructions.
  Arguments:
    path: The path from the start point to the goal location.
    with_path: Add path to map.
    with_end_point: Add end point to map.
  Returns:
    OSM maps from the GeoDataFrame.
  '''

  map_osms_instructions = []
  entities = util.load_entities(path)
  for entity_idx, entity in enumerate(entities):
    if specific_sample !=-1 and entity_idx!=specific_sample:
      continue
    map_osm = get_osm_map(
      entity, with_path, with_end_point, with_landmarks)
    features_list = []
    for feature_type, feature in entity.geo_features.items():
      features_list.append(feature_type + ": " + str(feature))

    landmark_list = []
    for landmark_type, landmark in entity.geo_landmarks.items():
      if landmark_type == 'end_point':
        goal_icon = get_goal_icon(landmark)
      landmark_list.append(str(landmark.main_tag))

    instruction = '; '.join(features_list) + '; '.join(landmark_list)
    map_osms_instructions.append(
      (map_osm, instruction, landmark_list, entity, goal_icon))

  return map_osms_instructions
